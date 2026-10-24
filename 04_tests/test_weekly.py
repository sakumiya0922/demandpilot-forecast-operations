from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg
import pytest

from daily.daily_batch import run_daily_batch
from training.training_data import build_training_data, FEATURE_COLUMNS
from weekly.forecast import prediction_features, save_forecasts, verified_history, forecast


def daily_data(start='2011-04-01', end='2011-06-20'):
    return pd.DataFrame([(code,day,float(day.day)) for code in ['A','B'] for day in pd.date_range(start,end)],
                        columns=['stock_code','sales_date','quantity'])


def test_inference_features_match_training_and_ignore_future():
    daily=daily_data()
    train=build_training_data(daily,['A','B'],'2011-04-01','2011-06-20')
    expected=train.loc[train.forecast_origin.eq('2011-06-05'),FEATURE_COLUMNS].reset_index(drop=True)
    actual=prediction_features(daily,['A','B'],'2011-06-05')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-9)
    daily.loc[daily.sales_date.gt('2011-06-05'),'quantity']=999999
    pd.testing.assert_frame_equal(actual,prediction_features(daily,['A','B'],'2011-06-05'))


def test_origin_and_unknown_history_are_rejected():
    with pytest.raises(ValueError,match='日曜日'):
        prediction_features(daily_data(),['A','B'],'2011-06-06')
    with pytest.raises(ValueError,match='空白'):
        prediction_features(daily_data('2010-12-01','2011-02-01'),['A','B'],'2011-01-23')


def test_reforecast_retains_history_and_duplicate_id_does_not_overwrite(database):
    features=prediction_features(daily_data(),['A','B'],'2011-06-05')
    first,second=uuid4(),uuid4()
    save_forecasts(database,features,[10,20],'2011-06-05','run-a','models:/model-a',first)
    save_forecasts(database,features,[30,40],'2011-06-05','run-b','models:/model-b',second)
    with pytest.raises(psycopg.errors.UniqueViolation):
        save_forecasts(database,features,[999,999],'2011-06-05','run-a','models:/model-a',first)
    with psycopg.connect(database) as connection:
        rows=connection.execute('SELECT predicted_quantity,mlflow_run_id,sales_sum_7 FROM weekly_forecasts ORDER BY predicted_quantity').fetchall()
    assert [float(r[0]) for r in rows]==[10,20,30,40]
    assert [r[1] for r in rows]==['run-a','run-a','run-b','run-b']
    assert all(float(r[2])==features.sales_sum_7.iloc[0] for r in rows)


def test_one_bad_forecast_rolls_back_entire_run(database):
    features=prediction_features(daily_data(),['A','B'],'2011-06-05')
    features.loc[1,'stock_code']=''
    with pytest.raises(psycopg.errors.CheckViolation):
        save_forecasts(database,features,[10,20],'2011-06-05','run','models:/model',uuid4())
    with psycopg.connect(database) as connection:
        assert connection.execute('SELECT COUNT(*) FROM weekly_forecasts').fetchone()[0]==0


def test_nonfinite_predictions_are_rejected(database):
    features=prediction_features(daily_data(),['A','B'],'2011-06-05')
    with pytest.raises(ValueError,match='予測値'):
        save_forecasts(database,features,[10,np.nan],'2011-06-05','run','models:/model',uuid4())


def test_csv_db_mismatch_stops_prediction_and_future_rows_are_ignored(tmp_path,database):
    for number,day in [(2,'2011-05-09'),(3,'2011-06-05'),(4,'2011-06-06')]:
        pd.DataFrame([{'source_row':number,'InvoiceNo':'123','StockCode':'A','Description':'test',
                       'Quantity':5,'InvoiceDate':day+' 10:00:00','UnitPrice':'1.0','CustomerID':'','Country':'UK'}]).to_csv(tmp_path/(day+'.csv'),index=False)
    run_daily_batch(tmp_path/'2011-05-09.csv','2011-05-09',database)
    with pytest.raises(ValueError,match='不一致'):
        verified_history(database,'2011-06-05',tmp_path)
    run_daily_batch(tmp_path/'2011-06-05.csv','2011-06-05',database)
    run_daily_batch(tmp_path/'2011-06-06.csv','2011-06-06',database)
    result=verified_history(database,'2011-06-05',tmp_path)
    assert len(result)==2 and result.sales_date.max().isoformat()=='2011-06-05'
    with psycopg.connect(database) as connection:
        connection.execute('UPDATE daily_sales SET quantity=100 WHERE sales_date=%s',('2011-06-05',))
    with pytest.raises(ValueError,match='不一致'):
        verified_history(database,'2011-06-05',tmp_path)


OLD_TABLE='''CREATE TABLE weekly_forecasts (
    week_start DATE NOT NULL, stock_code TEXT NOT NULL, forecast_origin DATE NOT NULL,
    predicted_quantity NUMERIC NOT NULL CHECK (predicted_quantity>=0), model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(week_start,stock_code))'''


def test_empty_old_forecast_schema_migrates_and_reruns_safely(database):
    migration=(Path(__file__).parents[1]/'02_model/weekly/migrate_forecasts.sql').read_text()
    with psycopg.connect(database,autocommit=True) as connection:
        # 旧形式の再現前に、新設した空の評価明細だけを隔離スキーマ内で外す。
        connection.execute('DROP TABLE weekly_evaluation_items')
        connection.execute('DROP TABLE weekly_forecasts')
        connection.execute(OLD_TABLE)
        connection.execute(migration)
        connection.execute(migration)
    features=prediction_features(daily_data(),['A','B'],'2011-06-05')
    save_forecasts(database,features,[10,20],'2011-06-05','run','models:/model',uuid4())


def test_migration_refuses_to_discard_old_predictions(database):
    migration=(Path(__file__).parents[1]/'02_model/weekly/migrate_forecasts.sql').read_text()
    with psycopg.connect(database,autocommit=True) as connection:
        # 旧形式の再現前に、新設した空の評価明細だけを隔離スキーマ内で外す。
        connection.execute('DROP TABLE weekly_evaluation_items')
        connection.execute('DROP TABLE weekly_forecasts')
        connection.execute(OLD_TABLE)
        connection.execute("INSERT INTO weekly_forecasts(week_start,stock_code,forecast_origin,predicted_quantity,model_name) VALUES ('2011-06-06','A','2011-06-05',10,'old')")
        with pytest.raises(psycopg.errors.RaiseException,match='旧形式'):
            connection.execute(migration)
        connection.execute('ROLLBACK')
        assert connection.execute('SELECT COUNT(*) FROM weekly_forecasts').fetchone()[0]==1


def test_model_trained_on_future_is_rejected(tmp_path):
    import mlflow
    previous=mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri('sqlite:///'+str(tmp_path/'mlflow.db'))
        experiment_id=mlflow.create_experiment('future-model',artifact_location=(tmp_path/'artifacts').as_uri())
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_dict({'run_id':run.info.run_id,'features':FEATURE_COLUMNS,
                             'final_training':{'last_label_date':'2011-06-12'}},'conditions.json')
            identifier=run.info.run_id
        with pytest.raises(ValueError,match='未来'):
            forecast('unused','2011-06-05',identifier,'unused','unused',tmp_path)
    finally:
        mlflow.set_tracking_uri(previous)


def test_known_product_with_no_sales_in_last_28_days_has_zero_features(tmp_path,database):
    # Bは期間前にのみ販売がある。対象期間のCSV・DB照合はそのまま行う。
    for number,day,code in [(2,'2011-05-08','B'),(3,'2011-05-09','A'),
                            (4,'2011-06-05','A'),(5,'2011-06-06','C')]:
        path=tmp_path/(day+'.csv')
        pd.DataFrame([{'source_row':number,'InvoiceNo':'123','StockCode':code,'Description':'test',
            'Quantity':5,'InvoiceDate':day+' 10:00:00','UnitPrice':'1','CustomerID':'','Country':'UK'}]).to_csv(path,index=False)
        run_daily_batch(path,day,database)
    original=verified_history(database,'2011-06-05',tmp_path)
    daily=verified_history(database,'2011-06-05',tmp_path,['A','B'])
    features=prediction_features(daily,['A','B'],'2011-06-05')
    b=features.loc[features.stock_code.eq('B')].iloc[0]
    assert b.sales_sum_7==b.sales_sum_28==b.sales_std_28==0
    expected=prediction_features(original,['A'],'2011-06-05')
    pd.testing.assert_frame_equal(features.loc[features.stock_code.eq('A')].reset_index(drop=True),expected)
    # 未来にだけ存在する品番は、過去の既存商品とみなさない。
    future=verified_history(database,'2011-06-05',tmp_path,['A','C'])
    with pytest.raises(ValueError,match='実績のない'):
        prediction_features(future,['A','C'],'2011-06-05')
    with psycopg.connect(database) as c:
        assert c.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]==4
        assert c.execute('SELECT COUNT(*) FROM daily_sales').fetchone()[0]==4
