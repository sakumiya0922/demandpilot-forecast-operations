import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import psycopg
import pytest

from daily.daily_batch import run_daily_batch
from weekly.forecast import save_forecasts
from evaluation.evaluate import compare_actuals, save_evaluation


def prepare_week(database, source_dir, complete=True):
    features=pd.DataFrame({'stock_code':['A','B'],'sales_sum_7':[8.,3.],
                           'sales_sum_28':[32.,12.],'sales_std_28':[1.,1.],'month':[6,6]})
    identifier=uuid4()
    save_forecasts(database,features,[9,2],'2011-06-05','learning-run','models:/model',identifier)
    for number,day,quantity in [(2,'2011-06-06',4),(3,'2011-06-12',6)]:
        path=source_dir/(day+'.csv')
        pd.DataFrame([{'source_row':number,'InvoiceNo':'123','StockCode':'A','Description':'test',
                       'Quantity':quantity,'InvoiceDate':day+' 10:00:00','UnitPrice':'1.0','CustomerID':'','Country':'UK'}]).to_csv(path,index=False)
        if complete or day=='2011-06-06':
            run_daily_batch(path,day,database)
    return identifier


def test_sql_actuals_baseline_saved_at_prediction_and_zero_sales(tmp_path,database):
    identifier=prepare_week(database,tmp_path)
    with psycopg.connect(database) as connection:
        before=connection.execute('SELECT * FROM weekly_forecasts ORDER BY stock_code').fetchall()
        # 予測前の履歴を後日修正しても、ベースラインは当時保存した値を使用する。
        connection.execute("INSERT INTO daily_sales VALUES ('2011-06-05','A',999)")
    frame,scores=compare_actuals(database,identifier,'2011-06-12',tmp_path)
    assert frame.actual.tolist()==[10,0]
    assert frame.baseline.tolist()==[8,3]
    assert pd.isna(frame.loc[frame.stock_code.eq('B'),'lightgbm_wape']).all()
    assert scores['lightgbm']['wape']==pytest.approx(.3)
    assert scores['baseline']['wape']==pytest.approx(.5)
    with psycopg.connect(database) as connection:
        assert connection.execute('SELECT * FROM weekly_forecasts ORDER BY stock_code').fetchall()==before


def test_missing_import_is_not_evaluated_as_zero(tmp_path,database):
    identifier=prepare_week(database,tmp_path,complete=False)
    with pytest.raises(ValueError,match='不一致'):
        compare_actuals(database,identifier,'2011-06-12',tmp_path)


def test_week_must_finish_before_evaluation(tmp_path,database):
    identifier=prepare_week(database,tmp_path)
    with pytest.raises(ValueError,match='終了'):
        compare_actuals(database,identifier,'2011-06-11',tmp_path)


def test_unknown_prediction_id_is_rejected(tmp_path,database):
    with pytest.raises(ValueError,match='ありません'):
        compare_actuals(database,uuid4(),'2011-06-12',tmp_path)


def test_common_unknown_week_is_not_evaluated(tmp_path,database):
    features=pd.DataFrame({'stock_code':['A'],'sales_sum_7':[8.],'sales_sum_28':[32.],
                           'sales_std_28':[1.],'month':[12]})
    identifier=uuid4()
    save_forecasts(database,features,[9],'2010-12-26','learning-run','models:/model',identifier)
    with pytest.raises(ValueError,match='空白'):
        compare_actuals(database,identifier,'2011-01-02',tmp_path)


def test_reevaluation_creates_new_snapshot_without_overwriting(tmp_path,database):
    identifier=prepare_week(database,tmp_path)
    frame,scores=compare_actuals(database,identifier,'2011-06-12',tmp_path)
    first=save_evaluation(frame,scores,'2011-06-12',tmp_path,tmp_path/'results')
    old=(first/'comparison.csv').read_bytes()
    # 訂正CSVで再取込した後も、最初の評価ファイルを保持する。
    path=tmp_path/'2011-06-12.csv'
    raw=pd.read_csv(path,keep_default_na=False)
    raw['Quantity']=16
    raw.to_csv(path,index=False)
    run_daily_batch(path,'2011-06-12',database)
    revised,new_scores=compare_actuals(database,identifier,'2011-06-12',tmp_path)
    second=save_evaluation(revised,new_scores,'2011-06-12',tmp_path,tmp_path/'results')
    assert first!=second and (first/'comparison.csv').read_bytes()==old
    assert (second/'comparison.csv').read_bytes()!=old
    assert (first/'comparison.png').is_file() and (second/'report.md').is_file()
    assert json.loads((first/'metrics.json').read_text())['scores']['lightgbm']['wape']==pytest.approx(.3)
