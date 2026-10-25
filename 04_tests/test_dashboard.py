from pathlib import Path
from uuid import uuid4

import pandas as pd
import psycopg
from streamlit.testing.v1 import AppTest

from daily.daily_batch import run_daily_batch
from weekly.forecast import save_forecasts

APP=Path(__file__).parents[1]/'03_app/app.py'


def set_up(database,tmp_path,monkeypatch,complete=False):
    monkeypatch.setenv('DATABASE_URL',database)
    monkeypatch.setenv('SOURCE_DIR',str(tmp_path))
    features=pd.DataFrame({'stock_code':['A','B'],'sales_sum_7':[8.,3.],
        'sales_sum_28':[32.,12.],'sales_std_28':[1.,1.],'month':[6,6]})
    identifier=uuid4()
    save_forecasts(database,features,[9,2],'2011-06-05','learning-run','models:/model',identifier)
    for number,day,quantity in [(2,'2011-06-06',4),(3,'2011-06-12',6)]:
        path=tmp_path/(day+'.csv')
        pd.DataFrame([{'source_row':number,'InvoiceNo':'123','StockCode':'A','Description':'test',
            'Quantity':quantity,'InvoiceDate':day+' 10:00:00','UnitPrice':'1.0','CustomerID':'','Country':'UK'}]).to_csv(path,index=False)
        if complete or day=='2011-06-06': run_daily_batch(path,day,database)
    return str(identifier),features


def test_dashboard_handles_no_forecasts(database,monkeypatch):
    monkeypatch.setenv('DATABASE_URL',database)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    assert any('保存済みの予測がありません' in i.value for i in app.info)


def test_pending_actuals_do_not_show_wape_or_zero_actual(database,tmp_path,monkeypatch):
    set_up(database,tmp_path,monkeypatch)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    assert any('実績の確認待ち' in i.value for i in app.info)
    assert not app.metric
    app.radio(key='view').set_value('商品別分析').run()
    assert not app.exception
    assert next(item.value for item in app.metric if item.label=='実績数量')=='未確認'


def test_evaluated_metrics_product_and_run_selection_keep_db_unchanged(database,tmp_path,monkeypatch):
    identifier,features=set_up(database,tmp_path,monkeypatch,complete=True)
    next_id=str(uuid4())
    save_forecasts(database,features,[5,6],'2011-06-12','learning-run','models:/model',next_id)
    with psycopg.connect(database) as connection:
        before={table:connection.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()
                for table in ['transactions','daily_sales','weekly_forecasts']}
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    app.selectbox(key='week').select(pd.Timestamp('2011-06-06').date()).run()
    assert not app.exception
    metrics=dict(zip(['LightGBM WAPE','ベースライン WAPE','LightGBM MAE','ベースライン MAE'],[item.value for item in app.metric]))
    assert metrics['LightGBM WAPE']=='30.00%' and metrics['ベースライン WAPE']=='50.00%'
    assert app.metric[0].delta.startswith('+20.00 pp')
    app.radio(key='view').set_value('商品別分析').run()
    app.selectbox(key='product').select('B').run()
    assert not app.exception
    assert next(item.value for item in app.metric if item.label=='実績数量')=='0.00'
    app.selectbox(key='week').select(pd.Timestamp('2011-06-13').date()).run()
    assert not app.exception
    assert [m.label for m in app.metric]==['予測数量','実績数量','絶対誤差']
    with psycopg.connect(database) as connection:
        after={table:connection.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()
               for table in ['transactions','daily_sales','weekly_forecasts']}
    assert before==after


def test_inconsistent_actuals_are_shown_as_pending(database,tmp_path,monkeypatch):
    set_up(database,tmp_path,monkeypatch,complete=True)
    with psycopg.connect(database) as connection:
        connection.execute("UPDATE daily_sales SET quantity=99 WHERE sales_date='2011-06-12'")
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    assert any('不一致' in i.value for i in app.info)
    assert not app.metric


def test_model_training_snapshot_is_linked_and_verified(tmp_path,monkeypatch):
    import hashlib
    import json
    import sqlite3
    from app import load_model_info, period_text

    monkeypatch.setenv('MLFLOW_DATA_DIR',str(tmp_path))
    artifact=tmp_path/'artifacts/model/artifacts'
    (artifact/'data').mkdir(parents=True)
    training=pd.DataFrame({
        'stock_code':['A','A'],'forecast_origin':['2011-01-30','2011-05-22'],
        'history_start':['2011-01-03','2011-04-25'],
        'target_end':['2011-02-06','2011-05-29'],
        'sales_sum_7':[1,2],'sales_sum_28':[4,8],'sales_std_28':[1,2],'month':[1,5]})
    raw=training.to_csv(index=False).encode()
    (artifact/'data/training.csv').write_bytes(raw)
    conditions={'run_id':'model-run','model_uri':'models:/actual-model',
                'data_sha256':hashlib.sha256(raw).hexdigest()}
    (artifact/'conditions.json').write_text(json.dumps(conditions))
    with sqlite3.connect(tmp_path/'mlflow.db') as c:
        c.execute('CREATE TABLE runs (run_uuid TEXT,artifact_uri TEXT,start_time INTEGER,status TEXT)')
        c.execute('INSERT INTO runs VALUES (?,?,?,?)',('model-run',artifact.as_uri(),1000,'FINISHED'))
    before=(tmp_path/'mlflow.db').read_bytes()
    info=load_model_info('model-run','models:/actual-model')
    assert not info['issue']
    assert period_text(info)=='2011-01-30 ～ 2011-05-22'
    assert len(info['training'])==2 and info['status']=='FINISHED'
    assert load_model_info('model-run','models:/other-model')['training'] is None
    assert period_text(load_model_info('missing','models:/actual-model'))=='未記録'
    (artifact/'data/training.csv').write_bytes(raw+b'\n')
    assert load_model_info('model-run','models:/actual-model')['training'] is None
    assert (tmp_path/'mlflow.db').read_bytes()==before


def test_top_error_navigation_and_execution_information(database,tmp_path,monkeypatch):
    set_up(database,tmp_path,monkeypatch,complete=True)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    # Aの絶対誤差1よりBの誤差2が大きい。誤差の大きい品番から詳細へ移動する。
    assert app.selectbox(key='top_product').value=='B'
    button=next(button for button in app.button if '商品別分析へ' in button.label)
    button.click().run()
    assert not app.exception
    assert app.radio(key='view').value=='商品別分析'
    assert app.selectbox(key='product').value=='B'
    assert next(item.value for item in app.metric if item.label=='絶対誤差')=='2.00'
    app.radio(key='view').set_value('実行情報').run()
    assert not app.exception
    records=app.dataframe[0].value.set_index('項目')['記録']
    assert records['学習時のMLflow実行ID']=='learning-run'
    assert records['販売実績のDB最終更新日時']=='未記録'
    assert records['現在の実績照合']=='一致確認済み'


def test_overview_is_compact_and_one_week_has_no_charts(database,tmp_path,monkeypatch):
    set_up(database,tmp_path,monkeypatch,complete=True)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    assert [m.label for m in app.metric]==['LightGBM','ベースライン','LightGBM','ベースライン']
    assert any('kpi-group-title' in m.value and '>WAPE<' in m.value for m in app.markdown)
    assert any('kpi-group-title' in m.value and '>MAE<' in m.value for m in app.markdown)
    assert [s.key for s in app.sidebar.selectbox]==['week']
    assert not app.sidebar.title and not app.sidebar.info and not app.sidebar.caption
    assert not app.get('arrow_vega_lite_chart')
    assert list(app.dataframe[0].value.columns)==['品番','実績数量','予測数量','絶対誤差','WAPE']
    assert app.dataframe[0].value['品番'].tolist()==['B','A']
    assert not any('models:/' in m.value or 'learning-run' in m.value for m in app.markdown)
    app.radio(key='view').set_value('商品別分析').run()
    assert not app.exception and len(app.get('arrow_vega_lite_chart'))==1
    assert [m.label for m in app.metric]==['予測数量','実績数量','絶対誤差']
    assert app.dataframe[0].value['学習時（中央値）'].tolist()==['未記録']*3
    assert app.selectbox(key='product').value=='B'
    assert app.dataframe[0].value['予測時'].tolist()==['3.00','12.00','1.00']


def test_multiple_evaluated_weeks_show_trends(database,tmp_path,monkeypatch):
    _,features=set_up(database,tmp_path,monkeypatch,complete=True)
    save_forecasts(database,features,[5,6],'2011-06-12','learning-run','models:/model',uuid4())
    for number,day in [(4,'2011-06-13'),(5,'2011-06-19')]:
        path=tmp_path/(day+'.csv')
        pd.DataFrame([{'source_row':number,'InvoiceNo':'123','StockCode':'A','Description':'test',
            'Quantity':3,'InvoiceDate':day+' 10:00:00','UnitPrice':'1','CustomerID':'','Country':'UK'}]).to_csv(path,index=False)
        run_daily_batch(path,day,database)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    assert len(app.get('arrow_vega_lite_chart'))==1
    app.radio(key='view').set_value('商品別分析').run()
    assert not app.exception
    assert len(app.get('arrow_vega_lite_chart'))==1


def test_execution_history_selection_survives_navigation(database,tmp_path,monkeypatch):
    old_id,features=set_up(database,tmp_path,monkeypatch,complete=True)
    new_id=str(uuid4())
    save_forecasts(database,features,[8,0],'2011-06-05','learning-run','models:/model',new_id)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    app.radio(key='view').set_value('実行情報').run()
    app.selectbox(key='prediction').select(old_id).run()
    assert not app.exception
    assert app.session_state['selected_prediction']==old_id
    app.radio(key='view').set_value('全体評価').run()
    assert not app.exception
    assert app.metric[0].value=='30.00%'
    app.radio(key='view').set_value('実行情報').run()
    assert not app.exception
    assert app.selectbox(key='prediction').value==old_id


def test_nine_week_actuals_without_historical_predictions(database,tmp_path,monkeypatch):
    from datetime import date
    from app import load_product_weekly_sales, product_sales_chart
    _,features=set_up(database,tmp_path,monkeypatch,complete=True)
    # 期間外の実績、同週内の複数日、別品番だけが売れた週を含める。
    with psycopg.connect(database) as c:
        with c.cursor() as cursor:
            cursor.executemany('INSERT INTO daily_sales VALUES (%s,%s,%s)',[
                ('2011-04-04','A',999),('2011-04-11','A',2),('2011-04-17','A',3),
                ('2011-04-18','B',4),('2011-04-25','A',8)])
    frame=load_product_weekly_sales(database,'A',date(2011,6,6))
    assert len(frame)==9
    assert frame.week.tolist()==list(pd.date_range('2011-04-11',periods=9,freq='7D').date)
    assert frame.actual.iloc[0]==5 and frame.actual.iloc[1]==0
    assert pd.isna(frame.actual.iloc[3])  # 全商品記録なしを販売ゼロにしない
    assert frame.actual.iloc[-1]==10
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    app.radio(key='view').set_value('商品別分析').run()
    assert not app.exception and len(app.get('arrow_vega_lite_chart'))==1
    weekly=pd.DataFrame({'stock_code':['A'],'week':['2011-06-06'],'lightgbm':[9.], 'actual':[10.]})
    spec=product_sales_chart(frame,weekly,'A',date(2011,6,6)).to_dict()
    values=[row for data in spec['datasets'].values() for row in data]
    predictions=[row for row in values if row.get('種類')=='LightGBM']
    assert len(predictions)==1 and predictions[0]['週開始日']=='2011-06-06'
    actuals=[row for row in values if row.get('種類')=='実績']
    assert len(actuals)==9
    assert spec['layer'][0]['mark']['type']=='rect'


def test_partial_week_and_unverified_actuals_stay_missing(database,tmp_path,monkeypatch):
    from datetime import date
    from app import load_product_weekly_sales, product_sales_chart
    set_up(database,tmp_path,monkeypatch)
    frame=load_product_weekly_sales(database,'A',date(2011,6,6))
    assert pd.isna(frame.actual.iloc[-1])
    # DBに週合計が存在しても、既存照合で未確認なら表示しない。
    frame.loc[8,'actual']=99
    weekly=pd.DataFrame({'stock_code':['A'],'week':['2011-06-06'],'lightgbm':[9.], 'actual':[float('nan')]})
    spec=product_sales_chart(frame,weekly,'A',date(2011,6,6)).to_dict()
    values=[row for data in spec['datasets'].values() for row in data]
    assert next(row['数量'] for row in values if row.get('種類')=='実績' and row['週開始日']=='2011-06-06') is None


def test_product_wape_ranking_and_zero_actuals():
    from app import error_ranking, metric_change
    data=pd.DataFrame({'stock_code':['A','B','C','D','E','F','Z'],
        'actual':[1000,100,100,100,100,1,0], 'lightgbm':[1100,120,119,118,117,11,3],
        'lightgbm_abs_error':[100,20,19,18,17,10,3]})
    absolute=error_ranking(data,'絶対誤差が大きい順')
    relative=error_ranking(data,'WAPEが大きい順')
    assert absolute.head(5).stock_code.tolist()==['A','B','C','D','E']
    assert relative.head(5).stock_code.tolist()==['F','B','C','D','E']
    assert relative.product_wape.iloc[0]==1000
    zero=error_ranking(data.tail(1),'WAPEが大きい順')
    assert pd.isna(zero.product_wape.iloc[0])
    assert metric_change(.3,.5,100,'pp')=='+20.00 pp 改善'
    assert metric_change(3,2,1,'個')=='-1.00 個 悪化'
    assert metric_change(2,2,1,'個')=='+0.00 個 同等'
    assert metric_change(None,None,100,'pp') is None


def test_mae_sorting_zero_wape_and_selection_event(database,tmp_path,monkeypatch):
    import json
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
    set_up(database,tmp_path,monkeypatch,complete=True)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    metrics=dict(zip(['LightGBM WAPE','ベースライン WAPE','LightGBM MAE','ベースライン MAE'],app.metric))
    assert metrics['LightGBM MAE'].value=='1.50'
    assert metrics['ベースライン MAE'].value=='2.50'
    assert metrics['LightGBM MAE'].delta=='+1.00 個 改善'
    assert pd.isna(app.dataframe[0].value.WAPE.iloc[0])
    display=convert_arrow_bytes_to_pandas_df(app.dataframe[0].proto.styler.display_values)
    assert display.WAPE.tolist()==['—','10.00%']
    assert 'background-color' in app.dataframe[0].proto.styler.styles
    # AppTest 1.42には行クリックAPIがないため、標準ウィジェットの選択イベントを送る。
    # 行番号1は初期の絶対誤差順ではA。登録済みon_selectコールバックも実行する。
    states=app._tree.get_widget_states()
    event=states.widgets.add()
    event.id=app.dataframe[0].proto.id
    event.string_value=json.dumps({'selection':{'rows':[1],'columns':[]}})
    app._run(widget_state=states)
    assert not app.exception
    assert app.selectbox(key='top_product').value=='A'
    next(b for b in app.button if '商品別分析へ' in b.label).click().run()
    assert not app.exception and app.selectbox(key='product').value=='A'
    app.radio(key='view').set_value('全体評価').run()
    assert not app.exception and app.selectbox(key='top_product').value=='A'
    app.selectbox(key='error_order').select('WAPEが大きい順').run()
    assert not app.exception
    assert app.dataframe[0].value['品番'].tolist()==['A','B']
    # 並べ替え後はその表の行番号で品番を対応させる。
    states=app._tree.get_widget_states()
    event=states.widgets.add(); event.id=app.dataframe[0].proto.id
    event.string_value=json.dumps({'selection':{'rows':[1],'columns':[]}})
    app._run(widget_state=states)
    assert not app.exception and app.selectbox(key='top_product').value=='B'
    app.selectbox(key='top_product').select('A').run()
    assert not app.exception and app.selectbox(key='top_product').value=='A'


def test_worse_model_shows_negative_changes(database,tmp_path,monkeypatch):
    _,features=set_up(database,tmp_path,monkeypatch,complete=True)
    save_forecasts(database,features,[20,20],'2011-06-05','learning-run','models:/model',uuid4())
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    metrics=dict(zip(['LightGBM WAPE','ベースライン WAPE','LightGBM MAE','ベースライン MAE'],app.metric))
    assert metrics['LightGBM WAPE'].value=='300.00%'
    assert metrics['LightGBM WAPE'].delta=='-250.00 pp 悪化'
    assert metrics['LightGBM MAE'].delta=='-12.50 個 悪化'


def test_requested_row_sort_navigation_sequence(database,tmp_path,monkeypatch):
    import json
    _,features=set_up(database,tmp_path,monkeypatch,complete=True)
    # 6品番でTOP5外も選択欄に残ること、WAPE未定義が末尾になることを確認する。
    features=pd.DataFrame({'stock_code':list('ABCDEF'),'sales_sum_7':[8.]*6,
        'sales_sum_28':[32.]*6,'sales_std_28':[1.]*6,'month':[6]*6})
    save_forecasts(database,features,[9,2,18,20,4,6],'2011-06-05','learning-run','models:/model',uuid4())
    path=tmp_path/'2011-06-06.csv'
    records=[]
    for source_row,code,quantity in [(2,'A',4),(4,'C',20),(5,'D',10),(6,'E',5)]:
        records.append({'source_row':source_row,'InvoiceNo':'123','StockCode':code,'Description':'test',
            'Quantity':quantity,'InvoiceDate':'2011-06-06 10:00:00','UnitPrice':'1','CustomerID':'','Country':'UK'})
    pd.DataFrame(records).to_csv(path,index=False)
    run_daily_batch(path,'2011-06-06',database)
    app=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not app.exception
    assert app.selectbox(key='top_product').options==['D','F','B','C','A','E']
    def click_row(index):
        states=app._tree.get_widget_states()
        event=states.widgets.add(); event.id=app.dataframe[0].proto.id
        event.string_value=json.dumps({'selection':{'rows':[index],'columns':[]}})
        app._run(widget_state=states)
        assert not app.exception
    second=app.dataframe[0].value['品番'].iloc[1]
    click_row(1)
    assert second=='F' and app.selectbox(key='top_product').value==second
    # 追加の再描画でも戻らない。
    app.run()
    assert app.selectbox(key='top_product').value==second
    app.selectbox(key='error_order').select('WAPEが大きい順').run()
    assert not app.exception
    assert app.selectbox(key='top_product').options==['D','E','C','A','F','B']
    assert app.selectbox(key='top_product').value==second
    assert app.dataframe[0].value['品番'].tolist()==['D','E','C','A','F']
    third=app.dataframe[0].value['品番'].iloc[2]
    click_row(2)
    assert app.selectbox(key='top_product').value==third=='C'
    assert '#e6f0ff' in app.dataframe[0].proto.styler.styles
    next(b for b in app.button if '商品別分析へ' in b.label).click().run()
    assert not app.exception and app.selectbox(key='product').value==third
    app.radio(key='view').set_value('全体評価').run()
    assert not app.exception and app.selectbox(key='top_product').value==third
    # 手動変更後も、既に適用した行選択を再適用しない。
    app.selectbox(key='top_product').select('A').run()
    app.run()
    assert not app.exception and app.selectbox(key='top_product').value=='A'


def test_selection_event_uses_product_identity_and_is_not_replayed():
    from app import select_error_row
    state={'table_abs':{'selection':{'rows':[1]}}}
    assert select_error_row(state,'table_abs',['A','B','C'])
    assert state['analysis_product']=='B'
    state['analysis_product']='C'
    assert not select_error_row(state,'table_abs',['A','B','C'])
    assert state['analysis_product']=='C'
    state['table_wape']={'selection':{'rows':[]}}
    assert not select_error_row(state,'table_wape',['C','A','B'])
    assert state['analysis_product']=='C'
    state['table_wape']['selection']['rows']=[2]
    assert select_error_row(state,'table_wape',['C','A','B'])
    assert state['analysis_product']=='B'
    assert not select_error_row(state,'table_abs',['A','B','C'])
