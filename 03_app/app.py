"""保存済み予測を閲覧するダッシュボード。学習・予測・DB更新は行わない。"""
import hashlib
import io
import json
import os
import sqlite3
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from urllib.parse import unquote, urlparse

import altair as alt
import numpy as np
import pandas as pd
import psycopg
import streamlit as st

from evaluation.evaluate import compare_actuals

COLORS = alt.Scale(domain=['LightGBM', 'ベースライン', '実績'],
                   range=['#25496e', '#94a3b8', '#268b80'])
FEATURES = {'sales_sum_7': '過去7日販売数量', 'sales_sum_28': '過去28日販売数量',
            'sales_std_28': '過去28日の日次数量の標準偏差'}


def load_runs(database_url):
    with psycopg.connect(database_url) as connection:
        connection.execute('SET TRANSACTION READ ONLY')
        rows=connection.execute('''
            SELECT prediction_run_id::text,week_start,forecast_origin,
                   MIN(created_at) AS created_at,COUNT(*) AS products
            FROM weekly_forecasts GROUP BY prediction_run_id,week_start,forecast_origin
            ORDER BY created_at DESC,prediction_run_id
        ''').fetchall()
        latest=connection.execute('SELECT MAX(sales_date) FROM daily_sales').fetchone()[0]
    return pd.DataFrame(rows,columns=['prediction_run_id','week_start','forecast_origin','created_at','products']),latest


def load_forecast_history(database_url, prediction_id, week_start, observed_through):
    with psycopg.connect(database_url) as connection:
        connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        cursor=connection.execute('''
            SELECT stock_code,predicted_quantity::double precision AS lightgbm,
                   sales_sum_7::double precision AS baseline,sales_sum_28::double precision,
                   sales_std_28,month,mlflow_run_id,model_uri,created_at
            FROM weekly_forecasts WHERE prediction_run_id=%s ORDER BY stock_code
        ''',(prediction_id,))
        frame=pd.DataFrame(cursor.fetchall(),columns=[c.name for c in cursor.description])
        end=min(week_start+timedelta(days=6),observed_through) if observed_through else week_start-timedelta(days=1)
        history=connection.execute('''
            SELECT stock_code,sales_date,quantity FROM daily_sales
            WHERE stock_code=ANY(%s) AND sales_date BETWEEN %s AND %s
            ORDER BY stock_code,sales_date
        ''',(frame.stock_code.tolist(),week_start-timedelta(days=56),end)).fetchall()
    return frame,pd.DataFrame(history,columns=['stock_code','sales_date','quantity'])


def load_model_info(run_id, model_uri):
    """選択モデルの記録だけを読む。MLflow APIによるDB初期化・更新を避ける。"""
    root = Path(os.environ.get('MLFLOW_DATA_DIR', '02_model/mlflow')).resolve()
    result = {'training': None, 'conditions': {}, 'status': '未記録',
              'trained_at': '未記録', 'issue': ''}
    try:
        # mode=roなので、ファイルがなければ新規DBは作成されない。
        with sqlite3.connect((root / 'mlflow.db').as_uri() + '?mode=ro', uri=True) as connection:
            record = connection.execute(
                'SELECT artifact_uri,start_time,status FROM runs WHERE run_uuid=?',
                (run_id,)).fetchone()
        if record is None:
            result['issue'] = 'このモデルのMLflow学習記録はデータなしです。'
            return result
        uri, timestamp, result['status'] = record
        if timestamp is not None:
            result['trained_at'] = pd.to_datetime(timestamp, unit='ms', utc=True).tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S JST')
        parsed = urlparse(uri)
        if parsed.scheme != 'file' or parsed.netloc:
            raise ValueError('ローカル以外の学習成果物は、この画面では取得できません。')
        artifact = Path(unquote(parsed.path))
        # Docker内の既存保存パスを、同じマウント元のホストパスにも対応させる。
        docker_root = Path('/app/02_model/mlflow')
        if artifact.is_relative_to(docker_root):
            artifact = root / artifact.relative_to(docker_root)
        if not artifact.resolve().is_relative_to(root):
            raise ValueError('学習成果物の保存先が参照範囲外です。')
        conditions = json.loads((artifact / 'conditions.json').read_text())
        if conditions.get('run_id') != run_id or conditions.get('model_uri') != model_uri:
            raise ValueError('モデルIDと学習記録が一致しません。')
        result['conditions'] = conditions
        raw = (artifact / 'data/training.csv').read_bytes()
        if hashlib.sha256(raw).hexdigest() != conditions.get('data_sha256'):
            raise ValueError('保存された学習CSVのハッシュが一致しません。')
        training = pd.read_csv(io.BytesIO(raw), dtype={'stock_code': str})
        for column in ['forecast_origin', 'history_start', 'target_end']:
            training[column] = pd.to_datetime(training[column], errors='raise')
        if training.empty:
            raise ValueError('保存された学習CSVが空です。')
        result['training'] = training
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
        result['issue'] = '学習CSVまたはMLflow情報を取得・照合できません。未記録の項目は表示を省略します。'
    return result


def period_text(info):
    """予測週ではなく、そのモデルが実際に学習した行から期間を取得する。"""
    training = info['training']
    if training is not None:
        return f'{training.forecast_origin.min():%Y-%m-%d} ～ {training.forecast_origin.max():%Y-%m-%d}'
    final = info['conditions'].get('final_training', {})
    if final.get('origin_start') and final.get('origin_end'):
        return f'{final["origin_start"]} ～ {final["origin_end"]}'
    return '未記録'


def evaluated(database_url, identifier, start, latest):
    """既存の照合・評価関数を再利用し、未取り込みをゼロ実績にしない。"""
    if latest is None or latest < start + timedelta(days=6):
        return None, None, '対象週の実績がまだそろっていません。'
    try:
        comparison, scores = compare_actuals(database_url, identifier, latest,
            Path(os.environ.get('SOURCE_DIR', '01_data/storage/daily')))
        return comparison, scores, ''
    except (ValueError, OSError) as error:
        return None, None, str(error)
    except psycopg.Error:
        return None, None, 'DBから実績を確認できませんでした。'


def weekly_results(database_url, runs, selected_id, selected_start, latest, saved, comparison, scores):
    """各週の最新予測を比較。選択週は選択した実行を使い、架空の週は追加しない。"""
    history = runs.loc[runs.week_start.le(selected_start)].drop_duplicates('week_start').copy()
    history.loc[history.week_start.eq(selected_start), 'prediction_run_id'] = selected_id
    points = []
    products = []
    pending = 0
    for run in history.sort_values('week_start').itertuples(index=False):
        if run.prediction_run_id == selected_id:
            frame, actual, metrics = saved, comparison, scores
        else:
            frame, _ = load_forecast_history(database_url, run.prediction_run_id, run.week_start, latest)
            actual, metrics, _ = evaluated(database_url, run.prediction_run_id, run.week_start, latest)
        if metrics is None:
            pending += 1
        for model, key in [('LightGBM', 'lightgbm'), ('ベースライン', 'baseline')]:
            value = metrics[key]['wape'] if metrics else None
            points.append({'週': str(run.week_start), 'モデル': model,
                           'WAPE': None if value is None else value * 100,
                           '予測実行ID': run.prediction_run_id})
        frame = frame[['stock_code', 'lightgbm', 'baseline']].copy()
        if actual is not None:
            frame = frame.merge(actual[['stock_code', 'actual']], on='stock_code', validate='one_to_one')
        else:
            frame['actual'] = np.nan
        frame['week'] = str(run.week_start)
        products.append(frame)
    return pd.DataFrame(points), pd.concat(products, ignore_index=True), pending


def line_chart(frame, x, y, category, title):
    chart = alt.Chart(frame).mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.5).encode(
        x=alt.X(f'{x}:O', title='予測対象週（月曜日）', axis=alt.Axis(labelAngle=-30)),
        y=alt.Y(f'{y}:Q', title=title, scale=alt.Scale(zero=True)),
        color=alt.Color(f'{category}:N', scale=COLORS, title=None),
        tooltip=list(frame.columns)).properties(height=255)
    st.altair_chart(chart, use_container_width=True)


def error_ranking(comparison, order):
    """表示用の品番別WAPE。実績0は欠損にし、全体指標の計算は変更しない。"""
    table = comparison.copy()
    table['product_wape'] = table.lightgbm_abs_error.div(table.actual.where(table.actual.ne(0))) * 100
    column = 'product_wape' if order == 'WAPEが大きい順' else 'lightgbm_abs_error'
    return table.sort_values([column, 'lightgbm_abs_error', 'stock_code'] if column == 'product_wape'
        else [column, 'stock_code'], ascending=[False, False, True] if column == 'product_wape'
        else [False, True], na_position='last').reset_index(drop=True)


def select_error_row(state, table_key, codes):
    """選択イベントを一度だけ適用し、保存する選択状態は行番号ではなく品番にする。"""
    rows = state.get(table_key, {}).get('selection', {}).get('rows', [])
    event = tuple(rows)
    applied = dict(state.get('applied_error_selections', {}))
    if applied.get(table_key) == event:
        return False
    applied[table_key] = event
    state['applied_error_selections'] = applied
    if rows and 0 <= rows[0] < len(codes):
        # Streamlitの行番号はクライアント側ソート後も元データの位置を指す。
        # 描画時の品番一覧で変換し、選択値は品番として保持する。
        state['analysis_product'] = codes[rows[0]]
        return True
    return False


def remember_product(widget_key):
    # ウィジェットが別画面で非表示になっても、品番を失わないよう別キーにも保存する。
    st.session_state['analysis_product'] = st.session_state[widget_key]


def metric_change(model, baseline, multiplier, unit):
    if model is None or baseline is None:
        return None
    difference = (baseline - model) * multiplier
    status = '改善' if difference > 0 else '悪化' if difference < 0 else '同等'
    return f'{difference:+,.2f} {unit} {status}'


def show_overview(comparison, scores, trend, info, run, count):
    # 期間をひとつの帯にまとめ、学習サンプルの期間と予測期間を区別する。
    fields = [
        ('学習データ期間（学習サンプルの基準日）', period_text(info)),
        ('予測基準日', str(run.forecast_origin)),
        ('予測対象期間', f'{run.week_start} ～ {run.week_start + timedelta(days=6)}'),
        ('予測対象品番数', f'{count} 品番'),
    ]
    st.markdown('<div class="period-summary">' + ''.join(
        f'<div><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in fields) + '</div>', unsafe_allow_html=True)
    st.subheader('予測精度')
    if scores is not None:
        with st.container(key='accuracy_kpis'):
            groups = st.columns(2, gap='medium')
            for group, metric, unit, multiplier in zip(groups, ['wape', 'mae'], ['pp', '個'], [100, 1]):
                with group, st.container(key=f'kpi_{metric}'):
                    st.markdown(f'<div class="kpi-group-title">{metric.upper()}</div>', unsafe_allow_html=True)
                    columns = st.columns(2)
                    model, baseline = scores['lightgbm'][metric], scores['baseline'][metric]
                    def formatted(value):
                        return '未定義' if value is None else f'{value:.2%}' if metric == 'wape' else f'{value:,.2f}'
                    columns[0].metric('LightGBM', formatted(model),
                        delta=metric_change(model, baseline, multiplier, unit),
                        delta_color='off' if model == baseline else 'normal')
                    columns[1].metric('ベースライン', formatted(baseline))
    else:
        st.caption('実績の確認後にWAPE・MAEを表示します。')
    st.subheader('予測誤差が大きい商品 TOP 5')
    if comparison is None:
        st.caption('実績の確認後に表示します。')
    else:
        order = st.selectbox('並び順', ['絶対誤差が大きい順', 'WAPEが大きい順'], key='error_order')
        ranked = error_ranking(comparison, order)
        table = ranked.head(5)
        options = ranked.stock_code.tolist()
        if st.session_state.get('analysis_product') not in options:
            st.session_state['analysis_product'] = options[0]
        visible = table[['stock_code', 'actual', 'lightgbm', 'lightgbm_abs_error', 'product_wape']].rename(columns={
            'stock_code': '品番', 'actual': '実績数量', 'lightgbm': '予測数量',
            'lightgbm_abs_error': '絶対誤差', 'product_wape': 'WAPE'})
        styled = visible.style.format({c: '{:,.2f}%' if c == 'WAPE' else '{:,.2f}'
            for c in visible if c != '品番'}, na_rep='—').set_properties(
                subset=visible.columns[1:], **{'text-align': 'right'})
        for column in ['絶対誤差', 'WAPE']:
            maximum = visible[column].max()
            def shade(value, maximum=maximum):
                ratio = 0 if pd.isna(value) or pd.isna(maximum) or maximum <= 0 else min(max(value / maximum, 0), 1)
                return f'background-color: rgb(255, {255-round(29*ratio)}, {255-round(37*ratio)}); color: #26364a'
            styled = styled.map(shade, subset=[column])
        selected_code = st.session_state['analysis_product']
        def highlight_row(row):
            return ['background-color: #e6f0ff; color: #183153; font-weight: 600' if row['品番'] == selected_code else '' for _ in row]
        styled = styled.apply(highlight_row, axis=1)
        codes = table.stock_code.tolist()
        # 並び替え・実行切り替えごとに選択イベントを分け、古い行番号を再利用しない。
        table_key = f'error_table_{run.prediction_run_id}_{order}'
        st.dataframe(styled, hide_index=True, use_container_width=True, key=table_key,
            selection_mode='single-row', on_select='rerun',
            column_config={'WAPE': st.column_config.NumberColumn(help='絶対誤差 ÷ 実績数量 × 100。実績0は未定義。')})
        # コールバック内の更新だけに依存せず、再描画で届いた最新イベントを読み取る。
        # 適用済みイベントは無視し、プルダウンでの変更を維持する。
        if select_error_row(st.session_state, table_key, codes):
            st.rerun()
        st.caption('WAPEは実績が少ないと大きくなります。絶対誤差と併せて確認してください。実績0は「—」。')
        # 表とプルダウンは同じ全品番のランキングを使用。選択値は品番で復元する。
        st.session_state['top_product'] = st.session_state['analysis_product']
        selection, action = st.columns([3, 1], vertical_alignment='bottom')
        code = selection.selectbox('詳細を確認する品番', options, key='top_product',
            on_change=remember_product, args=('top_product',))
        st.caption(f'選択中：{code}')
        def open_product():
            st.session_state['analysis_product'] = code
            st.session_state['view'] = '商品別分析'
            st.session_state['product'] = code
        action.button('商品別分析へ →', on_click=open_product, use_container_width=True)
    st.subheader('予測精度の推移')
    if trend.loc[trend.WAPE.notna(), '週'].nunique() >= 2:
        line_chart(trend[['週', 'モデル', 'WAPE']], '週', 'WAPE', 'モデル', 'WAPE（%）')
    else:
        st.caption('評価済みの予測が2週以上になると表示します。')


def load_product_weekly_sales(database_url, code, week_start):
    """選択週を含む9週をSQLで集計。DBの範囲外・全商品記録なしは欠損にする。"""
    first = week_start - timedelta(weeks=8)
    with psycopg.connect(database_url) as connection:
        connection.execute('SET TRANSACTION READ ONLY')
        rows = connection.execute('''
            WITH weeks AS (
                SELECT generate_series(%s::date, %s::date, interval '7 days')::date AS week
            ), bounds AS (
                SELECT MIN(sales_date) AS first_day, MAX(sales_date) AS last_day FROM daily_sales
            ), totals AS (
                SELECT date_trunc('week', sales_date)::date AS week,
                       SUM(quantity) FILTER (WHERE stock_code = %s) AS quantity,
                       COUNT(*) AS records
                FROM daily_sales WHERE sales_date BETWEEN %s AND %s
                GROUP BY 1
            )
            SELECT weeks.week,
                   CASE WHEN weeks.week >= bounds.first_day
                         AND weeks.week + 6 <= bounds.last_day AND totals.records > 0
                        THEN COALESCE(totals.quantity, 0)::double precision END AS actual
            FROM weeks CROSS JOIN bounds LEFT JOIN totals USING (week)
            ORDER BY weeks.week
        ''', (first, week_start, code, first, week_start + timedelta(days=6))).fetchall()
    return pd.DataFrame(rows, columns=['week', 'actual'])


def product_sales_chart(actuals, weekly, code, week_start):
    """実績9週と保存済み予測だけを重ねる。未保存週の予測を補完しない。"""
    frame = actuals.copy()
    frame['week'] = pd.to_datetime(frame.week)
    saved = weekly.loc[weekly.stock_code.eq(code), ['week', 'lightgbm', 'actual']].copy()
    saved['week'] = pd.to_datetime(saved.week)
    saved = saved.loc[saved.week.between(frame.week.min(), frame.week.max())]
    # 保存済み予測の対象週は、既存の照合処理を通った実績を優先する。
    # 不一致や未完了の場合は欠損を維持し、確定した販売ゼロと混同しない。
    for row in saved.itertuples(index=False):
        frame.loc[frame.week.eq(row.week), 'actual'] = row.actual
    frame['actual'] = pd.to_numeric(frame.actual, errors='coerce').astype(float)
    actual_values = frame.rename(columns={'actual': '数量'})
    actual_values['種類'] = '実績'
    predictions = saved[['week', 'lightgbm']].rename(columns={'lightgbm': '数量'})
    predictions['数量'] = predictions['数量'].astype(float)
    predictions['種類'] = 'LightGBM'
    values = pd.concat([actual_values, predictions], ignore_index=True)
    values['数量'] = pd.to_numeric(values['数量'], errors='coerce')
    # 週をカテゴリとして等間隔に配置。選択週の列だけに薄い帯を引く。
    values['週開始日'] = values.week.dt.strftime('%Y-%m-%d')
    x = alt.X('週開始日:O', sort=frame.week.dt.strftime('%Y-%m-%d').tolist(),
              title='週開始日（月曜日）', axis=alt.Axis(labelAngle=-30))
    highlight = alt.Chart(pd.DataFrame({'週開始日': [str(week_start)]})).mark_rect(
        color='#25496e', opacity=.08).encode(x=x)
    lines = alt.Chart(values).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=65)).encode(
        x=x, y=alt.Y('数量:Q', title='週間販売数量', scale=alt.Scale(zero=True)),
        color=alt.Color('種類:N', scale=alt.Scale(domain=['実績', 'LightGBM'], range=['#268b80', '#25496e']), title=None),
        tooltip=['週開始日:N', '種類:N', alt.Tooltip('数量:Q', format=',.2f')])
    # 予測は点で示し、未保存の週をまたぐ線も生成しない。
    actual_line = lines.transform_filter(alt.datum['種類'] == '実績')
    prediction_points = alt.Chart(values).transform_filter(alt.datum['種類'] == 'LightGBM').mark_point(
        filled=True, size=95).encode(
        x=x, y=alt.Y('数量:Q', title='週間販売数量', scale=alt.Scale(zero=True)),
        color=alt.Color('種類:N', scale=alt.Scale(domain=['実績', 'LightGBM'], range=['#268b80', '#25496e']), title=None),
        tooltip=['週開始日:N', '種類:N', alt.Tooltip('数量:Q', format=',.2f')])
    return (highlight + actual_line + prediction_points).properties(height=285)


def show_product(saved, comparison, weekly, info, database_url, week_start):
    options = saved.stock_code.tolist()
    preferred = st.session_state.get('analysis_product', st.session_state.get('product'))
    st.session_state['product'] = preferred if preferred in options else '22197' if '22197' in options else options[0]
    code = st.selectbox('品番', options, key='product', on_change=remember_product, args=('product',))
    row = saved.loc[saved.stock_code.eq(code)].iloc[0]
    actual = None if comparison is None else float(comparison.loc[comparison.stock_code.eq(code), 'actual'].iloc[0])
    columns = st.columns(3)
    for column, label, value in zip(columns,
        ['予測数量', '実績数量', '絶対誤差'],
        [row.lightgbm, actual, None if actual is None else abs(row.lightgbm - actual)]):
        column.metric(label, '未確認' if value is None else f'{value:,.2f}')
    st.subheader('販売数量の推移')
    actuals = load_product_weekly_sales(database_url, code, week_start)
    st.altair_chart(product_sales_chart(actuals, weekly, code, week_start), use_container_width=True)
    st.caption('薄い帯は予測対象週。予測は保存済みの週だけを点で表示。未完了・記録なしの実績は空欄です。')
    st.subheader('学習時と予測時の特徴量')
    training = info['training']
    selected = None if training is None else training.loc[training.stock_code.eq(code)]
    records = []
    # 学習時は同じ商品の中央値、予測時はDBに保存された入力値。欠損は推測しない。
    for key, label in FEATURES.items():
        prediction = row.get('baseline' if key == 'sales_sum_7' else key)
        median = None if selected is None or selected.empty or key not in selected else selected[key].median()
        records.append({'特徴量': label,
            '学習時（中央値）': '未記録' if pd.isna(median) else f'{median:,.2f}',
            '予測時': '未記録' if pd.isna(prediction) else f'{prediction:,.2f}'})
    st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)


def show_metadata(info, saved, run, latest, scores, reason):
    st.subheader('実行情報')
    training = info['training']
    records = {
        '保存モデルID / URI': saved.model_uri.iloc[0],
        '学習時のMLflow実行ID': saved.mlflow_run_id.iloc[0],
        'モデルの学習開始日時': info['trained_at'],
        '学習データの対象期間（学習サンプルの基準日）': period_text(info),
        '学習に使った履歴・正解の範囲': '未記録' if training is None else f'{training.history_start.min():%Y-%m-%d} ～ {training.target_end.max():%Y-%m-%d}',
        '予測実行ID': run.prediction_run_id,
        '予測保存日時': pd.Timestamp(run.created_at).tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S JST'),
        '予測基準日': str(run.forecast_origin),
        '予測対象期間': f'{run.week_start} ～ {run.week_start + timedelta(days=6)}',
        '販売実績の最終記録日': str(latest) if latest else 'データなし',
        '販売実績のDB最終更新日時': '未記録',
        'MLflow学習の実行状態': info['status'],
        '予測実行の状態': '保存済み',
        '現在の実績照合': '一致確認済み' if scores else '確認待ち：' + reason,
        '予測実行のエラーログ': '未記録',
        '画面読込日時': datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
    }
    st.dataframe(pd.DataFrame(records.items(), columns=['項目', '記録']), hide_index=True, use_container_width=True, height=600)
    if info['issue']:
        st.caption(info['issue'])
    st.link_button('MLflowの学習記録を開く', os.environ.get('MLFLOW_UI_URL', 'http://localhost:5001'))


def main():
    st.set_page_config(page_title='DemandPilot | 週次需要予測', page_icon='📊', layout='wide')
    st.markdown('''<style>
    .block-container {padding-top:1.5rem;padding-bottom:2rem;max-width:1400px}
    h1,h2,h3 {color:#183153}
    h1 {font-size:1.9rem!important} h3 {font-size:1.05rem!important;margin-top:.6rem}
    [data-testid="stSidebar"] {border-right:1px solid #edf0f4}
    [data-testid="stMetric"] {padding:0 0 8px}
    [data-testid="stMetricValue"] {font-size:2.3rem;color:#183153}
    .page-heading {background:linear-gradient(110deg,#fff 10%,#edf5ff 100%);padding:12px 18px;margin-bottom:16px;border-radius:8px}
    .page-heading h1 {padding:0!important;margin:0}
    .page-heading p {color:#64748b;font-size:.85rem;margin:3px 0 0}
    .st-key-kpi_wape,.st-key-kpi_mae {background:linear-gradient(115deg,#f8fbff,#edf4fd);padding:10px 16px;border-radius:7px}
    .st-key-kpi_mae {border-left:2px solid #d2e1f3}
    .kpi-group-title {color:#345b80;font-size:.8rem;font-weight:650;letter-spacing:.08em;margin-bottom:5px}
    .st-key-accuracy_kpis [data-testid="stMetric"] {padding:0}
    .st-key-accuracy_kpis [data-testid="stMetricValue"] {font-size:2rem}
    .period-summary {display:grid;grid-template-columns:1.5fr 1fr 1.5fr .6fr;gap:16px;
        padding:16px 0;border-top:1px solid #e6eaf0;border-bottom:1px solid #e6eaf0;margin:6px 0 16px}
    .period-summary span {display:block;color:#64748b;font-size:.78rem;margin-bottom:5px}
    .period-summary strong {display:block;color:#183153;font-size:.95rem;font-weight:600}
    @media(max-width:1100px) {.period-summary {grid-template-columns:1fr 1fr}}
    @media(min-width:900px) {.st-key-accuracy_kpis [data-testid="stHorizontalBlock"] {flex-wrap:nowrap!important}
        .st-key-accuracy_kpis [data-testid="stColumn"] {min-width:0!important}}
    @media(max-width:640px) {.period-summary {grid-template-columns:1fr}.block-container {padding:1rem}}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="page-heading"><h1>DemandPilot</h1><p>週次需要予測</p></div>', unsafe_allow_html=True)
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        st.error('DB接続の設定がありません。READMEの起動手順を確認してください。')
        st.stop()
    try:
        runs, latest = load_runs(database_url)
    except psycopg.Error:
        st.error('PostgreSQLに接続できません。DBの起動状態を確認してください。')
        st.stop()
    if runs.empty:
        st.info('保存済みの予測がありません。予測コマンドで予測を保存すると表示されます。')
        st.stop()
    weeks = sorted(runs.week_start.unique(), reverse=True)
    start = st.sidebar.selectbox('予測週', weeks,
        format_func=lambda day: f'{day} ～ {day + timedelta(days=6)}', key='week')
    view = st.sidebar.radio('表示画面', ['全体評価', '商品別分析', '実行情報'], key='view')
    if st.sidebar.button('データを更新'):
        st.rerun()
    choices = runs.loc[runs.week_start.eq(start)]
    # 実行履歴の選択は画面を切り替えても保持。週を変更したらその週の最新実行を使用。
    if st.session_state.get('selected_prediction') not in choices.prediction_run_id.tolist():
        st.session_state['selected_prediction'] = choices.prediction_run_id.iloc[0]
    if view == '実行情報':
        st.session_state['prediction'] = st.session_state['selected_prediction']
        def select_run():
            st.session_state['selected_prediction'] = st.session_state['prediction']
        st.selectbox('予測実行履歴', choices.prediction_run_id.tolist(), key='prediction', on_change=select_run,
            format_func=lambda value: str(choices.loc[choices.prediction_run_id.eq(value), 'created_at'].iloc[0]) + ' / ' + value[:8])
    identifier = st.session_state['selected_prediction']
    run = choices.loc[choices.prediction_run_id.eq(identifier)].iloc[0]
    try:
        saved, _ = load_forecast_history(database_url, identifier, start, latest)
        comparison, scores, reason = evaluated(database_url, identifier, start, latest)
        trend, weekly, _ = weekly_results(database_url, runs, identifier, start, latest, saved, comparison, scores)
    except psycopg.Error:
        st.error('予測データを読み込めません。DBの状態を確認してください。')
        st.stop()
    info = load_model_info(saved.mlflow_run_id.iloc[0], saved.model_uri.iloc[0])
    if scores is None and view != '実行情報':
        st.info('実績の確認待ちです。' + reason)
    if view == '全体評価':
        show_overview(comparison, scores, trend, info, run, len(saved))
    elif view == '商品別分析':
        show_product(saved, comparison, weekly, info, database_url, start)
    else:
        show_metadata(info, saved, run, latest, scores, reason)


if __name__ == '__main__':
    main()
