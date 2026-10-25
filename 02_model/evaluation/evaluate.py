"""保存済みの予測を、取り込み確認済みの翌週実績と照合する。予測値は更新しない。"""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import psycopg

from evaluation.metrics import metrics
from training.training_data import UNKNOWN_PERIODS
from weekly.forecast import verify_period, verify_processed_period


# 実績は日次集計からSQLで合計し、商品・予測実行IDで対応させる。
# ベースラインは予測時に記録した過去7日合計を使用する。
COMPARISON_SQL = '''
    SELECT f.prediction_run_id::text, f.stock_code, f.week_start,
           f.forecast_origin, f.predicted_quantity::double precision AS lightgbm,
           f.sales_sum_7::double precision AS baseline,
           f.sales_sum_28::double precision AS sales_sum_28, f.sales_std_28, f.month,
           f.mlflow_run_id, f.model_uri, f.created_at,
           COALESCE(SUM(d.quantity),0)::bigint AS actual
    FROM weekly_forecasts f
    LEFT JOIN daily_sales d ON d.stock_code = f.stock_code
        AND d.sales_date >= f.week_start AND d.sales_date < f.week_start + 7
    WHERE f.prediction_run_id = %s
    GROUP BY f.prediction_run_id, f.stock_code
    ORDER BY f.stock_code
'''


def compare_actuals(database_url, prediction_id, as_of, source_dir):
    prediction_id = UUID(str(prediction_id))
    as_of = pd.Timestamp(as_of).date()
    with psycopg.connect(database_url) as connection:
        # 予測・明細・日次実績は同一スナップショットで読み、評価中の変更を混ぜない。
        connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        cursor = connection.execute(COMPARISON_SQL, (prediction_id,))
        rows = cursor.fetchall()
        if not rows:
            raise ValueError('指定した予測実行IDがありません')
        frame = pd.DataFrame(rows, columns=[c.name for c in cursor.description])
        if any(frame[c].nunique() != 1 for c in ['week_start','forecast_origin','mlflow_run_id','model_uri']):
            raise ValueError('同じ予測実行IDに異なる期間またはモデルが混在しています')
        start = frame.week_start.iloc[0]
        end = start + timedelta(days=6)
        if as_of < end:
            raise ValueError('予測対象の7日間が終了していません')
        for first, last in UNKNOWN_PERIODS:
            if pd.Timestamp(start) <= pd.Timestamp(last) and pd.Timestamp(end) >= pd.Timestamp(first):
                raise ValueError('実績の対象週が共通の空白期間を含んでいます')
        # 7日間の正本CSVに対し、明細・日次集計を全件照合。未取込なら評価しない。
        # 正本CSV群の完全性が前提。商品別に販売なしならSQLのLEFT JOINで0になる。
        if connection.execute("SELECT 1 FROM weekly_forecast_runs WHERE prediction_run_id=%s", (prediction_id,)).fetchone():
            verify_processed_period(connection, start, end, source_dir)
        verify_period(connection, start, end, source_dir)
    return comparison_result(frame)


def comparison_result(frame):
    """取込実績と保存済み実績に共通の誤差・指標計算。"""
    for model in ['baseline','lightgbm']:
        frame[model+'_signed_error'] = frame[model] - frame.actual
        frame[model+'_abs_error'] = frame[model+'_signed_error'].abs()
        frame[model+'_wape'] = frame[model+'_abs_error'] / frame.actual.replace(0,np.nan)
    scores = {model:metrics(frame.actual,frame[model]) for model in ['baseline','lightgbm']}
    return frame, scores


def save_database_evaluation(database_url, prediction_id, frame, scores, evaluated_at, streak, run_id):
    from psycopg.types.json import Jsonb
    if frame.empty or frame.stock_code.duplicated().any() or not frame.prediction_run_id.eq(str(prediction_id)).all():
        raise ValueError("評価対象の予測ID・品番が不正です")
    with psycopg.connect(database_url) as connection:
        connection.execute("""
            INSERT INTO weekly_evaluations VALUES (%s,%s,%s,%s,%s)
        """, (prediction_id, evaluated_at, Jsonb(scores), streak, run_id))
        # 予測値はコピー・JSON化せず、同じ商品行に確定実績だけを一度保存する。
        # 不整合があればヘッダーも含めてロールバックする。
        for row in frame.itertuples(index=False):
            cursor = connection.execute("""
                UPDATE weekly_forecasts SET actual_quantity=%s
                WHERE prediction_run_id=%s AND stock_code=%s AND actual_quantity IS NULL
            """, (int(row.actual), prediction_id, row.stock_code))
            if cursor.rowcount != 1:
                raise ValueError("予測が存在しない、または確定実績が既に保存されています")


def load_database_evaluation(database_url, prediction_id):
    with psycopg.connect(database_url) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        row = connection.execute("SELECT scores FROM weekly_evaluations WHERE prediction_run_id=%s",
                                 (prediction_id,)).fetchone()
        if row is None:
            return None, None
        cursor = connection.execute("""
            SELECT prediction_run_id::text,stock_code,week_start,forecast_origin,
                   predicted_quantity::double precision AS lightgbm,
                   sales_sum_7::double precision AS baseline,sales_sum_28::double precision,
                   sales_std_28,month,mlflow_run_id,model_uri,created_at,actual_quantity AS actual
            FROM weekly_forecasts WHERE prediction_run_id=%s ORDER BY stock_code
        """, (prediction_id,))
        frame = pd.DataFrame(cursor.fetchall(), columns=[c.name for c in cursor.description])
    if frame.empty or frame.actual.isna().any():
        raise ValueError("評価ヘッダーに対応する確定実績が不足しています")
    frame, _ = comparison_result(frame)
    return frame, row[0]


def save_evaluation(frame, scores, as_of, source_dir, output_root):
    # 再評価も新しいフォルダへ保存し、以前の実績スナップショットを上書きしない。
    evaluation_id = str(uuid4())
    output = Path(output_root)/evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output/'comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=frame.week_start.iloc[0]; end=start+timedelta(days=6)
    sources=[]
    for path in sorted(Path(source_dir).glob('*.csv')):
        if start.isoformat()<=path.stem<=end.isoformat():
            sources.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    record={'evaluation_id':evaluation_id,'prediction_run_id':frame.prediction_run_id.iloc[0],
            'mlflow_run_id':frame.mlflow_run_id.iloc[0],'model_uri':frame.model_uri.iloc[0],
            'week_start':str(start),'week_end':str(end),'as_of':str(as_of),
            'evaluated_at_jst':datetime.now(timezone(timedelta(hours=9))).isoformat(),
            'scores':scores,'source_csv':sources,
            'comparison_sha256':hashlib.sha256((output/'comparison.csv').read_bytes()).hexdigest(),
            'evaluation_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'status':'historical operational replay; one known dataset week, no model reselection'}
    (output/'metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for model,color in [('baseline','#bc7b35'),('lightgbm','#277e9d')]:
        axes[0].scatter(frame.actual,frame[model],label=model,alpha=.65,s=25,color=color)
    maximum=max(frame.actual.max(),frame.baseline.max(),frame.lightgbm.max())
    axes[0].plot([0,maximum],[0,maximum],color='gray',linestyle='--',linewidth=1)
    axes[0].set(xlabel='Actual 7-day units',ylabel='Predicted 7-day units',title=f'{start} – {end}: all {len(frame)} products')
    axes[0].legend()
    worst=frame.nlargest(12,'lightgbm_abs_error').sort_values('lightgbm_abs_error')
    y=np.arange(len(worst))
    axes[1].barh(y-.18,worst.baseline_abs_error,height=.35,label='baseline',color='#bc7b35')
    axes[1].barh(y+.18,worst.lightgbm_abs_error,height=.35,label='lightgbm',color='#277e9d')
    axes[1].set_yticks(y,worst.stock_code)
    axes[1].set(xlabel='Absolute error (7-day units)',title='12 largest LightGBM errors')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    def percent(value): return '未定義' if value is None else f'{value:.2%}'
    report=f'''# 予測と実績の照合：{start}〜{end}

予測実行ID：`{record['prediction_run_id']}`。学習実行ID：`{record['mlflow_run_id']}`。
実績の確認基準日：{as_of}。評価日時：{record['evaluated_at_jst']}。

予測対象週のCSVとDB明細・日次集計の一致を確認した後、SQLで7日合計を照合した。
ベースラインは予測時に保存した直近7日合計を使用し、予測値・入力特徴量は変更していない。

| 指標 | ベースライン | LightGBM |
|---|---:|---:|
| MAE（7日合計数量） | {scores['baseline']['mae']:.2f} | {scores['lightgbm']['mae']:.2f} |
| WAPE | {percent(scores['baseline']['wape'])} | {percent(scores['lightgbm']['wape'])} |
| 商品数 | {len(frame)} | {len(frame)} |

実績数量合計：{int(frame.actual.sum()):,}。
全体WAPEは誤差合計÷実績合計。商品別WAPEの単純平均ではない。
実績ゼロの商品のWAPEは未定義（CSVは空欄）。販売ゼロと未取り込みは、元CSVとの照合で区別する。

![予測と実績・誤差の大きい商品](comparison.png)

[全商品の入力・予測・実績・誤差](comparison.csv) / [評価条件とハッシュ](metrics.json)

## 誤差の大きい商品

| 品番 | 実績 | ベースライン | LightGBM | LightGBM絶対誤差 |
|---|---:|---:|---:|---:|
'''
    for row in frame.nlargest(10,'lightgbm_abs_error').itertuples(index=False):
        report+=f'| {row.stock_code} | {row.actual} | {row.baseline:.2f} | {row.lightgbm:.2f} | {row.lightgbm_abs_error:.2f} |\n'
    report+='''
## 判断の範囲

この結果は過去データによる1週間・固定商品の運用再現であり、モデルの一般的な優位性を証明しない。
同じデータセットの旧版の結果も既知のため、完全に未知の最終評価とは呼ばない。
結果を見て商品・パラメータ・モデルを変更していない。
欠品・販促・大口注文などの原因は、この数量比較だけでは確定できない。
正本CSV自体の欠落とDB側の欠落が同時に起きた場合は、この照合だけでは検知できない。
後から実績が修正されれば再評価値は変わり得るが、このCSVには今回の照合値を保持する。
'''
    (output/'report.md').write_text(report)
    print(json.dumps({'output':str(output),'scores':scores,'evaluation_id':evaluation_id},ensure_ascii=False,indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-id',required=True,type=UUID)
    parser.add_argument('--as-of',required=True,type=date.fromisoformat,help='実績が確定したとする日付。対象週の日曜以降')
    parser.add_argument('--source-dir',type=Path,default=Path('01_data/storage/daily'))
    parser.add_argument('--output-root',type=Path,default=Path('02_model/evaluation/results'))
    args=parser.parse_args()
    if not os.environ.get('DATABASE_URL'): parser.error('DATABASE_URLを設定してください')
    frame,scores=compare_actuals(os.environ['DATABASE_URL'],args.prediction_id,args.as_of,args.source_dir)
    save_evaluation(frame,scores,args.as_of,args.source_dir,args.output_root)
