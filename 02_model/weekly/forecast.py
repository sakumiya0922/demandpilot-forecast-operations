"""日曜までの実績で翌週を予測し、入力特徴量とモデルIDを履歴としてDBへ保存する。"""
import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import psycopg

from daily.daily_batch import read_daily_csv
from training.training_data import build_daily_features, FEATURE_COLUMNS, UNKNOWN_PERIODS
from training.model import matrix


def prediction_features(daily, codes, origin):
    origin = pd.Timestamp(origin)
    if origin.dayofweek != 6:
        raise ValueError("予測基準日は日曜日を指定してください")
    start = origin - pd.Timedelta(days=27)
    frame = build_daily_features(daily, codes, start, origin, UNKNOWN_PERIODS)
    result = frame.loc[frame.sales_date.eq(origin), FEATURE_COLUMNS].copy()
    if len(result) != len(codes) or result.isna().any().any():
        raise ValueError("28日履歴が不足、または共通空白期間を含んでいます")
    return result.reset_index(drop=True)


def verify_period(connection, start, end, source_dir):
    # 学習時と同じ日別CSVが正本。対象期間の取込漏れを販売ゼロと誤認しないよう照合する。
    # CSV自体の欠落は検知できないため、分割済みの出力を完全に保持することが前提。
    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    expected_transactions = {}
    expected_daily = {}
    files = []
    for path in sorted(Path(source_dir).glob('*.csv')):
        if not start.isoformat() <= path.stem <= end.isoformat():
            continue
        day = date.fromisoformat(path.stem)
        accepted, _ = read_daily_csv(path, day)
        files.append(path.name)
        for row in accepted.itertuples(index=False):
            source_row = int(row.source_row)
            if source_row in expected_transactions:
                raise ValueError("履歴CSV間で元行番号が重複しています")
            expected_transactions[source_row] = (row.StockCode, day, int(row.Quantity))
        for code, quantity in accepted.groupby('StockCode').Quantity.sum().items():
            expected_daily[(day, code)] = int(quantity)
    if not files:
        raise ValueError("履歴CSVがありません")
    # 日次バッチとの並行実行でも、明細と集計を同じDBスナップショットで読む。
    transactions = connection.execute('''
        SELECT source_row, stock_code, invoice_date::date, quantity
        FROM transactions WHERE invoice_date >= %s AND invoice_date < %s
    ''', (start, end + timedelta(days=1))).fetchall()
    rows = connection.execute('''
        SELECT stock_code, sales_date, quantity FROM daily_sales
        WHERE sales_date BETWEEN %s AND %s ORDER BY stock_code, sales_date
    ''', (start, end)).fetchall()
    actual_transactions = {r[0]: (r[1], r[2], r[3]) for r in transactions}
    actual_daily = {(r[1], r[0]): r[2] for r in rows}
    if actual_transactions != expected_transactions or actual_daily != expected_daily:
        raise ValueError("履歴CSVとDBが不一致です。取込漏れ・集計・CSVの欠落を確認してください")
    return pd.DataFrame(rows, columns=['stock_code', 'sales_date', 'quantity'])

def verified_history(database_url, origin, source_dir, codes=None):
    origin = pd.Timestamp(origin).date()
    with psycopg.connect(database_url) as connection:
        connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        start = origin - timedelta(days=27)
        daily = verify_period(connection, start, origin, source_dir)
        # 28日間売れていない固定対象品番も、過去の実績で品番の存在を確認する。
        # 履歴範囲より前の1行は共通の特徴量処理で除かれるので、特徴量は28日販売0になる。
        # 正本CSVの照合済み期間内へ架空の取引を追加せず、DBにも何も書き込まない。
        missing = sorted(set(codes or []) - set(daily.stock_code))
        if missing:
            previous = connection.execute(''' 
                SELECT DISTINCT ON (stock_code) stock_code, sales_date, quantity
                FROM daily_sales WHERE stock_code = ANY(%s) AND sales_date < %s
                ORDER BY stock_code, sales_date DESC
            ''', (missing, start)).fetchall()
            if previous:
                daily = pd.concat([daily, pd.DataFrame(previous, columns=daily.columns)], ignore_index=True)
        return daily



def save_forecasts(database_url, features, predictions, origin, run_id, model_uri, prediction_id):
    origin = pd.Timestamp(origin).date()
    prediction_id = UUID(str(prediction_id))
    predictions = np.asarray(predictions, dtype=float)
    if origin.weekday() != 6 or not run_id or not model_uri:
        raise ValueError("基準日またはモデル情報が不正です")
    if features.empty or features.stock_code.duplicated().any() or predictions.shape != (len(features),):
        raise ValueError("商品と予測数量の対応が不正です")
    numeric = features[['sales_sum_7','sales_sum_28','sales_std_28','month']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any() or not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("特徴量または予測値が不正です")
    if not features.month.eq(origin.month).all():
        raise ValueError("特徴量の月が予測基準日と一致しません")
    # 1予測実行の全商品を一括確定する。失敗時はすべてロールバック。
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.executemany('''
                INSERT INTO weekly_forecasts (
                    prediction_run_id, week_start, stock_code, forecast_origin,
                    predicted_quantity, model_name, mlflow_run_id, model_uri,
                    sales_sum_7, sales_sum_28, sales_std_28, month
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ''', [(prediction_id, origin + timedelta(days=1), row.stock_code, origin,
                   float(predictions[i]), 'lightgbm', run_id, model_uri,
                   float(row.sales_sum_7), float(row.sales_sum_28), float(row.sales_std_28), int(row.month))
                  for i, row in enumerate(features.itertuples(index=False))])
    return str(prediction_id)


def forecast(database_url, origin, run_id, products_path, source_dir, tracking_dir, prediction_id=None):
    origin = pd.Timestamp(origin)
    if origin.dayofweek != 6:
        raise ValueError('予測基準日は日曜日を指定してください')
    tracking_dir = Path(tracking_dir).resolve()
    if not (tracking_dir/'mlflow.db').exists():
        raise ValueError('保存済みのMLflow管理DBがありません')
    mlflow.set_tracking_uri('sqlite:///'+str(tracking_dir/'mlflow.db'))
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    if run.info.status != 'FINISHED':
        raise ValueError('正常終了したMLflow実行を指定してください')
    conditions_path = client.download_artifacts(run_id, 'conditions.json')
    conditions = json.loads(Path(conditions_path).read_text())
    if conditions['run_id'] != run_id or conditions['features'] != FEATURE_COLUMNS:
        raise ValueError('モデルの実行IDまたは特徴量定義が一致しません')
    if pd.Timestamp(conditions['final_training']['last_label_date']) > origin:
        raise ValueError('指定モデルは予測基準日より未来の実績で学習されています')
    raw = Path(products_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != conditions['product_csv_sha256']:
        raise ValueError('学習時と対象商品一覧が異なります')
    codes = pd.read_csv(products_path, dtype={'stock_code':str}).stock_code.tolist()
    if len(codes) != 100 or codes != conditions['product_codes']:
        raise ValueError('学習時の固定100品番が必要です')
    daily = verified_history(database_url, origin, source_dir, codes)
    features = prediction_features(daily, codes, origin)
    model_uri = conditions['model_uri']
    model = mlflow.lightgbm.load_model(model_uri)
    predictions = np.maximum(0, model.predict(matrix(features, codes)))
    identifier = save_forecasts(database_url, features, predictions, origin, run_id, model_uri,
                                prediction_id or uuid4())
    report = {'prediction_run_id':identifier, 'mlflow_run_id':run_id, 'model_uri':model_uri,
              'forecast_origin':str(origin.date()), 'week_start':str((origin+pd.Timedelta(days=1)).date()),
              'week_end':str((origin+pd.Timedelta(days=7)).date()), 'products':len(features),
              'predicted_quantity_total':float(predictions.sum())}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', required=True, type=date.fromisoformat)
    parser.add_argument('--run-id', required=True, help='手動で指定する学習時のMLflow実行ID')
    parser.add_argument('--products', type=Path, default=Path('01_data/analysis/target_products.csv'))
    parser.add_argument('--source-dir', type=Path, default=Path('01_data/storage/daily'))
    parser.add_argument('--tracking-dir', type=Path, default=Path('02_model/mlflow'))
    parser.add_argument('--prediction-id', type=UUID, help='省略時は新規UUID。既存IDの再使用はエラー')
    args = parser.parse_args()
    if not os.environ.get('DATABASE_URL'):
        parser.error('DATABASE_URLを設定してください')
    forecast(os.environ['DATABASE_URL'], args.origin, args.run_id, args.products,
             args.source_dir, args.tracking_dir, args.prediction_id)
