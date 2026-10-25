"""固定100品番から、日曜基準の特徴量と翌7日間の正解を作る。学習はしない。"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg


# この週は全商品に共通して元の記録がなく、休業か欠落か未確定。
UNKNOWN_PERIODS = [("2010-12-27", "2011-01-02")]
FEATURE_COLUMNS = ["sales_sum_7", "sales_sum_28", "sales_std_28", "month", "stock_code"]


def build_daily_features(daily, product_codes, start, end, unknown_periods=UNKNOWN_PERIODS):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or not product_codes or len(product_codes) != len(set(product_codes)):
        raise ValueError("期間または商品一覧が不正です")
    daily = daily.copy()
    daily["sales_date"] = pd.to_datetime(daily["sales_date"])
    if daily.empty or daily[["stock_code", "sales_date", "quantity"]].isna().any().any():
        raise ValueError("販売実績が空、または欠損があります")
    if not daily["sales_date"].eq(daily["sales_date"].dt.normalize()).all():
        raise ValueError("日次実績は日付単位で指定してください")
    if daily.duplicated(["stock_code", "sales_date"]).any():
        raise ValueError("商品・日付が重複しています")
    if not np.isfinite(daily.quantity).all() or daily.quantity.lt(0).any():
        raise ValueError("販売数量は有限の非負値が必要です")
    if not set(product_codes).issubset(set(daily.stock_code)):
        raise ValueError("実績のない品番が対象一覧にあります")

    # 元のカレンダーを維持する。商品別の販売なしは0として扱う。
    # これは指定期間の日次取り込み完了が前提であり、欠品補正済み需要ではない。
    calendar = pd.MultiIndex.from_product(
        [product_codes, pd.date_range(start, end)], names=["stock_code", "sales_date"]
    )
    frame = daily.set_index(["stock_code", "sales_date"])[["quantity"]].reindex(calendar, fill_value=0).reset_index()
    frame = frame.sort_values(["stock_code", "sales_date"]).reset_index(drop=True)
    # 共通空白週は0にせず不明値に戻し、それを含む履歴や正解を除外する。
    for first, last in unknown_periods:
        frame.loc[frame.sales_date.between(first, last), "quantity"] = np.nan

    grouped = frame.groupby("stock_code", sort=False).quantity
    for window in [7, 28]:
        # 基準日当日まで。未来の値は特徴量に使わない。
        frame[f"sales_sum_{window}"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).sum()
        )
    frame["sales_std_28"] = grouped.transform(lambda s: s.rolling(28, min_periods=28).std())
    frame["month"] = frame.sales_date.dt.month
    return frame


def build_training_data(daily, product_codes, start, end, unknown_periods=UNKNOWN_PERIODS):
    end = pd.Timestamp(end)
    frame = build_daily_features(daily, product_codes, start, end, unknown_periods)
    grouped = frame.groupby("stock_code", sort=False).quantity
    # 翌日～7日後の全実績がある場合だけ正解を作る。
    future = pd.concat([grouped.shift(-offset) for offset in range(1, 8)], axis=1)
    frame["target"] = future.sum(axis=1, min_count=7)
    frame["history_start"] = frame.sales_date - pd.Timedelta(days=27)
    frame["target_start"] = frame.sales_date + pd.Timedelta(days=1)
    frame["target_end"] = frame.sales_date + pd.Timedelta(days=7)
    frame = frame.loc[frame.sales_date.dt.dayofweek.eq(6) & frame.target_end.le(end)]
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["target"])
    frame = frame.rename(columns={"sales_date": "forecast_origin"})
    return frame[["stock_code", "forecast_origin", "history_start", "target_start", "target_end",
                  "sales_sum_7", "sales_sum_28", "sales_std_28", "month", "target"]].reset_index(drop=True)


def create_training_data(database_url, products_path, start, end, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("既存の学習データは上書きしません。別の出力先を指定してください")
    products = pd.read_csv(products_path, dtype={"stock_code": str})
    if len(products) != 100 or products.stock_code.isna().any() or products.stock_code.duplicated().any():
        raise ValueError("固定した100品番が必要です")
    codes = products.stock_code.tolist()
    # 明細は再集計せず、既存のSQL日次集計を読み取り専用で使う。
    with psycopg.connect(database_url) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        rows = connection.execute("""
            SELECT stock_code, sales_date, quantity FROM daily_sales
            WHERE stock_code = ANY(%s) AND sales_date BETWEEN %s AND %s
            ORDER BY stock_code, sales_date
        """, (codes, start, end)).fetchall()
        observed = connection.execute("""
            SELECT MIN(sales_date), MAX(sales_date) FROM daily_sales
            WHERE sales_date BETWEEN %s AND %s
        """, (start, end)).fetchone()
    if observed != (pd.Timestamp(start).date(), pd.Timestamp(end).date()):
        raise ValueError("指定期間の端に取引記録がありません。取り込み済み期間を確認してください")
    daily = pd.DataFrame(rows, columns=["stock_code", "sales_date", "quantity"])
    training = build_training_data(daily, codes, start, end)
    if training.empty:
        raise ValueError("完全な履歴と正解を持つ学習行がありません")
    output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(output, index=False, date_format="%Y-%m-%d")
    print(f"保存先: {output}")
    print(f"{len(training):,}行 / {training.stock_code.nunique()}品番 / {training.forecast_origin.nunique()}基準日")
    print(f"予測基準日: {training.forecast_origin.min().date()}〜{training.forecast_origin.max().date()}")
    print(f"正解の期間: {training.target_start.min().date()}〜{training.target_end.max().date()}")
    print("商品選定に使った既知期間のデータです。学習・検証分割やモデル学習は未実施です。")
    return training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-12-01")
    parser.add_argument("--end", default="2011-05-31")
    parser.add_argument("--products", type=Path, default=Path("01_data/analysis/target_products.csv"))
    parser.add_argument("--output", type=Path, default=Path("01_data/storage/training/weekly_training.csv"))
    args = parser.parse_args()
    if not os.environ.get("DATABASE_URL"):
        parser.error("DATABASE_URLを設定してください")
    create_training_data(os.environ["DATABASE_URL"], args.products, args.start, args.end, args.output)
