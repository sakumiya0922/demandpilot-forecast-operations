
"""1日分の取引CSVを読み込み、PostgreSQLに保存して日次販売数量を集計する。"""

import argparse
import json
import os
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from preparation.prepare_csv import REQUIRED_COLUMNS


# CSVの読み込み・品質チェック

def read_daily_csv(csv_path, batch_date):
    """
    指定日のCSVを読み込み、不正なデータを確認する。

    戻り値
    -------
    valid_data : DataFrame
        販売実績として採用する取引明細
    report : dict
        入力件数や除外件数などの処理結果
    """

    # すべて文字列として読み込み、商品コード「NA」などを欠損値に変換しない
    data = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False
    )

    # 必要な列がそろっているか、データが空でないか確認する
    required_columns = ["source_row"] + REQUIRED_COLUMNS

    if set(data.columns) != set(required_columns):
        raise ValueError("CSVの列が不正です")

    if data.empty:
        # 単に空のCSVを正常扱いしない。元Excelの0件確認記録とハッシュを照合する。
        import hashlib
        try:
            evidence = json.loads((Path(csv_path).parent / "empty_days.json").read_text())
            entry = evidence["days"][batch_date.isoformat()]
            if (entry["rows"] != 0 or len(evidence["source_sha256"]) != 64
                    or entry["csv_sha256"] != hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()):
                raise ValueError("空CSVの根拠が不正です")
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise ValueError("CSVが空です。元Excelの取引0件確認がありません") from error
        return data, {"date": batch_date.isoformat(), "input_rows": 0, "cancelled": 0,
                      "nonpositive_quantity": 0, "negative_price": 0, "accepted_rows": 0,
                      "verified_empty": True, "source_sha256": evidence["source_sha256"]}

    # 単価は後で正確な数値（Decimal）に変換するため、元の文字列を保存する
    original_prices = data["UnitPrice"].copy()

    # 伝票番号と商品コードの前後の空白を削除し、空欄を確認する
    for column in ["InvoiceNo", "StockCode"]:
        data[column] = data[column].str.strip()

        if data[column].eq("").any():
            raise ValueError(f"{column}に空欄があります")

    # 行番号・販売数量・単価が正しい数値か確認する
    for column in ["source_row", "Quantity", "UnitPrice"]:

        values = pd.to_numeric(data[column], errors="raise")

        # NaNや無限大が含まれていないか確認する
        if not np.isfinite(values).all():
            raise ValueError(f"{column}に不正な数値があります")

        # 行番号と販売数量は整数である必要がある
        if column != "UnitPrice":
            if not values.mod(1).eq(0).all():
                raise ValueError(f"{column}は整数である必要があります")

        data[column] = values

    # 行番号が正しい範囲にあり、CSV内で重複していないか確認する
    invalid_source_row = (
        data["source_row"].lt(2).any()
        or data["source_row"].gt(2**63 - 1).any()
        or data["source_row"].duplicated().any()
    )

    if invalid_source_row:
        raise ValueError("source_rowの範囲または重複が不正です")

    # 販売数量がPostgreSQLのINTEGER型で扱える範囲か確認する
    if not data["Quantity"].between(-(2**31), 2**31 - 1).all():
        raise ValueError("Quantityが整数の範囲外です")

    # 取引日時を日時型に変換し、すべて指定日の取引か確認する
    invoice_dates = pd.to_datetime(
        data["InvoiceDate"],
        errors="raise"
    )

    if invoice_dates.isna().any():
        raise ValueError("取引日時に欠損があります")

    if invoice_dates.dt.tz is not None:
        raise ValueError("タイムゾーン付き日時は使用できません")

    if not invoice_dates.dt.date.eq(batch_date).all():
        raise ValueError("指定日以外の取引が含まれています")

    data["InvoiceDate"] = invoice_dates

    # キャンセル・数量0以下・単価マイナスの取引を除外する
    # 除外理由が重複して集計されないよう、順番に判定する

    # 伝票番号がCで始まる取引はキャンセルとする
    cancelled = data["InvoiceNo"].str.upper().str.startswith("C")

    # キャンセル以外で販売数量が0以下の取引
    nonpositive_quantity = (
        ~cancelled
        & data["Quantity"].le(0)
    )

    # 上記以外で単価がマイナスの取引
    negative_price = (
        ~cancelled
        & ~nonpositive_quantity
        & data["UnitPrice"].lt(0)
    )

    # いずれの除外条件にも該当しない取引だけを残す
    excluded = cancelled | nonpositive_quantity | negative_price
    valid_data = data.loc[~excluded].copy()

    # 単価0の取引は除外しない
    # 単価をDecimalに変換し、浮動小数点の丸め誤差を避ける
    valid_data["UnitPrice"] = (
        original_prices.loc[valid_data.index].map(Decimal)
    )

    report = {
        "date": batch_date.isoformat(),
        "input_rows": len(data),
        "cancelled": int(cancelled.sum()),
        "nonpositive_quantity": int(nonpositive_quantity.sum()),
        "negative_price": int(negative_price.sum()),
        "accepted_rows": len(valid_data),
    }

    return valid_data, report


# PostgreSQLへの保存・日次集計

def run_daily_batch(csv_path, batch_date, database_url):
    """
    指定日の取引データをDBに保存し、日次販売数量を集計する。

    同じ日を再実行しても二重計上されないよう、
    対象日の既存データを削除してから登録し直す。
    """

    if isinstance(batch_date, str):
        batch_date = date.fromisoformat(batch_date)

    valid_data, report = read_daily_csv(csv_path, batch_date)

    # 処理対象の日付範囲を決める
    # 例：2010-12-01 00:00以上、2010-12-02 00:00未満
    start_date = batch_date.isoformat()
    next_date = (batch_date + timedelta(days=1)).isoformat()

    # 正常終了ならCOMMIT、途中でエラーが出ればROLLBACKする
    with psycopg.connect(database_url) as connection:

        # 同じ日のバッチが同時にDBを更新しないようロックする
        # 異なる日付のバッチは独立して処理できる
        connection.execute(
            "SELECT pg_advisory_xact_lock(20261016, %s)",
            (batch_date.toordinal(),)
        )

        connection.execute(
            "DELETE FROM daily_sales WHERE sales_date = %s",
            (batch_date,)
        )

        # 対象日の既存取引明細も削除する
        connection.execute(
            """
            DELETE FROM transactions
            WHERE invoice_date >= %s
              AND invoice_date < %s
            """,
            (start_date, next_date)
        )

        with connection.cursor() as cursor:
            with cursor.copy(
                """
                COPY transactions (
                    source_row,
                    invoice_no,
                    stock_code,
                    invoice_date,
                    quantity,
                    unit_price,
                    description,
                    customer_id,
                    country
                )
                FROM STDIN
                """
            ) as copy:

                for row in valid_data.itertuples(index=False):
                    copy.write_row((
                        int(row.source_row),
                        row.InvoiceNo,
                        row.StockCode,
                        row.InvoiceDate.to_pydatetime(),
                        int(row.Quantity),
                        row.UnitPrice,
                        row.Description or None,
                        row.CustomerID or None,
                        row.Country or None,
                    ))

        # SQLで商品別・日別の販売数量を集計し、daily_salesに保存する
        connection.execute(
            """
            INSERT INTO daily_sales (
                sales_date,
                stock_code,
                quantity
            )
            SELECT
                invoice_date::date,
                stock_code,
                SUM(quantity)
            FROM transactions
            WHERE invoice_date >= %s
              AND invoice_date < %s
            GROUP BY
                invoice_date::date,
                stock_code
            """,
            (start_date, next_date)
        )

        result = connection.execute(
            """
            SELECT COUNT(*)
            FROM daily_sales
            WHERE sales_date = %s
            """,
            (batch_date,)
        )

        report["daily_sales_rows"] = result.fetchone()[0]

    # COMMITが完了した場合だけ結果を出力する
    print(json.dumps(report, ensure_ascii=False))

    return report


# コマンドライン実行

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--date",
        required=True,
        type=date.fromisoformat
    )

    parser.add_argument(
        "--csv",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    # Docker Composeから渡されるDB接続情報を取得する
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        parser.error("DATABASE_URLを設定してください")

    run_daily_batch(
        args.csv,
        args.date,
        database_url
    )
