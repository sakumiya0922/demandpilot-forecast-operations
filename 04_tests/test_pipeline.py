"""実際のPostgreSQLに隔離スキーマを作り、日次処理の整合性を検証する。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import psycopg
import pytest

from daily.daily_batch import run_daily_batch
from preparation.prepare_csv import prepare_csv, REQUIRED_COLUMNS




def write_csv(path, rows):
    # source_row, 商品, 数量以外はテスト用の共通値。
    frame = pd.DataFrame([{
        "source_row": number, "InvoiceNo": "536365", "StockCode": stock,
        "Description": "TEST", "Quantity": quantity, "InvoiceDate": "2010-12-01 08:26:00",
        "UnitPrice": "2.55", "CustomerID": "", "Country": "United Kingdom",
    } for number, stock, quantity in rows])
    frame.to_csv(path, index=False)
    return frame


def snapshot(database):
    with psycopg.connect(database) as connection:
        return {
            "transactions": connection.execute("SELECT * FROM transactions ORDER BY source_row").fetchall(),
            "daily_sales": connection.execute("SELECT * FROM daily_sales ORDER BY sales_date, stock_code").fetchall(),
            "weekly_forecasts": connection.execute("SELECT * FROM weekly_forecasts").fetchall(),
        }


def test_rerun_keeps_identical_rows_and_preserves_duplicate_business_rows(tmp_path, database):
    path = tmp_path / "day.csv"
    # 内容が同じでも元行番号が異なる2明細を両方保持する。
    write_csv(path, [(2, "001A", 3), (3, "001A", 3), (4, "NA", 2)])
    run_daily_batch(path, "2010-12-01", database)
    first = snapshot(database)
    run_daily_batch(path, "2010-12-01", database)
    assert snapshot(database) == first
    assert len(first["transactions"]) == 3
    assert first["daily_sales"] == [(date(2010, 12, 1), "001A", 6), (date(2010, 12, 1), "NA", 2)]
    assert first["weekly_forecasts"] == []


def test_corrected_file_replaces_day_and_does_not_change_other_days(tmp_path, database):
    path = tmp_path / "day.csv"
    other = write_csv(path, [(10, "OTHER", 7)])
    other["InvoiceDate"] = "2010-12-02 09:00:00"
    other.to_csv(path, index=False)
    run_daily_batch(path, "2010-12-02", database)
    write_csv(path, [(2, "A", 3), (3, "B", 4)])
    run_daily_batch(path, "2010-12-01", database)
    write_csv(path, [(2, "A", 9)])
    run_daily_batch(path, "2010-12-01", database)
    state = snapshot(database)
    assert len(state["transactions"]) == 2
    assert state["daily_sales"] == [(date(2010, 12, 1), "A", 9), (date(2010, 12, 2), "OTHER", 7)]


def test_exclusions_zero_price_and_all_excluded_rerun(tmp_path, database):
    path = tmp_path / "day.csv"
    frame = write_csv(path, [(2, "A", 5), (3, "A", -2), (4, "A", 3), (5, "A", 1)])
    frame.loc[0, "UnitPrice"] = "0"
    frame.loc[2, "InvoiceNo"] = "C123"
    frame.loc[3, "UnitPrice"] = "-1"
    frame.to_csv(path, index=False)
    report = run_daily_batch(path, "2010-12-01", database)
    assert (report['accepted_rows'], report['cancelled'], report['nonpositive_quantity'], report['negative_price']) == (1, 1, 1, 1)
    assert snapshot(database)["daily_sales"][0][2] == 5
    frame["InvoiceNo"] = "C123"
    frame.to_csv(path, index=False)
    run_daily_batch(path, "2010-12-01", database)
    assert snapshot(database)["transactions"] == []
    assert snapshot(database)["daily_sales"] == []


@pytest.mark.parametrize('column,value', [
    ('InvoiceDate', '2010-12-02 00:00:00'), ('InvoiceDate', ''),
    ('Quantity', '1.5'), ('UnitPrice', 'NaN'), ('StockCode', ''),
    ('source_row', '1'), ('Quantity', 'not-a-number'),
])
def test_bad_data_leaves_existing_day_unchanged(tmp_path, database, column, value):
    path = tmp_path / "day.csv"
    frame = write_csv(path, [(2, "A", 5)])
    run_daily_batch(path, "2010-12-01", database)
    first = snapshot(database)
    frame[column] = value
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, TypeError)):
        run_daily_batch(path, "2010-12-01", database)
    assert snapshot(database) == first


def test_duplicate_source_row_is_rejected(tmp_path, database):
    path = tmp_path / "day.csv"
    write_csv(path, [(2, "A", 5), (2, "A", 5)])
    with pytest.raises(ValueError, match="source_row"):
        run_daily_batch(path, "2010-12-01", database)
    assert snapshot(database)["transactions"] == []


def test_aggregation_failure_rolls_back_both_tables(tmp_path, database):
    path = tmp_path / "day.csv"
    write_csv(path, [(2, "A", 5)])
    run_daily_batch(path, "2010-12-01", database)
    first = snapshot(database)
    # 明細COPYは成功するが、続く集計INSERTが失敗する状況を作る。
    with psycopg.connect(database) as connection:
        connection.execute("ALTER TABLE daily_sales ADD CHECK (quantity < 10)")
    write_csv(path, [(2, "A", 20)])
    with pytest.raises(psycopg.errors.CheckViolation):
        run_daily_batch(path, "2010-12-01", database)
    assert snapshot(database) == first


def test_source_row_conflict_with_other_day_rolls_back(tmp_path, database):
    path = tmp_path / "day.csv"
    frame = write_csv(path, [(2, "A", 5)])
    run_daily_batch(path, "2010-12-01", database)
    first = snapshot(database)
    frame['InvoiceDate'] = '2010-12-02 00:00:00'
    frame.to_csv(path, index=False)
    with pytest.raises(psycopg.errors.UniqueViolation):
        run_daily_batch(path, "2010-12-02", database)
    assert snapshot(database) == first


def test_same_day_concurrent_reruns_are_serialized(tmp_path, database):
    path = tmp_path / "day.csv"
    write_csv(path, [(2, "A", 5)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [pool.submit(run_daily_batch, path, "2010-12-01", database) for _ in range(2)]
        for run in runs:
            run.result()
    assert len(snapshot(database)["transactions"]) == 1
    assert snapshot(database)["daily_sales"] == [(date(2010, 12, 1), 'A', 5)]


def test_excel_split_preserves_rows_and_dates(tmp_path):
    frame = write_csv(tmp_path / 'source.csv', [(2, '001A', 5), (3, 'NA', -2)])
    frame.loc[1, 'InvoiceDate'] = '2010-12-02 00:00:00'
    frame['InvoiceDate'] = pd.to_datetime(frame.InvoiceDate)
    excel = tmp_path / 'source.xlsx'
    frame[REQUIRED_COLUMNS].to_excel(excel, index=False)
    directory = tmp_path / 'daily'
    assert prepare_csv(excel, directory) == 2
    first = pd.read_csv(directory / '2010-12-01.csv', dtype=str, keep_default_na=False)
    second = pd.read_csv(directory / '2010-12-02.csv', dtype=str, keep_default_na=False)
    assert first.source_row.tolist() == ['2'] and second.source_row.tolist() == ['3']
    assert first.StockCode.tolist() == ['001A'] and second.StockCode.tolist() == ['NA']
    assert second.Quantity.tolist() == ['-2']
    with pytest.raises(ValueError, match='空'):
        prepare_csv(excel, directory)
