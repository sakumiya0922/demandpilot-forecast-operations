"""固定したUCI Online Retail Excelを、元行番号付きの日別CSVに分割する。"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country",
]


def prepare_csv(excel_path, output_dir):
    output_dir = Path(output_dir)
    # 再分割で既存CSVを上書きしない。修正時は新しい出力ディレクトリを指定する。
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("出力ディレクトリは空にしてください")
    frame = pd.read_excel(excel_path, engine="openpyxl", keep_default_na=False, dtype={
        "InvoiceNo": str, "StockCode": str, "CustomerID": str,
    })
    if set(frame.columns) != set(REQUIRED_COLUMNS) or frame.empty:
        raise ValueError("元Excelの列または件数が不正です")
    # ヘッダが1行目なので、明細の識別子はExcelの2行目から始まる。
    frame.insert(0, "source_row", range(2, len(frame) + 2))
    dates = pd.to_datetime(frame["InvoiceDate"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("日付の欠損・タイムゾーン付き日付は分割できません")
    frame["InvoiceDate"] = dates
    output_dir.mkdir(parents=True, exist_ok=True)
    # 返品・キャンセルも分割時には保持し、日次バッチで理由別に除外する。
    count = 0
    for day, rows in frame.groupby(dates.dt.date, sort=True):
        rows.to_csv(output_dir / f"{day.isoformat()}.csv", index=False,
                    date_format="%Y-%m-%d %H:%M:%S")
        count += 1
    print(f"{len(frame)}行を{count}日分に分割: {output_dir}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("01_data/storage/daily"))
    args = parser.parse_args()
    prepare_csv(args.excel, args.output_dir)
