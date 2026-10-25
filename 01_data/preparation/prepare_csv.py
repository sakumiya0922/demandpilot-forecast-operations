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


def prepare_verified_calendar(excel_path, existing_dir, output_dir, start, end):
    """元Excelで0件を確認できた日に限り、検証用の空CSVを作る。既存入力は保持。"""
    import hashlib
    import json
    import shutil
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    output_dir = Path(output_dir)
    if start > end or output_dir.exists():
        raise ValueError("期間が不正、または検証用出力先が既に存在します")
    original = pd.read_excel(excel_path, engine="openpyxl", usecols=["InvoiceDate"])
    dates = pd.to_datetime(original.InvoiceDate, errors="raise").dt.normalize()
    if dates.isna().any() or start < dates.min() or end > dates.max():
        raise ValueError("元Excelで対象期間を確認できません")
    missing = [day for day in pd.date_range(start, end)
               if not (Path(existing_dir) / f"{day:%Y-%m-%d}.csv").is_file()]
    counts = {f"{day:%Y-%m-%d}": int(dates.eq(day).sum()) for day in missing}
    if any(counts.values()):
        raise ValueError(f"取引がある日のCSVが欠落しています: {counts}")
    source_hash = hashlib.sha256(Path(excel_path).read_bytes()).hexdigest()
    output_dir.mkdir(parents=True)
    for path in Path(existing_dir).glob("*.csv"):
        shutil.copyfile(path, output_dir / path.name)
    evidence = {"source_sha256": source_hash, "days": {}}
    for day, count in counts.items():
        path = output_dir / f"{day}.csv"
        pd.DataFrame(columns=["source_row"] + REQUIRED_COLUMNS).to_csv(path, index=False)
        evidence["days"][day] = {"rows": count, "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output_dir / "empty_days.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"source_sha256": source_hash, "verified_zero_days": counts}, indent=2))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("01_data/storage/daily"))
    parser.add_argument("--existing-dir", type=Path)
    parser.add_argument("--calendar-start")
    parser.add_argument("--calendar-end")
    args = parser.parse_args()
    if args.existing_dir:
        if not args.calendar_start or not args.calendar_end:
            parser.error("--calendar-startと--calendar-endが必要です")
        prepare_verified_calendar(args.excel, args.existing_dir, args.output_dir,
                                  args.calendar_start, args.calendar_end)
    else:
        prepare_csv(args.excel, args.output_dir)
