"""専用の空DBで、初期登録→6月検証→7月独立テスト→運用開始モデルを準備する。"""
import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import psycopg

from daily.daily_batch import run_daily_batch
from training.training_data import create_training_data
from training.weekly_operations import run

START = date(2010, 12, 1)
END = date(2011, 7, 31)
ROOT = Path(__file__).resolve().parents[2]


def database_summary(database_url):
    with psycopg.connect(database_url) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        result = {}
        for table, column in [("transactions", "invoice_date"), ("daily_sales", "sales_date")]:
            first, last, count = connection.execute(
                f"SELECT MIN({column})::date, MAX({column})::date, COUNT(*) FROM {table}"
            ).fetchone()
            if first is not None and (first < START or last > END):
                raise ValueError("DBに初期期間外の実績があります")
            result[table] = {"start": str(first), "end": str(last), "rows": count}
        return result


def select_products(database_url):
    # 既存SQLは2010-12-06〜2011-05-29だけで選定する。6月以降は使わない。
    source = ROOT / "01_data/analysis/select_products.sql"
    query = source.read_text().split("COPY (", 1)[1].rsplit(") TO STDOUT", 1)[0]
    with psycopg.connect(database_url) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        cursor = connection.execute(query)
        products = pd.DataFrame(cursor.fetchall(), columns=[c.name for c in cursor.description])
    if len(products) != 100:
        raise ValueError("学習期間だけでは100品番を選定できません")
    return products


def initialize(database_url, source_dir, output_root, tracking_dir):
    # 明示的な専用環境だけで実行。既存のDBへの再投入・削除は拒否する。
    if os.environ.get("DEMANDPILOT_INITIAL_DB") != "1":
        raise ValueError("専用Compose環境でDEMANDPILOT_INITIAL_DB=1を指定してください")
    before = database_summary(database_url)
    if any(v["rows"] for v in before.values()):
        raise ValueError("初期登録先DBは空である必要があります。既存DBは変更しません")
    paths = sorted(p for p in Path(source_dir).glob("*.csv")
                   if START <= date.fromisoformat(p.stem) <= END)
    if not paths or paths[0].stem != str(START) or paths[-1].stem != str(END):
        raise ValueError("初期期間の両端の日次CSVがありません")
    output = Path(output_root) / ("initial-" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    manifest = [{"date": p.stem, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    # CSVのない日は元データに明細がない日。欠落・休業の区別はできない。
    absent = sorted(set(pd.date_range(START, END).strftime("%Y-%m-%d")) - {p.stem for p in paths})
    reports = [run_daily_batch(p, date.fromisoformat(p.stem), database_url) for p in paths]
    counts = database_summary(database_url)
    if counts["transactions"]["rows"] != sum(row["accepted_rows"] for row in reports):
        raise ValueError("DB明細件数と前処理後件数が一致しません")
    if counts["daily_sales"]["rows"] != sum(row["daily_sales_rows"] for row in reports):
        raise ValueError("DB日次集計件数が一致しません")
    products = select_products(database_url)
    products_path = output / "products.csv"
    products.to_csv(products_path, index=False)
    training_path = output / "training.csv"
    create_training_data(database_url, products_path, str(START), str(END), training_path)
    summary = run(training_path, products_path, tracking_dir,
                  experiment_name="demandpilot-initial-training", initial_only=True)
    ingestion = {"database": counts, "source_files": manifest, "no_source_dates": absent,
                 "daily_reports": reports, "selection_start": "2010-12-06",
                 "selection_end": "2011-05-29"}
    import mlflow
    client = mlflow.tracking.MlflowClient()
    client.log_dict(summary["parent_run_id"], ingestion, "input/ingestion.json")
    client.log_artifact(summary["parent_run_id"], str(ROOT / "01_data/analysis/select_products.sql"), "source")
    client.log_artifact(summary["parent_run_id"], str(Path(__file__)), "source")
    result = {"ingestion": ingestion, "training": summary}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("初期処理結果: " + str(output / "summary.json"), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("01_data/storage/daily"))
    parser.add_argument("--output-root", type=Path, default=Path("02_model/results"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("02_model/mlflow"))
    args = parser.parse_args()
    initialize(os.environ["DATABASE_URL"], args.source_dir, args.output_root, args.tracking_dir)
