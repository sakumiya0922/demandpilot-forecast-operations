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

# 8〜9月の運用はこの制御経路に集約する。モデル採用・シャドー運用は行わない。
from contextlib import contextmanager
from datetime import timedelta
from uuid import uuid5, NAMESPACE_URL
import mlflow
import mlflow.lightgbm
from psycopg.types.json import Jsonb
from training.training_data import FEATURE_COLUMNS
from training.weekly_operations import (model_fit, register_version, update_bad_streak,
                                        log_weekly_history)
from weekly.forecast import forecast_model, verify_period, verify_processed_period as processed_period
from evaluation.evaluate import (compare_actuals, save_database_evaluation,
                                 load_database_evaluation)

OPS_START = date(2011, 8, 1)
OPS_END = date(2011, 10, 2)
FINAL_EVALUATION = date(2011, 10, 3)


def _read(database_url, query, params=()):
    with psycopg.connect(database_url) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        return connection.execute(query, params).fetchall()


def _clock(day, hour):
    return f"{day}T{hour:02}:00:00+09:00"


def _cutoff(database_url, origin):
    rows = _read(database_url, """
        SELECT MAX(invoice_date)::date FROM transactions
        UNION ALL SELECT MAX(sales_date) FROM daily_sales
    """)
    if any(row[0] is not None and row[0] > origin for row in rows):
        raise ValueError("再現日時より未来の実績がDBにあります")


def check_calendar(source_dir):
    missing = [str(day.date()) for day in pd.date_range(OPS_START, OPS_END)
               if not (Path(source_dir) / f"{day:%Y-%m-%d}.csv").is_file()]
    if missing:
        raise ValueError("日次CSVが欠落しています: " + ", ".join(missing))



@contextmanager
def _event(client, experiment_id, key, name, resume_finished=False):
    found = client.search_runs([experiment_id], filter_string=f"tags.event_key = '{key}'")
    if len(found) > 1:
        raise ValueError("同じ運用イベントに複数のMLflow Runがあります")
    if found and found[0].info.status == "FINISHED" and not resume_finished:
        yield found[0], True
        return
    kwargs = {"run_id": found[0].info.run_id} if found else {
        "experiment_id": experiment_id, "run_name": name, "tags": {"event_key": key}}
    with mlflow.start_run(**kwargs, nested=mlflow.active_run() is not None) as event:
        yield event, False


def _handoff(client, version):
    registered = client.get_model_version("DemandPilotWeeklyLightGBM", str(version))
    run = client.get_run(registered.run_id)
    info = json.loads(Path(client.download_artifacts(run.info.run_id, "handoff.json")).read_text())
    if (run.info.status != "FINISHED" or info["model_run_id"] != registered.run_id
            or str(info["registered_model_version"]) != str(version)
            or len(set(info["product_codes"])) != 100 or info["features"] != FEATURE_COLUMNS
            or info["final_training"]["last_label_date"] > str(END)):
        raise ValueError("運用開始モデルの受け渡し情報が不正です")
    return info


def _retrain(database_url, source_dir, output, active, day, client, event_id):
    origin = day - timedelta(days=1)
    _cutoff(database_url, origin)
    # 日次登録済みの実績だけから、既存の特徴量・正解生成を再利用する。
    folder = output / f"retrain-{day}"
    folder.mkdir(exist_ok=True)
    products = folder / "products.csv"
    if not products.exists():
        pd.DataFrame({"stock_code": active["product_codes"]}).to_csv(products, index=False)
    training_path = folder / "training.csv"
    if not training_path.exists():
        create_training_data(database_url, products, str(START), str(origin), training_path)
    training = pd.read_csv(training_path, dtype={"stock_code": str}, parse_dates=["forecast_origin", "target_end"])
    if training.empty or training.target_end.max().date() > origin:
        raise ValueError("再学習に未確定の正解があります")
    # 登録直後の中断にも、同じ再学習イベントのモデルを再利用する。
    previous = []
    for version in client.search_model_versions("name='DemandPilotWeeklyLightGBM'"):
        run = client.get_run(version.run_id)
        if run.data.tags.get("parent_lifecycle_run_id") == event_id:
            previous.append(version)
    if previous:
        version = max(previous, key=lambda v: int(v.version))
        info = {"logical_model_version": active.get("logical_model_version", 2) + 1,
                "model_run_id": version.run_id, "registered_model_name": version.name,
                "registered_model_version": str(version.version), "model_uri": version.source,
                "trained_through": str(origin), "reason": "2_consecutive_weeks_WAPE_worse_than_baseline"}
    else:
        model = model_fit(training, active["product_codes"], active["hyperparameters"])
        info = register_version(model, training, active["product_codes"],
                                active.get("logical_model_version", 2) + 1, origin,
                                "2_consecutive_weeks_WAPE_worse_than_baseline", event_id)
    info.update({"features": FEATURE_COLUMNS, "product_codes": active["product_codes"],
                 "hyperparameters": active["hyperparameters"], "updated_at": _clock(day, 4),
                 "old_model_uri": active["model_uri"], "old_model_run_id": active["model_run_id"],
                 "source_start": str(START), "source_end": str(origin),
                 "final_training": {"first_origin": str(training.forecast_origin.min().date()),
                                    "last_origin": str(training.forecast_origin.max().date()),
                                    "last_label_date": str(training.target_end.max().date()), "rows": len(training)}})
    client.log_dict(info["model_run_id"], info, "handoff.json")
    client.log_artifact(info["model_run_id"], str(training_path), "data")
    conditions = dict(info, run_id=info["model_run_id"],
                      data_sha256=hashlib.sha256(training_path.read_bytes()).hexdigest())
    client.log_dict(info["model_run_id"], conditions, "conditions.json")
    client.log_dict(event_id, info, "new_model.json")
    return info


def operate(database_url, source_dir, output_root, tracking_dir, *, initial_version=None,
            replay_key="august-september-2011"):
    if os.environ.get("DEMANDPILOT_OPERATIONS_DB") != "1":
        raise ValueError("分離した検証DBでDEMANDPILOT_OPERATIONS_DB=1を指定してください")
    if initial_version is None:
        raise ValueError("運用開始モデルの登録バージョンを明示してください")
    check_calendar(source_dir)  # ファイルの存在だけを確認。未来の販売数量は読まない。
    tracking_dir = Path(tracking_dir).resolve()
    mlflow.set_tracking_uri("sqlite:///" + str(tracking_dir / "mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    initial = _handoff(client, initial_version)
    replay_id = uuid5(NAMESPACE_URL, replay_key + ":" + initial["model_uri"])
    output = Path(output_root) / str(replay_id)
    output.mkdir(parents=True, exist_ok=True)
    # 他の運用実行との競合を拒否。手動実行用のセッションロック。
    with psycopg.connect(database_url, autocommit=True) as lock:
        if not lock.execute("SELECT pg_try_advisory_lock(352,20110801)").fetchone()[0]:
            raise ValueError("別の時系列運用が実行中です")
        rows = _read(database_url, "SELECT replay_id,status,result FROM operation_replays")
        if rows and any(str(row[0]) != str(replay_id) for row in rows):
            raise ValueError("別の運用履歴があるDBは使いません")
        if rows and rows[0][1] == "completed":
            with psycopg.connect(database_url) as c:
                processed_period(c, OPS_START, OPS_END, source_dir)
            return rows[0][2]
        if not rows:
            _cutoff(database_url, END)
            bounds = _read(database_url, "SELECT MIN(sales_date),MAX(sales_date) FROM daily_sales")[0]
            if bounds != (START, END):
                raise ValueError("7月31日までの初期DBが必要です")
        name = "demandpilot-chronological-operations"
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(
            name, artifact_location=(tracking_dir / "artifacts").as_uri())
        if not client.get_experiment(experiment_id).artifact_location.startswith((tracking_dir / "artifacts").as_uri()):
            raise ValueError("実験の保存先が永続領域ではありません")
        mlflow.set_experiment(experiment_id=experiment_id)
        with _event(client, experiment_id, f"{replay_id}:replay", "chronological-operations", resume_finished=True) as (parent, _):
            with psycopg.connect(database_url) as c:
                c.execute("""INSERT INTO operation_replays VALUES (%s,%s,%s,%s,'running',NULL)
                             ON CONFLICT (replay_id) DO NOTHING""",
                          (replay_id, replay_key, Jsonb(initial), parent.info.run_id))
            active, streak, trace, updates = initial, 0, [], []
            for stamp in pd.date_range(OPS_START, FINAL_EVALUATION):
                day = stamp.date()
                if day.weekday() == 0:
                    previous_start = day - timedelta(days=7)
                    previous_id = uuid5(replay_id, str(previous_start))
                    forecasts = _read(database_url, "SELECT prediction_run_id FROM weekly_forecast_runs WHERE prediction_run_id=%s", (previous_id,))
                    if forecasts:
                        old = _read(database_url, "SELECT scores,consecutive_bad_weeks FROM weekly_evaluations WHERE prediction_run_id=%s", (previous_id,))
                        with _event(client, experiment_id, f"{replay_id}:evaluate:{day}", f"evaluate-{day}") as (event, finished):
                            if old:
                                scores, streak = old[0]
                                comparison, _ = load_database_evaluation(database_url, previous_id)
                            else:
                                _cutoff(database_url, day - timedelta(days=1))
                                with psycopg.connect(database_url) as c:
                                    processed_period(c, previous_start, day - timedelta(days=1), source_dir)
                                comparison, scores = compare_actuals(database_url, previous_id,
                                                                      day - timedelta(days=1), source_dir)
                                if len(comparison) != 100:
                                    raise ValueError("100品番の評価がそろっていません")
                                streak, _ = update_bad_streak(streak, scores)
                            if not finished:
                                mlflow.set_tags({"stage": "weekly_monitor", "prediction_run_id": str(previous_id),
                                                 "evaluated_on": _clock(day, 4),
                                                 "model_uri": comparison.model_uri.iloc[0]})
                                mlflow.log_params({"consecutive_bad_weeks": streak, "trigger_limit": 2})
                                index = (previous_start - OPS_START).days // 7
                                log_weekly_history(client, parent.info.run_id, "operations_weekly", index, scores, idempotent=True)
                                log_weekly_history(client, event.info.run_id, "weekly", 0, scores, idempotent=True)
                                mlflow.log_dict(scores, "scores.json")
                            if not old:
                                save_database_evaluation(database_url, previous_id, comparison, scores,
                                                         _clock(day, 4), streak, event.info.run_id)
                        trace.append({"event": "evaluate", "day": str(day), "prediction_id": str(previous_id)})
                        if streak >= 2:
                            previous_model = active
                            # モデル更新はMLflowの一意なイベントと成果物を正本にする。
                            # 再開時も完了済みモデルを読み、重複登録しない。
                            with _event(client, experiment_id, f"{replay_id}:retrain:{day}", f"retrain-{day}") as (event, finished):
                                if finished:
                                    active = json.loads(Path(client.download_artifacts(event.info.run_id, "new_model.json")).read_text())
                                else:
                                    active = _retrain(database_url, source_dir, output, previous_model, day, client, event.info.run_id)
                                    mlflow.set_tags({"stage": "retraining", "triggered_on": _clock(day, 4),
                                                     "old_model_uri": previous_model["model_uri"], "new_model_uri": active["model_uri"]})
                            updates.append({"on": str(day), "old": previous_model, "new": active})
                            streak = 0
                            trace.append({"event": "retrain", "day": str(day), "model_uri": active["model_uri"]})
                    if day <= date(2011, 9, 26):
                        prediction_id = uuid5(replay_id, str(day))
                        existing = _read(database_url, "SELECT prediction_run_id FROM weekly_forecast_runs WHERE prediction_run_id=%s", (prediction_id,))
                        with _event(client, experiment_id, f"{replay_id}:forecast:{day}", f"forecast-{day}") as (event, finished):
                            if not existing:
                                _cutoff(database_url, day - timedelta(days=1))
                                if day > OPS_START:
                                    with psycopg.connect(database_url) as c:
                                        processed_period(c, OPS_START, day - timedelta(days=1), source_dir)
                                forecast_model(database_url, day - timedelta(days=1), active, source_dir, prediction_id,
                                               {"replay_id": replay_id, "simulated_at": _clock(day, 5),
                                                "forecast_run_id": event.info.run_id, "model_info": active})
                            if not finished:
                                mlflow.set_tags({"stage": "weekly_forecast", "simulated_at": _clock(day, 5),
                                                 "prediction_run_id": str(prediction_id), "model_uri": active["model_uri"],
                                                 "model_run_id": active["model_run_id"]})
                                mlflow.log_dict(active, "used_model.json")
                        trace.append({"event": "forecast", "day": str(day), "prediction_id": str(prediction_id)})
                if day <= OPS_END:
                    path = Path(source_dir) / f"{day}.csv"
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    done = _read(database_url, "SELECT source_sha256 FROM daily_batches WHERE batch_date=%s", (day,))
                    if done:
                        if done[0][0] != digest:
                            raise ValueError(f"処理済みCSVが変更されています: {day}")
                    else:
                        _cutoff(database_url, day)  # 同日バッチの再実行は許容する。
                        report = run_daily_batch(path, day, database_url)
                        with psycopg.connect(database_url) as c:
                            c.execute("INSERT INTO daily_batches VALUES (%s,%s,%s,%s,%s,%s,%s,DEFAULT)",
                                      (day, digest, report["input_rows"], report["accepted_rows"],
                                       report.get("verified_empty", False), _clock(day, 23), Jsonb(report)))
                    trace.append({"event": "daily", "day": str(day)})
            with psycopg.connect(database_url) as c:
                processed_period(c, OPS_START, OPS_END, source_dir)
                verify_period(c, OPS_START, OPS_END, source_dir)
            weeks = _read(database_url, """
                SELECT f.week_start,e.scores,e.consecutive_bad_weeks,f.model_info,
                       f.prediction_run_id::text,e.mlflow_run_id
                FROM weekly_forecast_runs f JOIN weekly_evaluations e USING(prediction_run_id)
                WHERE replay_id=%s ORDER BY week_start
            """, (replay_id,))
            if len(weeks) != 9:
                raise ValueError("予測・評価が9週そろっていません")
            counts = {table: _read(database_url, f"SELECT COUNT(*) FROM {table}")[0][0]
                      for table in ["transactions", "daily_sales", "daily_batches", "weekly_forecasts",
                                    "weekly_forecast_runs", "weekly_evaluations", "operation_replays"]}
            result = {"replay_id": str(replay_id), "parent_run_id": parent.info.run_id,
                      "weekly": [{"week_start": str(w[0]), "scores": w[1], "bad_streak": w[2],
                                  "model": w[3], "prediction_id": w[4], "evaluation_run_id": w[5]} for w in weeks],
                      "updates": updates,
                      "evaluated_forecasts": _read(database_url, "SELECT COUNT(*) FROM weekly_forecasts WHERE actual_quantity IS NOT NULL")[0][0],
                      "counts": counts, "timeline": trace}
            mlflow.log_dict(result, "summary.json")
            mlflow.log_artifact(str(Path(__file__)), "source")
            (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        # MLflowの親Run確定後にDBを完了扱いにする。この間の中断は同じRunを再開する。
        with psycopg.connect(database_url) as c:
            c.execute("UPDATE operation_replays SET status='completed',result=%s WHERE replay_id=%s",
                      (Jsonb(result), replay_id))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("01_data/storage/daily"))
    parser.add_argument("--output-root", type=Path, default=Path("02_model/results"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("02_model/mlflow"))
    parser.add_argument("--mode", choices=["initial", "operations"], default="initial")
    parser.add_argument("--initial-model-version", type=int, help="初期処理のhandoffに保存された登録バージョン")
    parser.add_argument("--replay-key", default="august-september-2011")
    args = parser.parse_args()
    if args.mode == "initial":
        initialize(os.environ["DATABASE_URL"], args.source_dir, args.output_root, args.tracking_dir)
    else:
        if args.initial_model_version is None:
            parser.error("operationsでは--initial-model-versionを指定してください")
        result = operate(os.environ["DATABASE_URL"], args.source_dir, args.output_root, args.tracking_dir,
                         initial_version=args.initial_model_version, replay_key=args.replay_key)
        print(json.dumps({"replay_id": result["replay_id"], "counts": result["counts"]}, indent=2))
