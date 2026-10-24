"""時系列分割・ラベルの利用可能時点・運用の再学習判断を単体検証。"""
from pathlib import Path
import pandas as pd
import pytest

from training.weekly_operations import (
    TEST_ORIGINS, VALIDATION_ORIGINS, train_before, validation_folds,
    update_bad_streak, is_worse,
)


def example():
    return pd.DataFrame([
        {"stock_code": c, "forecast_origin": week,
         "target_start": week + pd.Timedelta(days=1),
         "target_end": week + pd.Timedelta(days=7)}
        for week in pd.date_range("2011-01-30", "2011-07-24", freq="W-SUN")
        for c in ("A", "B")
    ])


def test_fold_dates_and_no_future_labels():
    folds = validation_folds(example())
    assert tuple(x[0] for x in folds) == VALIDATION_ORIGINS
    for week, train, valid in folds:
        assert train.target_end.max() <= week
        assert (train.forecast_origin < week).all()
        assert valid.forecast_origin.eq(week).all()
        assert valid.stock_code.nunique() == 2


def test_test_weeks_not_used_to_choose_parameters():
    rows = example()
    for _, training, validation in validation_folds(rows):
        assert not training.forecast_origin.isin(TEST_ORIGINS).any()
        assert not validation.forecast_origin.isin(TEST_ORIGINS).any()


def test_retraining_cannot_use_unfinished_label():
    eligible = train_before(example(), pd.Timestamp("2011-07-17"))
    assert eligible.target_end.max() <= pd.Timestamp("2011-07-17")
    assert not eligible.forecast_origin.eq(pd.Timestamp("2011-07-17")).any()


def test_two_bad_weeks_trigger_and_recover():
    bad = {"lightgbm": {"wape": 0.6}, "baseline": {"wape": 0.5}}
    good = {"lightgbm": {"wape": 0.4}, "baseline": {"wape": 0.5}}
    count, worse = update_bad_streak(0, bad)
    assert count == 1 and worse is True
    count, worse = update_bad_streak(count, bad)
    assert count == 2 and worse is True
    count, worse = update_bad_streak(count, good)
    assert count == 0 and worse is False


def test_zero_actual_cannot_be_treated_as_bad_week():
    unknown = {"lightgbm": {"wape": None}, "baseline": {"wape": None}}
    assert is_worse(unknown) is None
    assert update_bad_streak(1, unknown) == (0, None)



def test_first_fold_uses_only_initial_training_period():
    from training.weekly_operations import TRAIN_END
    week, train, valid = validation_folds(example())[0]
    assert train.target_end.max() <= TRAIN_END
    assert valid.forecast_origin.eq(VALIDATION_ORIGINS[0]).all()


def test_july_changes_cannot_change_validation_data():
    original = example()
    changed = original.copy()
    changed.loc[changed.forecast_origin.isin(TEST_ORIGINS), "target_end"] += pd.Timedelta(days=365)
    for first, second in zip(validation_folds(original), validation_folds(changed)):
        pd.testing.assert_frame_equal(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])


def test_initial_mode_rejects_august_labels_before_mlflow(tmp_path):
    from training.weekly_operations import run
    codes = [str(i) for i in range(100)]
    frame = pd.DataFrame([{
        "stock_code": c, "forecast_origin": "2011-07-31", "history_start": "2011-07-04",
        "target_start": "2011-08-01", "target_end": "2011-08-07", "target": 7,
        "sales_sum_7": 7, "sales_sum_28": 28, "sales_std_28": 0, "month": 7,
    } for c in codes])
    frame.to_csv(tmp_path / "training.csv", index=False)
    pd.DataFrame({"stock_code": codes}).to_csv(tmp_path / "products.csv", index=False)
    with pytest.raises(ValueError, match="8月"):
        run(tmp_path / "training.csv", tmp_path / "products.csv", tmp_path / "mlflow", initial_only=True)
    assert not (tmp_path / "mlflow").exists()


def test_initial_registration_refuses_existing_database(database, tmp_path, monkeypatch):
    import psycopg
    from weekly.replay import initialize
    monkeypatch.setenv("DEMANDPILOT_INITIAL_DB", "1")
    with psycopg.connect(database) as connection:
        connection.execute("INSERT INTO daily_sales VALUES ('2011-01-03', 'A', 1)")
    with pytest.raises(ValueError, match="空"):
        initialize(database, tmp_path, tmp_path / "output", tmp_path / "mlflow")
    with psycopg.connect(database) as connection:
        assert connection.execute("SELECT quantity FROM daily_sales").fetchone()[0] == 1
    assert not (tmp_path / "output").exists()


def test_initial_database_refuses_future_actuals(database):
    import psycopg
    from weekly.replay import database_summary
    with psycopg.connect(database) as connection:
        connection.execute("INSERT INTO daily_sales VALUES ('2011-08-01', 'A', 1)")
    with pytest.raises(ValueError, match="期間外"):
        database_summary(database)


def test_product_selection_cannot_use_june_sales(database):
    import psycopg
    from weekly.replay import select_products
    with psycopg.connect(database) as connection:
        for index in range(100):
            for day in pd.date_range("2010-12-06", periods=25, freq="W-MON"):
                connection.execute("INSERT INTO daily_sales VALUES (%s, %s, 1)", (day.date(), f"P{index:03}"))
    before = select_products(database)
    with psycopg.connect(database) as connection:
        for day in pd.date_range("2011-06-01", "2011-07-31"):
            connection.execute("INSERT INTO daily_sales VALUES (%s, 'FUTURE', 999999)", (day.date(),))
    pd.testing.assert_frame_equal(before, select_products(database))


def test_july31_training_cannot_use_august_labels():
    rows = example()
    future = rows.iloc[:2].copy()
    future["forecast_origin"] = pd.Timestamp("2011-07-31")
    future["target_end"] = pd.Timestamp("2011-08-07")
    combined = pd.concat([rows, future])
    result = train_before(combined, pd.Timestamp("2011-07-31"))
    pd.testing.assert_frame_equal(result, rows)



def test_initial_lifecycle_persists_all_child_runs_and_models(tmp_path):
    import mlflow
    import mlflow.lightgbm
    from training.weekly_operations import run
    from training.model import matrix
    codes = [f"P{i:03}" for i in range(100)]
    rows = pd.DataFrame([{
        "stock_code": code, "forecast_origin": week, "history_start": week - pd.Timedelta(days=27),
        "target_start": week + pd.Timedelta(days=1), "target_end": week + pd.Timedelta(days=7),
        "target": 7, "sales_sum_7": 7., "sales_sum_28": 28., "sales_std_28": 0., "month": week.month,
    } for week in pd.date_range("2011-01-30", "2011-07-24", freq="W-SUN") for code in codes])
    csv_path, products_path = tmp_path / "training.csv", tmp_path / "products.csv"
    rows.to_csv(csv_path, index=False)
    pd.DataFrame({"stock_code": codes}).to_csv(products_path, index=False)
    tracking = tmp_path / "tracking"
    previous = mlflow.get_tracking_uri()
    try:
        result = run(csv_path, products_path, tracking, experiment_name="persistent-initial", initial_only=True)
        client = mlflow.tracking.MlflowClient()
        parent = client.get_run(result["parent_run_id"])
        runs = client.search_runs([parent.info.experiment_id])
        # 親1 + 初期学習1 + 候補3 + Fold12 + モデル2 + テスト4
        assert len(runs) == 23
        assert all(x.info.artifact_uri.startswith((tracking / "artifacts").as_uri()) for x in runs)
        for child in runs:
            for artifact in client.list_artifacts(child.info.run_id):
                assert Path(client.download_artifacts(child.info.run_id, artifact.path)).exists()
        handoff = result["handoff"]
        model_id = handoff["model_uri"].split("/")[-1]
        assert client.get_logged_model(model_id).artifact_location.startswith((tracking / "artifacts").as_uri())
        assert Path(client.download_artifacts(handoff["model_run_id"], "handoff.json")).exists()
        model = mlflow.lightgbm.load_model(handoff["model_uri"])
        assert len(model.predict(matrix(rows.head(100), codes))) == 100
    finally:
        mlflow.set_tracking_uri(previous)



def test_calendar_missing_file_is_an_error(tmp_path):
    from weekly.replay import check_calendar
    with pytest.raises(ValueError, match="2011-08-01"):
        check_calendar(tmp_path)
    assert not list(tmp_path.iterdir())


def test_verified_empty_csv_and_unknown_missing_day(tmp_path, database):
    from preparation.prepare_csv import prepare_verified_calendar, REQUIRED_COLUMNS
    from daily.daily_batch import run_daily_batch, read_daily_csv
    import psycopg
    frame = pd.DataFrame([{
        "InvoiceNo": "1", "StockCode": "A", "Description": "test", "Quantity": 1,
        "InvoiceDate": day, "UnitPrice": 1, "CustomerID": "", "Country": "UK",
    } for day in ["2011-08-01", "2011-08-03"]])
    excel = tmp_path / "source.xlsx"
    frame.to_excel(excel, index=False)
    existing = tmp_path / "existing"
    existing.mkdir()
    for n, day in enumerate(["2011-08-01", "2011-08-03"], 2):
        row = frame.loc[frame.InvoiceDate.eq(day)].copy()
        row.insert(0, "source_row", n)
        row.to_csv(existing / (day + ".csv"), index=False)
    output = tmp_path / "verified"
    proof = prepare_verified_calendar(excel, existing, output, "2011-08-01", "2011-08-03")
    assert proof["days"]["2011-08-02"]["rows"] == 0
    report = run_daily_batch(output / "2011-08-02.csv", "2011-08-02", database)
    assert report["verified_empty"] and report["accepted_rows"] == 0
    with psycopg.connect(database) as c:
        assert c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM daily_sales").fetchone()[0] == 0
    # 元データに取引がある日の欠落は、0件CSVとして生成しない。
    (existing / "2011-08-03.csv").unlink()
    with pytest.raises(ValueError, match="取引がある"):
        prepare_verified_calendar(excel, existing, tmp_path / "bad", "2011-08-01", "2011-08-03")
    assert not (tmp_path / "bad").exists()
    # 確認記録のない単なるヘッダーのみCSVは拒否。
    unknown = tmp_path / "2011-08-02.csv"
    pd.DataFrame(columns=["source_row"] + REQUIRED_COLUMNS).to_csv(unknown, index=False)
    with pytest.raises(ValueError, match="0件確認"):
        read_daily_csv(unknown, pd.Timestamp("2011-08-02").date())


def test_processed_zero_day_is_distinct_from_missing_batch(tmp_path, database):
    import hashlib
    import psycopg
    from psycopg.types.json import Jsonb
    from weekly.forecast import verify_processed_period
    day = pd.Timestamp("2011-08-06").date()
    path = tmp_path / f"{day}.csv"
    path.write_text("header\n")
    with psycopg.connect(database) as c:
        with pytest.raises(ValueError, match="未完了"):
            verify_processed_period(c, day, day, tmp_path)
        c.execute("INSERT INTO daily_batches VALUES (%s,%s,0,0,true,%s,%s,DEFAULT)",
                  (day, hashlib.sha256(path.read_bytes()).hexdigest(), "2011-08-06T23:00:00+09:00", Jsonb({})))
        verify_processed_period(c, day, day, tmp_path)
        path.write_text("changed\n")
        with pytest.raises(ValueError, match="変更"):
            verify_processed_period(c, day, day, tmp_path)


def test_chronological_operations_resume_retrain_and_exactly_once(tmp_path, database, monkeypatch):
    import json
    import psycopg
    import mlflow
    from datetime import date
    from psycopg.types.json import Jsonb
    from preparation.prepare_csv import REQUIRED_COLUMNS
    from training.model import PARAMETERS
    from training.weekly_operations import model_fit, register_version
    from training.training_data import FEATURE_COLUMNS
    import weekly.replay as replay
    codes = [f"P{i:03}" for i in range(100)]
    source = tmp_path / "daily"
    source.mkdir()
    initial_transactions = []
    for index, day in enumerate(pd.date_range("2010-12-01", "2011-10-02")):
        data = []
        for i, code in enumerate(codes):
            number = 2 + index * 100 + i
            data.append([number, "1", code, "test", 1, f"{day:%Y-%m-%d} 12:00:00", "1", "", "UK"])
            if day <= pd.Timestamp("2011-07-31"):
                initial_transactions.append((number, "1", code, day.to_pydatetime(), 1, 1, "test", None, "UK"))
        pd.DataFrame(data, columns=["source_row"] + REQUIRED_COLUMNS).to_csv(source / f"{day:%Y-%m-%d}.csv", index=False)
    with psycopg.connect(database) as c:
        with c.cursor().copy("COPY transactions FROM STDIN") as copy:
            for row in initial_transactions:
                copy.write_row(row)
        c.execute("INSERT INTO daily_sales SELECT invoice_date::date,stock_code,SUM(quantity) FROM transactions GROUP BY 1,2")
    # 意図的に外れる初期モデルで、2週連続悪化の分岐を確実に通す。
    seed_rows = pd.DataFrame({"stock_code": codes, "forecast_origin": pd.Timestamp("2011-07-24"),
                             "target_end": pd.Timestamp("2011-07-31"), "target": 0.,
                             "sales_sum_7": 7., "sales_sum_28": 28., "sales_std_28": 0., "month": 7})
    tracking = tmp_path / "tracking"
    tracking.mkdir()
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri("sqlite:///" + str(tracking / "mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    experiment = client.create_experiment("seed", artifact_location=(tracking / "artifacts").as_uri())
    mlflow.set_experiment(experiment_id=experiment)
    with mlflow.start_run(experiment_id=experiment) as parent:
        info = register_version(model_fit(seed_rows, codes, PARAMETERS), seed_rows, codes, 1,
                                pd.Timestamp("2011-07-31"), "operations_start", parent.info.run_id)
        info.update({"features": FEATURE_COLUMNS, "product_codes": codes, "hyperparameters": PARAMETERS,
                     "final_training": {"last_label_date": "2011-07-31"}})
        client.log_dict(info["model_run_id"], info, "handoff.json")
    monkeypatch.setenv("DEMANDPILOT_OPERATIONS_DB", "1")
    real_batch = replay.run_daily_batch
    failed = [False]
    def interrupted(path, day, url):
        if str(day) == "2011-08-04" and not failed[0]:
            failed[0] = True
            raise RuntimeError("simulated interruption")
        return real_batch(path, day, url)
    monkeypatch.setattr(replay, "run_daily_batch", interrupted)
    try:
        with pytest.raises(RuntimeError, match="interruption"):
            replay.operate(database, source, tmp_path / "output", tracking, initial_version=1)
        result = replay.operate(database, source, tmp_path / "output", tracking, initial_version=1)
        assert result["counts"]["daily_batches"] == 63
        assert result["counts"]["weekly_forecasts"] == 900
        assert result["counts"]["weekly_evaluations"] == 9
        assert result["counts"]["weekly_evaluation_items"] == 900
        assert len(result["updates"]) == 1
        assert result["updates"][0]["on"] == "2011-08-15"
        assert result["weekly"][2]["model"]["model_uri"] == result["updates"][0]["new"]["model_uri"]
        assert all(row["model"]["final_training"]["last_label_date"] < row["week_start"] for row in result["weekly"])
        assert result["timeline"][-1] == {"event": "evaluate", "day": "2011-10-03", "prediction_id": result["weekly"][-1]["prediction_id"]}
        daily = [x["day"] for x in result["timeline"] if x["event"] == "daily"]
        assert daily == list(pd.date_range("2011-08-01", "2011-10-02").strftime("%Y-%m-%d"))
        for event in result["timeline"]:
            if event["event"] == "forecast":
                assert result["timeline"].index(event) < result["timeline"].index({"event": "daily", "day": event["day"]})
        parent = client.get_run(result["parent_run_id"])
        experiment_id = parent.info.experiment_id
        run_count = len(client.search_runs([experiment_id]))
        history = client.get_metric_history(parent.info.run_id, "operations_weekly_lightgbm_wape")
        assert len(history) == 9
        repeated = replay.operate(database, source, tmp_path / "output", tracking, initial_version=1)
        assert repeated == result
        assert len(client.search_runs([experiment_id])) == run_count
        assert len(client.get_metric_history(parent.info.run_id, "operations_weekly_lightgbm_wape")) == 9
        # MLflowの確定後、DBの完了印だけが未確定だった境界を再現する。
        with psycopg.connect(database) as c:
            c.execute("UPDATE operation_replays SET status='running',result=NULL")
        recovered = replay.operate(database, source, tmp_path / "output", tracking, initial_version=1)
        assert recovered == result
        assert len(client.search_runs([experiment_id])) == run_count
        assert len(client.get_metric_history(parent.info.run_id, "operations_weekly_lightgbm_wape")) == 9
        # 保存した評価は後日のDB実績変更から独立した履歴。
        from evaluation.evaluate import load_database_evaluation, compare_actuals
        frame, scores = load_database_evaluation(database, result["weekly"][0]["prediction_id"])
        assert len(frame) == 100 and scores["lightgbm"]["wape"] == 1
        # 正本CSVと集計が一致していても、処理済み記録がなければ評価できない。
        with psycopg.connect(database) as c:
            c.execute("DELETE FROM daily_batches WHERE batch_date='2011-08-06'")
        with pytest.raises(ValueError, match="未完了"):
            compare_actuals(database, result["weekly"][0]["prediction_id"], date(2011, 8, 8), source)
    finally:
        mlflow.set_tracking_uri(previous_uri)



def test_operations_cutoff_rejects_future_actuals(database):
    import psycopg
    from weekly.replay import _cutoff
    with psycopg.connect(database) as c:
        c.execute("INSERT INTO daily_sales VALUES ('2011-08-02','FUTURE',999)")
    with pytest.raises(ValueError, match="未来"):
        _cutoff(database, pd.Timestamp("2011-08-01").date())


def test_forecast_handoff_rejects_future_model_before_db_access():
    from weekly.forecast import forecast_model
    from training.training_data import FEATURE_COLUMNS
    with pytest.raises(ValueError, match="未来"):
        forecast_model("unused", "2011-07-31", {"features": FEATURE_COLUMNS,
                       "final_training": {"last_label_date": "2011-08-07"}}, "unused", "unused")
