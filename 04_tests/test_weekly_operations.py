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
