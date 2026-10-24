import numpy as np
import pandas as pd
import pytest
from evaluation.metrics import metrics
from training.model import PARAMETERS, matrix
from training.training_data import FEATURE_COLUMNS


def test_matrix_retains_product_order_and_features():
    values = {col: [1, 2] for col in FEATURE_COLUMNS if col != "stock_code"}
    values["stock_code"] = ["B", "A"]
    result = matrix(pd.DataFrame(values), ["A", "B"])
    assert result.columns.tolist() == FEATURE_COLUMNS
    assert result.stock_code.cat.categories.tolist() == ["A", "B"]
    assert result.stock_code.tolist() == ["B", "A"]


def test_model_parameters_include_reproducibility_seed():
    assert PARAMETERS["random_state"] == 42
    assert PARAMETERS["objective"] == "regression_l1"


def test_wape_is_weighted_and_zero_actual_is_undefined():
    assert metrics([10,90],[0,90])['wape']==pytest.approx(.1)
    assert metrics([0,0],[1,2])['wape'] is None
    with pytest.raises(ValueError): metrics([1],[np.inf])


def test_mlflow_sqlite_can_save_metrics_and_artifacts(tmp_path):
    import mlflow
    previous=mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri("sqlite:///"+str(tmp_path/"tracking.db"))
        experiment_id=mlflow.create_experiment("test-local-tracking",artifact_location=(tmp_path/"artifacts").as_uri())
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_param("seed",42)
            mlflow.log_metric("validation_wape",0.25)
            mlflow.log_text("snapshot", "data.txt")
            run_id=run.info.run_id
        client=mlflow.tracking.MlflowClient()
        assert client.get_run(run_id).data.metrics["validation_wape"]==0.25
        assert client.list_artifacts(run_id)[0].path=="data.txt"
    finally:
        mlflow.set_tracking_uri(previous)
