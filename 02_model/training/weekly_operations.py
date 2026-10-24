"""需要予測モデルの週次運用を、過去の販売データで再現する。

処理順: ①入力 → ②時系列検証・モデル選定 → ③独立テスト
      → ④週次予測・前週の評価・必要時再学習 → ⑤MLflowに記録

学習・運用の処理を理解するため、処理の流れはこのファイルに集約する。
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from training.model import PARAMETERS, matrix
from evaluation.metrics import metrics
from training.training_data import FEATURE_COLUMNS

# 期間・探索条件（コードの処理内容を変えず、最初に確認できるようにまとめる）
TRAIN_END = pd.Timestamp("2011-05-31")
VALIDATION_ORIGINS = tuple(pd.date_range("2011-06-05", periods=4, freq="W-SUN"))
TEST_ORIGINS = tuple(pd.date_range("2011-07-03", periods=4, freq="W-SUN"))
OPERATIONS_START = pd.Timestamp("2011-07-31")
DEFAULT_EXPERIMENT = "demandpilot-weekly-operations"
REGISTERED_MODEL = "DemandPilotWeeklyLightGBM"
# 推論はnative LightGBMとmatrix()を使い、商品コードを学習時と同じカテゴリへ戻す。
MODEL_SIGNATURE = ModelSignature(inputs=Schema([
    ColSpec("double", "sales_sum_7"), ColSpec("double", "sales_sum_28"),
    ColSpec("double", "sales_std_28"), ColSpec("long", "month"),
    ColSpec("string", "stock_code"),
]))
MODEL_REQUIREMENTS = ["lightgbm==4.6.0", "pandas==2.2.3", "numpy==2.2.3", "scikit-learn==1.6.1"]
# 小さい探索空間に固定。探索中にテスト期間を見ない。
CANDIDATES = (
    {"num_leaves": 7, "learning_rate": 0.05, "n_estimators": 100},
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 100},
    {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 200},
)



# ━━ 1. データ・時系列の扱い ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_before(frame, origin):
    """確定済み7日間の正解だけを使う。"""
    eligible = frame.loc[frame.target_end.le(pd.Timestamp(origin))]
    if eligible.empty or eligible.target_end.max() > pd.Timestamp(origin):
        raise ValueError("学習時点で確定したラベルがありません")
    return eligible.copy()


def validation_folds(frame):
    """週の基準日を単位に分割する。商品行を独立の時点としてsplitしない。"""
    dates = pd.DatetimeIndex(sorted(frame.loc[frame.forecast_origin.le(VALIDATION_ORIGINS[-1]), "forecast_origin"].unique()))
    if not set(VALIDATION_ORIGINS).issubset(set(dates)):
        raise ValueError("6月の4検証週が足りません")
    splitter = TimeSeriesSplit(n_splits=4, test_size=1)
    folds = []
    for train_indices, valid_indices in splitter.split(dates):
        week = pd.Timestamp(dates[valid_indices[0]])
        if week not in VALIDATION_ORIGINS:
            raise ValueError("検証週が6月の指定4週間と一致しません")
        train = frame.loc[frame.forecast_origin.isin(dates[train_indices]) & frame.target_end.le(week)].copy()
        if week == VALIDATION_ORIGINS[0]:
            train = train.loc[train.target_end.le(TRAIN_END)]
        valid = frame.loc[frame.forecast_origin.eq(week)].copy()
        if train.empty or valid.empty or train.target_end.max() > week:
            raise ValueError("未来の正解が混入、または検証週が空です")
        folds.append((week, train, valid))
    if tuple(x[0] for x in folds) != VALIDATION_ORIGINS:
        raise ValueError("TimeSeriesSplitの検証日が一致しません")
    return folds



# ━━ 2. モデル学習・予測・精度評価 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def record_predictions(frame, prediction, stage, model_version):
    rows = frame[["stock_code", "forecast_origin", "target_start", "target_end", "target"]].copy()
    rows["lightgbm"] = np.maximum(0, prediction)
    rows["baseline"] = frame.sales_sum_7.to_numpy(dtype=float)
    rows["stage"] = stage
    rows["model_version"] = model_version
    return rows


def model_fit(rows, codes, parameters):
    if rows.empty:
        raise ValueError("学習対象が空です")
    model = lgb.LGBMRegressor(**(PARAMETERS | parameters))
    model.fit(matrix(rows, codes), rows.target, categorical_feature=["stock_code"])
    return model


def score(rows):
    return {name: metrics(rows.target, rows[name]) for name in ("lightgbm", "baseline")}


def is_worse(scores):
    """評価可能な週のWAPEで採用モデルとベースラインを比較する。"""
    model = scores["lightgbm"]["wape"]
    baseline = scores["baseline"]["wape"]
    if model is None or baseline is None:
        return None
    return model > baseline


def update_bad_streak(current, scores):
    worse = is_worse(scores)
    return (current + 1 if worse else 0), worse



# ━━ 3. 学習前の入力チェック ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def checked_input(csv_path, products_path):
    raw, product_raw = Path(csv_path).read_bytes(), Path(products_path).read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), dtype={"stock_code": str}, parse_dates=["forecast_origin", "history_start", "target_start", "target_end"])
    codes = pd.read_csv(io.BytesIO(product_raw), dtype={"stock_code": str}).stock_code.tolist()
    if len(codes) != 100 or len(set(codes)) != 100 or set(frame.stock_code) != set(codes):
        raise ValueError("100品番の一覧と学習CSVが一致しません")
    if frame.empty or frame[FEATURE_COLUMNS + ["target"]].isna().any().any() or frame.duplicated(["forecast_origin", "stock_code"]).any():
        raise ValueError("欠損または重複があります")
    if not frame.forecast_origin.dt.dayofweek.eq(6).all():
        raise ValueError("予測基準日は日曜日だけにしてください")
    if not frame.target_end.eq(frame.forecast_origin + pd.Timedelta(days=7)).all():
        raise ValueError("7日先の正解期間に誤りがあります")
    if not frame.groupby("forecast_origin").stock_code.nunique().eq(100).all():
        raise ValueError("各週の100品番がそろっていません")
    if frame.target_end.max() > pd.Timestamp("2011-12-09"):
        raise ValueError("元データ終了後の正解が混入しています")
    return frame.sort_values(["forecast_origin", "stock_code"]).reset_index(drop=True), codes, raw, product_raw



# ━━ 4. MLflowへ結果を保存（モデルの判断ロジックとは別） ━━━━━━━━━━━
def put_scores(scores, prefix):
    for name, values in scores.items():
        for key in ("mae", "wape"):
            if values[key] is not None:
                mlflow.log_metric(f"{prefix}_{name}_{key}", values[key])


def log_weekly_history(client, parent_run_id, stage, index, scores):
    """親Runに時系列の指標を保存。stepは週番号（0起点）。"""
    for model_name in ("lightgbm", "baseline"):
        for metric_name in ("mae", "wape"):
            value = scores[model_name][metric_name]
            if value is not None:
                client.log_metric(parent_run_id, f"{stage}_{model_name}_{metric_name}",
                                  float(value), step=index)


def register_version(model, train_rows, codes, version, cutoff, source, parent_run_id):
    """学習済みモデルを子RunとMLflow Modelsに保存し、親Runとの対応を返す。"""
    with mlflow.start_run(run_name=f"model-v{version}-{source}", nested=True) as model_run:
        mlflow.set_tags({"stage": "model_version", "logical_model_version": str(version),
                         "retraining_reason": source, "parent_lifecycle_run_id": parent_run_id,
                         "train_cutoff": str(pd.Timestamp(cutoff).date())})
        mlflow.log_params({"training_rows": len(train_rows),
                           "last_known_label": str(train_rows.target_end.max().date())})
        # MLflow 3.x: 登録モデル一覧に履歴を表示する。登録バージョンは実行ごとに増える。
        result = mlflow.lightgbm.log_model(
            lgb_model=model, name=f"lightgbm_v{version}",
            registered_model_name=REGISTERED_MODEL,
            signature=MODEL_SIGNATURE, pip_requirements=MODEL_REQUIREMENTS,
        )
        return {"logical_model_version": version, "model_run_id": model_run.info.run_id,
                "registered_model_name": REGISTERED_MODEL,
                "registered_model_version": str(result.registered_model_version),
                "model_uri": result.model_uri,
                "trained_through": str(pd.Timestamp(cutoff).date()), "reason": source}


def put_csv(df, output, artifact_path):
    df.to_csv(output, index=False)
    mlflow.log_artifact(str(output), artifact_path)



# ━━ 5. 全体の実行順序 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run(csv_path, products_path, tracking_dir, *, experiment_name=DEFAULT_EXPERIMENT, bad_weeks=2, initial_only=False):
    """検証・テスト・週次運用をこの順に実行し、1つのMLflow実験に残す。"""
    if bad_weeks < 1:
        raise ValueError("再学習までの悪化週数は1以上")
    # ① 入力データ確認。1週100品番、正解の終了日などを検査する。
    frame, codes, raw, product_raw = checked_input(csv_path, products_path)
    if initial_only and frame.target_end.max() > pd.Timestamp("2011-07-31"):
        raise ValueError("初期処理に8月以降の正解があります")
    folds = validation_folds(frame)
    test = [(week, frame.loc[frame.forecast_origin.eq(week)].copy()) for week in TEST_ORIGINS]
    if any(rows.empty for _, rows in test):
        raise ValueError("7月のテスト4週がそろっていません")
    operations = [(pd.Timestamp(week), subset.copy()) for week, subset in frame.groupby("forecast_origin", sort=True) if week >= OPERATIONS_START]
    if not operations and not initial_only:
        raise ValueError("週次運用期間のデータがありません")
    # MLflowの保存先はDockerの永続マウント（02_model/mlflow）。
    tracking_dir = Path(tracking_dir).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + str(tracking_dir / "mlflow.db"))
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name, artifact_location=(tracking_dir / "artifacts").as_uri())
    else:
        experiment_id = experiment.experiment_id
        # 既存の作業環境で消失し得る一時パスに向いた実験は再利用しない。
        if not str(experiment.artifact_location).startswith((tracking_dir / "artifacts").as_uri()):
            raise ValueError("既存のMLflow実験のArtifacts保存先が永続領域ではありません。別のexperiment名を使用してください")
    # 子Runにも同じ実験を明示する。既定実験0の一時artifact領域へ保存しない。
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(experiment_id=experiment_id, run_name="weekly-model-lifecycle") as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"purpose": "historical_replay", "splitter": "TimeSeriesSplit", "test_policy": "frozen_model", "operational_policy": "prepare_only" if initial_only else "weekly_forecast_monitor_retrain", "cohort_limit": "100_products_preselected_from_historical_data"})
        mlflow.log_params({"initial_train_end": str(TRAIN_END.date()), "validation_first_origin": str(VALIDATION_ORIGINS[0].date()), "validation_last_origin": str(VALIDATION_ORIGINS[-1].date()), "test_first_origin": str(TEST_ORIGINS[0].date()), "test_last_origin": str(TEST_ORIGINS[-1].date()), "cv_splits": 4, "test_weeks": 4, "bad_weeks_to_retrain": bad_weeks, "drift_metric": "weekly_WAPE", "retrain_data_rule": "target_end<=origin", "baseline": "last_7_days", "csv_sha256": hashlib.sha256(raw).hexdigest(), "products_sha256": hashlib.sha256(product_raw).hexdigest()})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            mlflow.log_artifact(str(Path(__file__).resolve()), "source")
            mlflow.log_artifact(str(Path(csv_path).resolve()), "input")
            mlflow.log_artifact(str(Path(products_path).resolve()), "input")
            mlflow.log_artifact(str(Path(__file__).resolve().parent / "training_data.py"), "source")
            if initial_only:
                mlflow.log_artifact(str(Path(__file__).resolve().parent / "model.py"), "source")
            if initial_only:
                initial_rows = train_before(frame, TRAIN_END)
                with mlflow.start_run(run_name="initial-training", nested=True):
                    mlflow.set_tag("stage", "initial_training")
                    mlflow.log_params(PARAMETERS | CANDIDATES[0])
                    mlflow.log_params({"last_label_date": str(initial_rows.target_end.max().date()),
                                       "training_rows": len(initial_rows)})
                    initial_model = model_fit(initial_rows, codes, CANDIDATES[0])
                    mlflow.lightgbm.log_model(initial_model, name="initial_model",
                                              signature=MODEL_SIGNATURE, pip_requirements=MODEL_REQUIREMENTS)
                    put_csv(initial_rows, output / "initial_training.csv", "input")
            # ② 検証: 6月の4週で候補を比較。テスト期間は一切見ない。
            selection = []
            for candidate_index, candidate in enumerate(CANDIDATES, 1):
                weekly = []
                with mlflow.start_run(run_name=f"tuning-candidate-{candidate_index}", nested=True):
                    mlflow.set_tag("stage", "tuning")
                    mlflow.log_params(candidate)
                    for fold_number, (week, train, valid) in enumerate(folds, 1):
                        model = model_fit(train, codes, candidate)
                        rows = record_predictions(valid, model.predict(matrix(valid, codes)), "validation", candidate_index)
                        weekly.append(rows)
                        with mlflow.start_run(run_name=f"cv-fold-{fold_number}-{week.date()}", nested=True):
                            mlflow.set_tags({"stage": "validation", "week_origin": str(week.date()), "parent_candidate": str(candidate_index)})
                            mlflow.log_params({"train_rows": len(train), "last_known_label": str(train.target_end.max().date()), "validation_origin": str(week.date())})
                            mlflow.log_param("evaluated_as_of", str(valid.target_end.max().date()))
                            if initial_only:
                                put_csv(train, output / "fold_training.csv", "input")
                            fold_scores = score(rows)
                            put_scores(fold_scores, "validation")
                            log_weekly_history(client, parent_id,
                                               f"validation_candidate_{candidate_index}",
                                               fold_number - 1, fold_scores)
                            put_csv(rows, output / "predictions.csv", "predictions")
                    combined = pd.concat(weekly, ignore_index=True)
                    candidate_score = score(combined)
                    put_scores(candidate_score, "cv_all")
                    put_csv(combined, output / f"candidate_{candidate_index}_validation.csv", "predictions")
                    selection.append({"candidate": candidate_index, "parameters": candidate, "scores": candidate_score})
            ranking = sorted(selection, key=lambda x: (float('inf') if x['scores']['lightgbm']['wape'] is None else x['scores']['lightgbm']['wape'], x['scores']['lightgbm']['mae']))
            best = ranking[0]
            params = best["parameters"]
            mlflow.log_params({"selected_candidate": best["candidate"], **{f"selected_{k}": v for k,v in params.items()}})
            mlflow.log_dict({"ranking": ranking, "selection_metric": "pooled_validation_WAPE"}, "tuning/selection.json")
            # ③ テスト: 選んだ条件でモデルを固定し、7月の独立4週を評価する。
            cutoff = TEST_ORIGINS[0] if initial_only else VALIDATION_ORIGINS[-1]
            fixed_train = train_before(frame, cutoff)
            frozen = model_fit(fixed_train, codes, params)
            if initial_only:
                put_csv(fixed_train, output / "test_training.csv", "input")
            mlflow.log_param("test_model_last_known_label", str(fixed_train.target_end.max().date()))
            frozen.booster_.save_model(str(output / "frozen_test_model.txt"))
            mlflow.log_artifact(str(output / "frozen_test_model.txt"), "models")
            model_history = [register_version(frozen, fixed_train, codes, 1, cutoff,
                                              "initial_selected_model", parent_id)]
            active_model = model_history[-1]
            tested = []
            for test_index, (week, subset) in enumerate(test):
                with mlflow.start_run(run_name=f"test-{week.date()}", nested=True):
                    mlflow.set_tags({"stage": "holdout_test", "week_origin": str(week.date()), "model_version": "1"})
                    rows = record_predictions(subset, frozen.predict(matrix(subset, codes)), "test", 1)
                    weekly_scores = score(rows)
                    mlflow.log_param("evaluated_as_of", str(subset.target_end.max().date()))
                    put_scores(weekly_scores, "test")
                    log_weekly_history(client, parent_id, "test_weekly", test_index, weekly_scores)
                    put_csv(rows, output / "weekly_predictions.csv", "predictions")
                    tested.append(rows)
            test_results = pd.concat(tested, ignore_index=True)
            test_score = score(test_results)
            put_scores(test_score, "test_pooled")
            put_csv(test_results, output / "test_predictions.csv", "evaluation")
            if initial_only:
                final_cutoff = pd.Timestamp("2011-07-31")
                final_rows = train_before(frame, final_cutoff)
                operational = model_fit(final_rows, codes, params)
                handoff = register_version(operational, final_rows, codes, 2, final_cutoff,
                                           "operations_start", parent_id)
                reloaded = mlflow.lightgbm.load_model(handoff["model_uri"])
                np.testing.assert_allclose(reloaded.predict(matrix(final_rows, codes)),
                                           operational.predict(matrix(final_rows, codes)))
                handoff.update({"model_reload_verified": True, "features": FEATURE_COLUMNS, "product_codes": codes,
                                "product_csv_sha256": hashlib.sha256(product_raw).hexdigest(),
                                "hyperparameters": PARAMETERS | params,
                                "prediction_api": "mlflow.lightgbm.load_model + training.model.matrix",
                                "source_start": "2010-12-01", "source_end": "2011-07-31",
                                "first_forecast_origin": "2011-07-31",
                                "first_target_start": "2011-08-01",
                                "final_training": {"last_label_date": str(final_rows.target_end.max().date()),
                                                   "first_origin": str(final_rows.forecast_origin.min().date()),
                                                   "last_origin": str(final_rows.forecast_origin.max().date()),
                                                   "rows": len(final_rows)}})
                client.log_dict(handoff["model_run_id"], handoff, "handoff.json")
                client.log_batch(handoff["model_run_id"], params=[
                    mlflow.entities.Param(k, str(v)) for k, v in (PARAMETERS | params).items()])
                put_csv(final_rows, output / "operational_training.csv", "input")
                mlflow.log_dict(handoff, "handoff.json")
                selected_cv = pd.read_csv(output / f"candidate_{best['candidate']}_validation.csv")
                summary = {"parent_run_id": parent_id, "selected_hyperparameters": params,
                           "cv_overall": best["scores"], "test_overall": test_score,
                           "validation_weekly": [{"origin": str(w), "scores": score(g)}
                               for w, g in selected_cv.groupby("forecast_origin")],
                           "test_weekly": [{"origin": str(w.date()), "scores": score(g)}
                               for w, g in test_results.groupby("forecast_origin")],
                           "handoff": handoff, "operational_weeks": 0}
                mlflow.log_dict(summary, "summary.json")
                print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
                return summary
            # テストがベースラインより悪い場合は、運用で既存モデルを採用しない。
            # ただし運用リプレイでは両方のスコアを保存し、どちらを採用したか明示する。
            adopted = (is_worse(test_score) is False)
            mlflow.set_tag("adoption_decision", "lightgbm" if adopted else "baseline")
            model = frozen
            model_version = 1
            model_cutoff = cutoff
            streak = 0
            recovery_streak = 0
            pending = None
            predictions, monitoring, retrain_events = [], [], []
            week_calendar = []
            # ④ 週次運用: 前週の実績を評価→必要なら更新→今週の予測を保存。
            for week_index, (week, subset) in enumerate(operations):
                # 週初めに、前週の確定実績を確認。未来の実績を先読みしない。
                if pending is not None:
                    previous_week, previous_rows, previous_run = pending
                    if previous_rows.target_end.max() <= week:
                        weekly_score = score(previous_rows)
                        streak, worse = update_bad_streak(streak, weekly_score)
                        recovery_streak = recovery_streak + 1 if worse is False else 0
                        # テストで不採用だったモデルもシャドー予測で2週連続改善すれば適用する。
                        if not adopted and recovery_streak >= 2:
                            adopted = True
                        monitoring.append({"origin": str(previous_week.date()), "evaluated_on": str(week.date()), "worse_than_baseline": worse, "consecutive_bad_weeks": streak, "scores": weekly_score, "model_version": model_version})
                        log_weekly_history(client, parent_id, "operations_weekly", week_index - 1, weekly_score)
                        with mlflow.start_run(run_name=f"monitor-{previous_week.date()}", nested=True):
                            mlflow.set_tags({"stage": "monitor", "forecast_run_id": previous_run, "evaluated_on": str(week.date())})
                            mlflow.log_params({"consecutive_bad_weeks": streak, "trigger_limit": bad_weeks})
                            put_scores(weekly_score, "observed")
                            mlflow.log_metric("worse_than_baseline", int(bool(worse)))
                            if streak >= bad_weeks:
                                eligible = train_before(frame, week)
                                model = model_fit(eligible, codes, params)
                                model_version += 1
                                model_cutoff = week
                                event = {"triggered_on": str(week.date()), "reason": f"{bad_weeks}_consecutive_weeks_WAPE_worse_than_baseline", "previous_run": previous_run, "new_model_version": model_version, "training_rows": len(eligible), "last_known_label": str(eligible.target_end.max().date())}
                                retrain_events.append(event)
                                mlflow.log_dict(event, "retraining/event.json")
                                model.booster_.save_model(str(output / f"model_v{model_version}.txt"))
                                mlflow.log_artifact(str(output / f"model_v{model_version}.txt"), "retraining")
                                active_model = register_version(model, eligible, codes, model_version,
                                                                week, event["reason"], parent_id)
                                model_history.append(active_model)
                                event.update({"model_run_id": active_model["model_run_id"],
                                              "registered_model_version": active_model["registered_model_version"]})
                                mlflow.log_dict(event, "retraining/model_registry_link.json")
                                streak = 0
                                recovery_streak = 0
                                adopted = False  # 再学習モデルは次週以降の実績で採用を再判定する。
                # 実績の確定前にその週を予測。スコアは次の週に初めて計算される。
                with mlflow.start_run(run_name=f"forecast-{week.date()}", nested=True) as forecast_run:
                    mlflow.set_tags({"stage": "weekly_forecast", "forecast_origin": str(week.date()), "selected_policy": "lightgbm" if adopted else "baseline",
                                     "model_run_id": active_model["model_run_id"],
                                     "registered_model_name": REGISTERED_MODEL,
                                     "registered_model_version": active_model["registered_model_version"]})
                    mlflow.log_params({"model_version": model_version, "model_train_cutoff": str(model_cutoff.date()), "hyperparameter_candidate": best["candidate"]})
                    rows = record_predictions(subset, model.predict(matrix(subset, codes)), "operation", model_version)
                    rows["adopted_prediction"] = rows.lightgbm if adopted else rows.baseline
                    # 推論時に未確定の実績/正解を予測Runに保存しない。
                    put_csv(rows.drop(columns=["target"]), output / "weekly_predictions.csv", "predictions")
                    mlflow.log_metric("weekly_predicted_total", float(rows.adopted_prediction.sum()))
                    predictions.append(rows)
                    week_calendar.append({"step": week_index, "forecast_origin": str(week.date()),
                                          "model_run_id": active_model["model_run_id"],
                                          "logical_model_version": model_version,
                                          "registry_version": active_model["registered_model_version"],
                                          "selected_policy": "lightgbm" if adopted else "baseline"})
                    pending = (week, rows, forecast_run.info.run_id)
            # 最終週に7日後の実績がデータセット内でそろう場合だけ集計。実績未確定なら保留。
            if pending is not None:
                last_week, last_rows, last_run = pending
                if last_rows.target_end.max() <= frame.target_end.max():
                    last_score = score(last_rows)
                    monitoring.append({"origin": str(last_week.date()), "evaluated_on": "offline_data_end", "worse_than_baseline": is_worse(last_score), "scores": last_score, "model_version": model_version})
                    log_weekly_history(client, parent_id, "operations_weekly", len(operations) - 1, last_score)
                    with mlflow.start_run(run_name=f"monitor-{last_week.date()}", nested=True):
                        mlflow.set_tags({"stage": "monitor", "forecast_run_id": last_run, "evaluated_on": "offline_data_end"})
                        put_scores(last_score, "observed")
            # ⑤ 成果物: 全週の予測、モデルの履歴、週とモデルの対応を保存。
            all_predictions = pd.concat(predictions, ignore_index=True)
            put_csv(all_predictions, output / "operational_predictions.csv", "operations")
            mlflow.log_dict({"monitoring": monitoring, "retrain_events": retrain_events}, "operations/monitoring.json")
            mlflow.log_dict({"model_versions": model_history, "week_to_model": week_calendar},
                            "models/version_history.json")
            mlflow.log_dict({"test_origins": [str(w.date()) for w, _ in test],
                             "operation_origins": [str(w.date()) for w, _ in operations],
                             "step_is_zero_based": True}, "operations/metric_step_calendar.json")
            summary = {"parent_run_id": parent_id, "selected_hyperparameters": params, "cv_overall": best['scores'], "test_overall": test_score, "adopted_policy": "lightgbm" if adopted else "baseline", "operational_weeks": len(operations), "retraining_events": retrain_events, "note": "historical replay; no live scheduling, no automatic database publish"}
            mlflow.log_dict(summary, "summary.json")
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("01_data/storage/training/weekly_training_full.csv"))
    parser.add_argument("--products", type=Path, default=Path("01_data/analysis/target_products.csv"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("02_model/mlflow"))
    parser.add_argument("--bad-weeks", type=int, default=2)
    options = parser.parse_args()
    run(options.csv, options.products, options.tracking_dir, bad_weeks=options.bad_weeks)
