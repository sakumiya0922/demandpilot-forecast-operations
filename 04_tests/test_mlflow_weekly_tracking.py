"""週次メトリクスの履歴がMLflowにstep付きで保存されることを確認する。"""
from training.weekly_operations import log_weekly_history


class FakeClient:
    def __init__(self):
        self.calls = []

    def log_metric(self, run_id, name, value, step):
        self.calls.append((run_id, name, value, step))


def test_weekly_history_logs_both_models_with_step():
    client = FakeClient()
    scores = {"lightgbm": {"mae": 12.0, "wape": .25},
              "baseline": {"mae": 17.0, "wape": .36}}
    log_weekly_history(client, "parent-run", "operations_weekly", 3, scores)
    assert len(client.calls) == 4
    assert all(call[0] == "parent-run" and call[3] == 3 for call in client.calls)
    assert ("parent-run", "operations_weekly_lightgbm_wape", .25, 3) in client.calls
    assert ("parent-run", "operations_weekly_baseline_mae", 17.0, 3) in client.calls


def test_missing_metric_is_not_logged():
    client = FakeClient()
    scores = {"lightgbm": {"mae": 8.0, "wape": None},
              "baseline": {"mae": 9.0, "wape": None}}
    log_weekly_history(client, "run", "test_weekly", 0, scores)
    assert len(client.calls) == 2
    assert all("mae" in call[1] for call in client.calls)
