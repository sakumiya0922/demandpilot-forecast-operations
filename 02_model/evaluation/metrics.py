"""実績と予測値のMAE・WAPEを計算する。"""

import numpy as np

def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.size == 0 or actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("同じ長さの空でない配列が必要です")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual < 0).any():
        raise ValueError("有限値かつ非負の実績が必要です")
    error = np.abs(actual - predicted)
    total = float(actual.sum())
    return {"mae": float(error.mean()), "wape": float(error.sum()/total) if total else None,
            "rows": len(actual), "actual_sum": total}
