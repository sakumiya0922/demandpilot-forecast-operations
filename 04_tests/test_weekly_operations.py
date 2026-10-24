"""時系列分割・ラベルの利用可能時点・運用の再学習判断を単体検証。"""
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
