"""曜日・予測対象期間・未来情報と不明期間の扱いを検証する。"""
import pandas as pd
import pytest

from training.training_data import build_training_data, FEATURE_COLUMNS


def daily_data():
    rows = [(code, day, quantity) for code,quantity in [("A",1),("B",3)]
            for day in pd.date_range("2011-01-01", "2011-03-31")]
    return pd.DataFrame(rows,columns=["stock_code","sales_date","quantity"])


def test_sunday_origins_and_nonoverlapping_complete_labels():
    result = build_training_data(daily_data(),["A","B"],"2011-01-01","2011-03-31",unknown_periods=[])
    assert result.forecast_origin.dt.dayofweek.eq(6).all()
    assert result.target_start.dt.dayofweek.eq(0).all()
    assert result.target_end.dt.dayofweek.eq(6).all()
    assert result.target_end.le(pd.Timestamp("2011-03-31")).all()
    assert result.loc[result.stock_code.eq('A'),'target'].eq(7).all()
    assert result.loc[result.stock_code.eq('B'),'target'].eq(21).all()
    assert result.loc[result.stock_code.eq('B'),'sales_sum_28'].eq(84).all()
    origins = result.loc[result.stock_code.eq('A')].sort_values('forecast_origin')
    assert origins.target_start.iloc[1:].reset_index(drop=True).gt(origins.target_end.iloc[:-1].reset_index(drop=True)).all()


def test_future_changes_labels_but_cannot_change_features():
    daily=daily_data()
    first=build_training_data(daily,['A','B'],'2011-01-01','2011-03-31',unknown_periods=[])
    daily.loc[daily.sales_date.gt('2011-02-06'),'quantity']=1000
    second=build_training_data(daily,['A','B'],'2011-01-01','2011-03-31',unknown_periods=[])
    before=first.loc[first.forecast_origin.le('2011-02-06')].reset_index(drop=True)
    after=second.loc[second.forecast_origin.le('2011-02-06')].reset_index(drop=True)
    pd.testing.assert_frame_equal(before[FEATURE_COLUMNS],after[FEATURE_COLUMNS])
    assert not before.target.equals(after.target)


def test_unknown_period_invalidates_history_and_target_without_shifting_calendar():
    result=build_training_data(daily_data(),['A','B'],'2011-01-01','2011-03-31',
                               unknown_periods=[('2011-02-09','2011-02-15')])
    assert not ((result.history_start.le('2011-02-15')) & (result.forecast_origin.ge('2011-02-09'))).any()
    assert not ((result.target_start.le('2011-02-15')) & (result.target_end.ge('2011-02-09'))).any()
    assert not result.forecast_origin.between('2011-02-06','2011-03-13').any()
    assert result.forecast_origin.eq('2011-03-20').any()


def test_product_day_with_no_sales_is_zero():
    daily=daily_data()
    daily=daily.loc[~(daily.stock_code.eq('A') & daily.sales_date.eq('2011-02-09'))]
    result=build_training_data(daily,['A','B'],'2011-01-01','2011-03-31',unknown_periods=[])
    assert result.loc[result.stock_code.eq('A') & result.forecast_origin.eq('2011-02-06'),'target'].item()==6
    assert result.loc[result.stock_code.eq('B') & result.forecast_origin.eq('2011-02-06'),'target'].item()==21


def test_duplicate_product_days_are_rejected():
    daily=daily_data()
    with pytest.raises(ValueError,match='重複'):
        build_training_data(pd.concat([daily,daily.iloc[:1]]),['A','B'],'2011-01-01','2011-03-31')
