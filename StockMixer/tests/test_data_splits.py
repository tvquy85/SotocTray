import pytest
from src_tc.data import make_offsets, assert_no_target_leakage


def test_make_offsets_non_empty_and_ordered():
    train, valid, test = make_offsets(
        valid_index=100, test_index=150, trade_dates=220, lookback=16, steps=1
    )
    assert len(train) > 0 and len(valid) > 0 and len(test) > 0
    assert train.max() < valid.max() < test.max()


def test_no_target_leakage_train_valid_test():
    valid_index, test_index, trade_dates, lookback, steps = 100, 150, 220, 16, 1
    train, valid, test = make_offsets(valid_index, test_index, trade_dates, lookback, steps)
    assert_no_target_leakage(train, valid_index, test_index, lookback, steps, "train")
    assert_no_target_leakage(valid, valid_index, test_index, lookback, steps, "valid")
    assert_no_target_leakage(test, test_index, trade_dates, lookback, steps, "test")


def test_empty_split_raises_assertion():
    with pytest.raises(AssertionError):
        make_offsets(valid_index=10, test_index=15, trade_dates=20, lookback=16, steps=1)
