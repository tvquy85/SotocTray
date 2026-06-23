from scripts.make_walkforward_configs import make_folds


def test_make_folds_monotonic():
    folds = make_folds(valid_index=300, test_index=400, fold_count=3, step_size=50, min_train_end=200)
    assert folds == [(0, 300, 400), (1, 350, 450), (2, 400, 500)]


def test_make_folds_rejects_invalid_training_window():
    try:
        make_folds(valid_index=100, test_index=200, fold_count=1, step_size=50, min_train_end=200)
    except ValueError:
        return
    raise AssertionError("Expected invalid walk-forward split to fail")
