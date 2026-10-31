from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "arf_outer_folds_seed42.csv"
)


def load_splits():
    assert SPLIT_PATH.exists(), f"Split file does not exist: {SPLIT_PATH}"
    return pd.read_csv(SPLIT_PATH)


def test_expected_columns():
    df = load_splits()
    assert list(df.columns) == [
        "sample_index",
        "ID",
        "label",
        "outer_fold",
    ]


def test_sample_count():
    df = load_splits()
    assert len(df) == 9921


def test_sample_index_is_unique_and_complete():
    df = load_splits()
    assert df["sample_index"].is_unique
    assert sorted(df["sample_index"].tolist()) == list(range(9921))


def test_id_is_unique():
    df = load_splits()
    assert df["ID"].is_unique
    assert df["ID"].nunique() == 9921


def test_labels_and_folds_are_valid():
    df = load_splits()
    assert set(df["label"].unique()) == {0, 1}
    assert set(df["outer_fold"].unique()) == {0, 1, 2, 3, 4}


def test_fold_sizes_and_positive_rates_are_balanced():
    df = load_splits()

    fold_sizes = df.groupby("outer_fold").size()
    positive_rates = df.groupby("outer_fold")["label"].mean()

    assert fold_sizes.max() - fold_sizes.min() <= 1
    assert positive_rates.max() - positive_rates.min() < 0.001
