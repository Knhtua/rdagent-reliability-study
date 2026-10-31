import numpy as np
import pytest

from src.data.arf_loader import (
    build_sum_features,
    load_arf_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    return load_arf_dataset()


def test_tensor_shape(dataset):
    assert dataset.tensor_shape == (9921, 12, 4816)


def test_metadata_and_split_alignment(dataset):
    assert len(dataset.metadata) == 9921
    assert len(dataset.splits) == 9921

    assert np.array_equal(
        dataset.metadata["ID"].to_numpy(),
        dataset.splits["ID"].to_numpy(),
    )


def test_class_distribution(dataset):
    labels = dataset.metadata["ARF_LABEL"]

    assert int((labels == 0).sum()) == 8961
    assert int((labels == 1).sum()) == 960


def test_sparse_tensor_structure(dataset):
    assert dataset.coordinates.shape[0] == 3
    assert dataset.coordinates.shape[1] == len(dataset.values)
    assert dataset.fill_value == 0


def test_sum_feature_shape(dataset):
    features = build_sum_features(dataset)

    assert features.shape == (9921, 4816)
    assert features.nnz > 0


def test_fixed_folds(dataset):
    assert set(dataset.splits["outer_fold"]) == {
        0, 1, 2, 3, 4
    }
