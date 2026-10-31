import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = (
    Path.home()
    / "projects"
    / "RD-Agent"
    / "git_ignore_folder"
    / "ds_data"
    / "arf-12-hours-prediction-task"
    / "train"
)

DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "arf_outer_folds_seed42.csv"
)


@dataclass
class ARFDataset:
    metadata: pd.DataFrame
    splits: pd.DataFrame
    tensor_shape: tuple[int, int, int]
    values: np.ndarray
    coordinates: np.ndarray
    fill_value: int | float


def get_data_dir() -> Path:
    configured_path = os.environ.get("ARF_DATA_DIR")

    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return DEFAULT_DATA_DIR


def load_arf_dataset(
    data_dir: Path | None = None,
    split_path: Path = DEFAULT_SPLIT_PATH,
) -> ARFDataset:
    data_dir = get_data_dir() if data_dir is None else Path(data_dir)

    metadata_path = data_dir / "ARF_12h.csv"
    tensor_path = data_dir / "X.npz"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {metadata_path}"
        )

    if not tensor_path.exists():
        raise FileNotFoundError(
            f"Tensor file not found: {tensor_path}"
        )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Fixed split file not found: {split_path}"
        )

    metadata = pd.read_csv(metadata_path)
    splits = pd.read_csv(split_path)

    archive = np.load(tensor_path)

    required_keys = {
        "data",
        "shape",
        "fill_value",
        "coords",
    }

    missing_keys = required_keys - set(archive.files)
    if missing_keys:
        raise ValueError(
            f"X.npz is missing fields: {sorted(missing_keys)}"
        )

    tensor_shape = tuple(
        int(value) for value in archive["shape"]
    )
    values = archive["data"]
    coordinates = archive["coords"]
    fill_value = archive["fill_value"].item()

    if tensor_shape[0] != len(metadata):
        raise ValueError(
            "The tensor sample count does not match the label-file row count."
        )

    if len(splits) != len(metadata):
        raise ValueError(
            "The fixed-split sample count does not match the label-file row count."
        )

    expected_indices = np.arange(len(metadata))

    if not np.array_equal(
        splits["sample_index"].to_numpy(),
        expected_indices,
    ):
        raise ValueError(
            "sample_index order in the fixed split is incorrect."
        )

    if not np.array_equal(
        splits["ID"].to_numpy(),
        metadata["ID"].to_numpy(),
    ):
        raise ValueError(
            "IDs in the fixed split do not match the label file."
        )

    if not np.array_equal(
        splits["label"].to_numpy(),
        metadata["ARF_LABEL"].to_numpy(),
    ):
        raise ValueError(
            "Labels in the fixed split do not match the source labels."
        )

    if coordinates.shape[0] != 3:
        raise ValueError(
            "Sparse-tensor coordinates must include sample, time, and feature dimensions."
        )

    if len(values) != coordinates.shape[1]:
        raise ValueError(
            "The number of sparse-tensor values does not match the number of coordinates."
        )

    if fill_value != 0:
        raise ValueError(
            f"This implementation requires a fill value of 0; received {fill_value}."
        )

    return ARFDataset(
        metadata=metadata,
        splits=splits,
        tensor_shape=tensor_shape,
        values=values,
        coordinates=coordinates,
        fill_value=fill_value,
    )


def build_sum_features(
    dataset: ARFDataset,
) -> sparse.csr_matrix:
    sample_count, _, feature_count = dataset.tensor_shape

    sample_coordinates = dataset.coordinates[0]
    feature_coordinates = dataset.coordinates[2]

    matrix = sparse.coo_matrix(
        (
            dataset.values.astype(np.float32),
            (sample_coordinates, feature_coordinates),
        ),
        shape=(sample_count, feature_count),
        dtype=np.float32,
    ).tocsr()

    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    return matrix
