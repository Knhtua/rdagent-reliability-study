import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = (
    Path.home()
    / "projects"
    / "RD-Agent"
    / "git_ignore_folder"
    / "ds_data"
    / "arf-12-hours-prediction-task"
    / "train"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "arf_outer_folds_seed42.csv"
)


def get_data_dir():
    configured = os.environ.get("ARF_DATA_DIR")

    if configured:
        return Path(configured).expanduser().resolve()

    return DEFAULT_DATA_DIR


def main():
    metadata_path = get_data_dir() / "ARF_12h.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    required_columns = {"ID", "ARF_LABEL"}
    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if metadata["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found.")

    if not set(
        metadata["ARF_LABEL"].unique()
    ).issubset({0, 1}):
        raise ValueError("ARF_LABEL is not binary.")

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    output = pd.DataFrame({
        "sample_index": range(len(metadata)),
        "ID": metadata["ID"].values,
        "label": metadata[
            "ARF_LABEL"
        ].astype(int).values,
        "outer_fold": -1,
    })

    for fold, (_, valid_index) in enumerate(
        splitter.split(
            metadata["ID"],
            metadata["ARF_LABEL"],
        )
    ):
        output.loc[
            valid_index,
            "outer_fold",
        ] = fold

    if (output["outer_fold"] < 0).any():
        raise RuntimeError(
            "Some samples have no assigned outer fold."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(OUTPUT_PATH, index=False)

    print("Generated:", OUTPUT_PATH)
    print("Samples:", len(output))
    print("Unique IDs:", output["ID"].nunique())
    print("\nSamples per fold:")
    print(output.groupby("outer_fold").size())
    print("\nPositive rate per fold:")
    print(
        output.groupby("outer_fold")[
            "label"
        ].mean()
    )


if __name__ == "__main__":
    main()
