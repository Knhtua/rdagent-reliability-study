import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

OOF_PATH = TABLE_DIR / "nested_cv_oof_predictions.csv"
FOLD_PATH = TABLE_DIR / "nested_cv_fold_results.csv"
SUMMARY_PATH = TABLE_DIR / "nested_cv_summary.csv"
PARAMETER_PATH = (
    TABLE_DIR / "nested_cv_selected_parameters.csv"
)


def test_result_files_exist():
    for path in [
        OOF_PATH,
        FOLD_PATH,
        SUMMARY_PATH,
        PARAMETER_PATH,
    ]:
        assert path.exists(), f"Missing result file: {path}"


def test_oof_predictions_are_complete():
    df = pd.read_csv(OOF_PATH)

    assert len(df) == 9921
    assert df["sample_index"].is_unique
    assert df["ID"].is_unique
    assert not df.isna().any().any()


def test_probabilities_are_valid():
    df = pd.read_csv(OOF_PATH)

    for column in [
        "logistic_probability",
        "lightgbm_probability",
    ]:
        assert df[column].between(0, 1).all()


def test_outer_folds_are_complete():
    df = pd.read_csv(OOF_PATH)

    assert set(df["outer_fold"]) == {0, 1, 2, 3, 4}

    fold_sizes = df.groupby("outer_fold").size()
    assert fold_sizes.max() - fold_sizes.min() <= 1


def test_fold_metrics_can_be_reproduced():
    oof = pd.read_csv(OOF_PATH)
    recorded = pd.read_csv(FOLD_PATH)

    probability_columns = {
        "LogisticRegression": "logistic_probability",
        "LightGBM": "lightgbm_probability",
    }

    for _, row in recorded.iterrows():
        fold_data = oof[
            oof["outer_fold"] == row["outer_fold"]
        ]

        y_true = fold_data["label"]
        probability = fold_data[
            probability_columns[row["model"]]
        ]

        assert np.isclose(
            roc_auc_score(y_true, probability),
            row["auroc"],
        )

        assert np.isclose(
            average_precision_score(
                y_true,
                probability,
            ),
            row["average_precision"],
        )

        assert np.isclose(
            brier_score_loss(y_true, probability),
            row["brier_score"],
        )


def test_summary_oof_metrics_can_be_reproduced():
    oof = pd.read_csv(OOF_PATH)
    summary = pd.read_csv(SUMMARY_PATH).set_index(
        "model"
    )

    probability_columns = {
        "LogisticRegression": "logistic_probability",
        "LightGBM": "lightgbm_probability",
    }

    for model, probability_column in (
        probability_columns.items()
    ):
        y_true = oof["label"]
        probability = oof[probability_column]

        assert np.isclose(
            roc_auc_score(y_true, probability),
            summary.loc[model, "oof_auroc"],
        )

        assert np.isclose(
            average_precision_score(
                y_true,
                probability,
            ),
            summary.loc[
                model,
                "oof_average_precision",
            ],
        )


def test_selected_parameters_maximize_inner_auroc():
    parameters = pd.read_csv(PARAMETER_PATH)

    assert len(parameters) == 10

    for _, row in parameters.iterrows():
        selected = json.loads(
            row["selected_parameters"]
        )
        candidates = json.loads(
            row["all_candidates"]
        )

        best_candidate_score = max(
            candidate["mean_inner_auroc"]
            for candidate in candidates
        )

        assert np.isclose(
            selected["mean_inner_auroc"],
            best_candidate_score,
        )
