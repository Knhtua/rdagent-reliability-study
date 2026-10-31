from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

CALIBRATION_PATH = TABLE_DIR / "calibration_metrics.csv"
BIN_PATH = TABLE_DIR / "calibration_bins.csv"
THRESHOLD_FOLD_PATH = TABLE_DIR / "threshold_policy_fold_results.csv"
THRESHOLD_SUMMARY_PATH = TABLE_DIR / "threshold_policy_summary.csv"
NESTED_SUMMARY_PATH = TABLE_DIR / "nested_cv_summary.csv"
FIGURE_PATH = FIGURE_DIR / "calibration_curves_v2.png"

EXPECTED_MODELS = {"LogisticRegression", "LightGBM"}
EXPECTED_METHODS = {"raw", "platt", "isotonic"}
EXPECTED_POLICIES = {
    "fixed_0.5",
    "youden_j",
    "maximum_f1",
    "target_sensitivity_0.80",
    "target_sensitivity_0.90",
}


def test_output_files_exist():
    for path in [
        CALIBRATION_PATH,
        BIN_PATH,
        THRESHOLD_FOLD_PATH,
        THRESHOLD_SUMMARY_PATH,
        FIGURE_PATH,
    ]:
        assert path.exists(), f"Missing result file: {path}"
        assert path.stat().st_size > 0, f"Result file is empty: {path}"


def test_calibration_metrics_structure():
    df = pd.read_csv(CALIBRATION_PATH)

    assert len(df) == 6
    assert set(df["model"]) == EXPECTED_MODELS
    assert set(df["calibration_method"]) == EXPECTED_METHODS
    assert not df.duplicated(["model", "calibration_method"]).any()

    bounded_columns = [
        "auroc",
        "average_precision",
        "brier_score",
        "ece_10_quantile",
        "mean_predicted_probability",
    ]

    for column in bounded_columns:
        assert df[column].between(0, 1).all(), column

    assert (df["log_loss"] >= 0).all()
    assert np.isfinite(df["calibration_intercept"]).all()
    assert np.isfinite(df["calibration_slope"]).all()


def test_raw_metrics_match_nested_cv_results():
    calibration = pd.read_csv(CALIBRATION_PATH)
    calibration = calibration[
        calibration["calibration_method"] == "raw"
    ].set_index("model")

    nested = pd.read_csv(NESTED_SUMMARY_PATH).set_index("model")

    for model in EXPECTED_MODELS:
        assert np.isclose(
            calibration.loc[model, "auroc"],
            nested.loc[model, "oof_auroc"],
        )
        assert np.isclose(
            calibration.loc[model, "average_precision"],
            nested.loc[model, "oof_average_precision"],
        )
        assert np.isclose(
            calibration.loc[model, "brier_score"],
            nested.loc[model, "oof_brier_score"],
        )


def test_platt_improves_calibration_metrics():
    df = pd.read_csv(CALIBRATION_PATH).set_index(
        ["model", "calibration_method"]
    )

    for model in EXPECTED_MODELS:
        raw = df.loc[(model, "raw")]
        platt = df.loc[(model, "platt")]

        assert platt["brier_score"] < raw["brier_score"]
        assert platt["log_loss"] < raw["log_loss"]
        assert platt["ece_10_quantile"] < raw["ece_10_quantile"]


def test_calibration_bins_complete():
    df = pd.read_csv(BIN_PATH)

    assert len(df) == 2 * 3 * 10
    assert set(df["model"]) == EXPECTED_MODELS
    assert set(df["calibration_method"]) == EXPECTED_METHODS
    assert set(df["bin"]) == set(range(10))

    for _, group in df.groupby(["model", "calibration_method"]):
        assert len(group) == 10
        assert group["sample_count"].sum() == 9921
        assert group["mean_predicted_probability"].between(0, 1).all()
        assert group["observed_positive_rate"].between(0, 1).all()


def test_threshold_fold_results_complete():
    df = pd.read_csv(THRESHOLD_FOLD_PATH)

    assert len(df) == 2 * 3 * 5 * 5
    assert set(df["model"]) == EXPECTED_MODELS
    assert set(df["calibration_method"]) == EXPECTED_METHODS
    assert set(df["threshold_policy"]) == EXPECTED_POLICIES
    assert set(df["outer_fold"]) == set(range(5))
    assert df["selected_threshold"].between(0, 1).all()

    confusion_columns = [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    confusion_total = df[confusion_columns].sum(axis=1)
    assert (confusion_total == df["validation_sample_count"]).all()

    metric_columns = [
        "sensitivity_recall",
        "specificity",
        "precision_ppv",
        "negative_predictive_value",
        "f1",
        "balanced_accuracy",
        "predicted_positive_rate",
    ]

    for column in metric_columns:
        assert df[column].between(0, 1).all(), column

    fixed = df[df["threshold_policy"] == "fixed_0.5"]
    assert np.allclose(fixed["selected_threshold"], 0.5)


def test_threshold_summary_complete():
    df = pd.read_csv(THRESHOLD_SUMMARY_PATH)

    assert len(df) == 2 * 3 * 5
    assert set(df["model"]) == EXPECTED_MODELS
    assert set(df["calibration_method"]) == EXPECTED_METHODS
    assert set(df["threshold_policy"]) == EXPECTED_POLICIES
    assert df["mean_selected_threshold"].between(0, 1).all()

    confusion_columns = [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    assert (df[confusion_columns].sum(axis=1) == 9921).all()

    metric_columns = [
        "sensitivity_recall",
        "specificity",
        "precision_ppv",
        "negative_predictive_value",
        "f1",
        "balanced_accuracy",
        "predicted_positive_rate",
    ]

    for column in metric_columns:
        assert df[column].between(0, 1).all(), column
