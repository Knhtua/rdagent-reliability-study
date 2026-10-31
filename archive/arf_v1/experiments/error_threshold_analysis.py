from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "learning_records/arf_research/feature_importance_oof.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "learning_records/arf_research"
METRICS_PATH = OUTPUT_DIR / "threshold_metrics.csv"
ERROR_CASES_PATH = OUTPUT_DIR / "error_cases.csv"
CALIBRATION_PATH = OUTPUT_DIR / "calibration_bins.csv"
SUMMARY_PATH = OUTPUT_DIR / "threshold_analysis_summary.csv"
PLOT_PATH = OUTPUT_DIR / "threshold_analysis.png"


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def calculate_metrics(y_true, probability, threshold, name):
    predicted = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)

    return {
        "threshold_name": name,
        "threshold": threshold,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision,
        "negative_predictive_value": npv,
        "f1": f1_score(y_true, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "predicted_positive_rate": predicted.mean(),
    }


def threshold_for_target_sensitivity(
    y_true,
    probability,
    target_sensitivity,
):
    fpr, tpr, thresholds = roc_curve(y_true, probability)

    valid = np.where(tpr >= target_sensitivity)[0]

    if len(valid) == 0:
        return 0.0

    # Among thresholds reaching the target sensitivity, choose the one with the highest specificity.
    best_index = valid[np.argmin(fpr[valid])]
    return float(thresholds[best_index])


def main():
    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "ID",
        "ARF_LABEL",
        "oof_probability",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df[list(required_columns)].isna().any().any():
        raise ValueError("The input file contains missing values.")

    y_true = df["ARF_LABEL"].to_numpy(dtype=int)
    probability = df["oof_probability"].to_numpy(dtype=float)

    if not np.all((probability >= 0) & (probability <= 1)):
        raise ValueError("Predicted probabilities are outside the [0, 1] range.")

    overall_auroc = roc_auc_score(y_true, probability)
    overall_ap = average_precision_score(y_true, probability)
    brier_score = brier_score_loss(y_true, probability)

    fpr, tpr, roc_thresholds = roc_curve(y_true, probability)
    youden_index = tpr - fpr
    youden_threshold = float(
        roc_thresholds[np.argmax(youden_index)]
    )

    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(y_true, probability)
    )

    f1_values = (
        2
        * precision_curve[:-1]
        * recall_curve[:-1]
        / (
            precision_curve[:-1]
            + recall_curve[:-1]
            + 1e-12
        )
    )

    max_f1_threshold = float(
        pr_thresholds[np.argmax(f1_values)]
    )

    sensitivity_80_threshold = (
        threshold_for_target_sensitivity(
            y_true,
            probability,
            0.80,
        )
    )

    sensitivity_90_threshold = (
        threshold_for_target_sensitivity(
            y_true,
            probability,
            0.90,
        )
    )

    threshold_definitions = [
        ("fixed_0.5", 0.5),
        ("youden_j", youden_threshold),
        ("maximum_f1", max_f1_threshold),
        (
            "target_sensitivity_0.80",
            sensitivity_80_threshold,
        ),
        (
            "target_sensitivity_0.90",
            sensitivity_90_threshold,
        ),
    ]

    metric_records = [
        calculate_metrics(
            y_true,
            probability,
            threshold,
            name,
        )
        for name, threshold in threshold_definitions
    ]

    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(METRICS_PATH, index=False)

    # Describe false-positive and false-negative cases at the Youden J threshold.
    selected_threshold = youden_threshold
    selected_prediction = (
        probability >= selected_threshold
    ).astype(int)

    error_df = df.copy()
    error_df["predicted_label"] = selected_prediction
    error_df["threshold_name"] = "youden_j"
    error_df["threshold"] = selected_threshold

    error_df["error_type"] = np.select(
        [
            (y_true == 1) & (selected_prediction == 0),
            (y_true == 0) & (selected_prediction == 1),
        ],
        [
            "false_negative",
            "false_positive",
        ],
        default="correct",
    )

    error_df = error_df[
        error_df["error_type"] != "correct"
    ].copy()

    error_df["distance_from_threshold"] = (
        error_df["oof_probability"] - selected_threshold
    ).abs()

    error_df = error_df.sort_values(
        ["error_type", "distance_from_threshold"],
        ascending=[True, False],
    )

    error_df.to_csv(ERROR_CASES_PATH, index=False)

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probability,
        n_bins=10,
        strategy="quantile",
    )

    calibration_df = pd.DataFrame({
        "mean_predicted_probability": mean_predicted,
        "observed_positive_fraction": fraction_positive,
    })

    calibration_df["absolute_calibration_error"] = (
        calibration_df["mean_predicted_probability"]
        - calibration_df["observed_positive_fraction"]
    ).abs()

    calibration_df.to_csv(CALIBRATION_PATH, index=False)

    summary_df = pd.DataFrame([{
        "sample_count": len(df),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "oof_auroc": overall_auroc,
        "oof_average_precision": overall_ap,
        "brier_score": brier_score,
        "mean_predicted_probability": float(
            probability.mean()
        ),
        "youden_threshold": youden_threshold,
        "maximum_f1_threshold": max_f1_threshold,
    }])

    summary_df.to_csv(SUMMARY_PATH, index=False)

    # Plot figures.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    # ROC
    axes[0, 0].plot(
        fpr,
        tpr,
        label=f"AUROC = {overall_auroc:.4f}",
    )
    axes[0, 0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )
    axes[0, 0].set_title("OOF ROC Curve")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend()

    # Precision–Recall
    axes[0, 1].plot(
        recall_curve,
        precision_curve,
        label=f"AP = {overall_ap:.4f}",
    )
    axes[0, 1].axhline(
        y_true.mean(),
        linestyle="--",
        color="gray",
        label=f"Positive rate = {y_true.mean():.4f}",
    )
    axes[0, 1].set_title("OOF Precision–Recall Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()

    # Calibration
    axes[1, 0].plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label=f"Brier = {brier_score:.4f}",
    )
    axes[1, 0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect calibration",
    )
    axes[1, 0].set_title("OOF Calibration Curve")
    axes[1, 0].set_xlabel("Mean Predicted Probability")
    axes[1, 0].set_ylabel("Observed Positive Fraction")
    axes[1, 0].legend()

    # Threshold trade-offs.
    plot_metrics = metrics_df.set_index("threshold_name")
    x = np.arange(len(plot_metrics))

    axes[1, 1].plot(
        x,
        plot_metrics["sensitivity_recall"],
        marker="o",
        label="Sensitivity",
    )
    axes[1, 1].plot(
        x,
        plot_metrics["specificity"],
        marker="o",
        label="Specificity",
    )
    axes[1, 1].plot(
        x,
        plot_metrics["precision_ppv"],
        marker="o",
        label="Precision",
    )
    axes[1, 1].plot(
        x,
        plot_metrics["f1"],
        marker="o",
        label="F1",
    )

    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        plot_metrics.index,
        rotation=25,
        ha="right",
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Threshold Trade-offs")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    print("========== Experiment 008 summary ==========")
    print("Samples:", len(df))
    print("Positive cases:", int(y_true.sum()))
    print("Positive rate:", f"{y_true.mean():.6f}")
    print("OOF AUROC:", f"{overall_auroc:.6f}")
    print("OOF AP:", f"{overall_ap:.6f}")
    print("Brier score:", f"{brier_score:.6f}")
    print(
        "Mean predicted probability:",
        f"{probability.mean():.6f}",
    )

    print("\nThreshold comparison:")
    print(metrics_df.to_string(index=False))

    print("\nError counts at the Youden J threshold:")
    print(error_df["error_type"].value_counts())

    print("\nOutput files:")
    for path in [
        METRICS_PATH,
        ERROR_CASES_PATH,
        CALIBRATION_PATH,
        SUMMARY_PATH,
        PLOT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
