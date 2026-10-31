from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "nested_cv_oof_predictions.csv"
)

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

CALIBRATION_METRICS_PATH = TABLE_DIR / "calibration_metrics.csv"
CALIBRATION_BINS_PATH = TABLE_DIR / "calibration_bins.csv"
THRESHOLD_FOLDS_PATH = TABLE_DIR / "threshold_policy_fold_results.csv"
THRESHOLD_SUMMARY_PATH = TABLE_DIR / "threshold_policy_summary.csv"
CALIBRATION_FIGURE_PATH = FIGURE_DIR / "calibration_curves_v2.png"

MODEL_COLUMNS = {
    "LogisticRegression": "logistic_probability",
    "LightGBM": "lightgbm_probability",
}

CALIBRATION_METHODS = ["raw", "platt", "isotonic"]
THRESHOLD_POLICIES = [
    "fixed_0.5",
    "youden_j",
    "maximum_f1",
    "target_sensitivity_0.80",
    "target_sensitivity_0.90",
]

EPSILON = 1e-6
N_CALIBRATION_BINS = 10


def clip_probability(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        EPSILON,
        1.0 - EPSILON,
    )


def logit(probability):
    probability = clip_probability(probability)
    return np.log(probability / (1.0 - probability))


def fit_and_apply_calibrator(
    method,
    train_probability,
    train_label,
    test_probability,
):
    train_probability = clip_probability(train_probability)
    test_probability = clip_probability(test_probability)
    train_label = np.asarray(train_label, dtype=np.int64)

    if method == "raw":
        return train_probability, test_probability

    if method == "platt":
        calibrator = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        calibrator.fit(
            logit(train_probability).reshape(-1, 1),
            train_label,
        )

        calibrated_train = calibrator.predict_proba(
            logit(train_probability).reshape(-1, 1)
        )[:, 1]

        calibrated_test = calibrator.predict_proba(
            logit(test_probability).reshape(-1, 1)
        )[:, 1]

        return (
            clip_probability(calibrated_train),
            clip_probability(calibrated_test),
        )

    if method == "isotonic":
        calibrator = IsotonicRegression(
            y_min=EPSILON,
            y_max=1.0 - EPSILON,
            out_of_bounds="clip",
        )
        calibrator.fit(train_probability, train_label)

        return (
            clip_probability(calibrator.predict(train_probability)),
            clip_probability(calibrator.predict(test_probability)),
        )

    raise ValueError(f"Unknown calibration method: {method}")


def calibration_intercept_and_slope(label, probability):
    model = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        logit(probability).reshape(-1, 1),
        label,
    )

    return float(model.intercept_[0]), float(model.coef_[0, 0])


def make_quantile_calibration_bins(
    label,
    probability,
    model_name,
    calibration_method,
):
    label = np.asarray(label, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)

    order = np.argsort(probability)
    groups = np.array_split(order, N_CALIBRATION_BINS)

    records = []

    for bin_number, indices in enumerate(groups):
        bin_label = label[indices]
        bin_probability = probability[indices]

        records.append(
            {
                "model": model_name,
                "calibration_method": calibration_method,
                "bin": bin_number,
                "sample_count": len(indices),
                "mean_predicted_probability": float(
                    np.mean(bin_probability)
                ),
                "observed_positive_rate": float(
                    np.mean(bin_label)
                ),
                "absolute_calibration_error": float(
                    abs(
                        np.mean(bin_probability)
                        - np.mean(bin_label)
                    )
                ),
            }
        )

    return records


def expected_calibration_error(bin_records):
    frame = pd.DataFrame(bin_records)
    weights = frame["sample_count"] / frame["sample_count"].sum()

    return float(
        np.sum(
            weights
            * frame["absolute_calibration_error"]
        )
    )


def select_threshold(policy, label, probability):
    label = np.asarray(label, dtype=np.int64)
    probability = clip_probability(probability)

    if policy == "fixed_0.5":
        return 0.5

    if policy == "youden_j":
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            label,
            probability,
        )

        finite = np.isfinite(thresholds)
        scores = true_positive_rate[finite] - false_positive_rate[finite]
        finite_thresholds = thresholds[finite]

        return float(finite_thresholds[np.argmax(scores)])

    if policy == "maximum_f1":
        precision, recall, thresholds = precision_recall_curve(
            label,
            probability,
        )

        f1_values = (
            2.0
            * precision[:-1]
            * recall[:-1]
            / np.maximum(
                precision[:-1] + recall[:-1],
                EPSILON,
            )
        )

        return float(thresholds[np.argmax(f1_values)])

    if policy.startswith("target_sensitivity_"):
        target = float(policy.rsplit("_", 1)[1])

        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            label,
            probability,
        )

        valid = (
            np.isfinite(thresholds)
            & (true_positive_rate >= target)
        )

        if not np.any(valid):
            return float(np.min(probability))

        valid_thresholds = thresholds[valid]

        # Select the highest threshold that still reaches the target
        # sensitivity on the threshold-development data.
        return float(np.max(valid_thresholds))

    raise ValueError(f"Unknown threshold policy: {policy}")


def classification_metrics(label, probability, threshold):
    label = np.asarray(label, dtype=np.int64)
    predicted = (
        np.asarray(probability, dtype=np.float64) >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        label,
        predicted,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    negative_predictive_value = tn / max(tn + fn, 1)

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(precision),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "f1": float(f1_score(label, predicted)),
        "balanced_accuracy": float(
            (sensitivity + specificity) / 2.0
        ),
        "predicted_positive_rate": float(np.mean(predicted)),
    }


def main():
    start_time = time.perf_counter()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_index",
        "ID",
        "label",
        "outer_fold",
        "logistic_probability",
        "lightgbm_probability",
    }

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data["sample_index"].duplicated().any():
        raise ValueError("Duplicate sample_index detected.")

    if data["ID"].duplicated().any():
        raise ValueError("Duplicate ID detected.")

    if data.isna().any().any():
        raise ValueError("Missing values detected.")

    folds = sorted(data["outer_fold"].unique())
    label = data["label"].to_numpy(dtype=np.int64)

    calibration_metric_records = []
    calibration_bin_records = []
    calibrated_predictions = {}

    print("Starting cross-fitted calibration analysis...")

    for model_name, probability_column in MODEL_COLUMNS.items():
        raw_probability = data[
            probability_column
        ].to_numpy(dtype=np.float64)

        for method in CALIBRATION_METHODS:
            cross_fitted_probability = np.zeros(
                len(data),
                dtype=np.float64,
            )

            for held_out_fold in folds:
                train_mask = (
                    data["outer_fold"].to_numpy()
                    != held_out_fold
                )
                test_mask = ~train_mask

                _, calibrated_test = fit_and_apply_calibrator(
                    method=method,
                    train_probability=raw_probability[train_mask],
                    train_label=label[train_mask],
                    test_probability=raw_probability[test_mask],
                )

                cross_fitted_probability[test_mask] = calibrated_test

            calibrated_predictions[
                (model_name, method)
            ] = cross_fitted_probability

            bin_records = make_quantile_calibration_bins(
                label=label,
                probability=cross_fitted_probability,
                model_name=model_name,
                calibration_method=method,
            )

            calibration_bin_records.extend(bin_records)

            intercept, slope = calibration_intercept_and_slope(
                label,
                cross_fitted_probability,
            )

            record = {
                "model": model_name,
                "calibration_method": method,
                "auroc": roc_auc_score(
                    label,
                    cross_fitted_probability,
                ),
                "average_precision": average_precision_score(
                    label,
                    cross_fitted_probability,
                ),
                "brier_score": brier_score_loss(
                    label,
                    cross_fitted_probability,
                ),
                "log_loss": log_loss(
                    label,
                    cross_fitted_probability,
                    labels=[0, 1],
                ),
                "ece_10_quantile": expected_calibration_error(
                    bin_records
                ),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "mean_predicted_probability": float(
                    np.mean(cross_fitted_probability)
                ),
            }

            calibration_metric_records.append(record)

            print(
                f"{model_name} / {method}: "
                f"AUROC={record['auroc']:.6f}, "
                f"AP={record['average_precision']:.6f}, "
                f"Brier={record['brier_score']:.6f}, "
                f"ECE={record['ece_10_quantile']:.6f}"
            )

    calibration_metrics = pd.DataFrame(
        calibration_metric_records
    )
    calibration_bins = pd.DataFrame(
        calibration_bin_records
    )

    calibration_metrics.to_csv(
        CALIBRATION_METRICS_PATH,
        index=False,
    )
    calibration_bins.to_csv(
        CALIBRATION_BINS_PATH,
        index=False,
    )

    print("\nStarting leakage-controlled threshold analysis...")

    threshold_fold_records = []

    for model_name, probability_column in MODEL_COLUMNS.items():
        raw_probability = data[
            probability_column
        ].to_numpy(dtype=np.float64)

        for method in CALIBRATION_METHODS:
            for held_out_fold in folds:
                train_mask = (
                    data["outer_fold"].to_numpy()
                    != held_out_fold
                )
                test_mask = ~train_mask

                calibrated_train, calibrated_test = (
                    fit_and_apply_calibrator(
                        method=method,
                        train_probability=raw_probability[train_mask],
                        train_label=label[train_mask],
                        test_probability=raw_probability[test_mask],
                    )
                )

                for policy in THRESHOLD_POLICIES:
                    threshold = select_threshold(
                        policy=policy,
                        label=label[train_mask],
                        probability=calibrated_train,
                    )

                    metrics = classification_metrics(
                        label=label[test_mask],
                        probability=calibrated_test,
                        threshold=threshold,
                    )

                    threshold_fold_records.append(
                        {
                            "model": model_name,
                            "calibration_method": method,
                            "threshold_policy": policy,
                            "outer_fold": held_out_fold,
                            "selected_threshold": threshold,
                            "validation_sample_count": int(
                                np.sum(test_mask)
                            ),
                            **metrics,
                        }
                    )

    threshold_folds = pd.DataFrame(
        threshold_fold_records
    )
    threshold_folds.to_csv(
        THRESHOLD_FOLDS_PATH,
        index=False,
    )

    summary_records = []

    grouping_columns = [
        "model",
        "calibration_method",
        "threshold_policy",
    ]

    for keys, group in threshold_folds.groupby(
        grouping_columns,
        sort=True,
    ):
        model_name, method, policy = keys

        tn = int(group["true_negative"].sum())
        fp = int(group["false_positive"].sum())
        fn = int(group["false_negative"].sum())
        tp = int(group["true_positive"].sum())

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        f1 = (
            2.0
            * precision
            * sensitivity
            / max(precision + sensitivity, EPSILON)
        )

        total = tn + fp + fn + tp

        summary_records.append(
            {
                "model": model_name,
                "calibration_method": method,
                "threshold_policy": policy,
                "mean_selected_threshold": float(
                    group["selected_threshold"].mean()
                ),
                "std_selected_threshold": float(
                    group["selected_threshold"].std(ddof=1)
                ),
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "sensitivity_recall": sensitivity,
                "specificity": specificity,
                "precision_ppv": precision,
                "negative_predictive_value": npv,
                "f1": f1,
                "balanced_accuracy": (
                    sensitivity + specificity
                )
                / 2.0,
                "predicted_positive_rate": (
                    fp + tp
                )
                / max(total, 1),
            }
        )

    threshold_summary = pd.DataFrame(summary_records)
    threshold_summary.to_csv(
        THRESHOLD_SUMMARY_PATH,
        index=False,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        sharex=True,
        sharey=True,
    )

    for axis, model_name in zip(
        axes,
        MODEL_COLUMNS.keys(),
    ):
        model_bins = calibration_bins[
            calibration_bins["model"] == model_name
        ]

        for method in CALIBRATION_METHODS:
            subset = model_bins[
                model_bins["calibration_method"] == method
            ]

            axis.plot(
                subset["mean_predicted_probability"],
                subset["observed_positive_rate"],
                marker="o",
                linewidth=2,
                label=method,
            )

        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            linewidth=1,
            label="ideal",
        )

        axis.set_title(model_name)
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed positive rate")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(
        "Cross-fitted calibration curves",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        CALIBRATION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    runtime = time.perf_counter() - start_time

    print("\nCalibration metrics:")
    print(calibration_metrics.to_string(index=False))

    print("\nResult files:")
    print(CALIBRATION_METRICS_PATH)
    print(CALIBRATION_BINS_PATH)
    print(THRESHOLD_FOLDS_PATH)
    print(THRESHOLD_SUMMARY_PATH)
    print(CALIBRATION_FIGURE_PATH)
    print(f"\nTotal runtime: {runtime:.2f} seconds")


if __name__ == "__main__":
    main()
