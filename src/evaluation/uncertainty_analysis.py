from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FOLD_PATH = TABLE_DIR / "nested_cv_fold_results.csv"
OOF_PATH = TABLE_DIR / "nested_cv_oof_predictions.csv"

PAIRED_PATH = TABLE_DIR / "paired_fold_differences.csv"
INFERENCE_PATH = TABLE_DIR / "paired_inference_summary.csv"
BOOTSTRAP_PATH = TABLE_DIR / "oof_bootstrap_summary.csv"
BOOTSTRAP_DISTRIBUTION_PATH = (
    TABLE_DIR / "oof_bootstrap_differences.csv"
)

N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95


def confidence_interval(
    mean,
    standard_error,
    degrees_of_freedom,
):
    alpha = 1.0 - CONFIDENCE_LEVEL
    critical_value = t.ppf(
        1.0 - alpha / 2.0,
        degrees_of_freedom,
    )

    return (
        mean - critical_value * standard_error,
        mean + critical_value * standard_error,
    )


def paired_fold_analysis(fold_results):
    metrics = {
        "auroc": "auroc",
        "average_precision": "average_precision",
    }

    paired_rows = []
    inference_rows = []

    for metric_name, column in metrics.items():
        pivot = fold_results.pivot(
            index="outer_fold",
            columns="model",
            values=column,
        ).sort_index()

        differences = (
            pivot["LightGBM"]
            - pivot["LogisticRegression"]
        )

        for fold, difference in differences.items():
            paired_rows.append({
                "metric": metric_name,
                "outer_fold": int(fold),
                "logistic_regression": pivot.loc[
                    fold,
                    "LogisticRegression",
                ],
                "lightgbm": pivot.loc[
                    fold,
                    "LightGBM",
                ],
                "difference_lightgbm_minus_logistic":
                    difference,
            })

        number_of_folds = len(differences)
        mean_difference = differences.mean()
        standard_deviation = differences.std(ddof=1)

        naive_standard_error = (
            standard_deviation
            / np.sqrt(number_of_folds)
        )

        naive_lower, naive_upper = confidence_interval(
            mean_difference,
            naive_standard_error,
            number_of_folds - 1,
        )

        # Nadeau-Bengio-style correction for one 5-fold CV:
        # test/train ratio = 1/(k - 1) = 1/4.
        test_train_ratio = 1.0 / (
            number_of_folds - 1
        )

        corrected_standard_error = (
            standard_deviation
            * np.sqrt(
                1.0 / number_of_folds
                + test_train_ratio
            )
        )

        corrected_lower, corrected_upper = (
            confidence_interval(
                mean_difference,
                corrected_standard_error,
                number_of_folds - 1,
            )
        )

        lightgbm_wins = int(
            np.sum(differences > 0)
        )

        sign_test = binomtest(
            lightgbm_wins,
            n=number_of_folds,
            p=0.5,
            alternative="two-sided",
        )

        inference_rows.append({
            "metric": metric_name,
            "number_of_outer_folds": number_of_folds,
            "lightgbm_wins": lightgbm_wins,
            "mean_paired_difference":
                mean_difference,
            "paired_difference_std":
                standard_deviation,
            "naive_ci_lower": naive_lower,
            "naive_ci_upper": naive_upper,
            "corrected_ci_lower": corrected_lower,
            "corrected_ci_upper": corrected_upper,
            "exact_sign_test_p_value":
                sign_test.pvalue,
        })

    return (
        pd.DataFrame(paired_rows),
        pd.DataFrame(inference_rows),
    )


def stratified_bootstrap(oof):
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    labels = oof["label"].to_numpy()
    logistic = oof[
        "logistic_probability"
    ].to_numpy()
    lightgbm = oof[
        "lightgbm_probability"
    ].to_numpy()

    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)

    records = []

    for bootstrap_index in range(N_BOOTSTRAP):
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        sampled_indices = np.concatenate([
            sampled_negative,
            sampled_positive,
        ])

        sampled_labels = labels[sampled_indices]
        sampled_logistic = logistic[sampled_indices]
        sampled_lightgbm = lightgbm[sampled_indices]

        logistic_auroc = roc_auc_score(
            sampled_labels,
            sampled_logistic,
        )
        lightgbm_auroc = roc_auc_score(
            sampled_labels,
            sampled_lightgbm,
        )

        logistic_ap = average_precision_score(
            sampled_labels,
            sampled_logistic,
        )
        lightgbm_ap = average_precision_score(
            sampled_labels,
            sampled_lightgbm,
        )

        records.append({
            "bootstrap_index": bootstrap_index,
            "auroc_difference":
                lightgbm_auroc - logistic_auroc,
            "average_precision_difference":
                lightgbm_ap - logistic_ap,
        })

    distribution = pd.DataFrame(records)
    summary_rows = []

    for metric_column in [
        "auroc_difference",
        "average_precision_difference",
    ]:
        values = distribution[
            metric_column
        ].to_numpy()

        summary_rows.append({
            "metric": metric_column,
            "bootstrap_iterations": N_BOOTSTRAP,
            "mean_difference": np.mean(values),
            "median_difference": np.median(values),
            "ci_lower_2_5_percent":
                np.quantile(values, 0.025),
            "ci_upper_97_5_percent":
                np.quantile(values, 0.975),
            "proportion_greater_than_zero":
                np.mean(values > 0),
        })

    return distribution, pd.DataFrame(summary_rows)


def main():
    fold_results = pd.read_csv(FOLD_PATH)
    oof = pd.read_csv(OOF_PATH)

    paired, inference = paired_fold_analysis(
        fold_results
    )

    bootstrap_distribution, bootstrap_summary = (
        stratified_bootstrap(oof)
    )

    paired.to_csv(PAIRED_PATH, index=False)
    inference.to_csv(INFERENCE_PATH, index=False)
    bootstrap_summary.to_csv(
        BOOTSTRAP_PATH,
        index=False,
    )
    bootstrap_distribution.to_csv(
        BOOTSTRAP_DISTRIBUTION_PATH,
        index=False,
    )

    print("========== Five-fold paired differences ==========")
    print(paired.to_string(index=False))

    print("\n========== Paired-inference summary ==========")
    print(inference.to_string(index=False))

    print("\n========== OOF Bootstrap ==========")
    print(bootstrap_summary.to_string(index=False))

    print("\nInterpretation notes:")
    print(
        "1. Corrected confidence intervals account for dependence across cross-validation folds."
    )
    print(
        "2. The exact sign test has only five paired observations and therefore limited power."
    )
    print(
        "3. The OOF Bootstrap treats samples as independent and is descriptive only; "
        "it cannot replace external validation."
    )

    print("\nOutput files:")
    print(PAIRED_PATH)
    print(INFERENCE_PATH)
    print(BOOTSTRAP_PATH)
    print(BOOTSTRAP_DISTRIBUTION_PATH)


if __name__ == "__main__":
    main()
