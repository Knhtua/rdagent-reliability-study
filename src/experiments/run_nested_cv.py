import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.data.arf_loader import (
    build_sum_features,
    load_arf_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "arf_nested_cv.json"
RESULT_DIR = PROJECT_ROOT / "results" / "tables"

FOLD_RESULT_PATH = RESULT_DIR / "nested_cv_fold_results.csv"
SUMMARY_PATH = RESULT_DIR / "nested_cv_summary.csv"
OOF_PATH = RESULT_DIR / "nested_cv_oof_predictions.csv"
PARAMETER_PATH = RESULT_DIR / "nested_cv_selected_parameters.csv"


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def calculate_metrics(y_true, probabilities):
    return {
        "auroc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


def create_inner_splits(y, config):
    splitter = StratifiedKFold(
        n_splits=config["inner_cv"]["n_splits"],
        shuffle=config["inner_cv"]["shuffle"],
        random_state=config["inner_cv"]["seed"],
    )

    return list(
        splitter.split(
            np.zeros(len(y)),
            y,
        )
    )


def tune_logistic_regression(
    X,
    y,
    inner_splits,
    config,
):
    settings = config["logistic_regression"]
    candidate_results = []

    for c_value in settings["c_grid"]:
        fold_scores = []

        for train_index, valid_index in inner_splits:
            model = LogisticRegression(
                C=c_value,
                solver=settings["solver"],
                class_weight=settings["class_weight"],
                max_iter=settings["max_iter"],
                random_state=config["random_seed"],
            )

            model.fit(
                X[train_index],
                y[train_index],
            )

            probabilities = model.predict_proba(
                X[valid_index]
            )[:, 1]

            score = roc_auc_score(
                y[valid_index],
                probabilities,
            )
            fold_scores.append(score)

        candidate_results.append({
            "C": c_value,
            "mean_inner_auroc": float(
                np.mean(fold_scores)
            ),
            "std_inner_auroc": float(
                np.std(fold_scores, ddof=1)
            ),
        })

    best = max(
        candidate_results,
        key=lambda row: row["mean_inner_auroc"],
    )

    return best, candidate_results


def train_logistic_outer(
    X_train,
    y_train,
    X_valid,
    best_parameters,
    config,
):
    settings = config["logistic_regression"]

    model = LogisticRegression(
        C=best_parameters["C"],
        solver=settings["solver"],
        class_weight=settings["class_weight"],
        max_iter=settings["max_iter"],
        random_state=config["random_seed"],
    )

    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]


def make_lightgbm_parameters(
    candidate,
    y_train,
    config,
    seed,
):
    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))

    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": config["lightgbm"][
            "learning_rate"
        ],
        "num_leaves": candidate["num_leaves"],
        "min_data_in_leaf": candidate[
            "min_data_in_leaf"
        ],
        "feature_fraction": candidate[
            "feature_fraction"
        ],
        "bagging_fraction": candidate[
            "bagging_fraction"
        ],
        "bagging_freq": 1,
        "scale_pos_weight": (
            negative_count / positive_count
        ),
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_threads": config["lightgbm"][
            "num_threads"
        ],
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "force_col_wise": True,
        "verbose": -1,
    }


def tune_lightgbm(
    X,
    y,
    inner_splits,
    config,
    outer_fold,
):
    settings = config["lightgbm"]
    candidate_results = []

    for candidate_index, candidate in enumerate(
        settings["parameter_grid"]
    ):
        fold_scores = []
        best_iterations = []

        for inner_fold, (
            train_index,
            valid_index,
        ) in enumerate(inner_splits):
            seed = (
                config["random_seed"]
                + outer_fold * 100
                + candidate_index * 10
                + inner_fold
            )

            parameters = make_lightgbm_parameters(
                candidate,
                y[train_index],
                config,
                seed,
            )

            train_data = lgb.Dataset(
                X[train_index],
                label=y[train_index],
                free_raw_data=True,
            )

            valid_data = lgb.Dataset(
                X[valid_index],
                label=y[valid_index],
                reference=train_data,
                free_raw_data=True,
            )

            model = lgb.train(
                parameters,
                train_data,
                num_boost_round=settings[
                    "num_boost_round"
                ],
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(
                        settings[
                            "early_stopping_rounds"
                        ],
                        verbose=False,
                    )
                ],
            )

            probabilities = model.predict(
                X[valid_index],
                num_iteration=model.best_iteration,
            )

            fold_scores.append(
                roc_auc_score(
                    y[valid_index],
                    probabilities,
                )
            )
            best_iterations.append(
                int(model.best_iteration)
            )

        candidate_results.append({
            **candidate,
            "mean_inner_auroc": float(
                np.mean(fold_scores)
            ),
            "std_inner_auroc": float(
                np.std(fold_scores, ddof=1)
            ),
            "final_num_boost_round": int(
                np.median(best_iterations)
            ),
        })

    best = max(
        candidate_results,
        key=lambda row: row["mean_inner_auroc"],
    )

    return best, candidate_results


def train_lightgbm_outer(
    X_train,
    y_train,
    X_valid,
    best_parameters,
    config,
    outer_fold,
):
    parameters = make_lightgbm_parameters(
        best_parameters,
        y_train,
        config,
        seed=config["random_seed"] + outer_fold,
    )

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        free_raw_data=True,
    )

    model = lgb.train(
        parameters,
        train_data,
        num_boost_round=best_parameters[
            "final_num_boost_round"
        ],
    )

    return model.predict(X_valid)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    dataset = load_arf_dataset()

    print("Constructing 12-hour sum features...")
    X = build_sum_features(dataset)
    y = dataset.metadata["ARF_LABEL"].to_numpy(
        dtype=np.int32
    )
    outer_folds = dataset.splits[
        "outer_fold"
    ].to_numpy(dtype=np.int32)

    sample_count = len(y)

    logistic_oof = np.full(
        sample_count,
        np.nan,
        dtype=np.float64,
    )
    lightgbm_oof = np.full(
        sample_count,
        np.nan,
        dtype=np.float64,
    )

    fold_records = []
    parameter_records = []

    for outer_fold in range(5):
        print(
            f"\n========== Outer Fold "
            f"{outer_fold + 1}/5 =========="
        )

        outer_start = time.perf_counter()

        valid_index = np.flatnonzero(
            outer_folds == outer_fold
        )
        train_index = np.flatnonzero(
            outer_folds != outer_fold
        )

        X_train = X[train_index]
        X_valid = X[valid_index]
        y_train = y[train_index]
        y_valid = y[valid_index]

        inner_splits = create_inner_splits(
            y_train,
            config,
        )

        print("Selecting logistic-regression parameters...")
        best_logistic, logistic_candidates = (
            tune_logistic_regression(
                X_train,
                y_train,
                inner_splits,
                config,
            )
        )

        logistic_probabilities = train_logistic_outer(
            X_train,
            y_train,
            X_valid,
            best_logistic,
            config,
        )
        logistic_oof[valid_index] = (
            logistic_probabilities
        )

        logistic_metrics = calculate_metrics(
            y_valid,
            logistic_probabilities,
        )

        fold_records.append({
            "model": "LogisticRegression",
            "outer_fold": outer_fold,
            **logistic_metrics,
            "selected_C": best_logistic["C"],
            "selected_num_boost_round": np.nan,
        })

        parameter_records.append({
            "model": "LogisticRegression",
            "outer_fold": outer_fold,
            "selected_parameters": json.dumps(
                best_logistic,
                sort_keys=True,
            ),
            "all_candidates": json.dumps(
                logistic_candidates,
                sort_keys=True,
            ),
        })

        print(
            "Logistic regression: "
            f"AUROC={logistic_metrics['auroc']:.6f}, "
            f"AP={logistic_metrics['average_precision']:.6f}, "
            f"C={best_logistic['C']}"
        )

        print("Selecting LightGBM parameters...")
        best_lightgbm, lightgbm_candidates = (
            tune_lightgbm(
                X_train,
                y_train,
                inner_splits,
                config,
                outer_fold,
            )
        )

        lightgbm_probabilities = train_lightgbm_outer(
            X_train,
            y_train,
            X_valid,
            best_lightgbm,
            config,
            outer_fold,
        )
        lightgbm_oof[valid_index] = (
            lightgbm_probabilities
        )

        lightgbm_metrics = calculate_metrics(
            y_valid,
            lightgbm_probabilities,
        )

        fold_records.append({
            "model": "LightGBM",
            "outer_fold": outer_fold,
            **lightgbm_metrics,
            "selected_C": np.nan,
            "selected_num_boost_round":
                best_lightgbm[
                    "final_num_boost_round"
                ],
        })

        parameter_records.append({
            "model": "LightGBM",
            "outer_fold": outer_fold,
            "selected_parameters": json.dumps(
                best_lightgbm,
                sort_keys=True,
            ),
            "all_candidates": json.dumps(
                lightgbm_candidates,
                sort_keys=True,
            ),
        })

        elapsed = time.perf_counter() - outer_start

        print(
            "LightGBM："
            f"AUROC={lightgbm_metrics['auroc']:.6f}, "
            f"AP={lightgbm_metrics['average_precision']:.6f}, "
            f"rounds={best_lightgbm['final_num_boost_round']}"
        )
        print(f"Outer-fold runtime: {elapsed:.1f} seconds")

        pd.DataFrame(fold_records).to_csv(
            FOLD_RESULT_PATH,
            index=False,
        )
        pd.DataFrame(parameter_records).to_csv(
            PARAMETER_PATH,
            index=False,
        )

    if np.isnan(logistic_oof).any():
        raise RuntimeError("Logistic-regression OOF predictions are incomplete.")

    if np.isnan(lightgbm_oof).any():
        raise RuntimeError("LightGBM OOF predictions are incomplete.")

    oof_frame = pd.DataFrame({
        "sample_index": dataset.splits[
            "sample_index"
        ],
        "ID": dataset.metadata["ID"],
        "label": y,
        "outer_fold": outer_folds,
        "logistic_probability": logistic_oof,
        "lightgbm_probability": lightgbm_oof,
    })
    oof_frame.to_csv(OOF_PATH, index=False)

    fold_frame = pd.DataFrame(fold_records)
    summary_records = []

    for model_name, probabilities in [
        ("LogisticRegression", logistic_oof),
        ("LightGBM", lightgbm_oof),
    ]:
        model_folds = fold_frame[
            fold_frame["model"] == model_name
        ]

        oof_metrics = calculate_metrics(
            y,
            probabilities,
        )

        summary_records.append({
            "model": model_name,
            "fold_auroc_mean": model_folds[
                "auroc"
            ].mean(),
            "fold_auroc_std": model_folds[
                "auroc"
            ].std(ddof=1),
            "oof_auroc": oof_metrics["auroc"],
            "fold_ap_mean": model_folds[
                "average_precision"
            ].mean(),
            "fold_ap_std": model_folds[
                "average_precision"
            ].std(ddof=1),
            "oof_average_precision":
                oof_metrics["average_precision"],
            "oof_brier_score":
                oof_metrics["brier_score"],
        })

    summary_frame = pd.DataFrame(summary_records)
    summary_frame.to_csv(SUMMARY_PATH, index=False)

    print("\n========== Final summary ==========")
    print(summary_frame.to_string(index=False))

    print("\nOutput files:")
    print(FOLD_RESULT_PATH)
    print(SUMMARY_PATH)
    print(OOF_PATH)
    print(PARAMETER_PATH)


if __name__ == "__main__":
    main()
