from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler


SEED = 42
N_FOLDS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    PROJECT_ROOT
    / "git_ignore_folder"
    / "ds_data"
    / "arf-12-hours-prediction-task"
    / "train"
)
OUTPUT_DIR = PROJECT_ROOT / "learning_records" / "arf_research"

NPZ_PATH = DATA_DIR / "X.npz"
LABEL_PATH = DATA_DIR / "ARF_12h.csv"

FOLD_RESULTS_PATH = OUTPUT_DIR / "fair_comparison_folds.csv"
SUMMARY_PATH = OUTPUT_DIR / "fair_comparison_summary.csv"
OOF_PATH = OUTPUT_DIR / "fair_comparison_oof.csv"


def load_sum_features():
    """Sum a 12-hour sparse tensor over time to obtain an (N, D) sparse matrix."""
    with np.load(NPZ_PATH) as z:
        tensor_shape = tuple(int(v) for v in z["shape"])
        coords = z["coords"]
        values = z["data"].astype(np.float32)
        fill_value = z["fill_value"].item()

    if fill_value != 0:
        raise ValueError("This script requires the sparse tensor fill value to be 0.")

    n_samples, n_hours, n_features = tensor_shape

    sample_index = coords[0]
    feature_index = coords[2]

    X = coo_matrix(
        (values, (sample_index, feature_index)),
        shape=(n_samples, n_features),
        dtype=np.float32,
    ).tocsr()

    labels = pd.read_csv(LABEL_PATH)
    y = labels["ARF_LABEL"].to_numpy(dtype=np.int64)
    ids = labels["ID"].to_numpy()

    if len(y) != X.shape[0]:
        raise ValueError("The label count does not match the feature-matrix sample count.")

    print("Original tensor shape:", tensor_shape)
    print("Sum-feature matrix:", X.shape)
    print("Nonzero elements:", X.nnz)
    print("Positive samples:", int(y.sum()))
    print("Negative samples:", int((y == 0).sum()))

    return X, y, ids


def evaluate_predictions(y_true, probability):
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(
            average_precision_score(y_true, probability)
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, ids = load_sum_features()

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    logistic_oof = np.zeros(len(y), dtype=np.float32)
    lightgbm_oof = np.zeros(len(y), dtype=np.float32)

    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        print(f"\n========== Fold {fold + 1}/{N_FOLDS} ==========")

        X_train = X[train_idx]
        X_valid = X[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # Logistic regression
        logistic_start = perf_counter()

        logistic_model = make_pipeline(
            MaxAbsScaler(),
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=500,
                random_state=SEED,
            ),
        )

        logistic_model.fit(X_train, y_train)
        logistic_probability = logistic_model.predict_proba(X_valid)[:, 1]
        logistic_oof[valid_idx] = logistic_probability.astype(np.float32)

        logistic_seconds = perf_counter() - logistic_start
        logistic_metrics = evaluate_predictions(
            y_valid, logistic_probability
        )

        fold_rows.append(
            {
                "model": "LogisticRegression",
                "features": "12h_sum",
                "fold": fold,
                "auroc": logistic_metrics["auroc"],
                "average_precision": logistic_metrics[
                    "average_precision"
                ],
                "runtime_seconds": logistic_seconds,
            }
        )

        print(
            "Logistic regression:",
            f"AUROC={logistic_metrics['auroc']:.6f},",
            f"AP={logistic_metrics['average_precision']:.6f},",
            f"runtime={logistic_seconds:.1f}s",
        )

        # LightGBM
        lightgbm_start = perf_counter()

        positive_count = int(y_train.sum())
        negative_count = int(len(y_train) - positive_count)
        scale_pos_weight = negative_count / max(1, positive_count)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.01,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "verbosity": -1,
            "seed": SEED,
            "feature_fraction_seed": SEED,
            "bagging_seed": SEED,
            "data_random_seed": SEED,
            "num_threads": -1,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_data,
        )

        lightgbm_model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=8000,
            valid_sets=[valid_data],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
            ],
        )

        lightgbm_probability = lightgbm_model.predict(
            X_valid,
            num_iteration=lightgbm_model.best_iteration,
        )
        lightgbm_oof[valid_idx] = lightgbm_probability.astype(np.float32)

        lightgbm_seconds = perf_counter() - lightgbm_start
        lightgbm_metrics = evaluate_predictions(
            y_valid, lightgbm_probability
        )

        fold_rows.append(
            {
                "model": "LightGBM",
                "features": "12h_sum",
                "fold": fold,
                "auroc": lightgbm_metrics["auroc"],
                "average_precision": lightgbm_metrics[
                    "average_precision"
                ],
                "runtime_seconds": lightgbm_seconds,
            }
        )

        print(
            "LightGBM:",
            f"AUROC={lightgbm_metrics['auroc']:.6f},",
            f"AP={lightgbm_metrics['average_precision']:.6f},",
            f"best_iteration={lightgbm_model.best_iteration},",
            f"runtime={lightgbm_seconds:.1f}s",
        )

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(FOLD_RESULTS_PATH, index=False)

    summary_rows = []

    for model_name, oof_probability in [
        ("LogisticRegression", logistic_oof),
        ("LightGBM", lightgbm_oof),
    ]:
        model_folds = fold_results[
            fold_results["model"] == model_name
        ]

        oof_metrics = evaluate_predictions(y, oof_probability)

        summary_rows.append(
            {
                "model": model_name,
                "features": "12h_sum",
                "fold_auroc_mean": model_folds["auroc"].mean(),
                "fold_auroc_std": model_folds["auroc"].std(ddof=1),
                "oof_auroc": oof_metrics["auroc"],
                "fold_ap_mean": model_folds[
                    "average_precision"
                ].mean(),
                "fold_ap_std": model_folds[
                    "average_precision"
                ].std(ddof=1),
                "oof_average_precision": oof_metrics[
                    "average_precision"
                ],
                "total_runtime_seconds": model_folds[
                    "runtime_seconds"
                ].sum(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    oof_output = pd.DataFrame(
        {
            "ID": ids,
            "ARF_LABEL": y,
            "logistic_oof_probability": logistic_oof,
            "lightgbm_oof_probability": lightgbm_oof,
        }
    )
    oof_output.to_csv(OOF_PATH, index=False)

    print("\n========== Final summary ==========")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(FOLD_RESULTS_PATH)
    print(SUMMARY_PATH)
    print(OOF_PATH)


if __name__ == "__main__":
    main()
