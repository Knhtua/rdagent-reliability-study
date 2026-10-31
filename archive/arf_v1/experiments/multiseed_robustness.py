from pathlib import Path
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT
    / "learning_records/arf_research/cache/train_features_9ch.npy"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "git_ignore_folder/ds_data/"
    / "arf-12-hours-prediction-task/train/ARF_12h.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "learning_records/arf_research"
RESULT_PATH = OUTPUT_DIR / "multiseed_results.csv"
FOLD_PATH = OUTPUT_DIR / "multiseed_folds.csv"
SUMMARY_PATH = OUTPUT_DIR / "multiseed_summary.csv"
DIFFERENCE_PATH = OUTPUT_DIR / "multiseed_paired_differences.csv"

SEEDS = [7, 21, 42, 77, 2026]
N_FOLDS = 5
N_CHANNELS = 9
NUM_BOOST_ROUND = 8000
EARLY_STOPPING = 100

BASE_PARAMS = {
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
    "verbose": -1,
    "n_jobs": -1,
}


def save_results(results, folds):
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False)
    pd.DataFrame(folds).to_csv(FOLD_PATH, index=False)


def run_one_configuration(X, y, feature_name, seed, fold_records):
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    oof = np.zeros(len(y), dtype=np.float64)
    fold_aurocs = []
    fold_aps = []
    start_time = time.perf_counter()

    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(y)), y),
        start=1,
    ):
        fold_start = time.perf_counter()

        X_train = np.asarray(X[train_index], dtype=np.float32)
        X_valid = np.asarray(X[valid_index], dtype=np.float32)
        y_train = y[train_index]
        y_valid = y[valid_index]

        positive_count = int(y_train.sum())
        negative_count = int((y_train == 0).sum())

        params = dict(BASE_PARAMS)
        params.update({
            "scale_pos_weight": negative_count / max(1, positive_count),
            "random_state": seed,
            "seed": seed + fold_number,
            "bagging_seed": seed + fold_number,
            "feature_fraction_seed": seed + fold_number,
        })

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            free_raw_data=True,
        )

        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_data,
            free_raw_data=True,
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_data],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(
                    EARLY_STOPPING,
                    verbose=False,
                )
            ],
        )

        probability = model.predict(
            X_valid,
            num_iteration=model.best_iteration,
        )

        oof[valid_index] = probability

        auroc = roc_auc_score(y_valid, probability)
        ap = average_precision_score(y_valid, probability)
        runtime = time.perf_counter() - fold_start

        fold_aurocs.append(auroc)
        fold_aps.append(ap)

        fold_records.append({
            "seed": seed,
            "feature_set": feature_name,
            "fold": fold_number - 1,
            "auroc": auroc,
            "average_precision": ap,
            "best_iteration": model.best_iteration,
            "runtime_seconds": runtime,
        })

        print(
            f"seed={seed}, feature={feature_name}, "
            f"fold={fold_number}/{N_FOLDS}, "
            f"AUROC={auroc:.6f}, AP={ap:.6f}, "
            f"best_iteration={model.best_iteration}, "
            f"time={runtime:.1f}s",
            flush=True,
        )

        del (
            X_train,
            X_valid,
            y_train,
            y_valid,
            train_data,
            valid_data,
            model,
            probability,
        )
        gc.collect()

    return {
        "seed": seed,
        "feature_set": feature_name,
        "n_features": X.shape[1],
        "fold_auroc_mean": np.mean(fold_aurocs),
        "fold_auroc_std": np.std(fold_aurocs, ddof=1),
        "oof_auroc": roc_auc_score(y, oof),
        "fold_ap_mean": np.mean(fold_aps),
        "fold_ap_std": np.std(fold_aps, ddof=1),
        "oof_average_precision": average_precision_score(y, oof),
        "runtime_seconds": time.perf_counter() - start_time,
    }


def main():
    X_all = np.load(FEATURE_PATH, mmap_mode="r")
    metadata = pd.read_csv(LABEL_PATH)
    y = metadata["ARF_LABEL"].to_numpy(dtype=np.int32)

    if X_all.shape[0] != len(y):
        raise ValueError("Feature-row count does not match label count.")

    if X_all.shape[1] % N_CHANNELS != 0:
        raise ValueError("Feature-column count is not divisible by nine.")

    features_per_channel = X_all.shape[1] // N_CHANNELS
    X_sum = X_all[:, :features_per_channel]

    print("Full nine-channel shape:", X_all.shape, flush=True)
    print("Sum-feature shape:", X_sum.shape, flush=True)
    print("Random seeds:", SEEDS, flush=True)

    results = []
    fold_records = []

    for seed in SEEDS:
        print("\n" + "=" * 70, flush=True)
        print(f"Starting random seed {seed}", flush=True)
        print("=" * 70, flush=True)

        # Under the same random seed, both settings use identical five-fold splits and model seeds.
        for feature_name, X in [
            ("sum", X_sum),
            ("all_9_channels", X_all),
        ]:
            print(
                f"\nStarting seed={seed}, feature={feature_name}",
                flush=True,
            )

            result = run_one_configuration(
                X,
                y,
                feature_name,
                seed,
                fold_records,
            )

            results.append(result)
            save_results(results, fold_records)

            print(
                f"Completed seed={seed}, feature={feature_name}: "
                f"OOF AUROC={result['oof_auroc']:.6f}, "
                f"OOF AP={result['oof_average_precision']:.6f}",
                flush=True,
            )

            gc.collect()

    results_df = pd.DataFrame(results)

    summary = (
        results_df
        .groupby("feature_set")
        .agg(
            seed_count=("seed", "count"),
            oof_auroc_mean=("oof_auroc", "mean"),
            oof_auroc_std=("oof_auroc", "std"),
            oof_auroc_min=("oof_auroc", "min"),
            oof_auroc_max=("oof_auroc", "max"),
            oof_ap_mean=("oof_average_precision", "mean"),
            oof_ap_std=("oof_average_precision", "std"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )

    pivot_auc = results_df.pivot(
        index="seed",
        columns="feature_set",
        values="oof_auroc",
    )

    pivot_ap = results_df.pivot(
        index="seed",
        columns="feature_set",
        values="oof_average_precision",
    )

    differences = pd.DataFrame({
        "seed": pivot_auc.index,
        "sum_oof_auroc": pivot_auc["sum"],
        "all_9_oof_auroc": pivot_auc["all_9_channels"],
        "auroc_difference_sum_minus_all9": (
            pivot_auc["sum"] - pivot_auc["all_9_channels"]
        ),
        "sum_oof_ap": pivot_ap["sum"],
        "all_9_oof_ap": pivot_ap["all_9_channels"],
        "ap_difference_sum_minus_all9": (
            pivot_ap["sum"] - pivot_ap["all_9_channels"]
        ),
    }).reset_index(drop=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    differences.to_csv(DIFFERENCE_PATH, index=False)

    print("\n" + "=" * 70, flush=True)
    print("Cross-seed summary", flush=True)
    print("=" * 70, flush=True)
    print(summary.to_string(index=False), flush=True)

    print("\nPaired differences (sum - all_9_channels)", flush=True)
    print(differences.to_string(index=False), flush=True)

    print(
        "\nMean AUROC difference:",
        differences["auroc_difference_sum_minus_all9"].mean(),
        flush=True,
    )

    print(
        "Seeds won by sum features:",
        int(
            (
                differences["auroc_difference_sum_minus_all9"] > 0
            ).sum()
        ),
        "/",
        len(SEEDS),
        flush=True,
    )

    print("\nOutput files:", flush=True)
    print(RESULT_PATH, flush=True)
    print(FOLD_PATH, flush=True)
    print(SUMMARY_PATH, flush=True)
    print(DIFFERENCE_PATH, flush=True)


if __name__ == "__main__":
    main()
