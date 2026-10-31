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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_OUTPUT = OUTPUT_DIR / "feature_ablation_folds.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "feature_ablation_summary.csv"
OOF_OUTPUT = OUTPUT_DIR / "feature_ablation_oof.csv"

CHANNEL_NAMES = [
    "sum",
    "max",
    "mean",
    "std",
    "min",
    "first",
    "last",
    "slope",
    "nonzero",
]

COMBINATIONS = {
    "sum": ["sum"],
    "mean": ["mean"],
    "sum_mean": ["sum", "mean"],
    "sum_mean_max_min": ["sum", "mean", "max", "min"],
    "all_9_channels": CHANNEL_NAMES,
}

N_FOLDS = 5
SEED = 42
NUM_BOOST_ROUND = 8000
EARLY_STOPPING_ROUNDS = 100

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
    "random_state": SEED,
    "n_jobs": -1,
}


def selected_columns(channels, features_per_channel):
    """Return column indices for selected channels in the full nine-channel matrix."""
    column_groups = []

    for channel in channels:
        channel_index = CHANNEL_NAMES.index(channel)
        start = channel_index * features_per_channel
        end = start + features_per_channel
        column_groups.append(np.arange(start, end, dtype=np.int64))

    return np.concatenate(column_groups)


def prepare_combination(X_full, channels, features_per_channel):
    """
    Extract a feature combination.

    The nine-channel combination returns the memory map directly;
    other combinations copy only the required columns.
    """
    if channels == CHANNEL_NAMES:
        return X_full

    columns = selected_columns(channels, features_per_channel)
    X_selected = np.asarray(X_full[:, columns], dtype=np.float32)

    del columns
    gc.collect()

    return X_selected


def save_current_results(fold_records, summary_records, oof_table):
    """Save after each combination to avoid losing results if execution fails."""
    pd.DataFrame(fold_records).to_csv(FOLD_OUTPUT, index=False)
    pd.DataFrame(summary_records).to_csv(SUMMARY_OUTPUT, index=False)
    oof_table.to_csv(OOF_OUTPUT, index=False)


def main():
    print("Loading the nine-channel feature cache (memory-mapped)...", flush=True)
    X_full = np.load(FEATURE_PATH, mmap_mode="r")

    metadata = pd.read_csv(LABEL_PATH)
    y = metadata["ARF_LABEL"].to_numpy(dtype=np.int32)
    ids = metadata["ID"].to_numpy()

    if X_full.shape[0] != len(y):
        raise ValueError(
            f"Feature rows {X_full.shape[0]} do not match label count {len(y)}"
        )

    if X_full.shape[1] % len(CHANNEL_NAMES) != 0:
        raise ValueError("The full feature-column count is not divisible by nine channels.")

    features_per_channel = X_full.shape[1] // len(CHANNEL_NAMES)

    print("Full-feature shape:", X_full.shape, flush=True)
    print("Data type:", X_full.dtype, flush=True)
    print("Features per channel:", features_per_channel, flush=True)
    print("Positive samples:", int(y.sum()), flush=True)
    print("Negative samples:", int((y == 0).sum()), flush=True)

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    # Fix and save folds so every feature combination uses identical validation samples.
    folds = list(splitter.split(np.zeros(len(y)), y))

    fold_records = []
    summary_records = []

    oof_table = pd.DataFrame({
        "ID": ids,
        "ARF_LABEL": y,
    })

    for combination_name, channels in COMBINATIONS.items():
        print("\n" + "=" * 70, flush=True)
        print(
            f"Starting combination: {combination_name} -> {channels}",
            flush=True,
        )
        print("=" * 70, flush=True)

        combination_start = time.perf_counter()

        X = prepare_combination(
            X_full,
            channels,
            features_per_channel,
        )

        print("Current feature-matrix shape:", X.shape, flush=True)
        print(
            "Theoretical size of current matrix:",
            round(X.shape[0] * X.shape[1] * 4 / 1024**3, 3),
            "GiB",
            flush=True,
        )

        oof_predictions = np.zeros(len(y), dtype=np.float64)
        fold_aurocs = []
        fold_aps = []

        for fold_number, (train_index, valid_index) in enumerate(
            folds, start=1
        ):
            print(
                f"\n--- {combination_name}: Fold "
                f"{fold_number}/{N_FOLDS} ---",
                flush=True,
            )

            fold_start = time.perf_counter()

            y_train = y[train_index]
            y_valid = y[valid_index]

            positive_count = int(y_train.sum())
            negative_count = int((y_train == 0).sum())
            scale_pos_weight = negative_count / max(1, positive_count)

            params = dict(BASE_PARAMS)
            params["scale_pos_weight"] = scale_pos_weight
            params["seed"] = SEED + fold_number
            params["bagging_seed"] = SEED + fold_number
            params["feature_fraction_seed"] = SEED + fold_number

            # Copy only the training and validation portions for each fold.
            X_train = np.asarray(X[train_index], dtype=np.float32)
            X_valid = np.asarray(X[valid_index], dtype=np.float32)

            train_dataset = lgb.Dataset(
                X_train,
                label=y_train,
                free_raw_data=True,
            )

            valid_dataset = lgb.Dataset(
                X_valid,
                label=y_valid,
                reference=train_dataset,
                free_raw_data=True,
            )

            model = lgb.train(
                params,
                train_dataset,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[valid_dataset],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(
                        EARLY_STOPPING_ROUNDS,
                        verbose=False,
                    )
                ],
            )

            valid_predictions = model.predict(
                X_valid,
                num_iteration=model.best_iteration,
            )

            oof_predictions[valid_index] = valid_predictions

            fold_auroc = roc_auc_score(
                y_valid,
                valid_predictions,
            )

            fold_ap = average_precision_score(
                y_valid,
                valid_predictions,
            )

            fold_seconds = time.perf_counter() - fold_start

            fold_aurocs.append(fold_auroc)
            fold_aps.append(fold_ap)

            fold_records.append({
                "combination": combination_name,
                "channels": "+".join(channels),
                "n_features": X.shape[1],
                "fold": fold_number - 1,
                "auroc": fold_auroc,
                "average_precision": fold_ap,
                "best_iteration": model.best_iteration,
                "runtime_seconds": fold_seconds,
            })

            print(
                f"AUROC={fold_auroc:.6f}, "
                f"AP={fold_ap:.6f}, "
                f"best_iteration={model.best_iteration}, "
                f"runtime={fold_seconds:.1f}s",
                flush=True,
            )

            del (
                X_train,
                X_valid,
                y_train,
                y_valid,
                train_dataset,
                valid_dataset,
                model,
                valid_predictions,
            )
            gc.collect()

        oof_auroc = roc_auc_score(y, oof_predictions)
        oof_ap = average_precision_score(y, oof_predictions)
        combination_seconds = time.perf_counter() - combination_start

        summary_records.append({
            "combination": combination_name,
            "channels": "+".join(channels),
            "n_features": X.shape[1],
            "fold_auroc_mean": np.mean(fold_aurocs),
            "fold_auroc_std": np.std(fold_aurocs, ddof=1),
            "oof_auroc": oof_auroc,
            "fold_ap_mean": np.mean(fold_aps),
            "fold_ap_std": np.std(fold_aps, ddof=1),
            "oof_average_precision": oof_ap,
            "total_runtime_seconds": combination_seconds,
        })

        oof_table[f"prob_{combination_name}"] = oof_predictions

        save_current_results(
            fold_records,
            summary_records,
            oof_table,
        )

        print(
            f"\nCompleted combination: {combination_name}",
            flush=True,
        )
        print(
            f"OOF AUROC={oof_auroc:.6f}",
            flush=True,
        )
        print(
            f"OOF AP={oof_ap:.6f}",
            flush=True,
        )
        print(
            f"Results saved to: {SUMMARY_OUTPUT}",
            flush=True,
        )

        del X, oof_predictions
        gc.collect()

    print("\n" + "=" * 70, flush=True)
    print("Experiment 005 final results", flush=True)
    print("=" * 70, flush=True)

    summary = pd.DataFrame(summary_records)
    print(summary.to_string(index=False), flush=True)

    print("\nSaved:", flush=True)
    print(FOLD_OUTPUT, flush=True)
    print(SUMMARY_OUTPUT, flush=True)
    print(OOF_OUTPUT, flush=True)


if __name__ == "__main__":
    main()
