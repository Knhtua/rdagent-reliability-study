from pathlib import Path
import gc
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

PER_FOLD_PATH = OUTPUT_DIR / "feature_importance_per_fold.csv"
SUMMARY_PATH = OUTPUT_DIR / "feature_importance_summary.csv"
TOP30_PATH = OUTPUT_DIR / "feature_importance_top30.csv"
METRICS_PATH = OUTPUT_DIR / "feature_importance_fold_metrics.csv"
STABILITY_PATH = OUTPUT_DIR / "feature_importance_stability.csv"
OOF_PATH = OUTPUT_DIR / "feature_importance_oof.csv"
PLOT_PATH = OUTPUT_DIR / "feature_importance_top30.png"

SEED = 42
N_FOLDS = 5
N_CHANNELS = 9
TOP_K = 30
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


print("Loading feature cache...")

X_all = np.load(FEATURE_PATH, mmap_mode="r")
y_frame = pd.read_csv(LABEL_PATH)
y = y_frame["ARF_LABEL"].to_numpy(dtype=np.int32)
ids = y_frame["ID"].to_numpy()

if X_all.shape[0] != len(y):
    raise ValueError(
        f"Feature and label counts differ: {X_all.shape[0]} != {len(y)}"
    )

if X_all.shape[1] % N_CHANNELS != 0:
    raise ValueError(
        f"Full feature count {X_all.shape[1]} is not divisible by {N_CHANNELS}"
    )

feature_count = X_all.shape[1] // N_CHANNELS
X = X_all[:, :feature_count]

feature_names = np.array(
    [f"feature_{index}" for index in range(feature_count)]
)

print("Full nine-channel shape:", X_all.shape)
print("Sum-feature shape:", X.shape)
print("Label count:", len(y))
print("Positive cases:", int(y.sum()))
print("Negative cases:", int((y == 0).sum()))
print("Random seed:", SEED)

splitter = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

oof = np.zeros(len(y), dtype=np.float64)

gain_matrix = np.zeros(
    (N_FOLDS, feature_count),
    dtype=np.float64,
)

gain_share_matrix = np.zeros(
    (N_FOLDS, feature_count),
    dtype=np.float64,
)

split_matrix = np.zeros(
    (N_FOLDS, feature_count),
    dtype=np.float64,
)

fold_records = []
per_fold_records = []

experiment_start = time.perf_counter()

for fold_number, (train_index, valid_index) in enumerate(
    splitter.split(np.zeros(len(y)), y),
    start=1,
):
    fold_start = time.perf_counter()

    print()
    print(f"========== Fold {fold_number}/{N_FOLDS} ==========")

    X_train = np.asarray(
        X[train_index],
        dtype=np.float32,
    )
    X_valid = np.asarray(
        X[valid_index],
        dtype=np.float32,
    )

    y_train = y[train_index]
    y_valid = y[valid_index]

    positive_count = int(y_train.sum())
    negative_count = int((y_train == 0).sum())

    params = dict(BASE_PARAMS)
    params.update({
        "scale_pos_weight": (
            negative_count / max(1, positive_count)
        ),
        "random_state": SEED,
        "seed": SEED + fold_number,
        "bagging_seed": SEED + fold_number,
        "feature_fraction_seed": SEED + fold_number,
    })

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=list(feature_names),
        free_raw_data=True,
    )

    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=list(feature_names),
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

    fold_auroc = roc_auc_score(
        y_valid,
        probability,
    )
    fold_ap = average_precision_score(
        y_valid,
        probability,
    )

    gain = model.feature_importance(
        importance_type="gain",
        iteration=model.best_iteration,
    ).astype(np.float64)

    split = model.feature_importance(
        importance_type="split",
        iteration=model.best_iteration,
    ).astype(np.float64)

    gain_total = gain.sum()

    if gain_total > 0:
        gain_share = gain / gain_total
    else:
        gain_share = np.zeros_like(gain)

    gain_matrix[fold_number - 1] = gain
    gain_share_matrix[fold_number - 1] = gain_share
    split_matrix[fold_number - 1] = split

    fold_runtime = time.perf_counter() - fold_start

    fold_records.append({
        "fold": fold_number - 1,
        "auroc": fold_auroc,
        "average_precision": fold_ap,
        "best_iteration": model.best_iteration,
        "used_feature_count": int((gain > 0).sum()),
        "runtime_seconds": fold_runtime,
    })

    fold_frame = pd.DataFrame({
        "fold": fold_number - 1,
        "feature_index": np.arange(feature_count),
        "feature_name": feature_names,
        "gain": gain,
        "gain_share": gain_share,
        "split": split,
    })

    per_fold_records.append(fold_frame)

    print(
        f"AUROC={fold_auroc:.6f}, "
        f"AP={fold_ap:.6f}, "
        f"best_iteration={model.best_iteration}, "
        f"features used={int((gain > 0).sum())}, "
        f"runtime={fold_runtime:.1f}s"
    )

    del (
        X_train,
        X_valid,
        y_train,
        y_valid,
        train_data,
        valid_data,
        model,
    )
    gc.collect()


total_runtime = time.perf_counter() - experiment_start

fold_metrics = pd.DataFrame(fold_records)
per_fold_importance = pd.concat(
    per_fold_records,
    ignore_index=True,
)

mean_gain = gain_matrix.mean(axis=0)
std_gain = gain_matrix.std(axis=0, ddof=1)

mean_gain_share = gain_share_matrix.mean(axis=0)
std_gain_share = gain_share_matrix.std(axis=0, ddof=1)

mean_split = split_matrix.mean(axis=0)
std_split = split_matrix.std(axis=0, ddof=1)

folds_used = (gain_matrix > 0).sum(axis=0)

top30_count = np.zeros(feature_count, dtype=np.int32)

for fold_index in range(N_FOLDS):
    top_indices = np.argsort(
        gain_share_matrix[fold_index]
    )[-TOP_K:]

    top30_count[top_indices] += 1

summary = pd.DataFrame({
    "feature_index": np.arange(feature_count),
    "feature_name": feature_names,
    "mean_gain": mean_gain,
    "std_gain": std_gain,
    "mean_gain_share": mean_gain_share,
    "std_gain_share": std_gain_share,
    "mean_split": mean_split,
    "std_split": std_split,
    "folds_used": folds_used,
    "top30_fold_count": top30_count,
})

summary = summary.sort_values(
    [
        "mean_gain_share",
        "folds_used",
        "mean_split",
    ],
    ascending=[False, False, False],
).reset_index(drop=True)

summary["importance_rank"] = (
    np.arange(len(summary)) + 1
)

top30 = summary.head(TOP_K).copy()

top_indices = top30["feature_index"].to_numpy(
    dtype=np.int64
)

X_top = np.asarray(
    X[:, top_indices],
    dtype=np.float64,
)

positive_values = X_top[y == 1]
negative_values = X_top[y == 0]

top30["positive_mean"] = positive_values.mean(axis=0)
top30["negative_mean"] = negative_values.mean(axis=0)

top30["positive_median"] = np.median(
    positive_values,
    axis=0,
)
top30["negative_median"] = np.median(
    negative_values,
    axis=0,
)

top30["positive_nonzero_rate"] = (
    positive_values != 0
).mean(axis=0)

top30["negative_nonzero_rate"] = (
    negative_values != 0
).mean(axis=0)

top30["mean_difference_positive_minus_negative"] = (
    top30["positive_mean"]
    - top30["negative_mean"]
)

stability_records = []

for first_fold in range(N_FOLDS):
    for second_fold in range(
        first_fold + 1,
        N_FOLDS,
    ):
        correlation, p_value = spearmanr(
            gain_share_matrix[first_fold],
            gain_share_matrix[second_fold],
        )

        first_top = set(
            np.argsort(
                gain_share_matrix[first_fold]
            )[-TOP_K:]
        )
        second_top = set(
            np.argsort(
                gain_share_matrix[second_fold]
            )[-TOP_K:]
        )

        union = first_top | second_top
        intersection = first_top & second_top

        jaccard = (
            len(intersection) / len(union)
            if union
            else 0.0
        )

        stability_records.append({
            "fold_a": first_fold,
            "fold_b": second_fold,
            "gain_spearman_correlation": correlation,
            "spearman_p_value": p_value,
            "top30_intersection_count": len(intersection),
            "top30_jaccard": jaccard,
        })

stability = pd.DataFrame(stability_records)

overall_auroc = roc_auc_score(y, oof)
overall_ap = average_precision_score(y, oof)

oof_frame = pd.DataFrame({
    "ID": ids,
    "ARF_LABEL": y,
    "oof_probability": oof,
})

per_fold_importance.to_csv(
    PER_FOLD_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

top30.to_csv(
    TOP30_PATH,
    index=False,
)

fold_metrics.to_csv(
    METRICS_PATH,
    index=False,
)

stability.to_csv(
    STABILITY_PATH,
    index=False,
)

oof_frame.to_csv(
    OOF_PATH,
    index=False,
)

plot_frame = top30.sort_values(
    "mean_gain_share",
    ascending=True,
)

plt.figure(figsize=(10, 10))

plt.barh(
    plot_frame["feature_name"],
    plot_frame["mean_gain_share"],
    xerr=plot_frame["std_gain_share"],
    color="#4472C4",
    alpha=0.85,
)

plt.xlabel("Mean normalized gain importance")
plt.ylabel("Anonymous feature identifier")

plt.title(
    "Top 30 LightGBM Features\n"
    "Five-fold mean gain importance (seed=42)"
)

plt.tight_layout()
plt.savefig(
    PLOT_PATH,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print()
print("========== Experiment 007 summary ==========")
print(f"Overall OOF AUROC: {overall_auroc:.6f}")
print(f"Overall OOF AP: {overall_ap:.6f}")
print(
    "Mean fold AUROC ± SD: "
    f"{fold_metrics['auroc'].mean():.6f} ± "
    f"{fold_metrics['auroc'].std(ddof=1):.6f}"
)
print(
    "Mean fold AP ± SD: "
    f"{fold_metrics['average_precision'].mean():.6f} ± "
    f"{fold_metrics['average_precision'].std(ddof=1):.6f}"
)
print(
    "Mean number of features used by the model: "
    f"{fold_metrics['used_feature_count'].mean():.1f}"
)
print(
    "Number of features used in all five folds: "
    f"{int((folds_used == N_FOLDS).sum())}"
)
print(
    "Full-feature importance correlation beyond mean Top-30 Spearman correlation: "
    f"{stability['gain_spearman_correlation'].mean():.6f}"
)
print(
    "Mean Top-30 Jaccard similarity: "
    f"{stability['top30_jaccard'].mean():.6f}"
)
print(f"Total runtime: {total_runtime:.1f}s")

print()
print("Top 30 features:")
print(
    top30[
        [
            "importance_rank",
            "feature_name",
            "mean_gain_share",
            "std_gain_share",
            "folds_used",
            "top30_fold_count",
            "positive_mean",
            "negative_mean",
            "positive_nonzero_rate",
            "negative_nonzero_rate",
        ]
    ].to_string(index=False)
)

print()
print("Output files:")
for path in [
    PER_FOLD_PATH,
    SUMMARY_PATH,
    TOP30_PATH,
    METRICS_PATH,
    STABILITY_PATH,
    OOF_PATH,
    PLOT_PATH,
]:
    print(path)
