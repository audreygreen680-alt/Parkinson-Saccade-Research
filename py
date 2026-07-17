"""
run_analyses.py
================
Example of how every downstream analysis should consume the ONE
canonical dataset from ppmi_pipeline.py, instead of re-deriving it.

Run once at the top of a session:
    df, feature_pool, source_of_column, manifest = build_clean_dataset()

Then every analysis function below takes that same (df, feature_pool)
as input. No analysis re-reads a raw CSV, re-applies a blacklist, or
re-derives DIAGNOSIS. If you need a different target definition (e.g.
the 4-way PRIMDIAG grouping used for the manifold plots), derive it
FROM this same df_final, not from a separately exported file -- that
way you can prove both analyses ran on the identical patient rows.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ppmi_pipeline import PipelineConfig, build_clean_dataset


# =====================================================================
# 1. CLASSIFIER: 5-fold CV, leakage-safe feature selection per fold
# =====================================================================
def run_classifier_cv(df: pd.DataFrame, feature_pool: list[str], n_splits: int = 5):
    """
    NOTE on leakage: build_clean_dataset() already filters features by
    global non-null counts BEFORE this split. That is a mild leak (test
    rows influence which features exist at all). For a paper, either:
      (a) accept it and say so explicitly in Limitations, because the
          filter is about missingness, not about the label, and re-run
          with a much higher min_valid_entries so the effect is tiny; or
      (b) do the filtering inside each fold, using only the training
          rows, and intersect the resulting feature pool across folds.
    Below shows the safer per-fold approach for comparison against
    the simpler global version already used in the notebook.
    """
    X_full = df[feature_pool]
    y = df["DIAGNOSIS"].values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    for train_idx, test_idx in cv.split(X_full, y):
        # Fold-local feature validity check (train rows only)
        train_valid = X_full.iloc[train_idx].notna().sum() >= 200
        fold_features = X_full.columns[train_valid]

        X_train = X_full.iloc[train_idx][fold_features].values
        X_test = X_full.iloc[test_idx][fold_features].values
        y_train, y_test = y[train_idx], y[test_idx]

        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.015,
            max_iter=300,
            max_leaf_nodes=40,
            min_samples_leaf=30,
            l2_regularization=0.2,
            class_weight="balanced",
            early_stopping=True,
            n_iter_no_change=12,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        tp = np.sum((preds == 1) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        fold_metrics.append({"balanced_acc": (sens + spec) / 2, "sensitivity": sens, "specificity": spec})

    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.agg(["mean", "std"])
    print("\n--- Classifier CV summary (mean ± std across folds) ---")
    print(summary)
    return metrics_df


# =====================================================================
# 2. CLINICAL vs DIGITAL vs COMBINED feature-set comparison
# =====================================================================
def run_feature_set_comparison(
    df: pd.DataFrame, feature_pool: list[str], source_of_column: dict[str, str]
):
    clinical_cols = [c for c in feature_pool if source_of_column.get(c) == "clinical"]
    digital_cols = [c for c in feature_pool if source_of_column.get(c) == "digital"]

    y = df["DIAGNOSIS"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for label, cols in [
        ("Clinical-only", clinical_cols),
        ("Digital-only", digital_cols),
        ("Combined", feature_pool),
    ]:
        if not cols:
            continue
        X = df[cols].values
        fold_bal_acc = []
        for train_idx, test_idx in cv.split(X, y):
            clf = HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.015, max_iter=300,
                max_leaf_nodes=40, min_samples_leaf=30, l2_regularization=0.2,
                class_weight="balanced", early_stopping=True, n_iter_no_change=12,
                random_state=42,
            )
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            y_test = y[test_idx]
            tp = np.sum((preds == 1) & (y_test == 1))
            tn = np.sum((preds == 0) & (y_test == 0))
            fp = np.sum((preds == 1) & (y_test == 0))
            fn = np.sum((preds == 0) & (y_test == 1))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            fold_bal_acc.append((sens + spec) / 2)
        results.append({"Feature Set": label, "N Features": len(cols), "Balanced Accuracy": np.mean(fold_bal_acc)})

    results_df = pd.DataFrame(results)
    print("\n--- Clinical vs Digital vs Combined ---")
    print(results_df.to_string(index=False))
    return results_df


# =====================================================================
# 3. CLUSTERING / MANIFOLD (uses the SAME df_final -- no separate CSV)
# =====================================================================
def run_clustering(df: pd.DataFrame, feature_pool: list[str], k: int = 3):
    X_raw = df[feature_pool]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_out = df[["PATNO", "DIAGNOSIS"]].copy()
    df_out["cluster"] = labels
    print(f"\n--- KMeans (k={k}) cluster sizes ---")
    print(df_out["cluster"].value_counts().sort_index())
    return df_out


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    cfg = PipelineConfig()  # edit data_dir / output_dir for your machine
    df, feature_pool, source_of_column, manifest = build_clean_dataset(cfg)

    run_classifier_cv(df, feature_pool)
    run_feature_set_comparison(df, feature_pool, source_of_column)
    run_clustering(df, feature_pool, k=3)

    print(f"\nAll analyses above ran on dataset hash={manifest['content_hash']} "
          f"({manifest['n_patients']} patients, {manifest['n_features']} features). "
          f"Cite this hash in your methods section / supplementary code.")
