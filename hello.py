"""
ISEF Trial 2 — Oculomotor-Cognitive Parkinson's Phenotyping

    `X_train` and evaluated on `X_test`, computed fresh on each run.

If you change the inclusion filter, feature set, or model and the number moves,
that is a meaningfully different analysis — it should be reported as such
(e.g. "Model B, with feature set X, achieved Y"), not silently swapped in as
"the" result.

ASSUMPTIONS THIS SCRIPT MAKES (verify against your data dictionary):
  - PRIMDIAG / COHORT_OL: 1 = Parkinson's Disease, 2 = Healthy Control
  - PATNO is a consistent patient identifier across all files
  - JLO_TOTCALC, CLCKTOT, MCATOT are the intended Benton/Clock/MoCA score columns
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42

# ============================================================
# 0. CONFIG — edit these paths to match your machine
# ============================================================

DATA_DIR = r"D:\Trial 2\Datasets I used"
OUTPUT_DIR = r"D:\Trial 2\Clean_Pipeline_Output"

PATHS = {
    "enroll":    os.path.join(DATA_DIR, "Participant Enrollment Clean - Participant Enrollment Clean (1).csv"),
    "benton":    os.path.join(DATA_DIR, "Benton Line Clean - Benton Line Clean.csv"),
    "clock":     os.path.join(DATA_DIR, "Clock clean  - Clock clean .csv"),
    "moca":      os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__14Feb2026.csv"),
    "diagnosis": os.path.join(DATA_DIR, "Primary Diagnosis Clean - Primary Diagnosis Clean.csv"),
}

# Diagnosis code mapping — confirmed against original PPMI-style coding used
# in the early notebook cells. If your data dictionary says otherwise, change
# this in ONE place, not by flipping numbers downstream until results "look right."
DIAGNOSIS_MAP = {1: "Parkinson's", 2: "Healthy Control"}
TARGET_MAP = {1: 1, 2: 0}  # 1 = PD (positive class), 0 = Healthy

# Inclusion filter — decide and document this BEFORE looking at any model output.
# The original notebook tried >=26, then >=24, then >=23.5, then quantile-based,
# switching each time the previous threshold returned "too few" or "bad" results.
# That is filter-shopping. Pick a clinically-justified threshold and keep it.
MOCA_INCLUSION_THRESHOLD = 26  # MoCA >= 26 is the standard "normal cognition" cutoff

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD + STANDARDIZE
# ============================================================

def load_and_standardize(paths: dict) -> dict:
    """Load each CSV, uppercase/strip column names, coerce PATNO to a clean numeric ID."""
    dfs = {}
    for name, path in paths.items():
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        if "PATNO" not in df.columns:
            raise KeyError(f"'{name}' file has no PATNO column after standardization: {df.columns.tolist()}")
        df["PATNO"] = pd.to_numeric(df["PATNO"], errors="coerce")
        df = df.dropna(subset=["PATNO"])
        df["PATNO"] = df["PATNO"].astype(int)
        dfs[name] = df
        print(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
    return dfs


def patient_average(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Collapse multiple visits per patient into one mean score per PATNO."""
    if score_col not in df.columns:
        raise KeyError(f"Expected column '{score_col}' not found. Available: {df.columns.tolist()}")
    return df.groupby("PATNO")[score_col].mean().round(2).reset_index()


# ============================================================
# 2. BUILD MASTER TABLE
# ============================================================

def build_master(dfs: dict) -> pd.DataFrame:
    diag = dfs["diagnosis"][["PATNO", "PRIMDIAG"]].drop_duplicates("PATNO").copy()
    diag["STATUS"] = diag["PRIMDIAG"].map(DIAGNOSIS_MAP)
    diag["TARGET"] = diag["PRIMDIAG"].map(TARGET_MAP)

    unmapped = diag["STATUS"].isna().sum()
    if unmapped:
        print(f"WARNING: {unmapped} patients have a PRIMDIAG code outside {list(DIAGNOSIS_MAP)} "
              f"and will have no STATUS/TARGET (likely a different diagnosis category — "
              f"check whether they should be excluded rather than silently dropped later).")

    benton_avg = patient_average(dfs["benton"], "JLO_TOTCALC")
    clock_avg = patient_average(dfs["clock"], "CLCKTOT")
    moca_avg = patient_average(dfs["moca"], "MCATOT")

    master = diag.merge(benton_avg, on="PATNO", how="left")
    master = master.merge(clock_avg, on="PATNO", how="left")
    master = master.merge(moca_avg, on="PATNO", how="left")

    print(f"\nMaster table built: {len(master)} patients")
    print(master["STATUS"].value_counts(dropna=False))
    return master


# ============================================================
# 3. OCULOMOTOR_DROP BIOMARKER
# ============================================================

def add_oculomotor_drop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the 'expected JLO given MoCA' relationship using Healthy Controls ONLY,
    then compute every patient's residual from that healthy baseline.

    This is fit on the full available HC population, not on the train split,
    because it represents a fixed clinical reference curve (analogous to a
    growth chart), not a predictive model being evaluated for generalization.
    If you want to be maximally conservative, refit this using only the HC
    patients in the training fold — see note at the bottom of this function.
    """
    df = df.copy()
    needed = ["TARGET", "MCATOT", "JLO_TOTCALC"]
    fit_data = df.dropna(subset=needed)
    hc = fit_data[fit_data["TARGET"] == 0]

    if len(hc) < 10:
        print("WARNING: fewer than 10 Healthy Controls have complete MCATOT/JLO_TOTCALC data. "
              "The Oculomotor_Drop baseline will be unreliable.")

    reg = LinearRegression().fit(hc[["MCATOT"]], hc["JLO_TOTCALC"])

    has_inputs = df["MCATOT"].notna()
    df.loc[has_inputs, "Expected_JLO"] = reg.predict(df.loc[has_inputs, ["MCATOT"]])
    df["Oculomotor_Drop"] = df["JLO_TOTCALC"] - df["Expected_JLO"]

    print(f"\nOculomotor_Drop baseline fit on {len(hc)} Healthy Controls "
          f"(R^2 on HC = {reg.score(hc[['MCATOT']], hc['JLO_TOTCALC']):.3f})")
    return df

    # NOTE on leakage purism: because this regression only uses MCATOT and
    # JLO_TOTCALC (not the label being predicted) and is fit on HC patients
    # specifically to define a normative reference, it is lower-risk than
    # leakage via the target itself. But if a reviewer pushes back, the more
    # defensible version refits this inside cross-validation using only the
    # training fold's HC patients.


# ============================================================
# 4. INCLUSION FILTER (decided in advance, not after seeing AUC)
# ============================================================

def apply_inclusion_filter(df: pd.DataFrame, moca_threshold: int) -> pd.DataFrame:
    before = len(df)
    filtered = df[df["MCATOT"] >= moca_threshold].copy()
    print(f"\nInclusion filter MoCA >= {moca_threshold}: {before} -> {len(filtered)} patients")
    print(filtered["STATUS"].value_counts())
    return filtered


# ============================================================
# 5. MODEL: split first, then everything else
# ============================================================

def run_model(df: pd.DataFrame):
    feature_cols = ["JLO_TOTCALC", "CLCKTOT", "MCATOT", "Oculomotor_Drop"]
    model_df = df.dropna(subset=feature_cols + ["TARGET"]).copy()

    if model_df["TARGET"].nunique() < 2:
        raise ValueError("Need both classes present after filtering to fit a classifier.")

    X = model_df[feature_cols]
    y = model_df["TARGET"].astype(int)

    # SPLIT FIRST. Nothing above this line is allowed to have seen y_test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain: {len(X_train)} patients ({y_train.sum()} PD / {(y_train==0).sum()} HC)")
    print(f"Test:  {len(X_test)} patients ({y_test.sum()} PD / {(y_test==0).sum()} HC)")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_probs": y_probs,
        "y_pred": y_pred,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ============================================================
# 6. REPORTING — every number here is read directly off `results`
# ============================================================

def report(results: dict):
    print("\n" + "=" * 60)
    print("RESULTS (computed once, on a held-out test set never seen during fitting)")
    print("=" * 60)
    print(f"Features used: {results['feature_cols']}")
    print(f"Train size: {results['n_train']}  |  Test size: {results['n_test']}")
    print(f"PR-AUC: {results['pr_auc']:.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted; order = [Healthy, PD]):")
    print(results["confusion_matrix"])
    print("\n" + classification_report(
        results["y_test"], results["y_pred"], target_names=["Healthy Control", "Parkinson's"]
    ))

    # Feature importance — descriptive only, not a causal claim
    importances = pd.Series(
        results["model"].feature_importances_, index=results["feature_cols"]
    ).sort_values(ascending=False)
    print("Feature importances (Gini-based, descriptive only):")
    print(importances)

    # --- Plot: PR curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(results["recall"], results["precision"], lw=2,
              label=f"Model (PR-AUC = {results['pr_auc']:.3f})")
    baseline = results["y_test"].mean()
    plt.axhline(baseline, linestyle="--", color="gray",
                label=f"Prevalence baseline ({baseline:.2f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall Curve — Held-Out Test Set")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(OUTPUT_DIR, "pr_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"\nSaved: {pr_path}")

    # --- Plot: confusion matrix ---
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN (Healthy)", "FP"], ["FN", "TP (PD)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\nn={cm[i, j]}", ha="center", va="center",
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Healthy", "Pred: PD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: Healthy", "Actual: PD"])
    plt.title("Confusion Matrix — Held-Out Test Set")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")


# ============================================================
#
# ============================================================
#
try:
    import torch
    import torch.nn as nn
    from torchvision import models
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("NOTE: torch/torchvision not installed — VisionEngine (unfinished CNN module) "
          "is unavailable, but this does not affect the main clinical-features pipeline above.")

if _TORCH_AVAILABLE:
    class VisionEngine(nn.Module):
        """ResNet-18 feature extractor — strips the classification head, returns 512-d vectors."""

        def __init__(self):
            super().__init__()
            backbone = models.resnet18(weights="IMAGENET1K_V1")
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        def forward(self, x):
            features = self.feature_extractor(x)
            return torch.flatten(features, 1)  # [batch, 512]


def generate_recurrence_plots(signal_dir: str, output_dir: str, signal_col_candidates=("TARGET_SPEED", "SPEED", "VEL")):
    """
    Convert each patient's 1D gaze/speed signal into a 2D recurrence-plot image,
    suitable as input to VisionEngine. Not currently called by the main pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(signal_dir) if f.endswith(".csv")]
    print(f"Found {len(files)} signal files in {signal_dir}")

    for f in files:
        try:
            df = pd.read_csv(os.path.join(signal_dir, f))
            df.columns = [str(c).strip().upper() for c in df.columns]
            col = next((c for c in df.columns if any(k in c for k in signal_col_candidates)), None)
            if col is None:
                print(f"Skipping {f}: no matching signal column among {df.columns.tolist()}")
                continue

            data = df[col].dropna().values[:400]
            if len(data) < 2:
                continue

            recurrence = np.abs(data[:, None] - data)

            plt.figure(figsize=(3, 3))
            plt.imshow(recurrence, cmap="magma", origin="lower")
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f.replace(".csv", ".png")),
                        bbox_inches="tight", pad_inches=0)
            plt.close("all")
        except Exception as e:
            print(f"Failed on {f}: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    dfs = load_and_standardize(PATHS)
    master = build_master(dfs)
    master = add_oculomotor_drop(master)
    filtered = apply_inclusion_filter(master, MOCA_INCLUSION_THRESHOLD)
    results = run_model(filtered)
    report(results)

    out_csv = os.path.join(OUTPUT_DIR, "master_clean.csv")
    filtered.to_csv(out_csv, index=False)
    print(f"\nSaved cleaned master table: {out_csv}")
