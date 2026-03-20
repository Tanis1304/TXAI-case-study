"""
SHAP validation: baseline comparisons and stability analysis.

Baseline comparisons:
    - Permutation importance (sklearn)
    - XGBoost native gain-based importance
    - Spearman rank correlation between all three methods

Stability analysis:
    - Bootstrap retraining + SHAP ranking correlation across runs
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = Path("results/xai")


def _ensure_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
# Baseline comparisons
def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute sklearn permutation importance."""
    print("[validation] Computing permutation importance …")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    print(df.to_string(index=False))
    return df


def compute_gain_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract XGBoost's native gain-based feature importance."""
    print("[validation] Extracting gain-based importance …")
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "gain_importance": importances,
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)

    print(df.to_string(index=False))
    return df


def compare_importance_rankings(
    shap_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Compare feature rankings across three methods using Spearman correlation.

    Returns
    -------
    comparison_df : merged DataFrame with all rankings
    Also prints and optionally saves a Spearman correlation matrix.
    """
    _ensure_dir()
    print("\n[validation] Importance Method Comparison")

    # Merge all rankings
    merged = (
        shap_df[["feature", "mean_abs_shap"]]
        .merge(perm_df[["feature", "perm_importance_mean"]], on="feature")
        .merge(gain_df[["feature", "gain_importance"]], on="feature")
    )

    # Add rank columns
    merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False).astype(int)
    merged["perm_rank"] = merged["perm_importance_mean"].rank(ascending=False).astype(int)
    merged["gain_rank"] = merged["gain_importance"].rank(ascending=False).astype(int)

    print(merged[["feature", "shap_rank", "perm_rank", "gain_rank"]].to_string(index=False))

    # Spearman correlations
    methods = ["shap_rank", "perm_rank", "gain_rank"]
    labels = ["SHAP", "Permutation", "Gain"]
    corr_matrix = np.ones((3, 3))

    for i in range(3):
        for j in range(i + 1, 3):
            rho, _ = spearmanr(merged[methods[i]], merged[methods[j]])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    print("\nSpearman rank correlation matrix:")
    print(corr_df.to_string())

    return merged
# Stability analysis (bootstrap)
def bootstrap_shap_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_bootstraps: int = 10,
    random_state: int = 42,
    save: bool = True,
) -> dict:
    """Assess SHAP explanation stability via bootstrap retraining.

    Trains ``n_bootstraps`` XGBoost models on bootstrap samples, computes
    SHAP rankings for each, then reports pairwise Spearman correlations.

    Returns
    -------
    dict with mean_correlation, std_correlation, and all pairwise correlations.
    """
    _ensure_dir()
    print(f"\n[stability] Bootstrap SHAP stability analysis ({n_bootstraps} runs) …")

    rng = np.random.RandomState(random_state)
    all_rankings = []

    for i in range(n_bootstraps):
        # Bootstrap resample of training data
        X_boot, y_boot = resample(
            X_train, y_train,
            random_state=rng.randint(0, 10_000),
            stratify=y_train,
        )

        # Train fresh model with reasonable defaults
        model = XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_boot, y_boot)

        # Compute SHAP
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_test)
        vals = sv.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]

        mean_abs = np.mean(np.abs(vals), axis=0)
        ranking = pd.Series(mean_abs, index=X_test.columns).rank(ascending=False)
        all_rankings.append(ranking)

        print(f"  Bootstrap {i + 1}/{n_bootstraps} done")

    # Pairwise Spearman correlations
    n = len(all_rankings)
    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(all_rankings[i], all_rankings[j])
            correlations.append(rho)

    mean_corr = float(np.mean(correlations))
    std_corr = float(np.std(correlations))

    print(f"\n[stability] Mean Spearman correlation: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"[stability] Range: [{min(correlations):.4f}, {max(correlations):.4f}]")

    result = {
        "n_bootstraps": n_bootstraps,
        "mean_spearman": mean_corr,
        "std_spearman": std_corr,
        "min_spearman": float(min(correlations)),
        "max_spearman": float(max(correlations)),
        "all_correlations": [float(c) for c in correlations],
    }

    return result
