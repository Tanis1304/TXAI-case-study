"""
SHAP-based explainability analysis for XGBoost CVD predictions.

Focuses on:
    - Feature Importance by Gender (Core Research Question)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# Consistent Holistic Theme
sns.set_theme(style="whitegrid", font_scale=1.1)

# Unified Color Palette across all figures
COLORS = {
    "female": "#d95f02",       # Deep Orange
    "male": "#1b9e77",         # Teal
}

FIG_DIR = Path("results/xai")


def _ensure_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_shap_values(model, X_test: pd.DataFrame):
    """Compute TreeSHAP values for all test predictions."""
    print("[shap] Computing TreeSHAP values …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    print(f"[shap] Done. Shape: {shap_values.values.shape}")
    return shap_values


def global_summary_plot(
    shap_values,
    X_test: pd.DataFrame,
    save: bool = True,
):
    """Plot the global SHAP beeswarm plot for overall context."""
    _ensure_dir()
    print("\n[shap] Generating global beeswarm summary plot...")

    fig, ax = plt.subplots(figsize=(10, 7))
    if shap_values.values.ndim == 3:
        shap.summary_plot(shap_values.values[:, :, 1], X_test, show=False)
    else:
        shap.summary_plot(shap_values.values, X_test, show=False)
    
    plt.title("Global Feature Importance & Direction (All Patients)", fontsize=14, pad=20)
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "shap_summary_beeswarm.png", dpi=300)
        print("[shap] Saved shap_summary_beeswarm.png")
    plt.close("all")


def groupwise_shap_analysis(
    shap_values,
    X_test: pd.DataFrame,
    gender_test: pd.Series,
    save: bool = True,
) -> pd.DataFrame:
    """Compare mean |SHAP| values between gender groups in a unified 1x2 plot.
    Left Panel: Absolute feature importance for women vs men.
    Right Panel: Delta J (Difference in reliance).
    """
    _ensure_dir()
    print("\n[shap-gender] Generating unified Gender-Wise Feature Importance 1x2 Plot...")

    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    gender_array = np.asarray(gender_test)
    female_mask = gender_array == 1
    male_mask = gender_array == 2

    # Compute mean |SHAP|
    female_imp = np.mean(np.abs(vals[female_mask]), axis=0)
    male_imp = np.mean(np.abs(vals[male_mask]), axis=0)

    # Compute difference (Δ_j = Φ_female - Φ_male)
    delta_j = female_imp - male_imp
    features = X_test.columns

    comparison_df = pd.DataFrame({
        "feature": features,
        "female_importance": female_imp,
        "male_importance": male_imp,
        "delta_j": delta_j
    })

    sorted_df = comparison_df.sort_values(by="delta_j", ascending=False)
    print(sorted_df.to_string(index=False))

    # ---- Create 1x2 Subplot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    
    # Sort features by absolute reliance for the left plot
    display_order = comparison_df.sort_values(by=["female_importance", "male_importance"], ascending=True)
    features_sorted = display_order["feature"]
    
    # Left Panel: Grouped Absolute Importance
    ax0 = axes[0]
    x = np.arange(len(features_sorted))
    width = 0.35
    ax0.barh(x - width/2, display_order["female_importance"], width, label="Female", color=COLORS["female"], edgecolor="white")
    ax0.barh(x + width/2, display_order["male_importance"], width, label="Male", color=COLORS["male"], edgecolor="white")
    ax0.set_yticks(x)
    ax0.set_yticklabels(features_sorted, fontsize=11)
    ax0.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax0.set_title("Absolute Feature Importance by Gender", fontsize=14, pad=15)
    ax0.legend(loc="lower right", frameon=True, shadow=True, fontsize=11)

    # Right Panel: Delta J
    ax1 = axes[1]
    # Re-sort for Delta J but keep the y-axis (features_sorted) order so they align perfectly
    delta_colors = [COLORS["female"] if val > 0 else COLORS["male"] for val in display_order["delta_j"]]
    ax1.barh(x, display_order["delta_j"], color=delta_colors, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=1.5, zorder=3)
    ax1.set_xlabel("Δ_j  (Positive = Higher importance for Women)", fontsize=12)
    ax1.set_title("Difference in Importance (Female − Male)", fontsize=14, pad=15)

    sns.despine(left=True, ax=ax0)
    sns.despine(left=True, ax=ax1)

    plt.tight_layout(w_pad=4.0)
    if save:
        fig.savefig(FIG_DIR / "gender_importance_analysis.png", dpi=300)
        print("[shap-gender] Saved gender_importance_analysis.png")
    plt.close(fig)

    return comparison_df


def error_shap_analysis(
    shap_values,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    gender_test: pd.Series,
    save: bool = True,
) -> dict:
    """Analyse SHAP values for False Positives exclusively to explain the EO gap."""
    _ensure_dir()
    print("\n[shap-error] False Positive Error Drivers Analysis")

    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    y_t = np.asarray(y_test)
    y_p = np.asarray(y_pred)
    gender = np.asarray(gender_test)

    fp_mask = (y_t == 0) & (y_p == 1)  # false positives

    rows = []
    for grp_name, grp_val in [("Female", 1), ("Male", 2)]:
        combined_mask = fp_mask & (gender == grp_val)
        n = combined_mask.sum()
        if n == 0:
            continue
        mean_imp = np.mean(np.abs(vals[combined_mask]), axis=0)
        for feat, imp in zip(X_test.columns, mean_imp):
            rows.append({"gender": grp_name, "feature": feat, "mean_abs_shap": float(imp)})
    
    df = pd.DataFrame(rows)
    if df.empty:
        print("[shap-error] No false positives to analyze.")
        return {}

    pivot = df.pivot_table(index="feature", columns="gender", values="mean_abs_shap", aggfunc="first")

    # Sort heatmap by the feature that causes the most FP errors overall
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(by="total", ascending=False).drop(columns=["total"])

    fig, ax = plt.subplots(figsize=(6, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Mean |SHAP value|'})
    ax.set_title("False Positive Error Drivers by Gender\n(Why does the model falsely predict CVD?)", fontsize=12, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "fp_error_drivers_heatmap.png", dpi=300)
        print("[shap-error] Saved fp_error_drivers_heatmap.png")
    plt.close(fig)

    return {"FP": pivot}
