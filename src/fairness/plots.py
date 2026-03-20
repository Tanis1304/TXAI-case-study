"""
Fairness visualisations for Part 1.

Generates:
    - fairness_intervention_impact.png: Before/after fairness metric comparison
    - roc_by_gender.png: Per-group ROC curves
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Consistent Holistic Theme
sns.set_theme(style="whitegrid", font_scale=1.1)

# Unified Color Palette across all figures
COLORS = {
    "female": "#d95f02",       # Deep Orange
    "male": "#1b9e77",         # Teal
    "baseline": "#999999",     # Medium Gray
    "mitigated": "#2ca02c",    # Strong Green
}

FIG_DIR = Path("results/fairness")

def _ensure_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fairness_comparison(tradeoff: dict, save: bool = True):
    """Side-by-side grouped bar chart showing the Gap in EO and PP 
    before vs after threshold optimization (Holistic Refinement).
    """
    _ensure_dir()

    metrics = ["Equalized Odds Gap", "Predictive Parity Gap"]
    baseline_vals = [
        tradeoff["baseline"]["eo_combined_gap"],
        tradeoff["baseline"]["pp_gap"],
    ]
    mitigated_vals = [
        tradeoff["mitigated"]["eo_combined_gap"],
        tradeoff["mitigated"]["pp_gap"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Baseline Bars
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline (Pre-Intervention)",
                   color=COLORS["baseline"], edgecolor="white")
    # Mitigated Bars
    bars2 = ax.bar(x + width / 2, mitigated_vals, width, label="Mitigated (ThresholdOpt)",
                   color=COLORS["mitigated"], edgecolor="white")

    ax.set_ylabel("Absolute Disparity Gap\n(Lower is Fairer)", fontsize=11, labelpad=10)
    ax.set_title("Impact of Fairness Intervention on Bias Metrics", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="medium")
    ax.legend(loc="upper right", frameon=True, shadow=True)
    
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=10)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=10)
    
    # Cleanup visual clutter
    sns.despine(left=True)

    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "fairness_intervention_impact.png", dpi=300)
        print("[plots] Saved fairness_intervention_impact.png")
    plt.close(fig)


def plot_roc_by_gender(
    model,
    X_test,
    y_test,
    gender_test,
    save: bool = True,
):
    """ROC curves for female and male subgroups (Holistic Refinement)."""
    _ensure_dir()

    y_prob = model.predict_proba(X_test)[:, 1]
    gender = np.asarray(gender_test)

    fig, ax = plt.subplots(figsize=(7, 6))

    for label, mask, color in [
        ("Female", gender == 1, COLORS["female"]),
        ("Male",   gender == 2, COLORS["male"]),
    ]:
        fpr, tpr, _ = roc_curve(np.asarray(y_test)[mask], y_prob[mask])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Guess Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Baseline Predictive Power Disparity by Gender", fontsize=13, pad=15)
    
    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=11)
    
    sns.despine()

    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "roc_by_gender.png", dpi=300)
        print("[plots] Saved roc_by_gender.png")
    plt.close(fig)
