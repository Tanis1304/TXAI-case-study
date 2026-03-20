"""
Fairness metrics for binary classification with a binary protected attribute.

Implements:
    - Equalized Odds: equal TPR and FPR across groups
    - Predictive Parity: equal PPV (precision) across groups
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# Per-group confusion-matrix helper
def _group_confusion_metrics(y_true, y_pred, group_mask):
    """Return TPR, FPR, PPV for a single demographic group."""
    yt = np.asarray(y_true)[group_mask]
    yp = np.asarray(y_pred)[group_mask]

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false-positive rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # precision

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "ppv": float(ppv),
        "n": int(len(yt)),
    }
# Equalized Odds
def compute_equalized_odds(y_true, y_pred, gender):
    """Compute Equalized Odds: gap in TPR and FPR between gender groups.

    Equalized Odds requires:
        P(Ŷ=1 | Y=y, G=female) = P(Ŷ=1 | Y=y, G=male)   for y ∈ {0, 1}

    Returns
    -------
    dict with per-group TPR/FPR and the absolute differences.
    """
    gender = np.asarray(gender)
    female_mask = gender == 1
    male_mask = gender == 2

    female = _group_confusion_metrics(y_true, y_pred, female_mask)
    male = _group_confusion_metrics(y_true, y_pred, male_mask)

    tpr_gap = abs(female["tpr"] - male["tpr"])
    fpr_gap = abs(female["fpr"] - male["fpr"])

    result = {
        "female": {"tpr": female["tpr"], "fpr": female["fpr"]},
        "male":   {"tpr": male["tpr"],   "fpr": male["fpr"]},
        "tpr_gap": float(tpr_gap),
        "fpr_gap": float(fpr_gap),
        "equalized_odds_gap": float(tpr_gap + fpr_gap),  # combined measure
    }
    return result
# Predictive Parity
def compute_predictive_parity(y_true, y_pred, gender):
    """Compute Predictive Parity: gap in PPV (precision) between gender groups.

    Predictive Parity requires:
        P(Y=1 | Ŷ=1, G=female) = P(Y=1 | Ŷ=1, G=male)

    Returns
    -------
    dict with per-group PPV and the absolute difference.
    """
    gender = np.asarray(gender)
    female_mask = gender == 1
    male_mask = gender == 2

    female = _group_confusion_metrics(y_true, y_pred, female_mask)
    male = _group_confusion_metrics(y_true, y_pred, male_mask)

    ppv_gap = abs(female["ppv"] - male["ppv"])

    result = {
        "female": {"ppv": female["ppv"]},
        "male":   {"ppv": male["ppv"]},
        "ppv_gap": float(ppv_gap),
    }
    return result
# Combined report
def compute_all_fairness_metrics(y_true, y_pred, gender):
    """Compute both fairness metrics and per-group confusion details.

    Returns
    -------
    dict with equalized_odds, predictive_parity, and per_group details.
    """
    gender = np.asarray(gender)
    female_mask = gender == 1
    male_mask = gender == 2

    female = _group_confusion_metrics(y_true, y_pred, female_mask)
    male = _group_confusion_metrics(y_true, y_pred, male_mask)

    eo = compute_equalized_odds(y_true, y_pred, gender)
    pp = compute_predictive_parity(y_true, y_pred, gender)

    result = {
        "equalized_odds": eo,
        "predictive_parity": pp,
        "per_group": {
            "female": female,
            "male": male,
        },
    }

    print("\n[fairness] Fairness Metrics")
    print(f"  Equalized Odds:")
    print(f"    Female TPR: {eo['female']['tpr']:.4f} | Male TPR: {eo['male']['tpr']:.4f} | Gap: {eo['tpr_gap']:.4f}")
    print(f"    Female FPR: {eo['female']['fpr']:.4f} | Male FPR: {eo['male']['fpr']:.4f} | Gap: {eo['fpr_gap']:.4f}")
    print(f"    Combined EO gap: {eo['equalized_odds_gap']:.4f}")
    print(f"  Predictive Parity:")
    print(f"    Female PPV: {pp['female']['ppv']:.4f} | Male PPV: {pp['male']['ppv']:.4f} | Gap: {pp['ppv_gap']:.4f}")

    return result
