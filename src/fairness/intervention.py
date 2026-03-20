"""
Fairness intervention via threshold optimization.

Uses fairlearn.postprocessing.ThresholdOptimizer to find per-group
decision thresholds that satisfy Equalized Odds while minimising accuracy loss.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score

from src.fairness.metrics import compute_all_fairness_metrics


def apply_threshold_optimization(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    gender_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    gender_test: pd.Series,
    constraint: str = "equalized_odds",
) -> tuple:
    """Apply post-processing threshold optimization for fairness.

    Parameters
    ----------
    model : fitted classifier with ``predict_proba``
    X_train, y_train, gender_train : training data (needed to fit thresholds)
    X_test, y_test, gender_test : test data for evaluation
    constraint : fairness constraint for ThresholdOptimizer
                 (``"equalized_odds"`` or ``"demographic_parity"``)

    Returns
    -------
    y_pred_mitigated : np.ndarray of mitigated predictions on test set
    threshold_optimizer : the fitted ThresholdOptimizer object
    """
    print(f"\n[intervention] Applying ThresholdOptimizer (constraint={constraint}) …")

    # Monkey-patch model to return float64 arrays to avoid fairlearn + pandas assignment bugs
    original_predict_proba = model.predict_proba
    original_predict = model.predict
    model.predict_proba = lambda X, *args, **kwargs: np.asarray(original_predict_proba(X, *args, **kwargs), dtype=np.float64)
    model.predict = lambda X, *args, **kwargs: np.asarray(original_predict(X, *args, **kwargs), dtype=np.float64)

    threshold_optimizer = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        predict_method="predict_proba",
        prefit=True,  # model is already fitted
    )

    # Convert to numpy to avoid pandas internal assignment errors in fairlearn
    X_tr_np = X_train.to_numpy()
    y_tr_np = y_train.to_numpy()
    g_tr_np = gender_train.to_numpy()
    
    X_te_np = X_test.to_numpy()
    g_te_np = gender_test.to_numpy()

    # Fit thresholds on training data
    threshold_optimizer.fit(X_tr_np, y_tr_np, sensitive_features=g_tr_np)

    # Generate mitigated predictions on test data
    y_pred_mitigated = threshold_optimizer.predict(
        X_te_np, sensitive_features=g_te_np, random_state=42
    )

    # Restore original methods so downstream sklearn tools (like permutation_importance) don't break
    model.predict_proba = original_predict_proba
    model.predict = original_predict

    print("[intervention] Threshold optimization complete.")
    return y_pred_mitigated, threshold_optimizer


def fairness_tradeoff_analysis(
    y_test: pd.Series,
    y_pred_baseline: np.ndarray,
    y_pred_mitigated: np.ndarray,
    gender_test: pd.Series,
    save_path: Path | None = None,
) -> dict:
    """Compare fairness metrics before and after intervention.

    Returns
    -------
    dict with baseline_metrics, mitigated_metrics, and deltas.
    """
    baseline = compute_all_fairness_metrics(y_test, y_pred_baseline, gender_test)
    mitigated = compute_all_fairness_metrics(y_test, y_pred_mitigated, gender_test)

    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    acc_mitigated = accuracy_score(y_test, y_pred_mitigated)
    accuracy_cost = acc_baseline - acc_mitigated

    result = {
        "baseline": {
            "accuracy": float(acc_baseline),
            "eo_tpr_gap": baseline["equalized_odds"]["tpr_gap"],
            "eo_fpr_gap": baseline["equalized_odds"]["fpr_gap"],
            "eo_combined_gap": baseline["equalized_odds"]["equalized_odds_gap"],
            "pp_gap": baseline["predictive_parity"]["ppv_gap"],
            "per_group": baseline["per_group"],
        },
        "mitigated": {
            "accuracy": float(acc_mitigated),
            "eo_tpr_gap": mitigated["equalized_odds"]["tpr_gap"],
            "eo_fpr_gap": mitigated["equalized_odds"]["fpr_gap"],
            "eo_combined_gap": mitigated["equalized_odds"]["equalized_odds_gap"],
            "pp_gap": mitigated["predictive_parity"]["ppv_gap"],
            "per_group": mitigated["per_group"],
        },
        "deltas": {
            "accuracy_cost": float(accuracy_cost),
            "eo_combined_reduction": float(
                baseline["equalized_odds"]["equalized_odds_gap"]
                - mitigated["equalized_odds"]["equalized_odds_gap"]
            ),
            "pp_gap_change": float(
                mitigated["predictive_parity"]["ppv_gap"]
                - baseline["predictive_parity"]["ppv_gap"]
            ),
        },
    }

    print("\n[tradeoff] Fairness Trade-off Analysis")
    print(f"  Accuracy:  baseline {acc_baseline:.4f} → mitigated {acc_mitigated:.4f} "
          f"(cost: {accuracy_cost:+.4f})")
    print(f"  EO gap:    {result['baseline']['eo_combined_gap']:.4f} → "
          f"{result['mitigated']['eo_combined_gap']:.4f} "
          f"(Δ: {result['deltas']['eo_combined_reduction']:+.4f})")
    print(f"  PP gap:    {result['baseline']['pp_gap']:.4f} → "
          f"{result['mitigated']['pp_gap']:.4f} "
          f"(Δ: {result['deltas']['pp_gap_change']:+.4f})")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[tradeoff] Results saved to {save_path}")

    return result
