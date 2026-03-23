"""
Run XAI Analysis (Part 2).

Script to load data, train the baseline model (or use a cached one),
and run the SHAP explainability analysis focusing on Feature Importance by Gender.
"""

import json
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.preprocessing import load_dataset, preprocess_dataset, split_dataset
from src.models.training import train_xgboost
from src.xai.shap_analysis import (
    compute_shap_values,
    global_summary_plot,
    groupwise_shap_analysis,
    error_shap_analysis,
)
from src.xai.validation import (
    compute_permutation_importance,
    compute_gain_importance,
    compare_importance_rankings,
    bootstrap_shap_stability,
)

RESULTS_DIR = Path("results/xai")
MODELS_DIR = Path("models")


def run_xai_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[pipeline] STEP 1: DATA LOADING & MODEL TRAINING/LOADING")
    data = load_dataset(Path("data/cardio_train.csv"))
    data = preprocess_dataset(data)
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        gender_train, gender_val, gender_test
    ) = split_dataset(data)
    model_path = MODELS_DIR / "best_xgboost.joblib"

    if model_path.exists():
        print(f"[pipeline] Loading existing model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("[pipeline] No saved model found. Training new model...")
        model = train_xgboost(X_train, y_train, save_dir=MODELS_DIR)
        evaluate_performance(model, X_test, y_test, save_path=RESULTS_DIR / "baseline_performance.json")

    print("\n[pipeline] STEP 2: SHAP EXPLAINABILITY ANALYSIS")
    shap_values = compute_shap_values(model, X_test)

    # Global Context
    global_summary_plot(shap_values, X_test)

    # Core Analysis: Feature Importance by Gender
    gender_comparison = groupwise_shap_analysis(shap_values, X_test, gender_test)

    # False Positive Deep Dive
    y_pred_baseline = model.predict(X_test)
    error_shap_analysis(shap_values, X_test, y_test, y_pred_baseline, gender_test)

    print("\n[pipeline] STEP 3: SHAP VALIDATION")
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    mean_abs_shap = np.abs(vals).mean(axis=0)
    shap_importance = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs_shap}).sort_values(by="mean_abs_shap", ascending=False)
    
    perm_imp = compute_permutation_importance(model, X_test, y_test)
    gain_imp = compute_gain_importance(model, list(X_test.columns))
    ranking_comparison = compare_importance_rankings(shap_importance, perm_imp, gain_imp)

    stability = bootstrap_shap_stability(X_train, y_train, X_test, n_bootstraps=10)
    with open(RESULTS_DIR / "shap_stability.json", "w") as f:
        json.dump(stability, f, indent=2)

    print("\n[done] XAI pipeline complete. Outputs saved to results/xai/")


if __name__ == "__main__":
    run_xai_pipeline()
