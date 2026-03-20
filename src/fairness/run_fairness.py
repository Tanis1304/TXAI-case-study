"""
Run Fairness Analysis (Part 1).

Script to load data, train the baseline model (or use a cached one),
and run the fairness evaluation and threshold optimization.
"""

import sys
from pathlib import Path

# Add project root to Python path so `src` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.preprocessing import load_dataset, preprocess_dataset, split_dataset
from src.models.training import train_xgboost, evaluate_performance
from src.fairness.metrics import compute_all_fairness_metrics
from src.fairness.intervention import (
    apply_threshold_optimization,
    fairness_tradeoff_analysis,
)
from src.fairness.plots import (
    plot_fairness_comparison,
    plot_roc_by_gender,
)

RESULTS_DIR = Path("results/fairness")
MODELS_DIR = Path("models")


def run_fairness_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[pipeline] STEP 1: DATA LOADING & PREPROCESSING")
    data = load_dataset(Path("data/cardio_train.csv"))
    data = preprocess_dataset(data)
    X_train, X_test, y_train, y_test, gender_train, gender_test = split_dataset(data)

    print("\n[pipeline] STEP 2: MODEL TRAINING")
    model = train_xgboost(X_train, y_train, save_dir=MODELS_DIR)
    evaluate_performance(model, X_test, y_test, save_path=RESULTS_DIR / "baseline_performance.json")

    print("\n[pipeline] STEP 3: FAIRNESS EVALUATION (BASELINE)")
    y_pred_baseline = model.predict(X_test)
    compute_all_fairness_metrics(y_test, y_pred_baseline, gender_test)
    plot_roc_by_gender(model, X_test, y_test, gender_test)

    print("\n[pipeline] STEP 4: FAIRNESS INTERVENTION")
    y_pred_mitigated, _ = apply_threshold_optimization(
        model, X_train, y_train, gender_train,
        X_test, y_test, gender_test,
        constraint="equalized_odds",
    )

    tradeoff = fairness_tradeoff_analysis(
        y_test, y_pred_baseline, y_pred_mitigated, gender_test,
        save_path=RESULTS_DIR / "fairness_tradeoff.json",
    )

    plot_fairness_comparison(tradeoff)

    print("\n[done] Fairness pipeline complete. Outputs saved to results/fairness/")


if __name__ == "__main__":
    run_fairness_pipeline()
