"""
XGBoost model training with cross-validated hyperparameter tuning.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
# Config helpers
DEFAULT_CONFIG_PATH = Path("configs/xgboost_config.json")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load training configuration from JSON."""
    with open(config_path) as f:
        return json.load(f)
# Training
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_path: Path = DEFAULT_CONFIG_PATH,
    save_dir: Path | None = Path("models"),
) -> XGBClassifier:
    """Train an XGBoost classifier with grid-search cross-validation.

    Parameters
    ----------
    X_train, y_train : training data
    config_path : path to JSON config with param grid
    save_dir : if provided, save the best model here as ``best_xgboost.joblib``

    Returns
    -------
    best_model : fitted XGBClassifier with the best hyperparameters
    """
    config = load_config(config_path)
    model_params = config["model_params"]
    param_grid = config["cv_param_grid"]
    cv_folds = config.get("cv_folds", 5)
    random_state = config.get("random_state", 42)

    base_model = XGBClassifier(
        **model_params,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    print(f"[training] Starting GridSearchCV with {cv_folds}-fold CV …")
    print(f"[training] Param grid: {param_grid}")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"[training] Best params: {grid_search.best_params_}")
    print(f"[training] Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "best_xgboost.joblib"
        joblib.dump(best_model, model_path)
        print(f"[training] Model saved to {model_path}")

    return best_model
# Evaluation
def evaluate_performance(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> dict:
    """Compute standard classification metrics on the test set.

    Returns
    -------
    metrics : dict with accuracy, roc_auc, precision, recall, f1, and
              the confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n[evaluation] Test-set Performance")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No CVD', 'CVD'])}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[evaluation] Metrics saved to {save_path}")

    return metrics
