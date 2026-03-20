"""
Data loading, preprocessing, and train/test splitting for the CVD dataset.

Dataset: Kaggle Cardiovascular Disease Dataset (sulianova/cardiovascular-disease-dataset)
- 70,000 patient records
- Gender encoding: 1 = female, 2 = male
- Age is stored in days (converted to years in preprocessing)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Constants
GENDER_MAP = {1: "female", 2: "male"}
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Physiologically plausible blood-pressure bounds
AP_HI_RANGE = (80, 250)     # systolic
AP_LO_RANGE = (40, 200)     # diastolic


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the raw CSV (semicolon-separated) and validate required columns."""
    data = pd.read_csv(file_path, sep=";")

    required = {"id", "age", "gender", "height", "weight",
                "ap_hi", "ap_lo", "cholesterol", "gluc",
                "smoke", "alco", "active", "cardio"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"[preprocessing] Loaded {len(data)} records from {file_path}")
    return data


def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform the raw data.

    Steps:
        1. Drop the ``id`` column (not a model feature).
        2. Convert age from days to years.
        3. Remove blood-pressure outliers outside physiological bounds.
        4. Ensure diastolic < systolic.
        5. Add BMI feature (weight_kg / height_m²).
    """
    df = data.copy()

    # Drop id
    df = df.drop(columns=["id"])

    # Age: days -> years
    df["age"] = (df["age"] / 365.25).astype(int)

    # Blood-pressure outlier removal
    n_before = len(df)
    df = df[
        (df["ap_hi"].between(*AP_HI_RANGE))
        & (df["ap_lo"].between(*AP_LO_RANGE))
        & (df["ap_lo"] < df["ap_hi"])
    ]
    n_removed = n_before - len(df)
    print(f"[preprocessing] Removed {n_removed} records with implausible blood pressure "
          f"({n_removed / n_before * 100:.1f}%)")

    # BMI
    height_m = df["height"] / 100
    df["bmi"] = df["weight"] / (height_m ** 2)

    # Drop raw height/weight (BMI encodes them)
    df = df.drop(columns=["height", "weight"])

    df = df.reset_index(drop=True)
    print(f"[preprocessing] Final dataset: {len(df)} records, {df.shape[1]} features")
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Split into train/test, returning features, labels, and gender arrays.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices (gender is kept as a feature for the model).
    y_train, y_test : pd.Series
        Binary target (cardio).
    gender_train, gender_test : pd.Series
        Gender column aligned with train/test rows for fairness evaluation.
    """
    y = df["cardio"]
    gender = df["gender"]
    X = df.drop(columns=["cardio"])

    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, gender,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[preprocessing] Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[preprocessing] Gender distribution (test) — "
          f"Female: {(gender_test == 1).sum()}, Male: {(gender_test == 2).sum()}")
    return X_train, X_test, y_train, y_test, gender_train, gender_test
