"""Data loading utilities for Task 06 - Classification Algorithms Comparison.

Loads the feature-engineered Telco Customer Churn dataset produced in
Task 03 (Feature Engineering Mastery) and provides a stratified train/test
split for classification experiments.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_feature_engineered.csv"

TARGET_COL = "Churn_binary"

# Columns dropped before modeling: identifiers/leaky/duplicate target encodings
DROP_COLS = ["Churn", TARGET_COL]


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the engineered Telco churn dataset from disk."""
    return pd.read_csv(path)


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_SEED):
    """Split features/target into stratified train and test sets."""
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
