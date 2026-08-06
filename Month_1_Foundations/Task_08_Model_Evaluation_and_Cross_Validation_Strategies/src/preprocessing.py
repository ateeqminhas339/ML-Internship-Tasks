"""Cleaning and feature preparation for the heart disease model."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates/nulls, binarize target (0 = no disease, 1 = disease)."""
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = (df[TARGET] > 0).astype(int)  # UCI target is 0-4 severity; binarize
    return df.reset_index(drop=True)


def make_xy(df: pd.DataFrame):
    """Split into X (unscaled) and y. Scaling is fit inside CV folds, not here,
    to avoid leaking test-fold statistics into training (see modeling.py)."""
    return df[FEATURES].copy(), df[TARGET].copy()


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a scaler on numeric features of a *training* split only."""
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERIC_FEATURES])
    return scaler
