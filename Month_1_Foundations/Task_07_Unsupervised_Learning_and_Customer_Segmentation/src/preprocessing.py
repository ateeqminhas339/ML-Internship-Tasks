"""Cleaning and feature preparation for clustering."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: dedupe, rename for convenience, drop nulls in features."""
    df = df.drop_duplicates(subset="CustomerID").copy()
    df = df.rename(columns={"Genre": "Gender"})
    df = df.dropna(subset=FEATURES)
    return df.reset_index(drop=True)


def scale_features(df: pd.DataFrame, features: list[str] = FEATURES):
    """Standardize numeric features. Returns (scaled_array, fitted_scaler).

    Scaling is required because K-Means/DBSCAN use Euclidean distance and
    Age/Income/Spending Score live on very different numeric ranges.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, scaler
