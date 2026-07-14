"""Advanced cleaning, feature engineering, and preprocessing-pipeline
utilities for the Telco Customer Churn dataset.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"


# --------------------------------------------------------------------------
# Step 1 - Type fixes & structural cleaning
# --------------------------------------------------------------------------
def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges to numeric and impute the 11 blank rows.

    All blank TotalCharges values belong to customers with tenure == 0
    (brand-new customers with no billing history yet), so 0 is used
    instead of the column mean to avoid an unsupported assumption.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def drop_identifier_column(df: pd.DataFrame, col: str = "customerID") -> pd.DataFrame:
    """Drop the customer identifier column - it carries no predictive signal."""
    return df.drop(columns=[col]) if col in df.columns else df.copy()


def drop_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop fully duplicated rows. Returns (deduped_df, n_dropped)."""
    n_before = len(df)
    deduped = df.drop_duplicates()
    return deduped, n_before - len(deduped)


# --------------------------------------------------------------------------
# Step 2 - Outlier detection & treatment
# --------------------------------------------------------------------------
def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.Series:
    """Return a boolean mask of IQR-based outliers for a numeric column."""
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (df[column] < lower) | (df[column] > upper)


def cap_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    """Winsorize (cap) outliers in a numeric column to the IQR fence values."""
    df = df.copy()
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


# --------------------------------------------------------------------------
# Step 3 - Feature engineering
# --------------------------------------------------------------------------
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that summarize customer behaviour:

    - tenure_group: binned tenure (0-12, 13-24, 25-48, 49-60, 61-72 months)
    - num_services: count of subscribed add-on/utility services
    - avg_monthly_spend: TotalCharges / max(tenure, 1), a smoothed spend rate
    - has_internet: whether the customer has any internet service at all
    """
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    )

    def _count_services(row):
        count = 0
        for col in SERVICE_COLUMNS:
            if col in row and str(row[col]) not in ("No", "No phone service", "No internet service"):
                count += 1
        return count

    df["num_services"] = df.apply(_count_services, axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1).clip(lower=1)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    return df


# --------------------------------------------------------------------------
# Step 4 - Encoding & scaling pipeline
# --------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Split the dataframe's columns into numeric and categorical feature lists."""
    numeric_cols = [c for c in NUMERIC_FEATURES + ["num_services", "avg_monthly_spend"] if c in df.columns]
    exclude = set(numeric_cols) | {target_col, "Churn_binary"}
    categorical_cols = [c for c in df.columns if c not in exclude]
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer: standard-scales numeric
    features and one-hot-encodes categorical features.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ]
    )


def split_data(df: pd.DataFrame, target_col: str = TARGET_COL, test_size: float = 0.2):
    """Stratified train/test split on the (imbalanced) target column."""
    X = df.drop(columns=[target_col])
    y = (df[target_col] == "Yes").astype(int)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
