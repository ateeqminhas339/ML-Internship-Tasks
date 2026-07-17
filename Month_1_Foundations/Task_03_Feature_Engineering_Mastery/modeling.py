"""Data loading + baseline cleaning utilities for the Telco Customer Churn
dataset (self-contained for Task 03, mirrors the cleaning done in Task 02).
"""

import os
import pandas as pd

DEFAULT_DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")


def load_raw_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV exactly as shipped by Kaggle."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place it in the data/ folder."
        )
    return pd.read_csv(path)


def prepare_clean_dataframe(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw data and apply the baseline structural cleaning
    established in Task 02: fix TotalCharges, drop customerID, drop
    duplicate rows, and add a binary Churn target column.
    """
    df = load_raw_data(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop_duplicates()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """Small dictionary of high-level dataset facts."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "missing_per_column": df.isna().sum().to_dict(),
    }
