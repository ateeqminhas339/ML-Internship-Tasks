"""Data loading and cleaning utilities for the Medical Cost Personal
(insurance) dataset - reused from Task 04 for consistency across the
internship repository.
"""

import os
import pandas as pd

DEFAULT_DATA_PATH = os.path.join("data", "insurance.csv")


def load_raw_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Medical Cost Personal CSV exactly as shipped by Kaggle."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            "Download it from https://www.kaggle.com/datasets/mirichoi0218/insurance "
            "and place it in the data/ folder."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the single known exact duplicate row and standardize
    categorical text casing (documented and justified in Task 04).
    """
    df = df.copy()
    df = df.drop_duplicates()
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """Small dictionary of high-level dataset facts."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "missing_per_column": df.isna().sum().to_dict(),
    }
