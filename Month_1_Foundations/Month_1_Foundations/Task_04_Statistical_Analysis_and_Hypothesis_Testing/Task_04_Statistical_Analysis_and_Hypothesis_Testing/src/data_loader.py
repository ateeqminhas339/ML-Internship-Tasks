"""Data loading and cleaning utilities for the Medical Cost Personal
(insurance) dataset.
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
    """Apply documented, justified cleaning steps:

    1. Drop fully duplicated rows (this dataset is known to ship with one
       exact duplicate record - a data-entry artifact, not a real repeat
       patient, since insurance datasets should not contain literal
       row-for-row duplicates).
    2. Standardize categorical text columns to lowercase/stripped, in case
       of inconsistent casing or whitespace.
    3. Add a few analysis-ready derived columns used throughout the
       hypothesis tests: a BMI category (underweight/normal/overweight/
       obese, per WHO cutoffs) and an age group.
    """
    df = df.copy()

    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)

    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, float("inf")],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-64"],
    )

    df.attrs["n_duplicates_dropped"] = n_dropped
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """Small dictionary of high-level dataset facts."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "missing_per_column": df.isna().sum().to_dict(),
    }
