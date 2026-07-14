"""Data loading utilities for the Telco Customer Churn dataset."""

import os
import pandas as pd

RANDOM_SEED = 42
DEFAULT_DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")


def load_raw_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV exactly as shipped by Kaggle.

    Parameters
    ----------
    path : str
        Path to the CSV file. Defaults to data/Telco-Customer-Churn.csv
        relative to the project root.

    Returns
    -------
    pd.DataFrame
        The unmodified raw dataset (7,043 rows x 21 columns).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place it in the data/ folder."
        )
    return pd.read_csv(path)


def basic_info(df: pd.DataFrame) -> dict:
    """Return a small dictionary of high-level dataset facts, useful for
    quick sanity checks and for populating report tables.
    """
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "n_duplicate_customer_ids": int(df["customerID"].duplicated().sum())
        if "customerID" in df.columns
        else None,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isna().sum().to_dict(),
    }
