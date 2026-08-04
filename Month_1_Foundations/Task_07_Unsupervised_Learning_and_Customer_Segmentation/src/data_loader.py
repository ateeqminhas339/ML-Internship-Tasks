"""Load the Mall Customer Segmentation dataset."""
from pathlib import Path
import pandas as pd

RAW_COLUMNS = ["CustomerID", "Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_mall_customers(path: str | Path = "data/Mall_Customers.csv") -> pd.DataFrame:
    """Load the raw Mall Customers CSV and do minimal sanity checks.

    Parameters
    ----------
    path : str or Path
        Location of Mall_Customers.csv (download from Kaggle, see README).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'Mall_Customers.csv' from "
            "Kaggle (Mall Customer Segmentation Data) and place it in data/."
        )
    df = pd.read_csv(path)

    missing_cols = set(RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    return df
