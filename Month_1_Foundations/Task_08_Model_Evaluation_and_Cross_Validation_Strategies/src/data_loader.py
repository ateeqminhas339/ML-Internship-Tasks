"""Load the UCI Heart Disease dataset."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_heart_disease(path: str | Path = "data/heart.csv") -> pd.DataFrame:
    """Load the raw Heart Disease CSV and do minimal sanity checks.

    Parameters
    ----------
    path : str or Path
        Location of heart.csv (download from Kaggle/UCI, see README).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the UCI Heart Disease "
            "CSV (see README for source link) and place it in data/."
        )
    df = pd.read_csv(path)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    return df
