"""Feature engineering and preprocessing pipeline for the regression task."""

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET_COL = "charges"


def add_interaction_features(df):
    """Add a smoker x BMI interaction term - a well-documented, strong
    driver of medical charges in this dataset (obese smokers cost far
    more than either factor alone would suggest).
    """
    df = df.copy()
    df["smoker_bmi_interaction"] = (df["smoker"] == "yes").astype(int) * df["bmi"]
    return df


def get_feature_lists():
    """Numeric and categorical feature lists, including the engineered
    interaction term.
    """
    numeric = NUMERIC_FEATURES + ["smoker_bmi_interaction"]
    return numeric, CATEGORICAL_FEATURES


def build_preprocessing_pipeline(numeric_cols, categorical_cols) -> ColumnTransformer:
    """StandardScaler for numeric features, OneHotEncoder for categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ]
    )


def split_data(df, test_size: float = 0.2):
    """80/20 train/test split (regression target is continuous, so no
    stratification is used).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)
