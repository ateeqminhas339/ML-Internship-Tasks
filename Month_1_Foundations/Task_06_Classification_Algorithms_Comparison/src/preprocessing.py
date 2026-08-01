"""Preprocessing pipeline for Task 06 - Classification Algorithms Comparison.

Builds a scikit-learn ColumnTransformer that scales numeric features and
one-hot encodes categorical features, shared across all compared classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(X: pd.DataFrame):
    """Split feature columns into numeric and categorical lists."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer: StandardScaler for numeric, OneHotEncoder for categorical."""
    numeric_cols, categorical_cols = get_feature_types(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ]
    )
    return preprocessor


def build_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    """Wrap a preprocessor + estimator into a single sklearn Pipeline."""
    preprocessor = build_preprocessor(X)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
