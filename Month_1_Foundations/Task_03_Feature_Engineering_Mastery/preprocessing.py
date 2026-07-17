"""Modeling utilities for Task 03: build baseline vs. feature-engineered
pipelines, evaluate them, and extract feature importance - used to prove
the engineered features add real predictive value, not just complexity.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .preprocessing import RANDOM_SEED, build_preprocessing_pipeline


def build_pipeline(numeric_cols, categorical_cols, model=None) -> Pipeline:
    """Wrap the preprocessing ColumnTransformer and a classifier into a
    single sklearn Pipeline. Defaults to Logistic Regression.
    """
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def evaluate_pipeline(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, evaluate on test, return standard classification metrics."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }


def compare_feature_sets(
    baseline_numeric, baseline_categorical, baseline_split,
    engineered_numeric, engineered_categorical, engineered_split,
) -> pd.DataFrame:
    """Train identical Logistic Regression models on (a) a minimal baseline
    feature set and (b) the fully engineered feature set, and return a
    side-by-side metrics comparison table.
    """
    X_train_b, X_test_b, y_train_b, y_test_b = baseline_split
    X_train_e, X_test_e, y_train_e, y_test_e = engineered_split

    baseline_pipe = build_pipeline(baseline_numeric, baseline_categorical)
    baseline_metrics = evaluate_pipeline(baseline_pipe, X_train_b, y_train_b, X_test_b, y_test_b)

    engineered_pipe = build_pipeline(engineered_numeric, engineered_categorical)
    engineered_metrics = evaluate_pipeline(engineered_pipe, X_train_e, y_train_e, X_test_e, y_test_e)

    return pd.DataFrame({"baseline_features": baseline_metrics, "engineered_features": engineered_metrics})


def random_forest_feature_importance(numeric_cols, categorical_cols, X_train, y_train, top_n: int = 15) -> pd.DataFrame:
    """Fit a Random Forest through the same preprocessing pipeline and
    return the top-N most important features by Gini importance.
    """
    pipe = build_pipeline(
        numeric_cols, categorical_cols,
        model=RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, max_depth=8),
    )
    pipe.fit(X_train, y_train)

    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = pipe.named_steps["classifier"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return imp_df.sort_values("importance", ascending=False).head(top_n)
