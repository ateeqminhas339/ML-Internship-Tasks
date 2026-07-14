"""Baseline modeling utilities used purely as a sanity check that the
cleaned/preprocessed data is usable for machine learning. Full model
development is out of scope for Task 02 and is deferred to a later task.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .preprocessing import RANDOM_SEED, build_preprocessing_pipeline


def build_baseline_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    """Wrap the preprocessing ColumnTransformer and a simple Logistic
    Regression classifier into a single sklearn Pipeline.
    """
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def evaluate_baseline(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Fit the pipeline on the training split and report standard
    classification metrics on the held-out test split.
    """
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
