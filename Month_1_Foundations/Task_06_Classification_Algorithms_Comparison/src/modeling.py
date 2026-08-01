"""Modeling utilities for Task 06 - Classification Algorithms Comparison.

Defines the candidate classifiers being compared and helper functions to
train, evaluate, and cross-validate each one through a shared pipeline.
"""

import time

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_pipeline

RANDOM_SEED = 42


def get_candidate_models() -> dict:
    """Return the dict of classifiers being compared, keyed by display name."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_SEED),
    }


def evaluate_model(name: str, estimator, X_train, y_train, X_test, y_test) -> dict:
    """Fit a pipeline for one estimator and compute test-set metrics + timing."""
    pipeline = build_pipeline(estimator, X_train)

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def cross_validate_model(name: str, estimator, X, y, cv_folds: int = 5) -> dict:
    """5-fold stratified CV ROC-AUC score for one estimator."""
    pipeline = build_pipeline(estimator, X)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
    return {"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()}


def run_comparison(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Train + evaluate every candidate model, return a results DataFrame."""
    results = []
    fitted = {}
    for name, estimator in get_candidate_models().items():
        res = evaluate_model(name, estimator, X_train, y_train, X_test, y_test)
        fitted[name] = res.pop("pipeline")
        results.append(res)
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return results_df, fitted
