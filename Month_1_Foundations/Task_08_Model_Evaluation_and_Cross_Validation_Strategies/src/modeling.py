"""Cross-validation strategies, curves, and statistical model comparison.

A pipeline (scaler + model) is used everywhere so that scaling is fit only
on each training fold -- fitting the scaler on the full dataset before CV
would leak test-fold information into training (data leakage).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, cross_val_score,
    learning_curve, validation_curve,
)

RANDOM_SEED = 42


def make_pipeline(model=None) -> Pipeline:
    """Scaler + classifier pipeline; scaling is refit per-fold by cross_val_score."""
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def compare_cv_strategies(X: pd.DataFrame, y: pd.Series, model=None) -> pd.DataFrame:
    """Run the same model under K-Fold, Stratified K-Fold, and Leave-One-Out.

    Shows how naive K-Fold can misestimate performance on imbalanced data
    versus Stratified K-Fold, and how LOO is near-unbiased but high-variance.
    """
    pipe = make_pipeline(model)
    strategies = {
        "K-Fold (k=5)": KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        "Stratified K-Fold (k=5)": StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        "Leave-One-Out": LeaveOneOut(),
    }
    rows = []
    for name, cv in strategies.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        rows.append({
            "strategy": name,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "n_folds": len(scores),
        })
    return pd.DataFrame(rows)


def get_learning_curve(X: pd.DataFrame, y: pd.Series, model=None):
    """Train/validation accuracy vs. training set size (bias-variance diagnostic)."""
    pipe = make_pipeline(model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_SEED,
    )
    return train_sizes, train_scores, val_scores


def get_validation_curve(X: pd.DataFrame, y: pd.Series, param_range=None):
    """Train/validation accuracy vs. regularization strength C (LogisticRegression)."""
    if param_range is None:
        param_range = np.logspace(-3, 2, 10)
    pipe = make_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_scores, val_scores = validation_curve(
        pipe, X, y, param_name="model__C", param_range=param_range,
        cv=cv, scoring="accuracy",
    )
    return param_range, train_scores, val_scores


def paired_ttest_models(X: pd.DataFrame, y: pd.Series, model_a, model_b, n_splits=10) -> dict:
    """Paired t-test on per-fold accuracy to check if model_a beats model_b
    by more than chance, rather than comparing single mean scores."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores_a = cross_val_score(make_pipeline(model_a), X, y, cv=cv, scoring="accuracy")
    scores_b = cross_val_score(make_pipeline(model_b), X, y, cv=cv, scoring="accuracy")
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {
        "mean_a": scores_a.mean(), "mean_b": scores_b.mean(),
        "t_stat": t_stat, "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def baseline_scores(X: pd.DataFrame, y: pd.Series) -> dict:
    """Majority-class baseline: always predict the most frequent class."""
    majority_frac = y.value_counts(normalize=True).max()
    return {"strategy": "Majority-class baseline", "mean_accuracy": majority_frac, "std_accuracy": 0.0}
