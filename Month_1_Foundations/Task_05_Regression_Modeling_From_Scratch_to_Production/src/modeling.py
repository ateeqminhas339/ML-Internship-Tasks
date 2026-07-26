"""Regression modeling: from scratch (gradient descent) through to a
production-ready, serialized scikit-learn pipeline.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .preprocessing import RANDOM_SEED, build_preprocessing_pipeline


# --------------------------------------------------------------------------
# "From scratch" — plain-NumPy linear regression via gradient descent
# --------------------------------------------------------------------------
class LinearRegressionFromScratch:
    """Multiple linear regression trained with batch gradient descent,
    implemented directly in NumPy (no sklearn) to demonstrate the
    underlying mechanics: y_hat = X @ w + b, minimizing mean squared error.
    """

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        y = np.asarray(y)

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error)
            grad_b = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


# --------------------------------------------------------------------------
# Evaluation metrics
# --------------------------------------------------------------------------
def evaluate_regression(y_true, y_pred) -> dict:
    """Standard regression metrics: RMSE, MAE, R-squared."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Production candidate models
# --------------------------------------------------------------------------
def get_candidate_models() -> dict:
    """Candidate sklearn regressors to compare against the from-scratch
    baseline, ranging from simple (linear) to more flexible (ensembles).
    """
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=RANDOM_SEED),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }


def build_pipeline(numeric_cols, categorical_cols, model) -> Pipeline:
    """Wrap the preprocessing ColumnTransformer and a regressor into a
    single sklearn Pipeline - this exact object is what gets serialized
    and served in production.
    """
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def compare_models(numeric_cols, categorical_cols, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Train every candidate model through the identical preprocessing
    pipeline and return a comparison table of test-set metrics.
    """
    rows = []
    fitted_pipelines = {}
    for name, model in get_candidate_models().items():
        pipe = build_pipeline(numeric_cols, categorical_cols, model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate_regression(y_test, preds)
        metrics["model"] = name
        rows.append(metrics)
        fitted_pipelines[name] = pipe
    results_df = pd.DataFrame(rows)[["model", "rmse", "mae", "r2"]].sort_values("rmse")
    return results_df, fitted_pipelines


# --------------------------------------------------------------------------
# Production serialization
# --------------------------------------------------------------------------
def save_model(pipeline: Pipeline, path: str = "api/model.joblib"):
    """Serialize the fitted, best-performing pipeline for production
    serving (loaded directly by the FastAPI app in api/app.py).
    """
    joblib.dump(pipeline, path)
    return path


def load_model(path: str = "api/model.joblib") -> Pipeline:
    """Load a previously serialized production pipeline."""
    return joblib.load(path)
