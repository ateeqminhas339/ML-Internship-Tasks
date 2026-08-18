"""
Task 9: Wine Quality Prediction - Hyperparameter Tuning Comparison
--------------------------------------------------------------------
Compares 5 approaches on the Wine Quality (red) dataset:
    1. Baseline Random Forest (no tuning)
    2. GridSearchCV
    3. RandomizedSearchCV
    4. Optuna (Bayesian Optimization + Pruning)
    5. AutoML-style search across multiple model families

Each method is evaluated on test-set Accuracy and wall-clock Runtime,
then summarized in a comparison table.
"""

import time
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DATA_PATH = "dataset/winequality-red.csv"


# ---------------------------------------------------------------------
# 1. Load & Preprocess Data
# ---------------------------------------------------------------------
def load_and_preprocess_data(path: str):
    print("📂 Loading dataset...")
    df = pd.read_csv(path, sep=None, engine="python")

    print(f"   Initial shape: {df.shape}")

    # Remove duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"   Removed {before - df.shape[0]} duplicate rows")

    # No missing values expected, but guard anyway
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print("   Dropped rows with missing values")

    # Convert quality score (0-10) into 3 classes
    def quality_to_class(q):
        if q <= 4:
            return 0  # low
        elif q <= 6:
            return 1  # medium
        else:
            return 2  # high

    df["quality_class"] = df["quality"].apply(quality_to_class)

    X = df.drop(columns=["quality", "quality_class"])
    y = df["quality_class"]

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    print(f"   Class distribution (train): {dict(y_train.value_counts())}\n")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ---------------------------------------------------------------------
# 2. Baseline Model
# ---------------------------------------------------------------------
def run_baseline(X_train, X_test, y_train, y_test):
    print("🔹 Running Baseline Random Forest (no tuning)...")
    start = time.time()

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    elapsed = time.time() - start
    print(f"   Accuracy: {acc:.4f} | Time: {elapsed:.2f}s\n")
    return acc, elapsed


# ---------------------------------------------------------------------
# 3. GridSearchCV
# ---------------------------------------------------------------------
def run_grid_search(X_train, X_test, y_train, y_test):
    print("🔹 Running GridSearchCV...")
    start = time.time()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    preds = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, preds)

    elapsed = time.time() - start
    print(f"   Best params: {grid.best_params_}")
    print(f"   Accuracy: {acc:.4f} | Time: {elapsed:.2f}s\n")
    return acc, elapsed


# ---------------------------------------------------------------------
# 4. RandomizedSearchCV
# ---------------------------------------------------------------------
def run_random_search(X_train, X_test, y_train, y_test):
    print("🔹 Running RandomizedSearchCV...")
    start = time.time()

    param_dist = {
        "n_estimators": np.arange(50, 300, 25),
        "max_depth": [None, 5, 10, 15, 20, 25],
        "min_samples_split": np.arange(2, 10),
        "min_samples_leaf": np.arange(1, 5),
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    preds = search.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, preds)

    elapsed = time.time() - start
    print(f"   Best params: {search.best_params_}")
    print(f"   Accuracy: {acc:.4f} | Time: {elapsed:.2f}s\n")
    return acc, elapsed


# ---------------------------------------------------------------------
# 5. Optuna (Bayesian Optimization + Pruning)
# ---------------------------------------------------------------------
def run_optuna_search(X_train, X_test, y_train, y_test, n_trials=30):
    print("🔹 Running Optuna Bayesian Optimization (with pruning)...")
    start = time.time()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, model.predict(X_val))

        trial.report(val_acc, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_acc

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    model = RandomForestClassifier(random_state=RANDOM_STATE, **best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    elapsed = time.time() - start
    print(f"   Best params: {best_params}")
    print(f"   Completed trials: {len(study.trials)} "
          f"(pruned: {sum(t.state.name == 'PRUNED' for t in study.trials)})")
    print(f"   Accuracy: {acc:.4f} | Time: {elapsed:.2f}s\n")
    return acc, elapsed


# ---------------------------------------------------------------------
# 6. AutoML-style Search (multiple model families)
# ---------------------------------------------------------------------
def run_automl_style_search(X_train, X_test, y_train, y_test):
    print("🔹 Running AutoML-style search across model families...")
    start = time.time()

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVC": SVC(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"   {name:<20} Accuracy: {acc:.4f}")

    best_name = max(results, key=results.get)
    best_acc = results[best_name]

    elapsed = time.time() - start
    print(f"   🏆 Best model: {best_name} | Accuracy: {best_acc:.4f} | Time: {elapsed:.2f}s\n")
    return best_acc, elapsed


# ---------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    results = {}
    results["Baseline"] = run_baseline(X_train, X_test, y_train, y_test)
    results["GridSearchCV"] = run_grid_search(X_train, X_test, y_train, y_test)
    results["RandomizedSearchCV"] = run_random_search(X_train, X_test, y_train, y_test)
    results["Optuna (Bayesian + Pruning)"] = run_optuna_search(X_train, X_test, y_train, y_test)
    results["AutoML-style Search"] = run_automl_style_search(X_train, X_test, y_train, y_test)

    print("=" * 55)
    print("📊 FINAL COMPARISON")
    print("=" * 55)
    summary_df = pd.DataFrame(
        [(method, acc, round(t, 2)) for method, (acc, t) in results.items()],
        columns=["Method", "Accuracy", "Time (seconds)"],
    ).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    print(summary_df.to_string(index=False))
    summary_df.to_csv("results_summary.csv", index=False)
    print("\n✅ Results saved to results_summary.csv")


if __name__ == "__main__":
    main()
