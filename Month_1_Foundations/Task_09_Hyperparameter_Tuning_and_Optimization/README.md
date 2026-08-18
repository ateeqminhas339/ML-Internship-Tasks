# Wine Quality Prediction - Hyperparameter Tuning Project (Task 9)

## 📌 Project Overview
This project builds a machine learning workflow to predict wine quality (red wine) from chemical features.
The main goal of this task is to **demonstrate and compare multiple hyperparameter tuning methods** and understand the trade-off between:
- **Search speed**
- **Search quality (accuracy)**

We compare 4 tuning approaches:

1. **Baseline model** (no tuning)
2. **GridSearchCV**
3. **RandomizedSearchCV**
4. **Optuna Bayesian Optimization** with **early stopping / pruning**
5. **AutoML-style search** (small search over multiple model types)

---

## 🎯 Objectives
- Load and clean the wine quality dataset
- Prepare data for machine learning
- Train a baseline Random Forest model
- Improve performance using:
  - GridSearchCV
  - RandomizedSearchCV
  - Optuna (Bayesian + pruning)
  - Simple AutoML-style candidate model search
- Compare all methods using:
  - **Accuracy**
  - **Runtime (seconds)**

---

## 📊 Dataset
- File: `dataset/winequality-red.csv`
- Source: UCI Machine Learning Repository (Wine Quality dataset)
- Size:
  - **1599 rows**
  - **12 columns** (11 input features + 1 output column)

### Input Features (11)
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

### Target
- `quality` (0 to 10 score from tasters)

---

## 🧼 Data Preprocessing
1. **Missing values**
   - Dataset contains no missing values (no row removal needed).
2. **Duplicate removal**
   - Duplicates are removed using `drop_duplicates()`.
3. **Convert quality score into 3 classes**
   - `0 = low quality` (quality 3–4)
   - `1 = medium quality` (quality 5–6)
   - `2 = high quality` (quality 7–8)
4. **Feature scaling**
   - `StandardScaler()` applied to all input features.
5. **Train-test split**
   - 80% training, 20% testing
   - Uses `random_state=42` and `stratify=y` to keep class distribution.

---

## 🧠 Models and Tuning Methods Compared

### 1) Baseline (No Tuning)
- Random Forest with default hyperparameters

### 2) GridSearchCV
- Exhaustive search over a defined hyperparameter grid

### 3) RandomizedSearchCV
- Random sampling of hyperparameters from a distribution

### 4) Optuna (Bayesian Optimization + Pruning)
- Uses:
  - **TPESampler** (Bayesian optimization)
  - **MedianPruner** (prunes weak trials early)
- Reduces computation time by skipping poor configurations

### 5) AutoML-style Search
- Tries a small list of different ML models automatically
- Selects the best based on test accuracy
- Illustrates that sometimes switching model types beats tuning one model deeply

---

## ▶️ How to Run

### Option A: Run using Python
```bash
pip install -r requirements.txt
python app.py
```

### Option B: Run using Docker
```bash
docker build -t wine-quality-tuning .
docker run --rm wine-quality-tuning
```

> Make sure the dataset exists at:
> `dataset/winequality-red.csv`

---

## 📌 Output / Results
The script prints progress messages and a final table (also saved to `results_summary.csv`):

| Method | Accuracy | Time (seconds) |
|--------|----------|------------------|

Example summary (your exact results may vary slightly due to randomness):
- Baseline accuracy serves as a starting point
- Grid and Random tuning may or may not improve accuracy depending on search space
- Optuna typically reaches higher accuracy with fewer trials but takes longer
- AutoML-style search may achieve best accuracy quickly by trying multiple model families

---

## ✅ Learning Outcomes
By completing Task 9, we practice:

- Dataset cleaning and label engineering
- Feature scaling for ML pipelines
- Differences between GridSearchCV and RandomizedSearchCV
- Bayesian optimization concepts (Optuna)
- Early stopping / pruning with Optuna
- AutoML-style idea of trying multiple model types
- Computational budgeting: balancing runtime vs accuracy improvements

---

## 📝 Notes
- Runtime depends on CPU resources and the chosen number of trials/search sizes.
- Accuracy depends on train/test split randomness (fixed with `random_state=42` here).

---

## 📚 Files in Repository
- `app.py` — main pipeline script
- `requirements.txt` — dependencies
- `Dockerfile` — container setup
- `dataset/winequality-red.csv` — input dataset (add this file yourself before running)
