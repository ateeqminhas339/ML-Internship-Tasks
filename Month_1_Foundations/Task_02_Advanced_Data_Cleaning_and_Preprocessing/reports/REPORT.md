# Task 02 — Advanced Data Cleaning & Preprocessing — Report

Author: Ateeq (@ateeqminhas339)
Date: 2026-07-14

## 1. Executive Summary (≤5 bullets)

- Fixed the `TotalCharges` structural type issue (11 blank values, all belonging to `tenure == 0` customers) by imputing with `0` rather than a global mean.
- Confirmed the dataset has **0 fully duplicated rows** and **0 duplicate customer IDs**.
- Ran IQR-based outlier detection across all numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) — **0 outliers found**; a capping (winsorization) step is included defensively for future data batches.
- Engineered 4 new features (`tenure_group`, `num_services`, `avg_monthly_spend`, `has_internet`) and built a reusable `ColumnTransformer` pipeline (scaling + one-hot encoding).
- Verified the fully processed data is ML-ready with a baseline Logistic Regression, achieving **ROC-AUC ≈ 0.84** on a held-out, stratified test set.

## 2. Business Problem & Framing

Telecom providers lose recurring revenue every time a customer churns, and acquiring a
new customer is significantly more expensive than retaining an existing one. Before any
churn-prediction model can be trusted, the underlying data must be verified as clean,
consistent, and correctly typed — otherwise the model risks learning from noise or
structural artifacts rather than genuine behavioural signal. This task treats data
cleaning and preprocessing as a first-class deliverable in its own right: the goal is a
dataset and a reusable pipeline that any downstream model (this task's baseline, or a
future, more sophisticated one) can consume safely and reproducibly.

## 3. Data Overview

| Property | Value |
|---|---|
| Source | Kaggle — Telco Customer Churn (IBM sample dataset) |
| Raw shape | 7,043 rows x 21 columns |
| Target variable | `Churn` (Yes / No) |
| Numeric features | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical features | 17 (demographics, subscribed services, contract & billing details) |
| Known structural issue | `TotalCharges` read as text; 11 blank values, all at `tenure == 0` |
| Duplicate rows | 0 |

## 4. Methodology

The cleaning and preprocessing pipeline was implemented as reusable Python modules
under `src/`, orchestrated from `notebook.ipynb`:

1. **Structural cleaning** (`preprocessing.fix_total_charges`, `drop_identifier_column`,
   `drop_duplicate_rows`) — coerce `TotalCharges` to numeric, impute the 11 structurally
   missing values with `0`, drop the non-predictive `customerID` column, and confirm no
   duplicate rows exist.
2. **Outlier detection & treatment** (`detect_outliers_iqr`, `cap_outliers_iqr`) — apply
   the standard 1.5xIQR rule to each numeric feature and cap (winsorize) any outliers
   found, preserving sample size rather than dropping rows.
3. **Feature engineering** (`engineer_features`) — derive `tenure_group` (binned
   tenure), `num_services` (count of subscribed add-on services), `avg_monthly_spend`
   (TotalCharges normalised by tenure), and `has_internet` (binary flag).
4. **Encoding & scaling pipeline** (`build_preprocessing_pipeline`) — a single
   `sklearn.compose.ColumnTransformer` combining `StandardScaler` for numeric features
   and `OneHotEncoder` for categorical features, so the identical transformation can be
   reapplied to new data without leakage.
5. **Stratified train/test split** (`split_data`) — an 80/20 split stratified on the
   (imbalanced, ~27% positive) target.
6. **Baseline model sanity check** (`modeling.build_baseline_pipeline`,
   `evaluate_baseline`) — a Logistic Regression trained purely to confirm the processed
   data trains and predicts without error; this is not a tuned or final model.

## 5. Results & Key Visualizations

| Metric | Value |
|---|---|
| Duplicate rows removed | 0 |
| IQR outliers detected (tenure / MonthlyCharges / TotalCharges) | 0 / 0 / 0 |
| Engineered features added | 4 |
| Train / test rows | 5,634 / 1,409 |
| Train / test churn rate | ~26.5% / ~26.5% (preserved via stratification) |
| Baseline accuracy | 0.80 |
| Baseline precision | 0.66 |
| Baseline recall | 0.52 |
| Baseline F1 | 0.58 |
| Baseline ROC-AUC | 0.84 |

Figures generated in `figures/` (7 total, exceeding the 5-figure minimum):

1. `01_missing_data_before_after.png` — missing/invalid `TotalCharges` count before vs. after cleaning
2. `02_boxplots_before_outlier_treatment.png` — numeric feature boxplots pre-treatment
3. `03_boxplots_after_outlier_treatment.png` — numeric feature boxplots post-treatment
4. `04_engineered_feature_distributions.png` — `tenure_group` and `num_services` distributions
5. `05_correlation_engineered_features.png` — correlation matrix including engineered features
6. `06_train_test_churn_balance.png` — churn rate preserved across the stratified split
7. `07_baseline_model_metrics.png` — baseline Logistic Regression test-set metrics

## 6. Limitations & Risks

- The IQR rule found no outliers because Telco's numeric features are naturally
  bounded (contracts run 0–72 months with realistic charge ranges); this pipeline step
  is retained for robustness against future or production data that may not be as
  well-behaved.
- `avg_monthly_spend` divides by tenure (clipped at a minimum of 1 to avoid
  division-by-zero for brand-new customers), which slightly compresses the signal for
  `tenure == 0` customers — worth revisiting if this feature proves important in later
  modelling.
- The baseline Logistic Regression is a sanity check only, not a tuned model: no
  hyperparameter search, cross-validation, or class-imbalance handling (e.g. SMOTE,
  class weights) was applied, so its metrics should not be treated as a performance
  ceiling.
- `tenure` and `TotalCharges` remain highly correlated (see Task 01); this
  multicollinearity is preserved in the processed dataset and should be addressed
  explicitly in future feature-selection work.

## 7. Recommendation / Next Steps

- Use the exported `data/processed/telco_cleaned_engineered.csv` and the
  `ColumnTransformer` pipeline in `src/preprocessing.py` as the standard input for all
  subsequent modelling tasks, to guarantee consistency.
- In a future task, address class imbalance explicitly (e.g. class weighting, SMOTE)
  and run a proper model comparison (Logistic Regression, Random Forest, Gradient
  Boosting) with cross-validation and hyperparameter tuning.
- Consider dropping or combining `tenure` and `TotalCharges` before training more
  complex models, given their strong collinearity.
- Re-run the outlier-detection step whenever new data batches are ingested, since the
  current all-zero outlier result is specific to this static Kaggle snapshot.

## 8. References

- Dataset: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Task 01 (EDA): `Month_1_Foundations/Task_01_.../` in this repository
- scikit-learn documentation: [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
