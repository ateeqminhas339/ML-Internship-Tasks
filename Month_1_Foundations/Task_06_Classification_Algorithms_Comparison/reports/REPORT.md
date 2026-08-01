# Task 06 — Classification Algorithms Comparison — Report
Author: Mussa Khan (@musagithub1)
Date: 2026-08-01

## 1. Executive Summary
- Compared 6 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM) for predicting Telco customer churn on a shared preprocessing pipeline.
- **Logistic Regression and Gradient Boosting are effectively tied for best performance**, both reaching ROC-AUC ≈ 0.845–0.847, clearly ahead of the rest.
- **Decision Tree performed worst** (ROC-AUC 0.645), showing strong overfitting on this dataset.
- Logistic Regression is recommended for production: comparable accuracy to Gradient Boosting, ~15x faster to train, and fully interpretable coefficients.
- 5-fold stratified cross-validation confirms the single-split ranking is stable, not a lucky split.

## 2. Business Problem & Framing
Telco companies lose significant recurring revenue to customer churn. Being
able to flag at-risk customers before they leave lets the retention team
intervene (discounts, service fixes, proactive support) before cancellation.
This task frames churn prediction as a binary classification problem
(`Churn_binary`: 1 = churned, 0 = retained) and asks which off-the-shelf
algorithm gives the best accuracy/interpretability/speed trade-off for a
production deployment.

## 3. Data Overview
- Source: Telco Customer Churn dataset, feature-engineered in Task 03 (7,043 customers, 33 columns).
- Target: `Churn_binary` — 26.5% churn rate (moderately imbalanced).
- Features used: 31 predictors spanning demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account info (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`), services (`InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, etc.), billing (`MonthlyCharges`, `TotalCharges`), and Task 03's engineered features (`tenure_group`, `num_services`, `avg_monthly_spend`, `customer_segment`, `tenure_x_monthly`, `charges_per_service`, etc.).
- Split: 80/20 stratified train/test split (random_state=42), preserving the 26.5% churn rate in both sets.

## 4. Methodology
1. Loaded the Task 03 engineered dataset and dropped the raw `Churn` label and target column from the feature set.
2. Built one shared `ColumnTransformer` (`src/preprocessing.py`): `StandardScaler` for numeric columns, `OneHotEncoder(handle_unknown="ignore")` for categorical columns — identical preprocessing for every model to keep the comparison fair.
3. Wrapped each of the six candidate estimators (`src/modeling.py`) in an `sklearn.Pipeline` with that same preprocessor, so scaling/encoding is fit only on the training fold (no leakage).
4. Trained and evaluated every model on the identical train/test split, recording Accuracy, Precision, Recall, F1, ROC-AUC, and wall-clock training time.
5. Ran 5-fold stratified cross-validation (`cross_validate_model`) on ROC-AUC to check the single-split ranking wasn't due to a favorable split.
6. Inspected the confusion matrix of the best model and Random Forest's feature importances to understand *what* is driving predictions, not just *how well* the model scores.

## 5. Results & Key Visualizations

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.664 | 0.535 | 0.593 | **0.847** | 0.15 |
| Gradient Boosting | 0.809 | 0.687 | 0.516 | 0.589 | 0.846 | 2.45 |
| Random Forest | 0.779 | 0.605 | 0.484 | 0.538 | 0.822 | 3.40 |
| K-Nearest Neighbors | 0.779 | 0.592 | 0.543 | 0.566 | 0.820 | 0.04 |
| SVM (RBF) | 0.798 | 0.662 | 0.487 | 0.561 | 0.800 | 8.47 |
| Decision Tree | 0.720 | 0.472 | 0.479 | 0.475 | 0.645 | 0.12 |

Figures (see `figures/`):
1. `01_class_distribution.png` — churn class imbalance (26.5% churn).
2. `02_metric_comparison.png` — grouped bar chart of all 5 metrics across all 6 models.
3. `03_roc_curves.png` — ROC curves for all classifiers on one plot.
4. `04_confusion_matrix_best_model.png` — confusion matrix for the best model (Logistic Regression).
5. `05_training_time_comparison.png` — wall-clock training time per model.
6. `06_feature_importance_rf.png` — top 15 Random Forest feature importances (contract type, tenure, and internet service consistently dominate).

## 6. Limitations & Risks
- Recall is moderate for every model (48–54%), meaning roughly half of actual churners are still missed at the default 0.5 threshold — for a retention campaign, the classification threshold should likely be lowered to trade some precision for higher recall.
- Hyperparameters were left mostly at sensible defaults (with light tuning: `n_estimators=300` for Random Forest, `n_neighbors=15` for KNN); a dedicated hyperparameter search (e.g., `GridSearchCV`/`Optuna`) could shift the ranking, especially for SVM and Gradient Boosting.
- The dataset is static and reflects one snapshot in time; churn drivers can shift with pricing/plan changes, so the model should be retrained periodically on fresh data.
- Class imbalance (26.5% positive) was not explicitly corrected (no SMOTE/class-weighting); this may be limiting recall across all models.

## 7. Recommendation / Next Steps
- **Ship Logistic Regression** as the production churn classifier: best/tied ROC-AUC, cheapest to train and serve, and coefficients are directly explainable to non-technical stakeholders (e.g., "month-to-month contract increases churn odds by X").
- Before deployment, tune the decision threshold against a recall target set by the retention team's capacity (e.g., "correctly flag 70% of churners") rather than defaulting to 0.5.
- As a follow-up experiment, add class-weighting or SMOTE resampling and re-run this same comparison to see if recall improves without sacrificing precision.
- If interpretability constraints loosen later, Gradient Boosting is the fallback candidate given its near-identical ROC-AUC.

## 8. References
- Dataset: Telco Customer Churn (IBM sample dataset), engineered in Task 03 — Feature Engineering Mastery.
- scikit-learn documentation: https://scikit-learn.org/stable/
