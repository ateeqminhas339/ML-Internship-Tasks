# Task 03 — Feature Engineering Mastery — Report

Author: Ateeq (@ateeqminhas339)
Date: 2026-07-17

## 1. Executive Summary (≤5 bullets)

- Reduced `TotalCharges` skewness from **0.96 to -0.82** with a log1p transform, and engineered 3 interaction/ratio features plus 3 polynomial terms on `tenure` and `MonthlyCharges`.
- Built a KMeans-based `customer_segment` feature (k=4) and visualized both segments and churn in PCA-reduced 2D space to confirm meaningful structure.
- Compared three categorical encoding strategies (one-hot, frequency, smoothed target encoding) on `PaymentMethod` and justified the final choice (one-hot, given low cardinality).
- Ranked all engineered numeric features by mutual information with churn to guide feature selection.
- **Directly proved feature-engineering value**: an identical Logistic Regression improved from **ROC-AUC 0.836 → 0.848** and **F1 0.571 → 0.600** when moving from a minimal baseline feature set to the fully engineered set.

## 2. Business Problem & Framing

A clean dataset (Task 02) is necessary but not sufficient for a strong churn-prediction
model — raw columns often under-represent the signal actually driving customer
behaviour. This task's goal is to engineer features that make latent patterns (e.g.
"long-tenured but suddenly high-spending" customers, or coherent usage segments)
explicit and usable by a model, and — critically — to **prove** that this engineering
effort translates into measurable predictive improvement rather than just added
complexity. The comparison in Section 5 exists specifically to answer the question a
mentor or stakeholder would ask: "did the feature engineering actually help?"

## 3. Data Overview

| Property | Value |
|---|---|
| Source | Kaggle — Telco Customer Churn (IBM sample dataset), cleaned per Task 02 |
| Base shape (post-cleaning) | 7,043 rows x 20 columns |
| Target variable | `Churn` (Yes / No) |
| Base numeric features | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Engineered numeric features | `TotalCharges_log`, `num_services`, `avg_monthly_spend`, `tenure_x_monthly`, `charges_per_service`, `spend_to_tenure_ratio`, `tenure^2`, `MonthlyCharges^2`, `tenure x MonthlyCharges` |
| Engineered categorical feature | `customer_segment` (KMeans, k=4), `tenure_group` (binned), `has_internet` |
| Final feature count | 12 numeric + 19 categorical (post-encoding: one-hot expands categoricals further) |

## 4. Methodology

All feature engineering logic lives in reusable functions under `src/`, orchestrated
from `notebook.ipynb`:

1. **Numeric transformation** (`preprocessing.log_transform`) — log1p applied to the
   right-skewed `TotalCharges`, reducing skewness from 0.96 to -0.82.
2. **Interaction & ratio features** (`add_interaction_features`) — `tenure_x_monthly`
   (tenure x spend), `charges_per_service` (spend normalised by service count), and
   `spend_to_tenure_ratio` (an alternative smoothed spend rate).
3. **Polynomial features** (`add_polynomial_features`) — degree-2 expansion of
   `tenure` and `MonthlyCharges` via `sklearn.preprocessing.PolynomialFeatures`,
   adding squared terms and their interaction.
4. **Clustering-based feature** (`add_cluster_feature`) — KMeans (k=4) fit on
   standardized base numeric features, producing a `customer_segment` label;
   visualized against both segment and churn in PCA-reduced 2D space.
5. **Encoding comparison** (`frequency_encode`, `target_encode`) — one-hot (used in
   the final pipeline), frequency encoding, and smoothing-adjusted target (mean)
   encoding were computed side-by-side on `PaymentMethod` to demonstrate trade-offs;
   one-hot was selected for the final pipeline since all categorical features here are
   low-cardinality (2-4 levels), avoiding the target-leakage risk of mean encoding.
6. **Feature selection** (`mutual_information_ranking`) — mutual information between
   each engineered numeric feature and churn, to rank signal strength.
7. **Dimensionality reduction** (`pca_2d`) — PCA to 2 components for visualization
   only (not fed into the model), used to sanity-check that clusters and churn show
   visible structure in reduced space.
8. **Baseline vs. engineered comparison** (`modeling.compare_feature_sets`) — the
   same Logistic Regression architecture, same train/test split methodology
   (80/20, stratified, `random_state=42`), trained once on a minimal baseline feature
   set and once on the fully engineered set, to isolate the effect of feature
   engineering from any modelling choice.
9. **Feature importance** (`modeling.random_forest_feature_importance`) — a Random
   Forest through the same preprocessing pipeline, ranked by Gini importance, as a
   second, non-linear perspective on which features matter.

## 5. Results & Key Visualizations

| Metric | Baseline features | Engineered features | Change |
|---|---|---|---|
| Accuracy | 0.789 | 0.807 | +0.018 |
| Precision | 0.621 | 0.667 | +0.046 |
| Recall | 0.529 | 0.545 | +0.016 |
| F1 | 0.571 | 0.600 | +0.029 |
| ROC-AUC | 0.836 | 0.848 | +0.012 |

Top mutual-information features: `charges_per_service`, `tenure^2`, `tenure`,
`tenure x MonthlyCharges`, `tenure_x_monthly`. Top Random Forest importances:
`Contract_Month-to-month`, `tenure`, `tenure^2`, `charges_per_service`,
`OnlineSecurity_No`.

Figures generated in `figures/` (8 total, exceeding the 5-figure minimum):

1. `01_skew_before_after_log_transform.png` — TotalCharges distribution before/after log1p
2. `02_cluster_segments_pca.png` — KMeans customer segments in PCA-reduced space
3. `03_churn_pca_scatter.png` — churn vs. retained customers in the same PCA space
4. `04_target_encoding_by_payment_method.png` — smoothed target-encoded churn rate by payment method
5. `05_mutual_information_ranking.png` — mutual information of engineered features with churn
6. `06_engineered_correlation_matrix.png` — correlation matrix of all engineered numeric features
7. `07_baseline_vs_engineered_comparison.png` — baseline vs. engineered metrics, side by side
8. `08_random_forest_feature_importance.png` — top 15 Random Forest feature importances

## 6. Limitations & Risks

- `charges_per_service` and `spend_to_tenure_ratio` are both derived from
  `MonthlyCharges`/`TotalCharges` and `tenure`, so they are correlated with the base
  numeric features and with each other; this redundancy is visible in the correlation
  matrix (Figure 6) and should be pruned before any regularized or tree-based model
  that is sensitive to collinearity.
- KMeans cluster assignments are sensitive to the chosen `k` (set to 4 here based on a
  quick visual check) and to feature scaling; a more rigorous approach would sweep `k`
  with a silhouette or elbow analysis before finalizing the segment feature.
- Polynomial features were limited to degree 2 on two columns to avoid combinatorial
  explosion and overfitting risk; extending to more columns or a higher degree would
  need explicit regularization.
- The baseline-vs-engineered comparison uses a single Logistic Regression run (no
  cross-validation), so the reported improvement, while consistent with the mutual
  information and feature-importance rankings, should be confirmed with k-fold
  cross-validation before being treated as a stable result.
- Target encoding was demonstrated but not used in the final pipeline; if adopted in a
  future task with higher-cardinality categoricals, it must be fit only on the
  training fold (via cross-validation) to avoid target leakage.

## 7. Recommendation / Next Steps

- Adopt the engineered feature set (`data/processed/telco_feature_engineered.csv`) as
  the standard input for subsequent modelling tasks, given its confirmed, measurable
  improvement over the raw baseline.
- Run a proper k-fold cross-validated comparison (baseline vs. engineered) to confirm
  the improvement is stable, not an artifact of a single train/test split.
- Sweep KMeans `k` with silhouette analysis, and consider feeding the `customer_segment`
  feature into a future model as one-hot rather than a raw label if it proves useful.
- Prune or combine the most collinear engineered features (e.g.
  `spend_to_tenure_ratio` vs. `avg_monthly_spend`) before training regularization-
  sensitive models.
- Revisit target encoding with proper cross-validation folds if a future task
  introduces higher-cardinality categorical features where one-hot becomes impractical.

## 8. References

- Dataset: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Task 01 (EDA): `Week_1_Task/` in this repository
- Task 02 (Cleaning & Preprocessing): `Month_1_Foundations/Task_02_Advanced_Data_Cleaning_and_Preprocessing/` in this repository
- scikit-learn documentation: [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html), [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [mutual_info_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
