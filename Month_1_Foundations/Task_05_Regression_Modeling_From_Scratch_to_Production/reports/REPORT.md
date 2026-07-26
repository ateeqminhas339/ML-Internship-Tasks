# Task 05 — Regression Modeling: From Scratch to Production — Report

Author: Ateeq (@ateeqminhas339)
Date: 2026-07-26

## 1. Executive Summary (≤5 bullets)

- Implemented multiple linear regression **from scratch** using batch gradient descent in
  plain NumPy; it achieves R² = 0.887, nearly identical to scikit-learn's closed-form
  `LinearRegression` (R² = 0.886), confirming the from-scratch implementation is correct.
- Compared 4 production candidate models through an identical preprocessing pipeline;
  **GradientBoostingRegressor** performed best (RMSE $4,257, R² = 0.901), followed closely
  by RandomForestRegressor.
- Engineered a `smoker_bmi_interaction` feature, capturing the well-documented compounding
  cost effect of obesity combined with smoking.
- Serialized the best model and served it through a **FastAPI** production app with
  `/predict` and `/health` endpoints, tested locally with both smoker and non-smoker inputs.
- **Containerized the API with Docker**, producing a reproducible, portable deployment
  artifact (`Dockerfile` + `.dockerignore`) that runs identically on any machine with
  Docker installed.

## 2. Business Problem & Framing

Following on from Task 04's finding that smoking, age, BMI, and children count are
legitimate, statistically-supported drivers of insurance charges, this task turns that
statistical understanding into an actual predictive tool the business can use: given a
new applicant's attributes, what should their expected annual charge be? The task
requires building this "from scratch" first (to demonstrate the model isn't a black box)
and then "to production" (a served, containerized API), since a model that only exists in
a notebook cannot be integrated into an underwriting system.

## 3. Data Overview

| Property | Value |
|---|---|
| Source | Kaggle — Medical Cost Personal Datasets (same as Task 04) |
| Cleaned shape | 1,337 rows x 8 columns (1 duplicate dropped, `smoker_bmi_interaction` added) |
| Target variable | `charges` (continuous, USD) |
| Numeric features | `age`, `bmi`, `children`, `smoker_bmi_interaction` |
| Categorical features | `sex`, `smoker`, `region` |
| Train / test split | 1,069 / 268 rows (80/20, `random_state=42`) |

## 4. Methodology

1. **Cleaning** (`data_loader.clean_data`) — identical to Task 04: drop the single exact
   duplicate row, standardize categorical text casing.
2. **Feature engineering** (`preprocessing.add_interaction_features`) — added
   `smoker_bmi_interaction` = `(smoker == "yes") x bmi`, since Task 04's exploratory plots
   and prior literature on this dataset show smoking and BMI compound rather than act
   independently on charges.
3. **From-scratch model** (`modeling.LinearRegressionFromScratch`) — batch gradient
   descent over the standardized/one-hot-encoded feature matrix, minimizing mean squared
   error directly via NumPy matrix operations (no sklearn `fit()` call involved), trained
   for 2,000 iterations at a learning rate of 0.1.
4. **Production candidates** (`modeling.get_candidate_models`, `compare_models`) — Linear
   Regression, Ridge (alpha=1.0), Random Forest (300 trees, max depth 6), and Gradient
   Boosting, each wrapped in an identical `ColumnTransformer` preprocessing pipeline
   (`StandardScaler` for numeric, `OneHotEncoder` for categorical) to ensure a fair
   comparison - only the final regressor differs between candidates.
5. **Evaluation** (`modeling.evaluate_regression`) — RMSE, MAE, and R-squared on the
   held-out 20% test split for every model.
6. **Serialization** (`modeling.save_model` / `load_model`) — the best-performing fitted
   pipeline (preprocessing + regressor as one object) is saved with `joblib` to
   `api/model.joblib`, verified to produce identical predictions after a reload.
7. **Production API** (`api/app.py`) — a FastAPI app that loads `model.joblib` once at
   startup and exposes `POST /predict` (accepting age/sex/bmi/children/smoker/region as
   JSON) and `GET /health`; tested locally with `uvicorn` against both smoker and
   non-smoker example payloads.
8. **Containerization** (`Dockerfile`, `.dockerignore`) — a `python:3.11-slim` base image
   installs only the API's minimal runtime dependencies (`api/requirements.txt`, separate
   from the heavier notebook/dev `requirements.txt`), copies in `api/app.py` and the
   pre-trained `model.joblib`, and runs `uvicorn` on port 8000.

## 5. Results & Key Visualizations

| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| GradientBoostingRegressor | 4,256.61 | 2,434.20 | 0.9014 |
| RandomForestRegressor | 4,284.82 | 2,407.04 | 0.9001 |
| Ridge | 4,554.13 | 2,821.18 | 0.8871 |
| LinearRegression | 4,572.81 | 2,828.97 | 0.8862 |
| **From-scratch (NumPy gradient descent)** | 4,553.64 | 2,816.72 | 0.8872 |

**API validation (local test, before containerizing):**

| Input | Predicted charges |
|---|---|
| Age 35, male, BMI 28.5, 2 children, non-smoker, southeast | $6,881.47 |
| Age 35, male, BMI 28.5, 2 children, **smoker**, southeast | $22,349.24 |

The ~$15,468 gap between otherwise-identical smoker/non-smoker predictions is consistent
with Task 04's OLS finding (~$23,800 average effect) and confirms the model has learned
the smoking signal correctly, even after moving through the full pipeline into a live API
call.

Figures generated in `figures/` (5 total, meeting the minimum requirement):

1. `01_gradient_descent_loss_curve.png` — from-scratch training loss convergence
2. `02_model_comparison_error.png` — RMSE/MAE across all 4 production candidates
3. `03_model_comparison_r2.png` — R-squared across all 4 production candidates
4. `04_residuals_and_fit.png` — residuals vs. predicted, and predicted vs. actual (best model)
5. `05_feature_importance.png` — Gradient Boosting feature importances

## 6. Limitations & Risks

- The from-scratch implementation uses plain batch gradient descent with a fixed
  learning rate and iteration count; it is a pedagogical implementation, not a
  production-grade optimizer (no learning rate scheduling, no convergence tolerance
  check, no regularization) - it is used for verification against sklearn, not for
  the production model itself.
- GradientBoostingRegressor and RandomForestRegressor were not hyperparameter-tuned
  (default/lightly-set parameters only); the reported metrics represent a reasonable
  baseline for each model family, not their best possible performance.
- The FastAPI app has no input-range validation beyond Pydantic's basic type/bound
  checks, no authentication, and no logging/monitoring - all of which a real production
  deployment would need before serving external traffic.
- The Docker image was built and validated for correctness of the Dockerfile syntax and
  local API behavior, but was not tested inside an actual Docker daemon in this
  environment (no Docker available) - the user should verify `docker build`/`docker run`
  succeed on their own machine before considering this deployment-verified end-to-end.
- The model was trained on a single, relatively small (1,337-row) historical dataset;
  before real underwriting use, it would need validation against a larger, more current,
  and jurisdiction-appropriate dataset, plus the regulatory review flagged in Task 04
  regarding which factors may legally be used for pricing.

## 7. Recommendation / Next Steps

- Promote `GradientBoostingRegressor` as the production model given its lowest RMSE and
  highest R², but run a proper hyperparameter search (grid/random search with
  cross-validation) before final deployment.
- Add authentication, request logging, and basic monitoring (e.g. prediction distribution
  drift) to `api/app.py` before exposing it beyond internal testing.
- Push the built Docker image to a container registry (e.g. Docker Hub, GitHub Container
  Registry) and add a CI step that rebuilds and tests it automatically on every commit.
- Cross-reference with Task 04's regulatory finding: ensure `region` is not silently
  reintroduced as a pricing factor in any future retraining, given the lack of
  statistical support found there.

## 8. References

- Dataset: [Medical Cost Personal Datasets — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Task 04 (Statistical Analysis): `Month_1_Foundations/Task_04_Statistical_Analysis_and_Hypothesis_Testing/` in this repository
- scikit-learn documentation: [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- FastAPI documentation: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- Docker documentation: [docs.docker.com](https://docs.docker.com/)
