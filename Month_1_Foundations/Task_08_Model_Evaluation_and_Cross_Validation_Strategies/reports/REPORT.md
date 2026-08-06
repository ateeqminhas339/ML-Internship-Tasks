# Task 08 — Model Evaluation & Cross-Validation Strategies — Report
Author: Ateeq (@ateeqminhas339)
Date: 2026-08-06

## 1. Executive Summary
- Stratified K-Fold gave the most reliable estimate (86.7% accuracy, std 0.018) — lowest variance of the three strategies tested, because it holds disease prevalence constant across folds.
- Plain K-Fold underestimated and was noisier (84.3%, std 0.044); Leave-One-Out was similar on average (84.7%) but with far higher per-fold variance (std 0.36, since each "fold" is a single point).
- All scaling was fit inside each CV fold via a `Pipeline`, closing the exact leakage path (fitting a scaler on the full dataset before splitting) that plausibly explains the prior model's 8% overestimation.
- Logistic Regression vs. Random Forest showed no statistically significant difference (paired t-test, p = 1.0 on this dataset) — added model complexity bought nothing here, so the simpler model is preferable.
- Both models comfortably beat the majority-class baseline (50% accuracy) at ~85-87%, but with real Kaggle/UCI data these margins should be re-verified.

## 2. Business Problem & Framing
A hospital deploying a heart-disease prediction model previously overestimated accuracy by 8% due to flawed evaluation methodology, and the model failed after deployment, putting patients at risk. This task builds an evaluation pipeline specifically designed to prevent that failure mode: leak-free preprocessing, a comparison of CV strategies to show how much the evaluation method itself moves the reported number, learning/validation curves to diagnose bias vs. variance, and a statistical significance test before declaring one model "better" than another.

## 3. Data Overview
Placeholder run: 300 patients, 14 columns (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia type, target). Target was binarized (0 = no disease, 1 = disease present) from the UCI dataset's 0-4 severity scale, and the placeholder data is balanced 50/50. **This will differ once the real UCI/Kaggle CSV replaces the placeholder — re-check the class balance, since the real dataset is typically closer to ~54/46.**

## 4. Methodology
1. **Cleaning:** dropped duplicates and rows missing any required feature; binarized `target`.
2. **Leak-free preprocessing:** `StandardScaler` is wrapped in a `sklearn.Pipeline` with the classifier, so it's refit on the training portion of every single CV fold rather than once on the whole dataset — this is the single most important fix relative to the prior model's failure.
3. **CV strategy comparison:** the same Logistic Regression model evaluated under K-Fold, Stratified K-Fold, and Leave-One-Out, all at accuracy, to see how much the estimate moves with the strategy alone.
4. **Learning curve:** training vs. validation accuracy across increasing training-set sizes, to diagnose whether the model is high-bias (underfit) or high-variance (overfit).
5. **Validation curve:** training vs. validation accuracy across Logistic Regression's regularization strength `C`, to check the model isn't over/under-regularized.
6. **Statistical comparison:** paired t-test on matched per-fold accuracy scores (Logistic Regression vs. Random Forest, 10-fold Stratified CV) — comparing means alone can't say whether a difference is real or noise.
7. **Baseline:** majority-class classifier (predict the more common class every time).
8. **Final diagnostics:** ROC curve and confusion matrix built entirely from out-of-fold predictions (`cross_val_predict`), so no prediction was made by a model that saw that row during training.

## 5. Results & Key Visualizations
| CV Strategy | Mean Accuracy | Std | Folds |
|---|---|---|---|
| K-Fold (k=5) | 84.3% | 0.044 | 5 |
| Stratified K-Fold (k=5) | 86.7% | 0.018 | 5 |
| Leave-One-Out | 84.7% | 0.360 | 300 |

**Statistical comparison (Logistic Regression vs. Random Forest, paired t-test):** mean accuracy 85.0% vs. 85.0%, p = 1.00 — not statistically significant. Prefer the simpler, more interpretable Logistic Regression for a clinical setting.

**Baseline:** majority-class accuracy 50.0% vs. Logistic Regression (Stratified K-Fold) 86.7% — a meaningful, non-trivial lift over guessing.

Figures: `figures/01_feature_distributions.png`, `02_correlation_heatmap.png`, `03_cv_strategy_comparison.png`, `04_learning_curve.png`, `05_validation_curve.png`, `06_roc_and_confusion_matrix.png`.

## 6. Limitations & Risks
- **These numbers come from placeholder/synthetic data**, generated only to prove the pipeline runs end-to-end; they must be regenerated on the real UCI/Kaggle Heart Disease CSV before any of these figures go into a real report.
- ~300 rows is small for a hospital-scale deployment decision; confidence intervals on accuracy stay fairly wide, especially per-fold.
- Leave-One-Out is shown for comparison only — its per-fold variance and computational cost make it impractical as the default strategy at larger scale.
- The paired t-test assumes fold scores are roughly independent and normally distributed; with only 10 folds this is an approximation.
- Only Logistic Regression and Random Forest were compared; other families (SVM, gradient boosting) were out of scope for this task.
- Accuracy alone can be misleading for clinical screening; sensitivity/recall on the disease-positive class matters more in a hospital context and should be weighted accordingly in a production decision (the ROC/confusion matrix in Section 9 of the notebook is a starting point, not the final word).

## 7. Recommendation / Next Steps
Adopt Stratified K-Fold as the default evaluation strategy for this problem — it gave the lowest-variance, most representative estimate. Prefer Logistic Regression over Random Forest given no significant accuracy difference and its clinical interpretability advantage. Before any production decision: (1) re-run this entire pipeline on the real UCI/Kaggle data, (2) report sensitivity/specificity alongside accuracy given the clinical cost of false negatives, (3) validate on an external/held-out hospital dataset if available, since even correct CV can't fully substitute for out-of-population validation.

## 8. References
- Dataset: Heart Disease, UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/45/heart+disease
- scikit-learn documentation: KFold, StratifiedKFold, LeaveOneOut, learning_curve, validation_curve, Pipeline
- scipy.stats.ttest_rel documentation, for the paired significance test
