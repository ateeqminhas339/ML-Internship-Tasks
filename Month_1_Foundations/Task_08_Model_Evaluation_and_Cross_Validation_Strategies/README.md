# Task 08 — Model Evaluation & Cross-Validation Strategies

## Overview
Builds a bulletproof evaluation pipeline for a heart-disease classifier:
K-Fold vs. Stratified K-Fold vs. Leave-One-Out comparison, learning curves,
validation curves, and a paired statistical significance test between models
— so that reported accuracy can be trusted before anything goes near production.

## Business Problem
A hospital deploying a heart disease prediction model previously overestimated
accuracy by 8% due to flawed evaluation methodology, and the model failed in
production, putting patients at risk. This task builds an evaluation
methodology that is resistant to that failure mode: stratified, leakage-free,
statistically validated.

## Dataset
**Heart Disease** (UCI / Kaggle), 14 attributes including age, sex, chest
pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`),
max heart rate (`thalach`), and a `target` (presence of heart disease):
https://archive.ics.uci.edu/dataset/45/heart+disease
(mirrored on Kaggle as "Heart Disease UCI")

Download the CSV and place it as `data/heart.csv` (git-ignored, not
committed — see `.gitignore`).

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Run top-to-bottom on a fresh kernel. All reusable logic lives in `src/`.

## Structure
- `notebook.ipynb` — main analysis: EDA → leak-free preprocessing → CV strategy comparison → learning/validation curves → statistical model comparison → baseline
- `src/data_loader.py` — loads and validates the raw CSV
- `src/preprocessing.py` — cleaning, target binarization, leak-free scaler fitting
- `src/modeling.py` — CV strategies, learning/validation curves, paired t-test, baseline
- `figures/` — saved charts (CV strategy comparison, learning curve, validation curve, ROC, confusion matrix)
- `reports/REPORT.md` — full written report

## Key Results
See `reports/REPORT.md` for the executive summary, chosen evaluation
strategy, statistical comparison outcome, and business recommendations.
