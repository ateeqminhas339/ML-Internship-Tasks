# Task 03 — Feature Engineering Mastery

Advanced feature engineering on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset — building on the cleaned data from Task 02.

## What this task does

- Reduces skew in `TotalCharges` with a log1p transform
- Adds interaction/ratio features: `tenure_x_monthly`, `charges_per_service`, `spend_to_tenure_ratio`
- Adds degree-2 polynomial features on `tenure` and `MonthlyCharges`
- Builds a KMeans-based `customer_segment` feature and visualizes it (and churn) in PCA-reduced space
- Compares three categorical encoding strategies: one-hot, frequency, and smoothed target encoding
- Ranks engineered numeric features by mutual information with churn
- **Proves the engineered features add real value**: trains an identical Logistic Regression on a minimal baseline feature set vs. the fully engineered set and compares metrics head-to-head
- Extracts Random Forest feature importance on the final feature set
- Exports the feature-engineered dataset to `data/processed/telco_feature_engineered.csv`

## Folder structure

```
Task_03_Feature_Engineering_Mastery/
├── README.md                  # this file
├── notebook.ipynb             # main solution notebook (run this)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # loading + Task 02 baseline cleaning
│   ├── preprocessing.py       # transforms, interactions, polynomial/cluster features, encoding, selection
│   └── modeling.py            # baseline-vs-engineered comparison + feature importance
├── figures/                   # 8 generated charts (.png)
├── reports/
│   └── REPORT.md              # written report
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── processed/             # feature-engineered output (generated on run)
├── requirements.txt
└── .gitignore
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Run all cells top-to-bottom (Kernel → Restart & Run All). Figures are saved automatically to `figures/`,
and the feature-engineered dataset is exported to `data/processed/`.

## Key results

| Metric | Baseline features | Engineered features |
|---|---|---|
| Accuracy | 0.789 | 0.807 |
| Precision | 0.621 | 0.667 |
| Recall | 0.529 | 0.545 |
| F1 | 0.571 | 0.600 |
| ROC-AUC | 0.836 | 0.848 |

Identical Logistic Regression, same train/test split methodology — the only difference is the feature set.
See `reports/REPORT.md` for the full write-up.
