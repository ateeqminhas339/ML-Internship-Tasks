# Task 02 — Advanced Data Cleaning & Preprocessing

Advanced cleaning, outlier handling, feature engineering, and an encoding/scaling
pipeline for the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset — building on the EDA from Task 01.

## What this task does

- Fixes the `TotalCharges` type issue (11 blank values, all `tenure == 0`) and imputes with `0`
- Checks for and removes fully duplicated rows
- Detects outliers in numeric features using the IQR rule and caps (winsorizes) any found
- Engineers 4 new features: `tenure_group`, `num_services`, `avg_monthly_spend`, `has_internet`
- Builds a reusable `ColumnTransformer` pipeline (StandardScaler + OneHotEncoder)
- Performs a stratified train/test split
- Trains a baseline Logistic Regression purely as a sanity check that the processed data is ML-ready
- Exports the cleaned/engineered dataset to `data/processed/telco_cleaned_engineered.csv`

## Folder structure

```
Task_02_Advanced_Data_Cleaning_and_Preprocessing/
├── README.md                  # this file
├── notebook.ipynb             # main solution notebook (run this)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # dataset loading + basic info utilities
│   ├── preprocessing.py       # cleaning, outliers, feature engineering, pipeline
│   └── modeling.py            # baseline model sanity check
├── figures/                   # 7 generated charts (.png)
├── reports/
│   └── REPORT.md              # written report
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── processed/             # cleaned/engineered output (generated on run)
├── requirements.txt
└── .gitignore
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Run all cells top-to-bottom (Kernel → Restart & Run All). Figures are saved automatically to `figures/`,
and the cleaned dataset is exported to `data/processed/`.

## Key results

| Check | Result |
|---|---|
| Duplicate rows | 0 |
| IQR outliers (tenure, MonthlyCharges, TotalCharges) | 0 |
| New engineered features | 4 |
| Baseline Logistic Regression ROC-AUC (test set) | ≈ 0.84 |

See `reports/REPORT.md` for the full write-up.
