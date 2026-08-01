# Task 06 — Classification Algorithms Comparison

Compares six classification algorithms — Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting, K-Nearest Neighbors, and SVM (RBF kernel)
— for predicting customer churn, using the feature-engineered Telco Customer
Churn dataset produced in Task 03.

## Structure

```
Task_06_Classification_Algorithms_Comparison/
├── README.md
├── notebook.ipynb          # main analysis, runs top-to-bottom
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # loads data, builds train/test split
│   ├── preprocessing.py    # shared ColumnTransformer (scale + one-hot)
│   └── modeling.py         # candidate models, train/eval/cross-validate helpers
├── data/
│   ├── telco_feature_engineered.csv     # input dataset (from Task 03)
│   └── model_comparison_results.csv     # output metrics table
├── figures/                # 6 saved charts (.png)
├── reports/
│   └── REPORT.md
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Or open `notebook.ipynb` directly in Jupyter and run all cells.

## Key result

| Model | ROC-AUC | Accuracy | F1 | Train time (s) |
|---|---|---|---|---|
| Logistic Regression | **0.847** | 0.805 | 0.593 | 0.15 |
| Gradient Boosting | 0.846 | 0.809 | 0.589 | 2.45 |
| Random Forest | 0.822 | 0.779 | 0.538 | 3.40 |
| K-Nearest Neighbors | 0.820 | 0.779 | 0.566 | 0.04 |
| SVM (RBF) | 0.800 | 0.798 | 0.561 | 8.47 |
| Decision Tree | 0.645 | 0.720 | 0.475 | 0.12 |

**Logistic Regression** is recommended for production: it ties for the best
ROC-AUC, trains ~15x faster than Gradient Boosting, and is fully
interpretable. Full reasoning in `reports/REPORT.md`.
