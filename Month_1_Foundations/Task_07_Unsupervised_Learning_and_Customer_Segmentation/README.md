# Task 07 — Unsupervised Learning & Customer Segmentation

## Overview
Segments 200 mall customers into 4-6 marketing personas using K-Means,
hierarchical clustering, and DBSCAN, evaluated with silhouette score and
Davies-Bouldin index, and visualized with PCA.

## Business Problem
An e-commerce company (200K customers) needs 4-6 marketing personas to
target with different email campaigns. Poor segmentation risks ~$2M in
wasted marketing spend. This task prototypes the segmentation approach on
the Mall Customer dataset as a proxy for the larger customer base.

## Dataset
**Mall Customer Segmentation Data** (Kaggle):
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Download `Mall_Customers.csv` and place it in `data/` (git-ignored, not
committed — see `.gitignore`).

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Run top-to-bottom on a fresh kernel. All reusable logic lives in `src/`.

## Structure
- `notebook.ipynb` — main analysis: EDA → preprocessing → clustering → evaluation → profiling → PCA viz
- `src/data_loader.py` — loads and validates the raw CSV
- `src/preprocessing.py` — cleaning + feature scaling
- `src/modeling.py` — K-Means/DBSCAN/hierarchical fitting, metrics, PCA, cluster profiling
- `figures/` — saved charts (elbow, silhouette, PCA scatter, cluster profiles)
- `reports/REPORT.md` — full written report

## Key Results
See `reports/REPORT.md` for the executive summary, chosen k, cluster
personas, and business recommendations.
