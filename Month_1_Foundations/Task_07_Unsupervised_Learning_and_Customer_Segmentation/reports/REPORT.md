# Task 07 — Unsupervised Learning & Customer Segmentation — Report
Author: Ateeq (@ateeqminhas339)
Date: 2026-08-04

## 1. Executive Summary
- K-Means with **k=5** was selected via the elbow method and silhouette sweep (silhouette peaks climbing gently from k=4 onward; k=5 matches the business's 4-6 persona requirement).
- K-Means (silhouette 0.36, Davies-Bouldin 0.93) clearly outperformed a naive single-feature baseline (silhouette 0.03, Davies-Bouldin 4.51) — clustering on Age + Income + Spending Score jointly is far more separable than binning on spending alone.
- K-Means also outperformed Hierarchical/Ward (silhouette 0.31) and DBSCAN (silhouette 0.11) on this dataset, so it is the recommended production algorithm.
- Five personas emerged, ranging from ~13% to ~24% of customers — large enough to justify distinct campaigns, small enough to stay targeted.
- PCA (2 components, 69.9% combined variance explained) confirms the 5 segments are visually well-separated.

## 2. Business Problem & Framing
An e-commerce company with 200K customers wants to define 4-6 marketing personas so each segment can receive a tailored email campaign; poor segmentation risks ~$2M in wasted spend. This task prototypes the segmentation methodology on the Mall Customer dataset as a stand-in for the full customer base, using Age, Annual Income, and Spending Score as inputs.

## 3. Data Overview
200 customers, 5 raw columns (`CustomerID`, `Genre`/Gender, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`). No missing values or duplicate IDs after cleaning. Age ranges 18-70 (mean ~44), Income ~15-104k$ (mean ~55k), Spending Score 1-100 (mean ~49) — all three features carry independent signal, which is why segmentation uses all three rather than any single column.

## 4. Methodology
1. **Cleaning:** dropped duplicate `CustomerID`s, renamed `Genre` → `Gender`, dropped rows missing any clustering feature.
2. **Scaling:** `StandardScaler` on Age/Income/Spending Score — required because clustering distance metrics are scale-sensitive and these features have very different ranges.
3. **Model selection:** swept K-Means k=2..10, chose k via the elbow (inertia) plus silhouette score, cross-checked against the business's 4-6 persona constraint.
4. **Algorithm comparison:** K-Means vs. Agglomerative/Ward vs. DBSCAN, all at comparable cluster counts, scored with silhouette and Davies-Bouldin.
5. **Baseline:** naive single-feature binning (5 equal-width bins on Spending Score) — represents what a non-ML approach might do.
6. **Visualization:** PCA to 2 components for plotting only (not used to fit the clusters, to avoid discarding signal before clustering).
7. **Profiling:** per-cluster mean feature values and size, translated into personas.

## 5. Results & Key Visualizations
| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Clusters |
|---|---|---|---|
| K-Means | 0.36 | 0.93 | 5 |
| Hierarchical (Ward) | 0.31 | 1.02 | 5 |
| DBSCAN | 0.11 | 1.03 | 5 (+ noise) |
| Baseline (spending bins) | 0.03 | 4.51 | 5 |

**Cluster profiles (K-Means, k=5):**
| Cluster | Age | Income (k$) | Spending Score | Size | % of total | Persona |
|---|---|---|---|---|---|---|
| 0 | 27.1 | 55.8 | 31.6 | 48 | 24.0% | Young, mid-income, low spenders |
| 1 | 42.4 | 85.0 | 80.0 | 41 | 20.5% | High income, high spenders (VIP) |
| 2 | 46.3 | 25.8 | 77.8 | 44 | 22.0% | Low income, high spenders |
| 3 | 57.3 | 24.7 | 17.5 | 27 | 13.5% | Older, low income, disengaged |
| 4 | 56.5 | 75.2 | 28.2 | 40 | 20.0% | Older, high income, low spenders |

Figures: `figures/01_feature_distributions.png`, `02_income_vs_spending.png`, `03_elbow_and_silhouette.png`, `04_pca_clusters.png`, `05_cluster_profiles.png`.

## 6. Limitations & Risks
- 200 rows is a small proxy for a 200K-customer base; cluster boundaries may shift with the real, larger dataset.
- Only 3 numeric features used — no purchase history, recency/frequency, or channel engagement, which would likely sharpen segments.
- DBSCAN results depend heavily on `eps`/`min_samples`; only one setting was tried, not a full grid search.
- All evaluation is internal (silhouette/Davies-Bouldin) — no ground-truth persona labels exist to validate externally.
- Cluster interpretations (personas) are analyst judgment applied to numeric summaries, not confirmed by customer research.

## 7. Recommendation / Next Steps
Adopt K-Means (k=5) as the segmentation baseline for the full 200K-customer rollout. Before production use: (1) re-run the same pipeline on real transactional data with richer features, (2) validate persona stability across a train/holdout split, (3) pilot campaigns on a small sample per cluster before full rollout to confirm personas convert as expected.

## 8. References
- Dataset: Mall Customer Segmentation Data, Kaggle — https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- scikit-learn documentation: KMeans, AgglomerativeClustering, DBSCAN, silhouette_score, davies_bouldin_score
