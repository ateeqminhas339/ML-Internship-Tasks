"""Clustering pipeline: fit models, evaluate, reduce dims, profile clusters."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

RANDOM_SEED = 42


def kmeans_sweep(X: np.ndarray, k_range=range(2, 11)):
    """Fit K-Means for each k; return a DataFrame of inertia + silhouette."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(X)


def fit_hierarchical(X: np.ndarray, k: int) -> AgglomerativeClustering:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)


def fit_dbscan(X: np.ndarray, eps: float = 0.6, min_samples: int = 5) -> DBSCAN:
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)


def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette (higher/better) + Davies-Bouldin (lower/better).
    Noise-only or single-cluster DBSCAN runs are skipped (metrics undefined).
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": n_clusters}
    mask = labels != -1  # exclude DBSCAN noise points from scoring
    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "n_clusters": n_clusters,
    }


def pca_2d(X: np.ndarray):
    """Reduce to 2 components for visualization only (not used for clustering)."""
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean feature values + size per cluster, for business interpretation."""
    out = df[features].copy()
    out["cluster"] = labels
    profile = out.groupby("cluster")[features].mean().round(1)
    profile["size"] = out.groupby("cluster").size()
    profile["pct_of_total"] = (profile["size"] / len(out) * 100).round(1)
    return profile.sort_values("cluster")
