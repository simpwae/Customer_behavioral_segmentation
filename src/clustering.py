"""
clustering.py - K-Means clustering and evaluation for customer segmentation.

Includes Elbow Method, Silhouette Analysis, and cluster labeling.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples


RANDOM_STATE = 42


def find_optimal_k(X, k_range=range(2, 11), random_state=RANDOM_STATE):
    """
    Evaluate K-Means for different values of K using Elbow and Silhouette methods.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of K values to evaluate.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    dict
        Dictionary with 'k_values', 'inertias', 'silhouette_scores'.
    """
    inertias = []
    silhouette_scores_list = []
    
    print("Evaluating K-Means for different K values:")
    print(f"{'K':>4} | {'Inertia':>12} | {'Silhouette':>10}")
    print("-" * 35)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X, labels)
        
        inertias.append(inertia)
        silhouette_scores_list.append(sil_score)
        
        print(f"{k:>4} | {inertia:>12.2f} | {sil_score:>10.4f}")
    
    best_k_sil = list(k_range)[np.argmax(silhouette_scores_list)]
    print(f"\nBest K by Silhouette Score: {best_k_sil} "
          f"(score: {max(silhouette_scores_list):.4f})")
    
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores_list,
        'best_k_silhouette': best_k_sil
    }


def train_kmeans(X, n_clusters, random_state=RANDOM_STATE):
    """
    Train a K-Means model with the specified number of clusters.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random state for reproducibility.
    
    Returns
    -------
    tuple
        (kmeans_model, labels, silhouette_avg)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    
    print(f"K-Means trained with K={n_clusters}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Cluster {u}: {c:,} customers ({c/len(labels)*100:.1f}%)")
    
    return kmeans, labels, sil_score


def train_hierarchical(X, n_clusters):
    """
    Train Agglomerative (Hierarchical) Clustering for comparison.
    
    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.
    
    Returns
    -------
    tuple
        (model, labels, silhouette_avg)
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    
    print(f"Hierarchical Clustering (Ward) with K={n_clusters}")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Cluster {u}: {c:,} customers ({c/len(labels)*100:.1f}%)")
    
    return agg, labels, sil_score


def analyze_clusters(rfm_df, labels):
    """
    Analyze cluster characteristics based on mean RFM values.
    
    Parameters
    ----------
    rfm_df : pd.DataFrame
        RFM table with Recency, Frequency, Monetary columns.
    labels : np.ndarray
        Cluster labels for each customer.
    
    Returns
    -------
    pd.DataFrame
        Cluster summary with mean RFM values, counts, and labels.
    """
    rfm_clustered = rfm_df.copy()
    rfm_clustered['Cluster'] = labels
    
    # Compute cluster means and counts
    cluster_summary = rfm_clustered.groupby('Cluster').agg(
        Recency_mean=('Recency', 'mean'),
        Frequency_mean=('Frequency', 'mean'),
        Monetary_mean=('Monetary', 'mean'),
        Recency_median=('Recency', 'median'),
        Frequency_median=('Frequency', 'median'),
        Monetary_median=('Monetary', 'median'),
        Count=('Customer ID', 'count')
    ).round(2)
    
    cluster_summary['Pct'] = (cluster_summary['Count'] / cluster_summary['Count'].sum() * 100).round(1)
    
    # Label clusters based on RFM patterns
    cluster_summary['Label'] = cluster_summary.apply(_label_cluster, axis=1)
    
    print("\nCluster Analysis:")
    print("=" * 90)
    for idx, row in cluster_summary.iterrows():
        print(f"\nCluster {idx} - {row['Label']} ({row['Count']:,} customers, {row['Pct']}%)")
        print(f"  Avg Recency:  {row['Recency_mean']:.1f} days")
        print(f"  Avg Frequency: {row['Frequency_mean']:.1f} transactions")
        print(f"  Avg Monetary:  ${row['Monetary_mean']:,.2f}")
    print("=" * 90)
    
    return cluster_summary, rfm_clustered


def _label_cluster(row):
    """
    Assign a business label to a cluster based on its RFM characteristics.
    Uses relative ranking within the cluster summary.
    """
    r = row['Recency_mean']
    f = row['Frequency_mean']
    m = row['Monetary_mean']
    
    # High monetary + high frequency + low recency = Best customers
    if m > 2000 and f > 5 and r < 50:
        return "High-Value Loyal Customers"
    elif m > 1000 and f > 3 and r < 100:
        return "High-Value Customers"
    elif f > 5 and r < 50:
        return "Loyal Customers"
    elif r > 200:
        return "At-Risk / Lost Customers"
    elif r > 100 and f <= 3:
        return "At-Risk Customers"
    elif m < 300 and f <= 2:
        return "Low-Value Customers"
    elif r < 50 and f <= 3:
        return "New / Recent Customers"
    else:
        return "Mid-Value Customers"


def label_clusters_by_rank(cluster_summary):
    """
    Alternative labeling: rank clusters by each RFM dimension
    and assign labels based on combined rank.
    
    Parameters
    ----------
    cluster_summary : pd.DataFrame
        Cluster summary from analyze_clusters().
    
    Returns
    -------
    pd.DataFrame
        Updated cluster summary with rank-based labels.
    """
    cs = cluster_summary.copy()
    n = len(cs)
    
    # Rank: lower Recency is better (rank ascending -> low = good)
    cs['R_rank'] = cs['Recency_mean'].rank(ascending=True)
    # Rank: higher Frequency is better
    cs['F_rank'] = cs['Frequency_mean'].rank(ascending=False)
    # Rank: higher Monetary is better
    cs['M_rank'] = cs['Monetary_mean'].rank(ascending=False)
    cs['RFM_rank_sum'] = cs['R_rank'] + cs['F_rank'] + cs['M_rank']
    
    # Best rank sum = lowest = best customers
    cs['Rank_Label'] = cs['RFM_rank_sum'].apply(
        lambda x: "Champions" if x <= n * 0.4
        else "Potential Loyalists" if x <= n * 0.7
        else "At Risk" if x <= n * 0.85
        else "Needs Attention"
    )
    
    print("\nRank-based Cluster Labels:")
    for idx, row in cs.iterrows():
        print(f"  Cluster {idx}: {row['Rank_Label']} "
              f"(R_rank={row['R_rank']}, F_rank={row['F_rank']}, M_rank={row['M_rank']})")
    
    return cs
