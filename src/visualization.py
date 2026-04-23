"""
visualization.py - Plotting functions for RFM analysis and clustering.

All plots use matplotlib and seaborn for consistent styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 6)


def plot_rfm_distributions(rfm, figsize=(16, 5)):
    """Plot distributions of Recency, Frequency, and Monetary values."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    titles = ['Recency (days since last purchase)',
              'Frequency (# of transactions)',
              'Monetary (total spending $)']
    
    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        axes[i].hist(rfm[col], bins=50, color=colors[i], edgecolor='white', alpha=0.8)
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        
        # Add median line
        median_val = rfm[col].median()
        axes[i].axvline(median_val, color='red', linestyle='--', linewidth=1.5,
                        label=f'Median: {median_val:.0f}')
        axes[i].legend()
    
    plt.suptitle('RFM Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_rfm_log_distributions(rfm, figsize=(16, 5)):
    """Plot distributions of log-transformed RFM values."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for i, col in enumerate(['Recency_log', 'Frequency_log', 'Monetary_log']):
        if col in rfm.columns:
            axes[i].hist(rfm[col], bins=50, color=colors[i], edgecolor='white', alpha=0.8)
            axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
    
    plt.suptitle('Log-Transformed RFM Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_rfm_boxplots(rfm, figsize=(16, 5)):
    """Plot boxplots to visualize outliers in RFM features."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        bp = axes[i].boxplot(rfm[col], patch_artist=True, vert=True)
        bp['boxes'][0].set_facecolor(colors[i])
        bp['boxes'][0].set_alpha(0.7)
        axes[i].set_title(f'{col} Boxplot', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(col)
    
    plt.suptitle('RFM Outlier Analysis (Boxplots)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(rfm, figsize=(8, 6)):
    """Plot correlation heatmap for RFM features."""
    corr = rfm[['Recency', 'Frequency', 'Monetary']].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1,
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('RFM Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_country_distribution(df, top_n=15, figsize=(12, 6)):
    """Plot top countries by number of customers."""
    country_counts = df.groupby('Country')['Customer ID'].nunique().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    country_counts.head(top_n).plot(kind='barh', color='#2196F3', edgecolor='white', ax=ax)
    ax.set_xlabel('Number of Unique Customers')
    ax.set_title(f'Top {top_n} Countries by Customer Count', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(country_counts.head(top_n).values):
        ax.text(v + 10, i, f'{v:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_elbow_curve(eval_results, figsize=(10, 5)):
    """Plot Elbow curve (Inertia vs K)."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(eval_results['k_values'], eval_results['inertias'],
            'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.set_xticks(eval_results['k_values'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_silhouette_scores(eval_results, figsize=(10, 5)):
    """Plot Silhouette Scores vs K."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(eval_results['k_values'], eval_results['silhouette_scores'],
            'rs-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score for Different K Values', fontsize=14, fontweight='bold')
    ax.set_xticks(eval_results['k_values'])
    ax.grid(True, alpha=0.3)
    
    # Highlight best K
    best_idx = np.argmax(eval_results['silhouette_scores'])
    best_k = eval_results['k_values'][best_idx]
    best_score = eval_results['silhouette_scores'][best_idx]
    ax.annotate(f'Best K={best_k}\nScore={best_score:.4f}',
                xy=(best_k, best_score),
                xytext=(best_k + 1, best_score - 0.02),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_silhouette_diagram(X, labels, n_clusters, figsize=(10, 7)):
    """Plot silhouette diagram showing per-sample silhouette values."""
    fig, ax = plt.subplots(figsize=figsize)
    
    sil_vals = silhouette_samples(X, labels)
    sil_avg = sil_vals.mean()
    
    y_lower = 10
    cmap = plt.cm.get_cmap('tab10')
    
    for i in range(n_clusters):
        cluster_sil = sil_vals[labels == i]
        cluster_sil.sort()
        
        size = cluster_sil.shape[0]
        y_upper = y_lower + size
        
        color = cmap(i / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i), fontsize=12, fontweight='bold')
        y_lower = y_upper + 10
    
    ax.axvline(x=sil_avg, color='red', linestyle='--', linewidth=2,
               label=f'Average: {sil_avg:.4f}')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(f'Silhouette Diagram (K={n_clusters})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def plot_clusters_2d_pca(X, labels, n_clusters, figsize=(10, 8)):
    """Plot clusters in 2D using PCA dimensionality reduction."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = plt.cm.get_cmap('tab10')
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[cmap(i)], label=f'Cluster {i}',
                   alpha=0.5, s=20, edgecolors='none')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Customer Clusters (PCA 2D Projection)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, markerscale=3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")


def plot_clusters_2d_tsne(X, labels, n_clusters, figsize=(10, 8), sample_size=5000):
    """Plot clusters in 2D using t-SNE dimensionality reduction."""
    # t-SNE is slow on large datasets, sample if needed
    if len(X) > sample_size:
        np.random.seed(42)
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
        print(f"t-SNE: Sampled {sample_size:,} from {len(X):,} points")
    else:
        X_sample = X
        labels_sample = labels
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = plt.cm.get_cmap('tab10')
    for i in range(n_clusters):
        mask = labels_sample == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[cmap(i)], label=f'Cluster {i}',
                   alpha=0.5, s=20, edgecolors='none')
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('Customer Clusters (t-SNE 2D Projection)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, markerscale=3)
    
    plt.tight_layout()
    plt.show()


def plot_cluster_comparison(cluster_summary, figsize=(14, 5)):
    """Plot bar charts comparing mean RFM values across clusters."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(cluster_summary)))
    cluster_ids = cluster_summary.index.tolist()
    
    metrics = ['Recency_mean', 'Frequency_mean', 'Monetary_mean']
    titles = ['Avg Recency (days)', 'Avg Frequency (transactions)', 'Avg Monetary ($)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        bars = axes[i].bar(range(len(cluster_ids)), cluster_summary[metric],
                           color=colors, edgecolor='white', linewidth=1.5)
        axes[i].set_xticks(range(len(cluster_ids)))
        axes[i].set_xticklabels([f'C{c}' for c in cluster_ids])
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Cluster')
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Cluster Comparison: Mean RFM Values',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cluster_sizes(cluster_summary, figsize=(8, 6)):
    """Plot pie chart of cluster sizes."""
    fig, ax = plt.subplots(figsize=figsize)
    
    labels_list = []
    for idx, row in cluster_summary.iterrows():
        label_text = row.get('Label', f'Cluster {idx}')
        labels_list.append(f"C{idx}: {label_text}")
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(cluster_summary)))
    
    wedges, texts, autotexts = ax.pie(
        cluster_summary['Count'],
        labels=labels_list,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 10}
    )
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rfm_scatter_matrix(rfm_clustered, figsize=(12, 10)):
    """Plot pairwise scatter plots of RFM features colored by cluster."""
    plot_data = rfm_clustered[['Recency', 'Frequency', 'Monetary', 'Cluster']].copy()
    plot_data['Cluster'] = plot_data['Cluster'].astype(str)
    
    g = sns.pairplot(plot_data, hue='Cluster', palette='tab10',
                     diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15},
                     height=3)
    g.fig.suptitle('RFM Pairwise Scatter Matrix by Cluster',
                   fontsize=14, fontweight='bold', y=1.02)
    plt.show()
