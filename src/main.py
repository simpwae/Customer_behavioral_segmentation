"""
main.py - Main script to run the full RFM Customer Segmentation pipeline.

Usage:
    python src/main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_dataset, inspect_dataset, save_dataframe
from src.preprocessing import preprocess_data
from src.rfm import compute_rfm, apply_log_transformation, scale_rfm, remove_rfm_outliers
from src.clustering import find_optimal_k, train_kmeans, analyze_clusters
from src.visualization import (
    plot_rfm_distributions, plot_elbow_curve,
    plot_silhouette_scores, plot_clusters_2d_pca
)


def main():
    # 1. Load Data
    print("\n" + "=" * 70)
    print("STAGE 1: DATA LOADING")
    print("=" * 70)
    DATA_PATH = os.path.join('data', 'online_retail_II.xlsx')
    if not os.path.exists(DATA_PATH):
        DATA_PATH = 'online_retail_II.xlsx'
    df = load_dataset(DATA_PATH)

    # 2. Inspect
    print("\n" + "=" * 70)
    print("STAGE 2: DATA INSPECTION")
    print("=" * 70)
    inspect_dataset(df)

    # 3. Preprocess
    print("\n" + "=" * 70)
    print("STAGE 3: PREPROCESSING")
    print("=" * 70)
    df_clean = preprocess_data(df)

    # 4. RFM Feature Engineering
    print("\n" + "=" * 70)
    print("STAGE 4: RFM FEATURE ENGINEERING")
    print("=" * 70)
    rfm = compute_rfm(df_clean)
    rfm = apply_log_transformation(rfm)

    # 5. Outlier Removal
    print("\n" + "=" * 70)
    print("STAGE 5: OUTLIER REMOVAL")
    print("=" * 70)
    rfm_clean = remove_rfm_outliers(rfm, method='iqr', threshold=1.5)
    rfm_clean = apply_log_transformation(rfm_clean)

    # 6. Scale
    rfm_scaled, scaler, features = scale_rfm(rfm_clean)

    # 7. Find Optimal K
    print("\n" + "=" * 70)
    print("STAGE 6: FINDING OPTIMAL K")
    print("=" * 70)
    eval_results = find_optimal_k(rfm_scaled, k_range=range(2, 11))

    # 8. Train Final Model
    print("\n" + "=" * 70)
    print("STAGE 7: FINAL K-MEANS MODEL")
    print("=" * 70)
    optimal_k = eval_results['best_k_silhouette']
    kmeans, labels, sil_score = train_kmeans(rfm_scaled, n_clusters=optimal_k)

    # 9. Analyze Clusters
    print("\n" + "=" * 70)
    print("STAGE 8: CLUSTER ANALYSIS")
    print("=" * 70)
    cluster_summary, rfm_clustered = analyze_clusters(rfm_clean, labels)

    # 10. Save Results
    save_dataframe(rfm_clustered, 'data/rfm_clustered.csv', index=False)
    save_dataframe(cluster_summary.reset_index(), 'data/cluster_summary.csv', index=False)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
