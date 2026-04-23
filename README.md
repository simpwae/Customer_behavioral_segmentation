# Customer Segmentation using RFM Analysis & K-Means Clustering

> **Data Mining Project** — Production-grade unsupervised machine learning pipeline for customer behavioral segmentation using the Online Retail II dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Objective](#2-business-objective)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Pipeline Stages](#5-pipeline-stages)
6. [Data Cleaning Summary](#6-data-cleaning-summary)
7. [RFM Feature Engineering](#7-rfm-feature-engineering)
8. [Modeling & Evaluation](#8-modeling--evaluation)
9. [Experimentation Results](#9-experimentation-results)
10. [Cluster Profiles & Business Labels](#10-cluster-profiles--business-labels)
11. [Business Strategies per Segment](#11-business-strategies-per-segment)
12. [Visualizations Produced](#12-visualizations-produced)
13. [Output Files](#13-output-files)
14. [Source Modules](#14-source-modules)
15. [How to Run](#15-how-to-run)
16. [Requirements & Installation](#16-requirements--installation)
17. [Configuration / Reproducibility](#17-configuration--reproducibility)
18. [LaTeX Tables](#18-latex-tables)

---

## 1. Project Overview

This project implements a full **data mining and machine learning pipeline** to segment customers of a UK-based online retailer into distinct behavioral groups. The pipeline covers every stage of the CRISP-DM process: data loading, quality assessment, preprocessing, feature engineering, exploratory data analysis, clustering, evaluation, experimentation, interpretation, and export.

**Key techniques used:**

- RFM (Recency, Frequency, Monetary) feature construction
- Log-transformation (`log1p`) for skewness reduction
- IQR-based outlier removal
- StandardScaler normalization
- K-Means clustering with elbow + silhouette analysis
- Agglomerative (hierarchical) clustering comparison
- PCA and t-SNE dimensionality reduction for visualization
- Rank-based composite scoring for business label assignment

---

## 2. Business Objective

Segment **5,001 customers** into meaningful groups to enable:

| Goal                | Description                                           |
| ------------------- | ----------------------------------------------------- |
| Targeted marketing  | Tailor campaigns to each segment's behavior           |
| Customer retention  | Identify and act on at-risk customers early           |
| Resource allocation | Focus budget on high-value segments                   |
| Churn reduction     | Intervene before hibernating customers are fully lost |
| Revenue growth      | Upsell/cross-sell to loyal and champion customers     |

---

## 3. Dataset

| Property       | Value                                                        |
| -------------- | ------------------------------------------------------------ |
| **Name**       | Online Retail II                                             |
| **Source**     | UCI Machine Learning Repository                              |
| **File**       | `data/online_retail_II.xlsx`                                 |
| **Sheets**     | Year 2009-2010 (525,461 rows), Year 2010-2011 (541,910 rows) |
| **Total Rows** | 1,067,371 transactions                                       |
| **Countries**  | 43                                                           |
| **Date Range** | 2009-12-01 to 2011-12-09                                     |

**Columns:**

| Column        | Type     | Description                                      |
| ------------- | -------- | ------------------------------------------------ |
| `Invoice`     | str      | Transaction ID (prefix `C` = cancellation)       |
| `StockCode`   | str      | Product code                                     |
| `Description` | str      | Product name                                     |
| `Quantity`    | int      | Units purchased (negative = return/cancellation) |
| `InvoiceDate` | datetime | Date and time of transaction                     |
| `Price`       | float    | Unit price in GBP                                |
| `Customer ID` | float    | Unique customer identifier (nullable)            |
| `Country`     | str      | Customer country                                 |

---

## 4. Project Structure

```
data Mining/
├── data/
│   ├── online_retail_II.xlsx          # Raw input dataset
│   ├── rfm_clustered.csv              # Output: 5,001 customers with RFM + Cluster label
│   ├── cluster_summary.csv            # Output: Per-cluster profile statistics
│   ├── experiment_results.csv         # Output: Scaler / outlier experiment table
│   ├── clusters_pca.png               # PCA scatter plot coloured by cluster
│   ├── clusters_tsne.png              # t-SNE scatter plot coloured by cluster
│   ├── cluster_boxplots.png           # RFM boxplots per cluster
│   ├── cluster_comparison_bars.png    # Mean R/F/M bar chart per cluster
│   ├── cluster_pie.png                # Customer segment distribution pie chart
│   ├── cluster_radar.png              # Radar / spider chart of cluster RFM profiles
│   ├── country_distribution.png       # Top-15 countries by unique customers
│   ├── dendrogram.png                 # Hierarchical clustering dendrogram
│   ├── elbow_silhouette.png           # Elbow + silhouette score curves
│   ├── rfm_boxplots.png               # RFM feature boxplots (outlier view)
│   ├── rfm_correlation.png            # Correlation heatmap of log-transformed RFM
│   ├── rfm_raw_vs_log.png             # Raw vs log1p distribution comparison
│   └── silhouette_diagram.png         # Per-sample silhouette diagram for K=4
├── notebooks/
│   └── RFM_Customer_Segmentation.ipynb   # Main analysis notebook (54 cells, fully executed)
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Dataset loading, display helpers, logging
│   ├── preprocessing.py               # Full data cleaning pipeline
│   ├── rfm.py                         # RFM computation, log transform, scaling, outliers
│   ├── clustering.py                  # K-Means training, silhouette evaluation
│   ├── visualization.py               # All plotting functions
│   └── main.py                        # End-to-end CLI script
├── requirements.txt
└── README.md
```

---

## 5. Pipeline Stages

The notebook (`RFM_Customer_Segmentation.ipynb`) is organized into **20 sections across 54 cells**:

| #   | Notebook Section        | What It Does                                                                       |
| --- | ----------------------- | ---------------------------------------------------------------------------------- |
| 1   | Imports & Config        | Load all libraries, set `RANDOM_STATE=42`, plot defaults                           |
| 2   | Load Dataset            | Read both Excel sheets, concatenate, inspect dtypes                                |
| 3   | Inspect Schema          | Shape, dtypes, missing values, sample rows                                         |
| 4   | Data Quality            | Missing %, cancellations, negatives, duplicates — summary table                    |
| 5   | Preprocessing           | Remove nulls, cancellations (C-prefix), negative qty/price, duplicates             |
| 6   | RFM Computation         | Recency (days since last purchase), Frequency (# invoices), Monetary (total spend) |
| 7   | Log Transformation      | Apply `log1p` to Recency, Frequency, Monetary to reduce skewness                   |
| 8   | EDA — Distributions     | Histograms: raw vs log-transformed RFM                                             |
| 9   | EDA — Correlation       | Heatmap of Pearson correlations on log-transformed features                        |
| 10  | EDA — Country           | Bar chart of top-15 countries by unique customer count                             |
| 11  | Outlier Removal         | IQR method (threshold=1.5) on all three log RFM features                           |
| 12  | Scaling                 | StandardScaler on log-transformed features                                         |
| 13  | Elbow & Silhouette      | K-Means for K=2..10, compute Inertia + Silhouette Score → choose K=4               |
| 14  | K-Means (K=4)           | Fit final K-Means model, assign cluster labels                                     |
| 15  | Silhouette Diagram      | Per-sample silhouette diagram showing cluster quality                              |
| 16  | Hierarchical Comparison | Agglomerative clustering (Ward linkage), compare silhouette                        |
| 17  | Dendrogram              | Hierarchical dendrogram visualization                                              |
| 18  | PCA Visualization       | 2-component PCA scatter plot coloured by cluster                                   |
| 19  | t-SNE Visualization     | t-SNE (2D) on a 1,000-sample subset                                                |
| 20  | Cluster Profiling       | Mean/median RFM per cluster; rank-based business label assignment                  |
| —   | Business Insights       | Actionable marketing & retention strategies per segment                            |
| —   | Cluster Charts          | Bar chart, radar chart, boxplots, pie chart per cluster                            |
| —   | Experimentation         | Compare 3 scalers × 3 K values + with/without outlier removal                      |
| —   | Evaluation Summary      | Best config, final model metrics, full K evaluation table                          |
| —   | Export                  | Save CSVs and visualizations to `data/`                                            |
| —   | LaTeX Tables            | Generate LaTeX-formatted tables for academic reporting                             |
| —   | Final Conclusion        | Full CRISP-DM summary printout                                                     |

---

## 6. Data Cleaning Summary

| Step                                 | Rows Removed | Remaining           |
| ------------------------------------ | ------------ | ------------------- |
| Initial dataset                      | —            | 1,067,371           |
| Remove null Customer IDs             | 243,007      | 824,364             |
| Remove cancelled invoices (C-prefix) | 19,494       | 804,870             |
| Remove negative / zero Quantity      | varies       | ~800,000            |
| Remove negative / zero Price         | varies       | ~800,000            |
| Remove duplicate rows                | 34,335       | 779,425             |
| **Clean dataset**                    | —            | **779,425**         |
| Remove RFM outliers (IQR × 1.5)      | 877          | **5,001 customers** |

---

## 7. RFM Feature Engineering

RFM features are computed per unique `Customer ID`:

| Feature       | Formula                                                    | Meaning                            |
| ------------- | ---------------------------------------------------------- | ---------------------------------- |
| **Recency**   | Days between last purchase and reference date (2011-12-10) | How recently did the customer buy? |
| **Frequency** | Count of distinct `Invoice` numbers                        | How often do they buy?             |
| **Monetary**  | Sum of `Quantity × Price` per customer                     | How much do they spend in total?   |

**Transformations applied:**

- `log1p(Recency)` — reduces right skew
- `log1p(Frequency)` — reduces right skew
- `log1p(Monetary)` — reduces right skew
- `StandardScaler` — zero mean, unit variance before clustering

---

## 8. Modeling & Evaluation

### Algorithm: K-Means Clustering

| Parameter      | Value                                             |
| -------------- | ------------------------------------------------- |
| `n_clusters`   | 4                                                 |
| `random_state` | 42                                                |
| `n_init`       | 10                                                |
| `max_iter`     | 300                                               |
| Scaler         | StandardScaler                                    |
| Features       | log1p(Recency), log1p(Frequency), log1p(Monetary) |

### Why K=4?

K=2 yields the highest silhouette score (0.4149) but produces only two coarse groups with little business interpretability. K=4 gives 4 distinct, actionable segments with a solid silhouette score (0.3624).

### Silhouette Scores by K

| K     | Inertia      | Silhouette Score    |
| ----- | ------------ | ------------------- |
| 2     | 7,625.05     | **0.4149**          |
| 3     | 5,883.70     | 0.3219              |
| **4** | **4,404.41** | **0.3624** ✓ chosen |
| 5     | 3,763.18     | 0.3105              |
| 6     | 3,272.71     | 0.3218              |
| 7     | 2,912.64     | 0.2925              |
| 8     | 2,653.04     | 0.2927              |
| 9     | 2,456.93     | 0.2799              |
| 10    | 2,287.89     | 0.2825              |

### Hierarchical Clustering Comparison

Agglomerative clustering (Ward linkage, K=4) was also run and compared visually via dendrogram and cluster assignment overlap.

---

## 9. Experimentation Results

Three scaling techniques were compared across K=3, 4, 5 and with/without outlier removal:

| Scaler             | Outlier Handling | K     | Silhouette Score | Customers |
| ------------------ | ---------------- | ----- | ---------------- | --------- |
| MinMaxScaler       | IQR Removed      | 4     | **0.3944**       | 5,001     |
| MinMaxScaler       | IQR Removed      | 3     | 0.3826           | 5,001     |
| StandardScaler     | With Outliers    | 4     | 0.3650           | 5,878     |
| **StandardScaler** | **IQR Removed**  | **4** | **0.3624**       | **5,001** |
| RobustScaler       | IQR Removed      | 4     | 0.3603           | 5,001     |
| StandardScaler     | With Outliers    | 3     | 0.3477           | 5,878     |
| MinMaxScaler       | IQR Removed      | 5     | 0.3462           | 5,001     |
| StandardScaler     | With Outliers    | 5     | 0.3425           | 5,878     |
| RobustScaler       | IQR Removed      | 3     | 0.3277           | 5,001     |
| StandardScaler     | IQR Removed      | 3     | 0.3219           | 5,001     |
| RobustScaler       | IQR Removed      | 5     | 0.3171           | 5,001     |
| StandardScaler     | IQR Removed      | 5     | 0.3105           | 5,001     |

**Best by silhouette:** MinMaxScaler + IQR Removed + K=4 (0.3944)  
**Final model chosen:** StandardScaler + IQR Removed + K=4 — selected for stability and consistency with the log-transformation preprocessing.

---

## 10. Cluster Profiles & Business Labels

Business labels are assigned using **rank-based composite scoring**: each cluster is ranked on Recency (ascending = better), Frequency (descending = better), and Monetary (descending = better). The composite rank determines the label.

| Cluster | Business Label                | Customers | %     | Avg Recency | Avg Frequency | Avg Monetary | Composite Rank |
| ------- | ----------------------------- | --------- | ----- | ----------- | ------------- | ------------ | -------------- |
| 0       | Hibernating / Lost Customers  | 1,571     | 31.4% | 415 days    | 1.2           | $247         | 12 (worst)     |
| 1       | Loyal / Regular Customers     | 982       | 19.6% | 35 days     | 2.2           | $613         | 7              |
| 2       | Champions / High-Value Loyal  | 1,138     | 22.8% | 45 days     | 7.7           | $2,180       | 4 (best)       |
| 3       | At-Risk / Declining Customers | 1,310     | 26.2% | 301 days    | 3.5           | $1,151       | 7              |

---

## 11. Business Strategies per Segment

### Cluster 2 — Champions / High-Value Loyal (22.8%)

- **Marketing:** Exclusive VIP loyalty programs, early access to new products, personalized recommendations.
- **Retention:** Thank-you rewards, premium communications, invite to referral programs.

### Cluster 1 — Loyal / Regular Customers (19.6%)

- **Marketing:** Upsell and cross-sell related products, loyalty discounts, bundle offers.
- **Retention:** Regular engagement emails, birthday/anniversary rewards, exclusive member pricing.

### Cluster 3 — At-Risk / Declining Customers (26.2%)

- **Marketing:** Urgent win-back campaigns, competitive pricing, understand lapse reasons.
- **Retention:** Personal outreach, service quality surveys, special comeback incentives.

### Cluster 0 — Hibernating / Lost Customers (31.4%)

- **Marketing:** "We miss you" emails with strong discounts, win-back campaigns, surveys.
- **Retention:** Re-engagement offers, personalized recommendations based on past purchases.

---

## 12. Visualizations Produced

| File                          | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| `rfm_raw_vs_log.png`          | Side-by-side histograms: raw RFM vs log1p-transformed       |
| `rfm_correlation.png`         | Pearson correlation heatmap of log-transformed RFM features |
| `country_distribution.png`    | Top-15 countries by unique customer count                   |
| `rfm_boxplots.png`            | Boxplots showing spread and outliers in R, F, M             |
| `elbow_silhouette.png`        | Elbow curve (inertia) + silhouette scores across K=2..10    |
| `silhouette_diagram.png`      | Per-sample silhouette plot for chosen K=4                   |
| `dendrogram.png`              | Hierarchical clustering dendrogram (Ward linkage)           |
| `clusters_pca.png`            | 2D PCA scatter coloured by K-Means cluster                  |
| `clusters_tsne.png`           | 2D t-SNE scatter coloured by cluster (1,000-sample subset)  |
| `cluster_comparison_bars.png` | Mean Recency, Frequency, Monetary bar chart per cluster     |
| `cluster_radar.png`           | Radar (spider) chart of normalized RFM profiles per cluster |
| `cluster_boxplots.png`        | Boxplots of R, F, M split by cluster                        |
| `cluster_pie.png`             | Pie chart of customer count per segment                     |

---

## 13. Output Files

| File                          | Description                                                     |
| ----------------------------- | --------------------------------------------------------------- |
| `data/rfm_clustered.csv`      | 5,001 rows — Customer ID, Recency, Frequency, Monetary, Cluster |
| `data/cluster_summary.csv`    | 4 rows — per-cluster statistics and business label              |
| `data/experiment_results.csv` | 12 rows — all scaler/K/outlier experiment combinations          |
| `data/*.png`                  | 13 visualization images (see section above)                     |

---

## 14. Source Modules

| Module                 | Purpose                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| `src/utils.py`         | `load_dataset()`, `inspect_dataset()`, `save_dataframe()` — dataset I/O and display helpers |
| `src/preprocessing.py` | `preprocess_data()` — full cleaning pipeline (nulls, cancellations, negatives, duplicates)  |
| `src/rfm.py`           | `compute_rfm()`, `apply_log_transformation()`, `scale_rfm()`, `remove_rfm_outliers()`       |
| `src/clustering.py`    | `find_optimal_k()`, `train_kmeans()`, `analyze_clusters()` — model training and evaluation  |
| `src/visualization.py` | All plotting functions: distributions, elbow, silhouette, PCA, t-SNE, radar, boxplots       |
| `src/main.py`          | End-to-end CLI pipeline — runs all stages from data load to export                          |

---

## 15. How to Run

### Option 1: Jupyter Notebook (Recommended)

```bash
# 1. Clone / open the project folder
cd "data Mining"

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook notebooks/RFM_Customer_Segmentation.ipynb
```

Run all 54 cells top-to-bottom. All outputs, plots, and CSV exports are generated automatically.

### Option 2: Python Script (CLI)

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 16. Requirements & Installation

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
openpyxl>=3.0.0
scipy>=1.9.0
jinja2>=3.0.0       # required by pandas to_latex()
```

Install all at once:

```bash
pip install -r requirements.txt
pip install jinja2   # if not already installed
```

**Tested with:**

| Package      | Version |
| ------------ | ------- |
| Python       | 3.13    |
| pandas       | 3.0.2   |
| numpy        | 2.4.4   |
| scikit-learn | 1.8.0   |
| matplotlib   | 3.10.8  |
| seaborn      | 0.13.2  |

---

## 17. Configuration / Reproducibility

All stochastic operations use a fixed seed for full reproducibility:

| Parameter                          | Value   |
| ---------------------------------- | ------- |
| `RANDOM_STATE`                     | 42      |
| `np.random.seed`                   | 42      |
| `KMeans(random_state=42)`          | 42      |
| `TSNE(random_state=42)`            | 42      |
| `PCA(random_state=42)`             | 42      |
| `IQR_THRESHOLD`                    | 1.5     |
| `SAMPLE_SIZE` (silhouette diagram) | 1,000   |
| `TSNE_SAMPLE`                      | 1,000   |
| `K_RANGE`                          | 2 to 10 |
| `OPTIMAL_K`                        | 4       |

---

## 18. LaTeX Tables

The notebook generates three ready-to-use LaTeX tables (Section 20, cell `#VSC-80628c4c`):

| Table                      | Label              | Content                                                |
| -------------------------- | ------------------ | ------------------------------------------------------ |
| RFM Cluster Summary        | `tab:rfm_clusters` | Segment name, count, %, avg R/F/M                      |
| K-Means Evaluation Metrics | `tab:eval_metrics` | K, Inertia, Silhouette Score for K=2..10               |
| Experimentation Results    | `tab:experiments`  | All scaler/outlier/K combinations sorted by silhouette |

---

## License

This project is for academic and educational use. The Online Retail II dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II).
