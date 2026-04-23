"""
rfm.py - RFM (Recency, Frequency, Monetary) feature engineering.

Computes customer-level RFM features from transactional data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_rfm(df, reference_date=None):
    """
    Compute RFM features aggregated at the Customer ID level.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactional data with columns:
        Customer ID, Invoice, InvoiceDate, TotalPrice.
    reference_date : str or pd.Timestamp, optional
        Reference date for Recency calculation.
        If None, uses max(InvoiceDate) + 1 day.
    
    Returns
    -------
    pd.DataFrame
        RFM table indexed by Customer ID with columns:
        Recency, Frequency, Monetary.
    """
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)
    
    print(f"Reference date for Recency: {reference_date.date()}")
    
    # Recency: days since last purchase
    # Frequency: number of unique invoices per customer
    # Monetary: total spending per customer
    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    
    # Round monetary to 2 decimal places
    rfm['Monetary'] = rfm['Monetary'].round(2)
    
    print(f"\nRFM Table Shape: {rfm.shape}")
    print(f"  Unique Customers: {rfm.shape[0]:,}")
    print(f"\nRFM Statistics:")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2).to_string())
    
    return rfm


def apply_log_transformation(rfm):
    """
    Apply log1p transformation to RFM features to reduce skewness.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM table with Recency, Frequency, Monetary columns.
    
    Returns
    -------
    pd.DataFrame
        RFM table with additional log-transformed columns.
    """
    rfm_log = rfm.copy()
    
    for col in ['Recency', 'Frequency', 'Monetary']:
        log_col = f'{col}_log'
        rfm_log[log_col] = np.log1p(rfm_log[col])
    
    print("Applied log1p transformation to R, F, M")
    print("\nSkewness comparison:")
    for col in ['Recency', 'Frequency', 'Monetary']:
        orig_skew = rfm_log[col].skew()
        log_skew = rfm_log[f'{col}_log'].skew()
        print(f"  {col:12s} -> Original: {orig_skew:7.2f}, Log: {log_skew:7.2f}")
    
    return rfm_log


def scale_rfm(rfm, features=None):
    """
    Standardize RFM features using StandardScaler.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM table.
    features : list, optional
        Feature columns to scale. If None, uses log-transformed columns
        if available, else raw RFM columns.
    
    Returns
    -------
    tuple
        (scaled_array, scaler, feature_names)
    """
    if features is None:
        # Prefer log-transformed columns if available
        log_cols = [c for c in rfm.columns if c.endswith('_log')]
        if log_cols:
            features = log_cols
        else:
            features = ['Recency', 'Frequency', 'Monetary']
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])
    
    print(f"Scaled features: {features}")
    print(f"Scaled shape: {rfm_scaled.shape}")
    print(f"Mean (should be ~0): {rfm_scaled.mean(axis=0).round(4)}")
    print(f"Std  (should be ~1): {rfm_scaled.std(axis=0).round(4)}")
    
    return rfm_scaled, scaler, features


def remove_rfm_outliers(rfm, method='iqr', threshold=1.5):
    """
    Remove outliers from RFM data using IQR method.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM table with Recency, Frequency, Monetary columns.
    method : str
        Outlier detection method: 'iqr' or 'zscore'.
    threshold : float
        IQR multiplier (default 1.5) or Z-score threshold (default 3).
    
    Returns
    -------
    pd.DataFrame
        RFM table with outliers removed.
    """
    rfm_clean = rfm.copy()
    initial_count = len(rfm_clean)
    
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    
    if method == 'iqr':
        for col in rfm_cols:
            Q1 = rfm_clean[col].quantile(0.25)
            Q3 = rfm_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            before = len(rfm_clean)
            rfm_clean = rfm_clean[(rfm_clean[col] >= lower) & (rfm_clean[col] <= upper)]
            removed = before - len(rfm_clean)
            print(f"  {col}: IQR=[{Q1:.2f}, {Q3:.2f}], "
                  f"Bounds=[{lower:.2f}, {upper:.2f}], Removed: {removed:,}")
    
    elif method == 'zscore':
        from scipy import stats
        for col in rfm_cols:
            z_scores = np.abs(stats.zscore(rfm_clean[col]))
            before = len(rfm_clean)
            rfm_clean = rfm_clean[z_scores < threshold]
            removed = before - len(rfm_clean)
            print(f"  {col}: Z-score threshold={threshold}, Removed: {removed:,}")
    
    total_removed = initial_count - len(rfm_clean)
    print(f"\nTotal outliers removed: {total_removed:,} ({total_removed/initial_count*100:.1f}%)")
    print(f"Remaining customers: {len(rfm_clean):,}")
    
    return rfm_clean
