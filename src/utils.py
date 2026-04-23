"""
utils.py - Utility functions for the RFM Customer Segmentation pipeline.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_dataset(filepath):
    """
    Load the Online Retail II dataset from an Excel file.
    Handles multiple sheets and concatenates them.
    
    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe from all sheets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    xls = pd.ExcelFile(filepath)
    print(f"Sheet names found: {xls.sheet_names}")
    
    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet)
        print(f"  Sheet '{sheet}': {df.shape[0]:,} rows, {df.shape[1]} columns")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal dataset: {combined.shape[0]:,} rows, {combined.shape[1]} columns")
    return combined


def inspect_dataset(df):
    """
    Print a comprehensive inspection of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    
    Returns
    -------
    dict
        Dictionary with inspection results.
    """
    info = {}
    
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    
    # Shape
    print(f"\nShape: {df.shape}")
    info['shape'] = df.shape
    
    # Column names and types
    print(f"\nColumns and Data Types:")
    for col in df.columns:
        print(f"  {col:20s} -> {df[col].dtype}")
    info['dtypes'] = df.dtypes.to_dict()
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(f"\nMissing Values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col:20s} -> {missing[col]:,} ({missing_pct[col]}%)")
        else:
            print(f"  {col:20s} -> 0")
    info['missing'] = missing.to_dict()
    
    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    info['duplicates'] = dup_count
    
    # Numeric stats
    print(f"\nNumeric Feature Statistics:")
    print(df.describe().to_string())
    
    # Date range (if InvoiceDate exists)
    if 'InvoiceDate' in df.columns:
        print(f"\nDate Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
        info['date_range'] = (df['InvoiceDate'].min(), df['InvoiceDate'].max())
    
    # Cancelled transactions
    if 'Invoice' in df.columns:
        cancelled = df['Invoice'].astype(str).str.startswith('C').sum()
        print(f"\nCancelled Transactions (Invoice starts with 'C'): {cancelled:,}")
        info['cancelled'] = cancelled
    
    # Unique customers
    cid_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    if cid_col in df.columns:
        unique_customers = df[cid_col].nunique()
        print(f"Unique Customers: {unique_customers:,}")
        info['unique_customers'] = unique_customers
    
    # Countries
    if 'Country' in df.columns:
        print(f"Unique Countries: {df['Country'].nunique()}")
        info['unique_countries'] = df['Country'].nunique()
    
    print("=" * 60)
    return info


def save_dataframe(df, filepath, index=False):
    """Save a DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index)
    print(f"Saved: {filepath} ({df.shape[0]:,} rows)")
