"""
preprocessing.py - Data cleaning and preprocessing for Online Retail dataset.

Handles:
- Null Customer ID removal
- Cancelled transaction removal
- Invalid quantity/price removal  
- Date conversion
- Duplicate removal
"""

import pandas as pd
import numpy as np


def preprocess_data(df, verbose=True):
    """
    Clean and preprocess the Online Retail dataset.
    
    Steps:
    1. Drop rows with missing Customer ID
    2. Remove cancelled transactions (Invoice starts with 'C')
    3. Remove rows with Quantity <= 0
    4. Remove rows with Price <= 0
    5. Ensure InvoiceDate is datetime
    6. Remove duplicate rows
    7. Create TotalPrice feature
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with columns: Invoice, StockCode, Description,
        Quantity, InvoiceDate, Price, Customer ID, Country.
    verbose : bool
        Whether to print step-by-step info.
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataset with additional TotalPrice column.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    if verbose:
        print("=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        print(f"Initial rows: {initial_rows:,}")
    
    # Step 1: Remove null Customer ID
    null_cid = df_clean['Customer ID'].isnull().sum()
    df_clean = df_clean.dropna(subset=['Customer ID'])
    if verbose:
        print(f"\n[Step 1] Removed {null_cid:,} rows with null Customer ID")
        print(f"  Remaining: {len(df_clean):,} rows")
    
    # Step 2: Remove cancelled transactions (Invoice starts with 'C')
    cancelled_mask = df_clean['Invoice'].astype(str).str.startswith('C')
    cancelled_count = cancelled_mask.sum()
    df_clean = df_clean[~cancelled_mask]
    if verbose:
        print(f"\n[Step 2] Removed {cancelled_count:,} cancelled transactions")
        print(f"  Remaining: {len(df_clean):,} rows")
    
    # Step 3: Remove rows with Quantity <= 0
    neg_qty = (df_clean['Quantity'] <= 0).sum()
    df_clean = df_clean[df_clean['Quantity'] > 0]
    if verbose:
        print(f"\n[Step 3] Removed {neg_qty:,} rows with Quantity <= 0")
        print(f"  Remaining: {len(df_clean):,} rows")
    
    # Step 4: Remove rows with Price <= 0
    neg_price = (df_clean['Price'] <= 0).sum()
    df_clean = df_clean[df_clean['Price'] > 0]
    if verbose:
        print(f"\n[Step 4] Removed {neg_price:,} rows with Price <= 0")
        print(f"  Remaining: {len(df_clean):,} rows")
    
    # Step 5: Ensure InvoiceDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_clean['InvoiceDate']):
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        if verbose:
            print(f"\n[Step 5] Converted InvoiceDate to datetime")
    else:
        if verbose:
            print(f"\n[Step 5] InvoiceDate already datetime64")
    
    # Step 6: Remove duplicates
    dup_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    if verbose:
        print(f"\n[Step 6] Removed {dup_count:,} duplicate rows")
        print(f"  Remaining: {len(df_clean):,} rows")
    
    # Step 7: Create TotalPrice
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']
    if verbose:
        print(f"\n[Step 7] Created TotalPrice = Quantity * Price")
    
    # Convert Customer ID to int
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
    
    # Summary
    removed = initial_rows - len(df_clean)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"PREPROCESSING SUMMARY")
        print(f"  Initial rows:  {initial_rows:,}")
        print(f"  Final rows:    {len(df_clean):,}")
        print(f"  Removed:       {removed:,} ({removed/initial_rows*100:.1f}%)")
        print(f"  Columns:       {list(df_clean.columns)}")
        print(f"{'=' * 60}")
    
    return df_clean
