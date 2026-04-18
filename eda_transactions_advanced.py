import pandas as pd
import numpy as np
from eda_transactions import get_customer_model_data

def build_advanced_customer_features(df_input):
    from eda_transactions import build_customer_features as build_base
    
    # 1. Get base features
    customer_features = build_base(df_input)
    
    # 2. Add the unique advanced features
    df_local = df_input.copy()
    df_local['prod_size'] = pd.to_numeric(df_local['prod_size'], errors='coerce')
    
    # Size Progression
    size_stats = df_local.groupby('cust_id')['prod_size'].agg(['min', 'max', 'nunique']).reset_index()
    size_stats['size_growth'] = size_stats['max'] - size_stats['min']
    
    customer_features = customer_features.merge(
        size_stats[['cust_id', 'size_growth', 'nunique']].rename(columns={'nunique':'unique_sizes'}), 
        on='cust_id', how='left'
    )

    # Seasonal Pulse
    df_local['month'] = df_local['order_date'].dt.month
    seasonal_stats = df_local.groupby('cust_id')['month'].agg(['std', 'nunique']).reset_index()
    
    customer_features = customer_features.merge(
        seasonal_stats.rename(columns={'std':'month_std', 'nunique':'unique_months'}), 
        on='cust_id', how='left'
    )

    # Size Stability Ratio
    customer_features['size_stability_ratio'] = (
        customer_features['unique_sizes'] / (customer_features['n_products'] + 1)
    )

    # Fill NaNs from new features
    numeric_cols = ['size_growth', 'unique_sizes', 'month_std', 'unique_months', 'size_stability_ratio']
    customer_features[numeric_cols] = customer_features[numeric_cols].fillna(0)

    return customer_features

def get_advanced_customer_model_data():
    """
    Complete pipeline for advanced data loading.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled
    Matches the signature of eda_transactions.get_customer_model_data
    """
    from eda_transactions import (
        df_train as df_trans_train, 
        df_test as df_trans_test,
        cols_to_drop,
        brand_freq
    )
    
    target_col = 'revenue_2018_2019'
    print("Building advanced customer features...")
    customer_features = build_advanced_customer_features(df_trans_train)
    customer_features_test = build_advanced_customer_features(df_trans_test)
    
    # Merge Targets
    customer_target = df_trans_train[['cust_id', target_col]].drop_duplicates()
    customer_features = customer_features.merge(customer_target, on='cust_id', how='left')
    
    # Encoding & Cleanup
    def encode_and_clean(df, is_train=True):
        df_mod = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()
        
        # Brand encoding
        df_mod['favorite_brand_encoded'] = df_mod['favorite_brand'].astype('category').cat.codes
        df_mod['favorite_brand_freq'] = df_mod['favorite_brand'].map(brand_freq)
        df_mod = df_mod.drop(columns=['favorite_brand'])
        
        return df_mod.fillna(0)

    customer_model_df = encode_and_clean(customer_features)
    customer_model_df_test = encode_and_clean(customer_features_test, is_train=False)
    
    # Final Selection
    exclude = [target_col, 'first_order_date', 'last_order_date'] # Note: keep 'cust_id' if needed for join, but base drops it
    feature_cols = [c for c in customer_model_df.columns 
                   if c in customer_model_df_test.columns 
                   and c not in exclude + ['cust_id']]
    
    X = customer_model_df.set_index('cust_id')[feature_cols]
    y = customer_model_df.set_index('cust_id')[target_col]
    X_unlabelled = customer_model_df_test.set_index('cust_id')[feature_cols]
    
    # Split strategy (80/10/10) to match base implementation
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_unlabelled
