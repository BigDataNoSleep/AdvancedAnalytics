#%%
"""
XGBoost Baseline for CLV Prediction

Completes the "Big Three" ensemble (CatBoost, LightGBM, XGBoost).
Uses the same 10-Fold Stratified CV structure for consistent OOF saving.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# Add parent directory for eda_transactions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda_transactions import get_customer_model_data

# ============================================================
# CONSTANTS & EXPERIMENT FLAGS
# ============================================================
RANDOM_STATE = 42
N_FOLDS = 5
SEEDS = [42]

XGB_PARAMS = {
    'objective': 'reg:absoluteerror', # Optimized for MAE
    'tree_method': 'hist',            # Equivalent to CatBoost/LightGBM histograms
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 10,                    # L2 regularization
    'seed': RANDOM_STATE,
    'n_jobs': -1
}

# %%
# ============================================================
# 1. UTILS
# ============================================================

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    return mae, spearman

def make_stratified_folds(y, n_folds=10):
    bins = np.unique(np.percentile(y[y > 0], np.linspace(0, 100, 10)))
    y_binned = np.digitize(y, bins=bins)
    y_binned[y == 0] = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    return skf.split(np.zeros(len(y)), y_binned)

# %%
# ============================================================
# 2. TRAINING
# ============================================================

def train_xgboost_baseline(X_full, y_full, X_test, test_unlabelled):
    print(f"Starting {N_FOLDS}-Fold Stratified CV with XGBoost...")
    
    val_preds = np.zeros(len(X_full))
    test_preds_list = []
    comp_preds_list = []
    
    # XGBoost requires categorical encoding if not using its specific handler.
    # For this baseline, we drop string columns for simplicity.
    X_full_num = X_full.select_dtypes(exclude=['object'])
    X_test_num = X_test.select_dtypes(exclude=['object'])
    test_unlabelled_num = test_unlabelled.select_dtypes(exclude=['object'])

    for fold, (tr_idx, val_idx) in enumerate(make_stratified_folds(y_full, N_FOLDS), 1):
        X_tr, X_val = X_full_num.iloc[tr_idx], X_full_num.iloc[val_idx]
        y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
        
        seed_val_preds = []
        seed_test_preds = []
        seed_comp_preds = []

        for seed in SEEDS:
            params = XGB_PARAMS.copy()
            params['seed'] = seed
            params['early_stopping_rounds'] = 200 # Moved to constructor

            model = XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            seed_val_preds.append(model.predict(X_val))
            seed_test_preds.append(model.predict(X_test_num))
            seed_comp_preds.append(model.predict(test_unlabelled_num))

        p_val = np.mean(seed_val_preds, axis=0)
        p_test = np.mean(seed_test_preds, axis=0)
        p_comp = np.mean(seed_comp_preds, axis=0)

        p_val = np.maximum(p_val, 0)
        p_test = np.maximum(p_test, 0)
        p_comp = np.maximum(p_comp, 0)
        
        val_preds[val_idx] = p_val
        test_preds_list.append(p_test)
        comp_preds_list.append(p_comp)
        
        fold_mae, _ = evaluate(y_full.iloc[val_idx], p_val)
        print(f"Fold {fold}: MAE = {fold_mae:.4f}")

    cv_mae, cv_spearman = evaluate(y_full, val_preds)
    avg_test_preds = np.mean(test_preds_list, axis=0)
    avg_comp_preds = np.mean(comp_preds_list, axis=0)
    
    print("\n" + "="*50)
    print("FINAL CV RESULTS (XGBoost)")
    print("="*50)
    print(f"OOF MAE:        {cv_mae:.4f}")
    
    return {
        'val_preds': val_preds,
        'test_preds': avg_test_preds,
        'comp_preds': avg_comp_preds
    }

# %%
# ============================================================
# 3. EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_customer_model_data()
    
    X_full_train = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full_train = pd.concat([y_train, y_val]).reset_index(drop=True)
    
    # Train
    results = train_xgboost_baseline(X_full_train, y_full_train, X_test, test_unlabelled)
    
    # Robust pathing: find root directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(base_dir) == 'models':
        output_dir = os.path.join(os.path.dirname(base_dir), 'submissions')
    else:
        output_dir = os.path.join(base_dir, 'submissions')
    
    os.makedirs(output_dir, exist_ok=True)
    
    oof_df = pd.DataFrame({'prediction': results['val_preds']}, index=X_full_train.index)
    oof_path = os.path.join(output_dir, 'xgboost_oof.csv')
    oof_df.to_csv(oof_path, index=False)

    # Save Predictions
    test_predictions = pd.DataFrame({
        'prediction': results['comp_preds']
    }, index=test_unlabelled.index)
    
    # Recover original IDs
    try:
        _test_baseline = pd.read_csv("../test_predictions.csv")
        if len(_test_baseline) == len(test_predictions):
            test_predictions.index = sorted(_test_baseline['cust_id'].unique())
    except:
        pass

    output_path = os.path.join(output_dir, 'xgboost_baseline_predictions.csv')
    test_predictions.to_csv(output_path, index_label='cust_id')
    print(f"\nXGBoost predictions saved to `{output_path}`.")
    print(f"OOF predictions saved to `{oof_path}`.")

# %%
