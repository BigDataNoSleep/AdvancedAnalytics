#%%
"""
LightGBM Baseline for CLV Prediction

This script mirrors the structure of the CatBoost baseline but utilizes LightGBM.
By capturing different types of errors due to its leaf-wise growth strategy,
it provides excellent diversity for ensembling.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
from lightgbm import LGBMRegressor
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

# Set to True to run an automated Optuna hyperparameter search before training
# (Requires: pip install optuna)
TUNE_HYPERPARAMETERS = False

LGBM_PARAMS = {
    'objective': 'mae',
    'metric': 'mae',
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8, # Feature fraction
    'reg_lambda': 10,       # L2 regularization
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# %%
# ============================================================
# 1. UTILS & TUNING
# ============================================================

def run_optuna_study(X_full, y_full, n_trials=30):
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Skipping tuning. (Run `pip install optuna` to enable)")
        return LGBM_PARAMS
    
    print(f"\n--- Starting Optuna Hyperparameter Search ({n_trials} trials) ---")
    
    def objective(trial):
        params = {
            'objective': 'mae',
            'metric': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 2000, 6000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 30.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        cv_maes = []
        # Use 3 folds for fast tuning
        for tr_idx, val_idx in make_stratified_folds(y_full, 3):
            X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
            y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
            
            model = LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
            )
            
            p_val = np.maximum(model.predict(X_val), 0)
            mae, _ = evaluate(y_val, p_val)
            cv_maes.append(mae)
            
        return np.mean(cv_maes)

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest Parameters found by Optuna:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    best_params = LGBM_PARAMS.copy()
    best_params.update(study.best_params)
    return best_params

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

def preprocess_categoricals(df):
    """
    Convert object columns to category dtype for LightGBM to use its native 
    categorical handling built into the DataFrame.
    """
    df_proc = df.copy()
    cat_cols = df_proc.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df_proc[c] = df_proc[c].astype('category')
    return df_proc

# %%
# ============================================================
# 2. TRAINING
# ============================================================

def train_lightgbm_baseline(X_full, y_full, X_test, test_unlabelled):
    print(f"Starting {N_FOLDS}-Fold Stratified CV with LightGBM...")
    
    val_preds = np.zeros(len(X_full))
    test_preds_list = []
    comp_preds_list = []
    
    for fold, (tr_idx, val_idx) in enumerate(make_stratified_folds(y_full, N_FOLDS), 1):
        X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
        
        seed_val_preds = []
        seed_test_preds = []
        seed_comp_preds = []

        for seed in SEEDS:
            params = LGBM_PARAMS.copy()
            params['random_state'] = seed

            model = LGBMRegressor(**params)
            
            # Use early_stopping via callbacks to avoid warnings in newer LightGBM versions
            # Some older versions use early_stopping_rounds kwarg in fit. We'll use kwargs for max compatibility.
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
            )
            
            seed_val_preds.append(model.predict(X_val))
            seed_test_preds.append(model.predict(X_test))
            seed_comp_preds.append(model.predict(test_unlabelled))

        # Average across seeds
        p_val = np.mean(seed_val_preds, axis=0)
        p_test = np.mean(seed_test_preds, axis=0)
        p_comp = np.mean(seed_comp_preds, axis=0)

        # Force non-negative predictions
        p_val = np.maximum(p_val, 0)
        p_test = np.maximum(p_test, 0)
        p_comp = np.maximum(p_comp, 0)
        
        val_preds[val_idx] = p_val
        test_preds_list.append(p_test)
        comp_preds_list.append(p_comp)
        
        fold_mae, _ = evaluate(y_full.iloc[val_idx], p_val)
        print(f"Fold {fold}: MAE = {fold_mae:.4f}")

    # Aggregates
    cv_mae, cv_spearman = evaluate(y_full, val_preds)
    avg_test_preds = np.mean(test_preds_list, axis=0)
    avg_comp_preds = np.mean(comp_preds_list, axis=0)
    
    print("\n" + "="*50)
    print("FINAL CV RESULTS (LightGBM)")
    print("="*50)
    print(f"OOF MAE:        {cv_mae:.4f}")
    print(f"OOF Spearman:   {cv_spearman:.4f}")
    
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
    
    # Merge for K-Fold
    X_full_train = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full_train = pd.concat([y_train, y_val]).reset_index(drop=True)
    
    # LightGBM requires strings to be cast as pd.Categorical
    print("Preprocessing categoricals...")
    X_full_train = preprocess_categoricals(X_full_train)
    X_test_proc = preprocess_categoricals(X_test)
    test_unlabelled_proc = preprocess_categoricals(test_unlabelled)
    
    # Optional Tuning
    if TUNE_HYPERPARAMETERS:
        global LGBM_PARAMS
        LGBM_PARAMS = run_optuna_study(X_full_train, y_full_train, n_trials=30)
    
    # Train
    results = train_lightgbm_baseline(X_full_train, y_full_train, X_test_proc, test_unlabelled_proc)
    
    # Robust pathing: find root directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(base_dir) == 'models':
        output_dir = os.path.join(os.path.dirname(base_dir), 'submissions')
    else:
        output_dir = os.path.join(base_dir, 'submissions')
    
    os.makedirs(output_dir, exist_ok=True)
    
    oof_df = pd.DataFrame({'prediction': results['val_preds']}, index=X_full_train.index)
    oof_path = os.path.join(output_dir, 'lightgbm_oof.csv')
    oof_df.to_csv(oof_path, index=False)

    # Holdout Verification
    test_mae, test_spearman = evaluate(y_test, results['test_preds'])
    print(f"\nLocal Holdout Test MAE: {test_mae:.4f}")
    
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

    output_path = os.path.join(output_dir, 'lightgbm_tuning_predictions.csv')
    test_predictions.to_csv(output_path, index_label='cust_id')
    print(f"\nFinal baseline predictions saved to `{output_path}`.")
    print(f"OOF predictions saved to `{oof_path}`.")

# %%
