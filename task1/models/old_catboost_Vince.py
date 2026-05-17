#%%
"""
Refined CatBoost Baseline for CLV Prediction

Changes:
- Reverted to Single-Stage Regressor (Simple is better for MAE)
- Implemented Stratified K-Fold (via target binning) for stable CV
- Removed Hurdle logic and Target Clipping to improve fit on outliers
- Tuned for MAE competition metric
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

# Ignore warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Add parent directory for eda_transactions and post_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Select Data Source: Standard vs Advanced Features
# from eda_transactions import get_customer_model_data; EDA_TYPE = "standard"
from eda_transactions_advanced import get_advanced_customer_model_data as get_customer_model_data; EDA_TYPE = "advanced"
from post_processing import fit_apply_post_processing, evaluate_log_and_save

# ============================================================
# CONSTANTS & EXPERIMENT FLAGS
# ============================================================
RANDOM_STATE = 42

# Reduced to 5 folds for faster iteration (was 10)
N_FOLDS = 5

# Set to True to test Tweedie loss (great for zero-inflated tabular)
USE_TWEEDIE = False 

# Set to True to run an automated Optuna hyperparameter search before training
# (Requires: pip install optuna)
TUNE_HYPERPARAMETERS = False

# Single seed for faster training (was 3)
SEEDS = [42]

CB_PARAMS = {
    'iterations': 5000,
    'learning_rate': 0.005028904391981856,
    'depth': 7,
    'l2_leaf_reg': 6.972082245567455,
    'random_strength': 0.7929841577545105,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.9115725780402228,
    'min_data_in_leaf': 83,
    'loss_function': 'Tweedie:variance_power=1.5' if USE_TWEEDIE else 'MAE',
    'eval_metric': 'MAE',
    'random_seed': RANDOM_STATE,
    'verbose': 0,
    'early_stopping_rounds': 300
}

# (Removed USE_LOG_TARGET: MAE loss on log1p targets causes severe under-prediction bias)

# %%
# ============================================================
# 1. UTILS & TUNING
# ============================================================

def run_optuna_study(X_full, y_full, n_trials=30):
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Skipping tuning. (Run `pip install optuna` to enable)")
        return CB_PARAMS
    
    print(f"\n--- Starting Optuna Hyperparameter Search ({n_trials} trials) ---")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 2000, 6000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 30.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'bootstrap_type': 'Bernoulli',
            'subsample': trial.suggest_float('subsample', 0.5, 0.95),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'loss_function': 'Tweedie:variance_power=1.5' if USE_TWEEDIE else 'MAE',
            'eval_metric': 'MAE',
            'random_seed': RANDOM_STATE,
            'verbose': 0,
            'early_stopping_rounds': 200
        }
        
        cv_maes = []
        # Use 3 folds for fast tuning
        for tr_idx, val_idx in make_stratified_folds(y_full, 3):
            X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
            y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            
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
    
    best_params = CB_PARAMS.copy()
    best_params.update(study.best_params)
    return best_params


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    return mae, spearman

def make_stratified_folds(y, n_folds=5):
    """
    Creates stratified folds by binning the target revenue.
    Ensures each fold has a representative mix of non-spenders and big spenders.
    """
    # np.unique prevents ValueError if multiple percentiles land on the same value
    bins = np.unique(np.percentile(y[y > 0], np.linspace(0, 100, 10)))
    y_binned = np.digitize(y, bins=bins)
    # Ensure zero-spenders are in their own bin
    y_binned[y == 0] = 0
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    return skf.split(np.zeros(len(y)), y_binned)

# %%
# ============================================================
# 2. TRAINING
# ============================================================

def train_catboost_baseline(X_full, y_full, X_test, test_unlabelled):
    print(f"Starting {N_FOLDS}-Fold Stratified CV...")
    
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
            params = CB_PARAMS.copy()
            params['random_seed'] = seed

            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

            seed_val_preds.append(model.predict(X_val))
            seed_test_preds.append(model.predict(X_test))
            seed_comp_preds.append(model.predict(test_unlabelled))

        p_val = np.mean(seed_val_preds, axis=0)
        p_test = np.mean(seed_test_preds, axis=0)
        p_comp = np.mean(seed_comp_preds, axis=0)

        # Force non-negative
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
    print("FINAL CV RESULTS")
    print("="*50)
    print(f"OOF MAE:        {cv_mae:.4f}")
    print(f"OOF Spearman:   {cv_spearman:.4f}")
    
    return {
        'val_preds': val_preds,
        'test_preds': avg_test_preds,
        'comp_preds': avg_comp_preds,
        'last_model': model # Return the last trained model for post-processing
    }

# %%
# ============================================================
# 3. EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_customer_model_data()
    
    # Merge for K-Fold
    X_full_train = pd.concat([X_train, X_val])
    y_full_train = pd.concat([y_train, y_val])
    
    # Optional Tuning
    if TUNE_HYPERPARAMETERS:
        CB_PARAMS = run_optuna_study(X_full_train, y_full_train, n_trials=20)
    
    # Train
    results = train_catboost_baseline(X_full_train, y_full_train, X_test, test_unlabelled)
    
    # Holdout Verification
    test_mae, test_spearman = evaluate(y_test, results['test_preds'])
    print(f"\nLocal Holdout Test MAE: {test_mae:.4f}")
    
    # ============================================================
    # 4. POST-PROCESSING & LOGGING
    # ============================================================
    POST_PROCESS_METHOD = "best"
    
    print(f"\nApplying {POST_PROCESS_METHOD} post-processing...")
    print(f"\nApplying {POST_PROCESS_METHOD} post-processing...")
    oof_final, test_final = fit_apply_post_processing(
        oof_preds=results['val_preds'],
        test_preds=results['comp_preds'],
        y_true=y_full_train.values,
        method_name=POST_PROCESS_METHOD,
        known_values=y_full_train.values,
        recency_train=X_full_train['recency_days'],
        recency_test=test_unlabelled['recency_days']
    )
    
    metrics = evaluate_log_and_save(
        oof_preds=oof_final,
        test_preds=test_final,
        y_true=y_full_train,
        test_ids=test_unlabelled.index,
        model_name="CatBoost",
        eda_used=EDA_TYPE,
        postprocess_method=POST_PROCESS_METHOD
    )

# %%
