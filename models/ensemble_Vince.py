#%%

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

# Add parent directory for eda_transactions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda_transactions import get_customer_model_data

def find_optimal_weights(oofs, y_true):
    """
    Find weights for each model that minimize MAE using SciPy minimize.
    """
    n_models = oofs.shape[1]
    initial_weights = np.array([1.0 / n_models] * n_models)
    
    # Constraints: weights sum to 1, each weight between 0 and 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0)] * n_models
    
    def objective(w):
        weighted_pred = np.dot(oofs, w)
        return mean_absolute_error(y_true, weighted_pred)
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def main():
    print("--- Advanced OOF-Optimized Ensemble ---")
    
    # 1. Load Data
    print("Loading actual targets...")
    X_train, X_val, _, y_train, y_val, _, _ = get_customer_model_data()
    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    
    # 2. Load OOF Predictions
    # Robust pathing: find root directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(base_dir) == 'models':
        output_dir = os.path.join(os.path.dirname(base_dir), 'submissions')
    else:
        output_dir = os.path.join(base_dir, 'submissions')
    
    print(f"Searching for OOF files in: {output_dir}")
    
    try:
        cb_oof = pd.read_csv(os.path.join(output_dir, 'catboost_oof.csv'))['prediction'].values
        lgb_oof = pd.read_csv(os.path.join(output_dir, 'lightgbm_oof.csv'))['prediction'].values
        xgb_oof = pd.read_csv(os.path.join(output_dir, 'xgboost_oof.csv'))['prediction'].values
    except FileNotFoundError as e:
        print(f"Error: Missing OOF file! ({e})")
        print("Ensure you have run catboost_Vince.py, lightgbm_Vince.py, and xgboost_Vince.py first.")
        return

    oofs = np.column_stack([cb_oof, lgb_oof, xgb_oof])
    
    # 3. Optimize Weights
    print("Finding mathematically optimal weights...")
    best_weights = find_optimal_weights(oofs, y_full)
    
    print("\nOptimal Model Weights:")
    print(f"  CatBoost: {best_weights[0]:.4f}")
    print(f"  LightGBM: {best_weights[1]:.4f}")
    print(f"  XGBoost:  {best_weights[2]:.4f}")
    
    # 4. Load Final Test Predictions
    print("\nLoading final test predictions from `submissions/`...")
    cb_test = pd.read_csv(os.path.join(output_dir, 'catboost_baseline_predictions.csv'), index_col='cust_id')
    lgb_test = pd.read_csv(os.path.join(output_dir, 'lightgbm_tuning_predictions.csv'), index_col='cust_id')
    xgb_test = pd.read_csv(os.path.join(output_dir, 'xgboost_baseline_predictions.csv'), index_col='cust_id')
    
    # Align
    cb_test, lgb_test = cb_test.align(lgb_test, join='inner', axis=0)
    cb_test, xgb_test = cb_test.align(xgb_test, join='inner', axis=0)
    
    # Apply Weights
    final_preds = (best_weights[0] * cb_test['prediction']) + \
                  (best_weights[1] * lgb_test['prediction']) + \
                  (best_weights[2] * xgb_test['prediction'])
    
    # 5. Save Output
    output_df = pd.DataFrame({'prediction': final_preds}, index=cb_test.index)
    output_path = os.path.join(output_dir, 'ensemble_final_predictions.csv')
    output_df.to_csv(output_path, index_label='cust_id')
    
    print(f"\nSuccess! Optimized ensemble saved to `{output_path}`.")
    print(f"Predicted Local OOF MAE improvement: {mean_absolute_error(y_full, np.dot(oofs, best_weights)):.4f}")

if __name__ == "__main__":
    main()

# %%
