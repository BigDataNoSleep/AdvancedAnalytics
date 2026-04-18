# %% [markdown]
# # Advanced LightGBM (GOSS) + Manual Recency Filters
# This script uses Gradient-based One-Side Sampling (GOSS) and manual 
# human-coded recency logic to refine long-term CLV predictions.

# %%
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Select Data Source: Standard vs Advanced Features
# from eda_transactions import get_customer_model_data; EDA_TYPE = "standard"
from eda_transactions_advanced import get_advanced_customer_model_data; EDA_TYPE = "advanced"
from post_processing import (
    calculate_metrics, 
    run_full_post_processing
)

# Set plotting style
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load Advanced Modeling Data

X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_advanced_customer_model_data()

# Combine for cross-validation
X = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])

# Note: test_ids are the index of test_unlabelled (cust_id)
test_ids = test_unlabelled.index

print(f"Features: {X.shape[1]}")
print(f"Train rows: {len(X)}")
print(f"Test rows:  {len(X_test)}")

# %% [markdown]
# ## 2. GOSS Training with Cross-Validation

params = {
    'objective': 'regression_l1',
    'boosting_type': 'goss',
    'learning_rate': 0.005,
    'num_leaves': 70,           # Smaller trees = less overfitting
    'min_child_samples': 200,    # Forces the model to only learn from large groups
    'feature_fraction': 0.6,     # Only look at 60% of features at a time
    'lambda_l1': 15.0,           # High penalty for high guesses
    'seed': 42,
    'verbosity': -1,
    'device': 'cpu' # Ensure compatibility
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_unlabelled))

# We'll save the OOF for diagnostics
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_f_train, y_f_train = X.iloc[train_idx], y.iloc[train_idx]
    X_f_val, y_f_val = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_f_train, label=y_f_train)
    dval = lgb.Dataset(X_f_val, label=y_f_val, reference=dtrain)
    
    model = lgb.train(
        params, dtrain, num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(500)]
    )
    
    oof_preds[val_idx] = model.predict(X_f_val)
    test_preds += model.predict(test_unlabelled) / kf.get_n_splits()

# Post-Fold cleanup
oof_preds = np.maximum(0, oof_preds)
test_preds = np.maximum(0, test_preds)

print(f"\nOOF Baseline MAE: {mean_absolute_error(y, oof_preds):.4f}")

# %% [markdown]
# ## 3. Manual Filters (Floor & Recency)
# Implementing the "Human-Coded" rules for aggressive inactivity zeroing.

def apply_manual_filters(preds, feature_df):
    """
    Applies the floor threshold and the 400/500-day recency logic.
    """
    p = preds.copy()
    
    # 1. Floor Filter
    p[p < 5.0] = 0
    
    # 2. Recency Filter
    if 'recency_days' in feature_df.columns:
        # Penalize customers inactive for over 400 days
        mask_400 = (feature_df['recency_days'] > 400) & (feature_df['recency_days'] <= 500)
        p[mask_400] *= 0.2
        
        # Zero out customers inactive for over 500 days
        mask_500 = (feature_df['recency_days'] > 500)
        p[mask_500] = 0
        
    return p

# Apply to OOF and Test
oof_final = apply_manual_filters(oof_preds, X)
test_final = apply_manual_filters(test_preds, test_unlabelled)

print(f"Final OOF MAE (after filters): {mean_absolute_error(y, oof_final):.4f}")

# %% [markdown]
# ## 4. Interactive Diagnostics & Submission

# 1. Run Standard Post-Processing Diagnostics
run_full_post_processing(
    model=model,
    X_train=X,
    y_true=y,
    y_pred=oof_final,
    model_name="LightGBM_Advanced",
    eda_used=EDA_TYPE
)

# 2. Prepare Submission
submission = pd.DataFrame({
    'cust_id': test_ids,
    'revenue': test_final
})

# Save to standard submissions folder
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submissions', 'lightgbm_advanced_predictions.csv'))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
submission.to_csv(output_path, index=False)
print(f"\nFinal predictions saved to: {output_path}")

# 3. Final Verification of Zero Rates
print("\n=== ZERO CONCENTRATION (OOF) ===")
print(f"Actual Zeros:   {(y == 0).mean()*100:.2f}%")
print(f"Predicted Zeros: {(oof_final == 0).mean()*100:.2f}%")
# %%
