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
    fit_apply_post_processing,
    evaluate_log_and_save
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
# ## 3. Post-Processing Pipeline
# Applying automated data-driven scaling and mapping functions.

POST_PROCESS_METHOD = "recency_only_cv"

print(f"\nApplying {POST_PROCESS_METHOD} post-processing...")
oof_final, test_final = fit_apply_post_processing(
    oof_preds=oof_preds,
    test_preds=test_preds,
    y_true=y.values,
    method_name=POST_PROCESS_METHOD,
    known_values=y.values,
    recency_train=X['recency_days'],
    recency_test=test_unlabelled['recency_days']
)

print(f"Final OOF MAE (after post-processing): {mean_absolute_error(y, oof_final):.4f}")

# %% [markdown]
# ## 4. Interactive Diagnostics, Logging & Final Submission

# Unified Orchestrator covers plotting, logging to CSV, and standardized CSV submission saving
metrics = evaluate_log_and_save(
    oof_preds=oof_final,
    test_preds=test_final,
    y_true=y,
    test_ids=test_ids,
    model_name="LightGBM_Advanced",
    eda_used=EDA_TYPE,
    postprocess_method=POST_PROCESS_METHOD
)

print("\n=== ZERO CONCENTRATION (OOF) ===")
print(f"Actual Zeros:   {(y == 0).mean()*100:.2f}%")
print(f"Predicted Zeros: {(oof_final == 0).mean()*100:.2f}%")
# %%
