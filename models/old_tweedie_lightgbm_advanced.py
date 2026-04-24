# %% [markdown]
# # Advanced Tweedie Regression (LightGBM)
# Using the Tweedie distribution to natively handle zero-inflation and long-tails.
# Optimized for Mac speed and integrated with behavioral post-processing.

# %%
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda_transactions_advanced import get_advanced_customer_model_data; EDA_TYPE = "advanced"
from post_processing import fit_apply_post_processing, evaluate_log_and_save

# %% [markdown]
# ## 1. Load Data
X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw, test_unlabelled = get_advanced_customer_model_data()

# Combine for cross-validation
X = pd.concat([X_train_raw, X_val_raw, X_test_raw])
y = pd.concat([y_train_raw, y_val_raw, y_test_raw])
test_ids = test_unlabelled.index

print(f"Features: {X.shape[1]}")
print(f"Total Train rows: {len(X)}")

# %% [markdown]
# ## 2. Tweedie Training

params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.2, # 1.1 - 1.5 is ideal for CLV
    'metric': 'mae',
    'boosting_type': 'goss',
    'learning_rate': 0.005,
    'num_leaves': 70,
    'min_child_samples': 200,
    'feature_fraction': 0.6,
    'lambda_l1': 10.0,
    'seed': 42,
    'verbosity': -1,
    'n_jobs': -1,
    'force_col_wise': True
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_unlabelled))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Training Fold {fold+1} ---")
    X_f_train, y_f_train = X.iloc[train_idx], y.iloc[train_idx]
    X_f_val, y_f_val = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_f_train, label=y_f_train)
    dval = lgb.Dataset(X_f_val, label=y_f_val, reference=dtrain)
    
    model = lgb.train(
        params, dtrain, num_boost_round=10000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(1000)]
    )
    
    oof_preds[val_idx] = model.predict(X_f_val)
    test_preds += model.predict(test_unlabelled) / kf.get_n_splits()

# Post-Fold cleanup
oof_preds = np.maximum(0, oof_preds)
test_preds = np.maximum(0, test_preds)

print(f"\nTweedie OOF Baseline MAE: {mean_absolute_error(y, oof_preds):.4f}")

# %% [markdown]
# ## 3. Post-Processing Pipeline

POST_PROCESS_METHOD = "best"

print(f"\nApplying {POST_PROCESS_METHOD} post-processing to Tweedie output...")
oof_final, test_final = fit_apply_post_processing(
    oof_preds=oof_preds,
    test_preds=test_preds,
    y_true=y,
    method=POST_PROCESS_METHOD,
    known_values=y,
    recency_train=X['recency_days'],
    recency_test=test_unlabelled['recency_days'],
    returns_train=X['return_rate'],
    returns_test=test_unlabelled['return_rate']
)

# %% [markdown]
# ## 4. Evaluation, Logging & Submission

metrics = evaluate_log_and_save(
    oof=oof_final,
    test=test_final,
    y_true=y,
    test_ids=test_ids,
    model_name="Tweedie_LightGBM_Advanced",
    eda=EDA_TYPE,
    method=POST_PROCESS_METHOD
)

print("\n=== ZERO CONCENTRATION (OOF) ===")
print(f"Actual Zeros:   {(y == 0).mean()*100:.2f}%")
print(f"Predicted Zeros: {(oof_final == 0).mean()*100:.2f}%")

# %%
