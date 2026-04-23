# %% [markdown]
# # 2-Stage Hurdle Model (LightGBM)
# Stage 1: Binary Classification (Will they spend?)
# Stage 2: Log-Regression (How much will they spend?)
# Unified via the modular post-processing pipeline.

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
# ## 2. Hurdle Training (2-Stage CV)

# Hyperparameters
clf_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbosity': -1
}

reg_params = {
    'objective': 'regression_l1',
    'boosting_type': 'goss',
    'learning_rate': 0.005,
    'num_leaves': 70,
    'min_child_samples': 200,
    'feature_fraction': 0.6,
    'lambda_l1': 10.0,
    'seed': 42,
    'verbosity': -1
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_unlabelled))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Training Fold {fold+1} ---")
    X_f_train, y_f_train = X.iloc[train_idx], y.iloc[train_idx]
    X_f_val, y_f_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # --- STAGE 1: CLASSIFICATION (Buy vs No Buy) ---
    y_f_train_clf = (y_f_train > 0).astype(int)
    y_f_val_clf = (y_f_val > 0).astype(int)
    
    dtrain_clf = lgb.Dataset(X_f_train, label=y_f_train_clf)
    dval_clf = lgb.Dataset(X_f_val, label=y_f_val_clf, reference=dtrain_clf)
    
    clf_model = lgb.train(
        clf_params, dtrain_clf, num_boost_round=2000,
        valid_sets=[dval_clf],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    prob_val = clf_model.predict(X_f_val)
    prob_test = clf_model.predict(test_unlabelled)
    
    # --- STAGE 2: REGRESSION (How much, if they buy) ---
    # We only train on the positive spenders in the training fold
    spender_mask = y_f_train > 0
    X_f_train_reg = X_f_train[spender_mask]
    y_f_train_reg = np.log1p(y_f_train[spender_mask]) # LOG-TARGET Transformation
    
    dtrain_reg = lgb.Dataset(X_f_train_reg, label=y_f_train_reg)
    
    # Note: Validation for regression is also only on spenders
    spender_mask_val = y_f_val > 0
    X_f_val_reg = X_f_val[spender_mask_val]
    y_f_val_reg = np.log1p(y_f_val[spender_mask_val])
    dval_reg = lgb.Dataset(X_f_val_reg, label=y_f_val_reg, reference=dtrain_reg)
    
    reg_model = lgb.train(
        reg_params, dtrain_reg, num_boost_round=5000,
        valid_sets=[dval_reg],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)]
    )
    
    # Predict amounts (and reverse log)
    amount_val = np.expm1(reg_model.predict(X_f_val))
    amount_test = np.expm1(reg_model.predict(test_unlabelled))
    
    # --- COMBINE ---
    oof_preds[val_idx] = prob_val * amount_val
    test_preds += (prob_test * amount_test) / kf.get_n_splits()

print(f"\nFinal Hurdle OOF MAE (Baseline): {mean_absolute_error(y, oof_preds):.4f}")

# %% [markdown]
# ## 3. Post-Processing Pipeline (Automatic Search)

POST_PROCESS_METHOD = "best"

print(f"\nApplying {POST_PROCESS_METHOD} post-processing to Hurdle output...")
oof_final, test_final = fit_apply_post_processing(
    oof_preds=oof_preds,
    test_preds=test_preds,
    y_true=y.values,
    method_name=POST_PROCESS_METHOD,
    known_values=y.values,
    recency_train=X['recency_days'],
    recency_test=test_unlabelled['recency_days']
)

# %% [markdown]
# ## 4. Evaluation, Logging & Submission

metrics = evaluate_log_and_save(
    oof_preds=oof_final,
    test_preds=test_final,
    y_true=y,
    test_ids=test_ids,
    model_name="Hurdle_LightGBM_Advanced",
    eda_used=EDA_TYPE,
    postprocess_method=POST_PROCESS_METHOD
)

print("\n=== ZERO CONCENTRATION (OOF) ===")
print(f"Actual Zeros:   {(y == 0).mean()*100:.2f}%")
print(f"Predicted Zeros: {(oof_final == 0).mean()*100:.2f}%")
# %%
