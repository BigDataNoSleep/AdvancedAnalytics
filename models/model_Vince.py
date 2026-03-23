#%%
"""
Customer Revenue Prediction Framework

A clean, modular, high-quality modelling code to predict future customer revenue
with two-stage modelling, baseline parity, and competitive tree models.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

from xgboost import XGBRegressor, XGBClassifier

from lightgbm import LGBMRegressor, early_stopping

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Set visual style and pandas display options
sns.set_theme(style="whitegrid")
pd.set_option('display.max_rows', 200)

RANDOM_STATE = 42

# Add parent directory to path to import eda_transactions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda_transactions import get_customer_model_data

#%%
# ==============================================================================
# 1. EVALUATION & DATA PREP
# ==============================================================================

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using primary (MAE) and secondary (Spearman) metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    if np.isnan(spearman):
        spearman = 0.0
    return mae, spearman

def prepare_features(X_train, X_val, X_test, test_unlabelled, model_type='tree'):
    """
    Handle skewness for specific features and scale optionally.
    Tree models should not be scaled.
    """
    X_tr = X_train.copy()
    X_v = X_val.copy()
    X_te = X_test.copy()
    X_comp = test_unlabelled.copy()
    
    skewed_cols = ['total_spent', 'revenue_last_90d', 'avg_order_value']
    for col in skewed_cols:
        if col in X_tr.columns:
            # log1p to handle skewness
            X_tr[col] = np.log1p(np.maximum(0, X_tr[col]))
            X_v[col] = np.log1p(np.maximum(0, X_v[col]))
            X_te[col] = np.log1p(np.maximum(0, X_te[col]))
            X_comp[col] = np.log1p(np.maximum(0, X_comp[col]))
            
    if model_type == 'linear':
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_v = pd.DataFrame(scaler.transform(X_v), columns=X_v.columns, index=X_v.index)
        X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
        X_comp = pd.DataFrame(scaler.transform(X_comp), columns=X_comp.columns, index=X_comp.index)
        
    return X_tr, X_v, X_te, X_comp

#%%
# ==============================================================================
# 2. MODELING FUNCTIONS
# ==============================================================================

def train_linear_models(X_train, y_train, X_val, X_test, test_unlabelled):
    X_tr, X_v, X_te, X_comp = prepare_features(X_train, X_val, X_test, test_unlabelled, model_type='linear')
    
    models = {
        '1_Median_Baseline': DummyRegressor(strategy='median'),
        '2_Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        '3_Lasso': Lasso(alpha=0.1, random_state=RANDOM_STATE),
        '4_ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_train)
        val_preds = np.maximum(0, model.predict(X_v))
        test_preds = np.maximum(0, model.predict(X_te))
        comp_preds = np.maximum(0, model.predict(X_comp))
        results[name] = {
            'model': model, 
            'val_preds': val_preds, 
            'test_preds': test_preds, 
            'comp_preds': comp_preds
        }
    return results

def train_tree_models(X_train, y_train, X_val, X_test, test_unlabelled):
    X_tr, X_v, X_te, X_comp = prepare_features(X_train, X_val, X_test, test_unlabelled, model_type='tree')
    
    models = {
        '5_RandomForest': RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        '6_ExtraTrees': ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        '7_GradientBoosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_train)
        val_preds = np.maximum(0, model.predict(X_v))
        test_preds = np.maximum(0, model.predict(X_te))
        comp_preds = np.maximum(0, model.predict(X_comp))
        results[name] = {
            'model': model, 
            'val_preds': val_preds, 
            'test_preds': test_preds, 
            'comp_preds': comp_preds
        }
    return results

def train_boosting_models(X_train, y_train, X_val, y_val, X_test, test_unlabelled):
    X_tr, X_v, X_te, X_comp = prepare_features(X_train, X_val, X_test, test_unlabelled, model_type='tree')
    results = {}
    
    # Model 8: XGBoost
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=RANDOM_STATE, early_stopping_rounds=50)
    xgb.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
    
    val_preds = np.maximum(0, xgb.predict(X_v))
    test_preds = np.maximum(0, xgb.predict(X_te))
    comp_preds = np.maximum(0, xgb.predict(X_comp))
    results['8_XGBoost'] = {
        'model': xgb, 
        'val_preds': val_preds, 
        'test_preds': test_preds, 
        'comp_preds': comp_preds
    }
    
    # Model 9: LightGBM
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=RANDOM_STATE)
    lgb.fit(X_tr, y_train, eval_set=[(X_v, y_val)], callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
    val_preds = np.maximum(0, lgb.predict(X_v))
    test_preds = np.maximum(0, lgb.predict(X_te))
    comp_preds = np.maximum(0, lgb.predict(X_comp))
    results['9_LightGBM'] = {
        'model': lgb, 
        'val_preds': val_preds, 
        'test_preds': test_preds, 
        'comp_preds': comp_preds
    }
        
    return results

def train_two_stage_models(X_train, y_train, X_val, y_val, X_test, test_unlabelled):
    X_tr, X_v, X_te, X_comp = prepare_features(X_train, X_val, X_test, test_unlabelled, model_type='tree')
    results = {}
    
    # Target transformations for regression stage
    y_train_clf = (y_train > 0).astype(int)
    mask_train_pos = y_train > 0
    y_train_reg = np.log1p(y_train[mask_train_pos])
    X_train_reg = X_tr[mask_train_pos]
    
    y_val_clf = (y_val > 0).astype(int)
    mask_val_pos = y_val > 0
    y_val_reg = np.log1p(y_val[mask_val_pos])
    X_val_reg = X_v[mask_val_pos]
    
    # Model 10: Two-stage Random Forest
    rf_clf = RandomForestClassifier(n_estimators=500, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
    rf_reg = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
    
    rf_clf.fit(X_tr, y_train_clf)
    rf_reg.fit(X_train_reg, y_train_reg)
    
    rf_val_preds = rf_clf.predict_proba(X_v)[:, 1] * np.expm1(rf_reg.predict(X_v))
    rf_test_preds = rf_clf.predict_proba(X_te)[:, 1] * np.expm1(rf_reg.predict(X_te))
    rf_comp_preds = rf_clf.predict_proba(X_comp)[:, 1] * np.expm1(rf_reg.predict(X_comp))
    
    results['10_Two_Stage_RF'] = {
        'model': {'classifier': rf_clf, 'regressor': rf_reg},
        'val_preds': rf_val_preds,
        'test_preds': rf_test_preds,
        'comp_preds': rf_comp_preds
    }
    
    # Model 11: Two-stage XGBoost
    xgb_clf = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=RANDOM_STATE, early_stopping_rounds=50)
    xgb_reg = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=RANDOM_STATE)
    
    xgb_clf.fit(X_tr, y_train_clf, eval_set=[(X_v, y_val_clf)], verbose=False)
    
    if len(X_val_reg) > 0:
        xgb_reg.set_params(early_stopping_rounds=50)
        xgb_reg.fit(X_train_reg, y_train_reg, eval_set=[(X_val_reg, y_val_reg)], verbose=False)
    else:
        xgb_reg.fit(X_train_reg, y_train_reg)
        
    xgb_val_preds = xgb_clf.predict_proba(X_v)[:, 1] * np.expm1(xgb_reg.predict(X_v))
    xgb_test_preds = xgb_clf.predict_proba(X_te)[:, 1] * np.expm1(xgb_reg.predict(X_te))
    xgb_comp_preds = xgb_clf.predict_proba(X_comp)[:, 1] * np.expm1(xgb_reg.predict(X_comp))
    
    results['11_Two_Stage_XGB'] = {
        'model': {'classifier': xgb_clf, 'regressor': xgb_reg},
        'val_preds': xgb_val_preds,
        'test_preds': xgb_test_preds,
        'comp_preds': xgb_comp_preds
    }
    
    return results

#%%
# ==============================================================================
# 3. COMPARISON & ENSEMBLE
# ==============================================================================

def compare_all_models(all_results, y_val, y_test):
    metrics = []
    
    for name, res in all_results.items():
        val_mae, val_spearman = evaluate_predictions(y_val, res['val_preds'])
        test_mae, test_spearman = evaluate_predictions(y_test, res['test_preds'])
        
        metrics.append({
            'Model': name,
            'Validation_MAE': val_mae,
            'Validation_Spearman': val_spearman,
            'Test_MAE': test_mae,
            'Test_Spearman': test_spearman
        })
        
    results_df = pd.DataFrame(metrics).sort_values('Validation_MAE', ascending=True).reset_index(drop=True)
    return results_df

def build_ensemble(results_df, all_results, y_val, y_test):
    top_3_models = results_df['Model'].head(3).tolist()
    
    val_ens = np.mean([all_results[m]['val_preds'] for m in top_3_models], axis=0)
    test_ens = np.mean([all_results[m]['test_preds'] for m in top_3_models], axis=0)
    comp_ens = np.mean([all_results[m]['comp_preds'] for m in top_3_models], axis=0)
    
    val_mae, val_spearman = evaluate_predictions(y_val, val_ens)
    test_mae, test_spearman = evaluate_predictions(y_test, test_ens)
    
    ensemble_results = {
        'Model': '12_Ensemble_Top_3',
        'Validation_MAE': val_mae,
        'Validation_Spearman': val_spearman,
        'Test_MAE': test_mae,
        'Test_Spearman': test_spearman
    }
    
    results_df = pd.concat([results_df, pd.DataFrame([ensemble_results])]).sort_values('Validation_MAE', ascending=True).reset_index(drop=True)
    
    all_results['12_Ensemble_Top_3'] = {
        'model': 'Ensemble',
        'val_preds': val_ens,
        'test_preds': test_ens,
        'comp_preds': comp_ens
    }
    
    return results_df, all_results

#%%
# ==============================================================================
# 4. VISUALISATIONS
# ==============================================================================

def plot_model_performance(results_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Validation_MAE', y='Model', palette='viridis')
    plt.title('Model Comparison - Validation MAE')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Validation_Spearman', y='Model', palette='magma')
    plt.title('Model Comparison - Validation Spearman')
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='#3498db')
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title(f'{model_name} - Prediction vs Actual')
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, color='#e74c3c')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Predicted Revenue')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residual Plot')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, model_name, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='mako')
        plt.title(f'{model_name} - Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    elif isinstance(model, dict) and 'regressor' in model:
        plot_feature_importance(model['regressor'], feature_names, f"{model_name} (Regressor Stage)", top_n)

def plot_distribution_comparison(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true, label='Actual Distribution', cumulative=False, color='#2ecc71', lw=2)
    sns.kdeplot(y_pred, label='Predicted Distribution', cumulative=False, color='#9b59b6', lw=2)
    plt.title(f'{model_name} - True vs Predicted Distributions')
    plt.xlabel('Revenue')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
# ==============================================================================
# 5. EXECUTION: LOAD DATA
# ==============================================================================
print("Loading data...")
X_train, X_val, X_test, y_train, y_val, y_test, test = get_customer_model_data()
all_results = {}

#%%
# ==============================================================================
# 6. TRAIN BASELINE AND LINEAR MODELS
# ==============================================================================
print("Training Baseline and Linear Models...")
lin_res = train_linear_models(X_train, y_train, X_val, X_test, test)
all_results.update(lin_res)

#%%
# ==============================================================================
# 7. TRAIN TREE MODELS
# ==============================================================================
print("Training Tree Models...")
tree_res = train_tree_models(X_train, y_train, X_val, X_test, test)
all_results.update(tree_res)

#%%
# ==============================================================================
# 8. TRAIN BOOSTING MODELS
# ==============================================================================
print("Training Boosting Models...")
boost_res = train_boosting_models(X_train, y_train, X_val, y_val, X_test, test)
all_results.update(boost_res)

#%%
# ==============================================================================
# 9. TRAIN TWO-STAGE MODELS
# ==============================================================================
print("Training Two-Stage Models...")
two_stage_res = train_two_stage_models(X_train, y_train, X_val, y_val, X_test, test)
all_results.update(two_stage_res)

#%%
# ==============================================================================
# 10. COMPARE MODELS & ENSEMBLE
# ==============================================================================
print("Comparing models...")
results_df = compare_all_models(all_results, y_val, y_test)

print("Building ensemble...")
results_df, all_results = build_ensemble(results_df, all_results, y_val, y_test)

#%%
# ==============================================================================
# 11. FINAL OUTPUTS
# ==============================================================================
print("\n" + "="*50)
print("MODEL COMPARISON TABLE")
print("="*50)
print(results_df.head(10))

best_model_name = results_df.iloc[0]['Model']
best_res = all_results[best_model_name]

print("\n" + "="*50)
print("BEST MODEL SUMMARY:")
print("="*50)
print(f"Best model name:     {best_model_name}")
print(f"Validation MAE:      {results_df.iloc[0]['Validation_MAE']:.4f}")
print(f"Test MAE:            {results_df.iloc[0]['Test_MAE']:.4f}")
print(f"Spearman (Val):      {results_df.iloc[0]['Validation_Spearman']:.4f}")

if best_model_name != '12_Ensemble_Top_3':
    best_model_obj = all_results[best_model_name]['model']
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print("\nTop 10 Feature Importances:")
        for rank, idx in enumerate(indices, 1):
            col = X_train.columns[idx]
            score = importances[idx]
            print(f"{rank}. {col} ({score:.4f})")
    elif isinstance(best_model_obj, dict) and 'regressor' in best_model_obj:
        reg = best_model_obj['regressor']
        if hasattr(reg, 'feature_importances_'):
            importances = reg.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            print("\nTop 10 Feature Importances (Regressor Stage):")
            for rank, idx in enumerate(indices, 1):
                col = X_train.columns[idx]
                score = importances[idx]
                print(f"{rank}. {col} ({score:.4f})")

#%%
# ==============================================================================
# 12. GENERATE DIAGNOSTICS & PLOTS
# ==============================================================================
print("\nGenerating Visual Diagnostics...")
plot_model_performance(results_df)

# Top 3 Models diagnostics
top_3 = results_df['Model'].head(3).tolist()
for m_name in top_3:
    plot_prediction_vs_actual(y_val, all_results[m_name]['val_preds'], m_name)
    plot_residuals(y_val, all_results[m_name]['val_preds'], m_name)

# Feature Importances for RF and XGBoost (if available in results)
rf_name = '5_RandomForest'
if rf_name in all_results:
    plot_feature_importance(all_results[rf_name]['model'], X_train.columns, "RandomForest")

xgb_name = '8_XGBoost'
if xgb_name in all_results:
    plot_feature_importance(all_results[xgb_name]['model'], X_train.columns, "XGBoost")

# Distribution comparison for best model
plot_distribution_comparison(y_val, best_res['val_preds'], best_model_name)

# Output competition test predictions
test_predictions = pd.DataFrame({
    'prediction': best_res['comp_preds']
}, index=test.index)

print("\nFinal test predictions are ready in `test_predictions` dataframe and saved to `test_predictions.csv`.")

#%%
# Save to CSV for scoreboard
# RECOVERY BLOCK: If eda_transactions was not fully re-run, index might be numeric.
# We recover string IDs (sorted by cust_id as per groupby default in eda_transactions)
if isinstance(test_predictions.index[0], (int, np.integer)):
    print("\n[!] Numeric index detected. Recovering original string IDs...")
    _test_df = pd.read_csv("/Users/vincecoppens/Documents/Courses/Big Data/AdvancedAnalytics/data/transactions_2016_2017.csv")
    _cust_df = pd.read_csv("/Users/vincecoppens/Documents/Courses/Big Data/AdvancedAnalytics/data/customer_clv_train.csv")
    _merged = pd.merge(_test_df, _cust_df, on='cust_id', how='left')
    _actual_ids = sorted(_merged[_merged['revenue_2018_2019'].isna()]['cust_id'].unique())
    
    if len(_actual_ids) == len(test_predictions):
        test_predictions.index = _actual_ids
        print(f"Successfully recovered {len(_actual_ids)} string IDs.")
    else:
        print(f"Warning: Count mismatch during recovery ({len(_actual_ids)} vs {len(test_predictions)}).")

test_predictions.to_csv('test_predictions.csv', index_label='cust_id')
print("\nFinal test predictions saved to `test_predictions.csv` with correct IDs.")
# %%
