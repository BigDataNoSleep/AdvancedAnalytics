import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error, median_absolute_error
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold

# Set plotting style
sns.set_theme(style="whitegrid")

# ============================================================
# 1. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculates key metrics for regression.
    """
    mae = mean_absolute_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    
    if np.isnan(spearman):
        spearman = 0.0
        
    return {
        'MAE': mae,
        'MedianAE': med_ae,
        'Spearman': spearman
    }

# ============================================================
# 2. DATA TRANSFORMATION HELPERS
# ============================================================

def force_zero_rate(preds, zero_rate):
    """
    Forces the bottom 'zero_rate' % of predictions to zero.
    Useful for zero-inflated targets like CLV.
    """
    preds = np.asarray(preds, dtype=float).copy()
    preds = np.maximum(preds, 0)

    n_zero = int(round(len(preds) * zero_rate))
    if n_zero <= 0:
        return preds

    order = np.argsort(preds)
    preds[order[:n_zero]] = 0.0
    return preds

def snap_to_known_values(preds, known_values):
    """
    Snaps predictions to the nearest value seen in the training/validation set.
    """
    preds = np.asarray(preds, dtype=float)
    known_values = np.asarray(known_values, dtype=float)
    known_values = np.sort(np.unique(known_values))

    idx = np.searchsorted(known_values, preds)
    idx_left = np.clip(idx - 1, 0, len(known_values) - 1)
    idx_right = np.clip(idx, 0, len(known_values) - 1)

    left_vals = known_values[idx_left]
    right_vals = known_values[idx_right]

    choose_right = np.abs(preds - right_vals) < np.abs(preds - left_vals)
    return np.where(choose_right, right_vals, left_vals)

def scale_predictions(preds, factor):
    """
    Scales predictions by a constant factor.
    """
    preds = np.asarray(preds, dtype=float)
    return np.maximum(preds * factor, 0)

def learn_scaling_factor(y_true, y_pred):
    def loss(f):
        return mean_absolute_error(y_true, y_pred * f[0])
    
    res = minimize(loss, x0=[1.0])
    return res.x[0]

def clip_predictions(preds, y_true, upper_q=0.99):
    upper = np.percentile(y_true, upper_q * 100)
    return np.clip(preds, 0, upper)

def tune_zero_rate(y_true, y_pred):
    best_q, best_mae = 0, float('inf')
    
    for q in np.linspace(0.5, 0.9, 9):
        preds_z = force_zero_rate(y_pred, q)
        mae = mean_absolute_error(y_true, preds_z)
        
        if mae < best_mae:
            best_mae = mae
            best_q = q
            
    return best_q

def piecewise_scale(preds):
    preds = preds.copy()
    
    preds[preds < 50] *= 0.8
    preds[(preds >= 50) & (preds < 200)] *= 1.1
    preds[preds >= 200] *= 1.3
    
    return np.maximum(preds, 0)

def apply_matthias_filter(preds, recency):
    if recency is None: 
        return preds
        
    p_out = preds.copy()
    recency = np.asarray(recency)
    
    # Matthias rule 1: > 400 days -> 20% value
    p_out[(recency > 400) & (recency <= 500)] *= 0.2
    
    # Matthias rule 2: > 500 days -> 0 value
    p_out[recency > 500] = 0.0
    return p_out

def tune_recency_filter_cv(y_true, preds, recency):
    """
    Tunes the recency filter thresholds and penalties using 5-Fold nested CV
    on the OOF predictions. If a grid parameter doesn't beat the "Matthias" 
    defaults substantially across folds, we keep the defaults to avoid overfitting.
    """
    recency = np.asarray(recency)
    
    # 1. Base Score with Matthias rules
    base_preds = apply_matthias_filter(preds, recency)
    base_mae = mean_absolute_error(y_true, base_preds)
    
    # Search grid (restricted coarse grid)
    penalty_thresholds = [365, 400, 450, 500]
    zero_thresholds = [450, 500, 550, 600, 730]
    penalty_rates = [0.1, 0.15, 0.2, 0.25]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_params = {'penalty_thresh': 400, 'zero_thresh': 500, 'penalty_rate': 0.2} # Matthias defaults
    best_cv_mae = base_mae
    
    for p_thresh in penalty_thresholds:
        for z_thresh in zero_thresholds:
            if z_thresh <= p_thresh: continue
            for p_rate in penalty_rates:
                
                fold_maes = []
                for train_idx, val_idx in kf.split(preds):
                    y_val = y_true[val_idx]
                    p_val = preds[val_idx].copy()
                    r_val = recency[val_idx]
                    
                    mask_p = (r_val > p_thresh) & (r_val <= z_thresh)
                    mask_z = (r_val > z_thresh)
                    
                    p_val[mask_p] *= p_rate
                    p_val[mask_z] = 0.0
                    
                    fold_maes.append(mean_absolute_error(y_val, p_val))
                
                avg_cv_mae = np.mean(fold_maes)
                
                # Require at least a 1 dollar absolute improvement to overcome 'inertia'
                if avg_cv_mae < (best_cv_mae - 1.0):
                    best_cv_mae = avg_cv_mae
                    best_params = {'penalty_thresh': p_thresh, 'zero_thresh': z_thresh, 'penalty_rate': p_rate}
                    
    return best_params

def apply_recency_filter(preds, recency, params):
    if recency is None: 
        return preds
        
    p_out = preds.copy()
    recency = np.asarray(recency)
    
    mask_p = (recency > params['penalty_thresh']) & (recency <= params['zero_thresh'])
    mask_z = (recency > params['zero_thresh'])
    
    p_out[mask_p] *= params['penalty_rate']
    p_out[mask_z] = 0.0
    return p_out


def fit_apply_post_processing(oof_preds, test_preds, y_true, method_name, train_zero_rate=0.0, known_values=None, recency_train=None, recency_test=None):
    """
    Fits post-processing parameters on OOF data and identically applies them to both OOF and Test data.
    Ensures safe transformation mapping without data leakage!
    Returns: (oof_final, test_final)
    """
    oof_final = np.asarray(oof_preds, dtype=float).copy()
    test_final = np.asarray(test_preds, dtype=float).copy()
    
    oof_final = np.maximum(oof_final, 0)
    test_final = np.maximum(test_final, 0)

    if method_name == "raw":
        return oof_final, test_final

    if method_name == "force_zero_rate":
        oof_final = force_zero_rate(oof_final, train_zero_rate)
        test_final = force_zero_rate(test_final, train_zero_rate)
        return oof_final, test_final

    if method_name == "snap_only":
        if known_values is None: raise ValueError("known_values required")
        oof_final = snap_to_known_values(oof_final, known_values)
        test_final = snap_to_known_values(test_final, known_values)
        oof_final[oof_final <= 0.01] = 0.0
        test_final[test_final <= 0.01] = 0.0
        return oof_final, test_final

    if method_name == "zero_then_snap":
        if known_values is None: raise ValueError("known_values required")
        oof_final = force_zero_rate(oof_final, train_zero_rate)
        test_final = force_zero_rate(test_final, train_zero_rate)
        oof_final = snap_to_known_values(oof_final, known_values)
        test_final = snap_to_known_values(test_final, known_values)
        oof_final[oof_final <= 0.01] = 0.0
        test_final[test_final <= 0.01] = 0.0
        return oof_final, test_final

    if method_name.startswith("zeroq_"):
        q = float(method_name.split("_")[1])
        oof_final = force_zero_rate(oof_final, q)
        test_final = force_zero_rate(test_final, q)
        return oof_final, test_final

    if method_name.startswith("scale_"):
        factor = float(method_name.split("_")[1])
        oof_final = scale_predictions(oof_final, factor)
        test_final = scale_predictions(test_final, factor)
        return oof_final, test_final

    if method_name.startswith("zeroqscale_"):
        parts = method_name.split("_")
        q, factor = float(parts[1]), float(parts[2])
        oof_final = force_zero_rate(oof_final, q)
        oof_final = scale_predictions(oof_final, factor)
        test_final = force_zero_rate(test_final, q)
        test_final = scale_predictions(test_final, factor)
        return oof_final, test_final

    if method_name == "piecewise_scale":
        return piecewise_scale(oof_final), piecewise_scale(test_final)

    if method_name == "quantile_map":
        qt_pred = QuantileTransformer(output_distribution='uniform')
        qt_true = QuantileTransformer(output_distribution='uniform')
        
        # Fit logic using purely train (OOF) distribution
        qt_pred.fit(oof_final.reshape(-1, 1))
        qt_true.fit(y_true.reshape(-1, 1))
        
        p_oof = qt_pred.transform(oof_final.reshape(-1, 1))
        oof_final = qt_true.inverse_transform(p_oof).ravel()
        
        # Apply strict train-fitted map to Test to avoid test leakage
        p_test = qt_pred.transform(test_final.reshape(-1, 1))
        test_final = qt_true.inverse_transform(p_test).ravel()
        return oof_final, test_final

    if method_name == "ultra_combo":
        if known_values is None: raise ValueError("known_values required for ultra_combo")
        
        # 1. Distribution Mapping (Learn on OOF, apply same map to Test)
        qt_pred = QuantileTransformer(output_distribution='uniform')
        qt_true = QuantileTransformer(output_distribution='uniform')
        qt_pred.fit(oof_final.reshape(-1, 1))
        qt_true.fit(y_true.reshape(-1, 1))
        
        p_oof = qt_pred.transform(oof_final.reshape(-1, 1))
        oof_final = qt_true.inverse_transform(p_oof).ravel()
        p_test = qt_pred.transform(test_final.reshape(-1, 1))
        test_final = qt_true.inverse_transform(p_test).ravel()
        
        # 2. Tune zero rate on OOF
        q = tune_zero_rate(y_true, oof_final)
        oof_final = force_zero_rate(oof_final, q)
        test_final = force_zero_rate(test_final, q)
        
        # 3. Tune scaling on OOF
        factor = learn_scaling_factor(y_true, oof_final)
        oof_final = scale_predictions(oof_final, factor)
        test_final = scale_predictions(test_final, factor)
        
        # 4. Snap
        oof_final = snap_to_known_values(oof_final, known_values)
        test_final = snap_to_known_values(test_final, known_values)
        
        # 5. Recency Filter (Matthias Rules)
        if recency_train is not None and recency_test is not None:
            oof_final = apply_matthias_filter(oof_final, recency_train)
            test_final = apply_matthias_filter(test_final, recency_test)
            
        # 6. Clip limits based on Train Target dist
        oof_final = clip_predictions(oof_final, y_true)
        test_final = clip_predictions(test_final, y_true) 
        
        return oof_final, test_final

    if method_name == "ultra_combo_cv":
        if known_values is None: raise ValueError("known_values required for ultra_combo_cv")
        
        qt_pred = QuantileTransformer(output_distribution='uniform')
        qt_true = QuantileTransformer(output_distribution='uniform')
        qt_pred.fit(oof_final.reshape(-1, 1))
        qt_true.fit(y_true.reshape(-1, 1))
        
        p_oof = qt_pred.transform(oof_final.reshape(-1, 1))
        oof_final = qt_true.inverse_transform(p_oof).ravel()
        p_test = qt_pred.transform(test_final.reshape(-1, 1))
        test_final = qt_true.inverse_transform(p_test).ravel()
        
        q = tune_zero_rate(y_true, oof_final)
        oof_final = force_zero_rate(oof_final, q)
        test_final = force_zero_rate(test_final, q)
        
        factor = learn_scaling_factor(y_true, oof_final)
        oof_final = scale_predictions(oof_final, factor)
        test_final = scale_predictions(test_final, factor)
        
        oof_final = snap_to_known_values(oof_final, known_values)
        test_final = snap_to_known_values(test_final, known_values)
        
        # 5. Recency Filter (Tuned with Nested CV)
        if recency_train is not None and recency_test is not None:
            r_params = tune_recency_filter_cv(y_true, oof_final, recency_train)
            print(f"  [CV Auto-Tuned Recency Parameters]: {r_params}")
            oof_final = apply_recency_filter(oof_final, recency_train, r_params)
            test_final = apply_recency_filter(test_final, recency_test, r_params)
            
        oof_final = clip_predictions(oof_final, y_true)
        test_final = clip_predictions(test_final, y_true) 
        
        return oof_final, test_final

    if method_name == "matthias_only":
        if recency_train is not None and recency_test is not None:
            oof_final = apply_matthias_filter(oof_final, recency_train)
            test_final = apply_matthias_filter(test_final, recency_test)
        return oof_final, test_final

    if method_name == "recency_only_cv":
        if recency_train is not None and recency_test is not None:
            r_params = tune_recency_filter_cv(y_true, oof_final, recency_train)
            print(f"  [CV Auto-Tuned Recency Parameters]: {r_params}")
            oof_final = apply_recency_filter(oof_final, recency_train, r_params)
            test_final = apply_recency_filter(test_final, recency_test, r_params)
        return oof_final, test_final

    if method_name == "best":
        # Search all methods on OOF to find the winner
        best_name, _, _, _ = find_best_post_processing(oof_final, y_true, known_values, recency_train)
        print(f"Auto-selected best method: {best_name}")
        return fit_apply_post_processing(oof_preds, test_preds, y_true, best_name, train_zero_rate, known_values, recency_train, recency_test)

    raise ValueError(f"Unknown postprocess method: {method_name}")

def find_best_post_processing(oof_preds, y_true, known_values=None, recency_train=None):
    """
    Sweeps across all available post-processing methods and returns the name of the one
    that minimizes MAE on the OOF set.
    """
    train_zero_rate = (y_true == 0).mean()
    
    # Define search space
    short_list = ["raw", "snap_only", "piecewise_scale", "quantile_map", "ultra_combo", "ultra_combo_cv"]
    
    if recency_train is not None:
        short_list.append("matthias_only")
        short_list.append("recency_only_cv")
    
    # Add zero-rate grid
    for q in np.linspace(0.4, 0.85, 10):
        short_list.append(f"zeroq_{q:.4f}")
        
    # Add scaling grid (now more aggressive)
    for factor in np.linspace(0.8, 3.0, 23):
        short_list.append(f"scale_{factor:.4f}")
        
    results = []
    best_mae = float('inf')
    best_method = "raw"
    
    for method in short_list:
        try:
            # We use apply_post_processing_method for the search
            p = apply_post_processing_method(oof_preds, method, train_zero_rate, known_values, y_true, recency_train)
            mae = mean_absolute_error(y_true, p)
            results.append({"method": method, "MAE": mae})
            
            if mae < best_mae:
                best_mae = mae
                best_method = method
        except:
            continue
            
    df_results = pd.DataFrame(results).sort_values("MAE")
    print("\n--- Post-Processing Search Results (Top 5) ---")
    print(df_results.head(5).to_string(index=False))
    
    return best_method, best_mae, df_results, None


def apply_post_processing_method(preds, method_name, train_zero_rate=0.0, known_values=None, y_true=None, recency_train=None):
    """
    Standard switch for applying various post-processing techniques.
    """
    preds = np.asarray(preds, dtype=float).copy()
    preds = np.maximum(preds, 0)

    if method_name == "raw":
        return preds

    if method_name == "force_zero_rate":
        return force_zero_rate(preds, train_zero_rate)

    if method_name == "snap_only":
        if known_values is None: raise ValueError("known_values required for snap_only")
        out = snap_to_known_values(preds, known_values)
        out[out <= 0.01] = 0.0
        return out

    if method_name == "zero_then_snap":
        if known_values is None: raise ValueError("known_values required for zero_then_snap")
        out = force_zero_rate(preds, train_zero_rate)
        out = snap_to_known_values(out, known_values)
        out[out <= 0.01] = 0.0
        return out

    if method_name.startswith("zeroq_"):
        q = float(method_name.split("_")[1])
        return force_zero_rate(preds, q)

    if method_name.startswith("scale_"):
        factor = float(method_name.split("_")[1])
        return scale_predictions(preds, factor)

    if method_name.startswith("zeroqscale_"):
        parts = method_name.split("_")
        q, factor = float(parts[1]), float(parts[2])
        out = force_zero_rate(preds, q)
        out = scale_predictions(out, factor)
        return out

    if method_name == "piecewise_scale":
        return piecewise_scale(preds)

    if method_name == "quantile_map":
        if y_true is None: raise ValueError("y_true required for quantile_map")
        return quantile_map(preds, y_true)

    if method_name == "strong_combo":
        if y_true is None: raise ValueError("y_true required for strong_combo")
        if known_values is None: raise ValueError("known_values required for strong_combo")
        
        # 1. Distrbution Mapping
        out = quantile_map(preds, y_true)
        
        # 2. Zero-Rate
        q = tune_zero_rate(y_true, out)
        out = force_zero_rate(out, q)
        
        # 3. Scaling
        factor = learn_scaling_factor(y_true, out)
        out = scale_predictions(out, factor)
        # 4. Snap
        out = snap_to_known_values(out, known_values)
        
        # 5. Clip
        out = clip_predictions(out, y_true)
        
        return out

    if method_name == "ultra_combo":
        if y_true is None: raise ValueError("y_true required for ultra_combo")
        if known_values is None: raise ValueError("known_values required for ultra_combo")
        
        out = quantile_map(preds, y_true)
        
        q = tune_zero_rate(y_true, out)
        out = force_zero_rate(out, q)
        
        factor = learn_scaling_factor(y_true, out)
        out = scale_predictions(out, factor)
        
        out = snap_to_known_values(out, known_values)
        
        if recency_train is not None:
            out = apply_matthias_filter(out, recency_train)
            
        out = clip_predictions(out, y_true)
        
        return out

    if method_name == "ultra_combo_cv":
        if y_true is None: raise ValueError("y_true required for ultra_combo_cv")
        if known_values is None: raise ValueError("known_values required for ultra_combo_cv")
        
        out = quantile_map(preds, y_true)
        q = tune_zero_rate(y_true, out)
        out = force_zero_rate(out, q)
        factor = learn_scaling_factor(y_true, out)
        out = scale_predictions(out, factor)
        out = snap_to_known_values(out, known_values)
        
        if recency_train is not None:
            r_params = tune_recency_filter_cv(y_true, out, recency_train)
            out = apply_recency_filter(out, recency_train, r_params)
            
        out = clip_predictions(out, y_true)
        return out

    if method_name == "matthias_only":
        if recency_train is not None:
            return apply_matthias_filter(preds, recency_train)
        return preds

    if method_name == "recency_only_cv":
        if recency_train is not None:
            r_params = tune_recency_filter_cv(y_true, preds, recency_train)
            return apply_recency_filter(preds, recency_train, r_params)
        return preds

    raise ValueError(f"Unknown postprocess method: {method_name}")

# ============================================================
# 3. SINGLE MODEL PLOTS
# ============================================================

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Standardized feature importance plotter for CatBoost, LightGBM, and XGBoost.
    """
    plt.figure(figsize=(10, 8))
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'feature_importance'): # LightGBM Booster
            importances = model.feature_importance()
        elif hasattr(model, 'get_score'): # XGBoost Booster
            score_dict = model.get_score(importance_type='gain')
            importances = np.array([score_dict.get(f, 0) for f in feature_names])
        else:
            print(f"Warning: Model type {type(model)} does not support feature importance extraction.")
            return
            
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(top_n)
        
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

def plot_diagnostic_results(y_true, y_pred, model_name):
    """
    Generates diagnostic plots: Actual vs Predicted and Error Distribution.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.xlabel('Actual Future Revenue')
    plt.ylabel('Predicted Future Revenue')
    plt.show()
    
    errors = y_pred - y_true
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.title(f'Prediction Error Distribution - {model_name}')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.show()

# ============================================================
# 4. ENSEMBLE PLOTS
# ============================================================

def plot_model_correlations(oof_dict):
    """
    Plots a heatmap of Spearman correlations between models.
    """
    df = pd.DataFrame(oof_dict)
    corr = df.corr(method='spearman')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0.8, fmt=".4f")
    plt.title("Model OOF Correlation (Spearman)")
    plt.show()

def plot_ensemble_diagnostics(y_true, oof_dict, ensemble_pred, weights=None):
    """
    Comparison of individual models vs ensemble performance.
    """
    # 1. Weight Plot (if provided)
    if weights is not None:
        plt.figure(figsize=(8, 5))
        w_df = pd.DataFrame({'Model': list(weights.keys()), 'Weight': list(weights.values())})
        sns.barplot(x='Weight', y='Model', data=w_df, palette='magma')
        plt.title("Ensemble Model Weights")
        plt.show()

    # 2. Error Comparison Plot
    plt.figure(figsize=(10, 6))
    for name, pred in oof_dict.items():
        err = pred - y_true
        sns.kdeplot(err, label=f"{name} (MAE: {mean_absolute_error(y_true, pred):.2f})", alpha=0.5)
    
    ensemble_err = ensemble_pred - y_true
    sns.kdeplot(ensemble_err, label=f"Ensemble (MAE: {mean_absolute_error(y_true, ensemble_pred):.2f})", 
                linewidth=3, color='black', alpha=0.9)
    
    plt.title("OOF Error Distribution Comparison")
    plt.xlabel("Prediction Error")
    plt.legend()
    plt.show()

    # 3. Distribution Comparison (Train vs Pred)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true[y_true > 0], label="Train Target (>0)", color='red', fill=True, alpha=0.1)
    sns.kdeplot(ensemble_pred[ensemble_pred > 0], label="Ensemble Pred (>0)", color='blue', fill=True, alpha=0.1)
    plt.title("Positive Revenue Distribution Comparison")
    plt.xlabel("Revenue")
    plt.legend()
    plt.show()

# ============================================================
# 5. LOGGING & LEADERS
# ============================================================

def log_experiment(model_name, metrics, postprocess_method="none", zero_rate=0, eda_used="standard"):
    """
    Appends the run results to a central CSV file for easy comparison.
    """
    try:
        # Resolve submissions directory relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, 'submissions')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'experiment_log.csv')
        
        # Prepare the log entry
        entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Model': model_name,
            'EDA': eda_used,
            'MAE': f"{metrics.get('MAE', 0):.4f}",
            'Spearman': f"{metrics.get('Spearman', 0):.6f}",
            'ZeroRate': f"{zero_rate*100:.2f}%",
            'PostProcess': postprocess_method
        }
        
        df_entry = pd.DataFrame([entry])
        
        # Append or create
        if not os.path.isfile(log_path):
            df_entry.to_csv(log_path, index=False)
        else:
            df_entry.to_csv(log_path, mode='a', header=False, index=False)
            
        print(f"\n--- Experiment logged to: {log_path} ---")
        
    except Exception as e:
        print(f"Warning: Failed to log experiment: {e}")

# ============================================================
# 6. ORCHESTRATORS
# ============================================================

def evaluate_log_and_save(oof_preds, test_preds, y_true, test_ids, model_name, eda_used="standard", postprocess_method="none"):
    """
    Standard orchestrator that calculates metrics, logs to CSV, generates plots and 
    automatically saves the final predictions for OOF and Test data.
    """
    print(f"\n--- Running Unified Evaluation for {model_name} (EDA: {eda_used}) ---")
    metrics = calculate_metrics(y_true, oof_preds)
    print("\nModel Metrics:")
    for m, val in metrics.items():
        print(f"  {m:<10}: {val:.4f}")
    
    plot_diagnostic_results(y_true, oof_preds, model_name)
    
    errors = np.abs(oof_preds - y_true)
    print("\nDetailed Error Statistics:")
    print(f"  Mean Absolute Error:   {errors.mean():.4f}")
    print(f"  Median Absolute Error: {np.median(errors):.4f}")
    print(f"  Max Error:             {errors.max():.4f}")
    print(f"  90th Percentile Error: {np.percentile(errors, 90):.4f}")
    
    # Automated Logging
    zero_rate = (oof_preds == 0).mean()
    log_experiment(model_name, metrics, postprocess_method=postprocess_method, zero_rate=zero_rate, eda_used=eda_used)
    
    # Automated Submission Saves
    if test_preds is not None and test_ids is not None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sub_dir = os.path.join(base_dir, 'submissions')
        os.makedirs(sub_dir, exist_ok=True)
        
        # Save OOF
        oof_df = pd.DataFrame({'revenue': oof_preds}, index=y_true.index if hasattr(y_true, 'index') else range(len(oof_preds)))
        oof_path = os.path.join(sub_dir, f'{model_name}_oof.csv')
        oof_df.to_csv(oof_path, index=True) # Keep index for OOF debugging
        print(f"Saved OOF to: {oof_path}")
        
        # Save Test (Standard Submission Format)
        sub_df = pd.DataFrame({
            'cust_id': test_ids,
            'revenue': test_preds
        })
        sub_path = os.path.join(sub_dir, f'{model_name}_predictions.csv')
        
        sub_df.to_csv(sub_path, index=False)
        print(f"Saved Test Predictions to: {sub_path}")
        
    return metrics
