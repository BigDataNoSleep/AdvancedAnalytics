import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error, median_absolute_error
from scipy.stats import spearmanr

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

def apply_post_processing_method(preds, method_name, train_zero_rate=0.0, known_values=None):
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

def run_full_post_processing(model, X_train, y_true, y_pred, model_name, eda_used="standard"):
    """
    Orchestrator for single model tasks.
    """
    print(f"\n--- Running Post-Processing for {model_name} (EDA: {eda_used}) ---")
    metrics = calculate_metrics(y_true, y_pred)
    print("\nModel Metrics:")
    for m, val in metrics.items():
        print(f"  {m:<10}: {val:.4f}")
    
    plot_feature_importance(model, X_train.columns, model_name)
    plot_diagnostic_results(y_true, y_pred, model_name)
    
    errors = np.abs(y_pred - y_true)
    print("\nDetailed Error Statistics:")
    print(f"  Mean Absolute Error:   {errors.mean():.4f}")
    print(f"  Median Absolute Error: {np.median(errors):.4f}")
    print(f"  Max Error:             {errors.max():.4f}")
    print(f"  90th Percentile Error: {np.percentile(errors, 90):.4f}")
    
    # NEW: Automated Logging
    zero_rate = (y_pred == 0).mean()
    log_experiment(model_name, metrics, zero_rate=zero_rate, eda_used=eda_used)
    
    return metrics

def run_ensemble_post_processing(y_true, oof_dict, ensemble_pred, weights=None, postprocess_method="none", eda_used="standard"):
    """
    Orchestrator for ensemble analysis.
    """
    print(f"\n--- Running Ensemble Post-Processing (EDA: {eda_used}) ---")
    plot_model_correlations(oof_dict)
    plot_ensemble_diagnostics(y_true, oof_dict, ensemble_pred, weights)
    
    metrics = calculate_metrics(y_true, ensemble_pred)
    print("\nEnsemble Metrics:")
    for m, val in metrics.items():
        print(f"  {m:<10}: {val:.4f}")
        
    errors = np.abs(ensemble_pred - y_true)
    print("\nDetailed Ensemble Statistics:")
    print(f"  Mean Absolute Error:   {errors.mean():.4f}")
    print(f"  Median Absolute Error: {np.median(errors):.4f}")
    print(f"  Max Error:             {errors.max():.4f}")
    print(f"  90th Percentile Error: {np.percentile(errors, 90):.4f}")
    
    # NEW: Automated Logging
    zero_rate = (ensemble_pred == 0).mean()
    log_experiment("Final_Ensemble", metrics, postprocess_method=postprocess_method, zero_rate=zero_rate, eda_used=eda_used)
    
    return metrics
