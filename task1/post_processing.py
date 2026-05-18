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

# --- SETTINGS ---
sns.set_theme(style="whitegrid")

# Matthias proven defaults
MATTHIAS_DEFAULTS = {
    'penalty_thresh': 400,
    'zero_thresh': 500,
    'penalty_rate': 0.2,
    'floor': 5.0
}

# ============================================================
# 1. METRICS & DIAGNOSTICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculates key regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    return {
        'MAE': mae,
        'MedianAE': med_ae,
        'Spearman': 0.0 if np.isnan(spearman) else spearman
    }

# ============================================================
# 2. CORE TRANSFORMERS (Low-Level Math)
# ============================================================

def apply_zero_rate(preds, rate):
    """Forces the bottom 'rate' % of predictions to zero."""
    preds = np.maximum(np.asarray(preds, dtype=float).copy(), 0)
    n_zero = int(round(len(preds) * rate))
    if n_zero > 0:
        order = np.argsort(preds)
        preds[order[:n_zero]] = 0.0
    return preds

def apply_scaling(preds, factor):
    """Scales predictions by a constant factor."""
    return np.maximum(np.asarray(preds, dtype=float) * factor, 0)

def apply_clipping(preds, y_true, q=0.99):
    """Clips predictions to the q-th percentile of training data."""
    upper = np.percentile(y_true, q * 100)
    return np.clip(preds, 0, upper)

def apply_snapping(preds, known_values):
    """Snaps predictions to the nearest value seen in the training set."""
    preds = np.asarray(preds, dtype=float)
    known = np.sort(np.unique(np.asarray(known_values, dtype=float)))
    idx = np.searchsorted(known, preds)
    idx_l = np.clip(idx - 1, 0, len(known) - 1)
    idx_r = np.clip(idx, 0, len(known) - 1)
    choose_r = np.abs(preds - known[idx_r]) < np.abs(preds - known[idx_l])
    return np.where(choose_r, known[idx_r], known[idx_l])

def apply_matthias_filter(preds, recency, params=MATTHIAS_DEFAULTS):
    """Applies the manual recency-based penalties and noise floor."""
    if recency is None: return preds
    p = np.asarray(preds, dtype=float).copy()
    r = np.asarray(recency)
    
    # Recency penalties (Matthias legacy)
    p[(r > params['penalty_thresh']) & (r <= params['zero_thresh'])] *= params['penalty_rate']
    p[r > params['zero_thresh']] = 0.0
    
    # Noise floor
    p[p < params.get('floor', 5.0)] = 0.0
    return p

def apply_behavioral_filter(preds, recency=None, returns=None):
    """
    Applies smarter behavioral filters based on EDA insights:
    1. Smoother Recency Decay: Instead of hard zero at 500, use a decay after 450.
    2. Returns Penalty: High return rates (> 85%) are severe churn signals.
    """
    p = np.asarray(preds, dtype=float).copy()
    
    # 1. Smoother Recency Decay (Insight #1)
    if recency is not None:
        r = np.asarray(recency)
        # 450-600 days: 50% penalty
        p[r > 450] *= 0.5
        # 600+ days: 90% penalty (keep some 'zombie' hope alive)
        p[r > 600] *= 0.2
    
    # 2. Returns Penalty (Insight #2)
    if returns is not None:
        ret = np.asarray(returns)
        # Extremely high return rate (> 85%) is a death signal for CLV
        p[ret > 0.85] *= 0.5
        
    # Standard Noise floor
    p[p < 5.0] = 0.0
    return p

# ============================================================
# 3. TUNING LOGIC (Parameter Optimization)
# ============================================================

def tune_zero_rate(y_true, preds):
    """Finds optimal zero-rate by searching around the actual observed zero rate."""
    true_zero_rate = (y_true == 0).mean()
    
    # Search in a range around the actual zero rate (e.g., +/- 15%)
    # This prevents the model from becoming way too pessimistic (e.g. 90% zeros)
    q_min = max(0, true_zero_rate - 0.15)
    q_max = min(0.95, true_zero_rate + 0.15)
    
    best_q, best_mae = 0, float('inf')
    for q in np.linspace(q_min, q_max, 11):
        mae = mean_absolute_error(y_true, apply_zero_rate(preds, q))
        if mae < best_mae:
            best_mae, best_q = mae, q
    return best_q

def tune_scaling_factor(y_true, preds):
    """Finds optimal scaling factor using optimization."""
    res = minimize(lambda f: mean_absolute_error(y_true, preds * f[0]), x0=[1.0])
    return res.x[0]

def tune_recency_filter_cv(y_true, preds, recency, inertia=2.0):
    """Nested CV search for optimal recency thresholds."""
    recency, y_true, preds = np.asarray(recency), np.asarray(y_true), np.asarray(preds)
    base_mae = mean_absolute_error(y_true, apply_matthias_filter(preds, recency))
    
    best_params = MATTHIAS_DEFAULTS.copy()
    best_cv_mae = base_mae
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for p_t in [365, 400, 450]:
        for z_t in [500, 550, 600, 730]:
            if z_t <= p_t: continue
            for p_r in [0.1, 0.2, 0.3]:
                fold_maes = []
                for t_idx, v_idx in kf.split(preds):
                    p_v = apply_matthias_filter(preds[v_idx], recency[v_idx], 
                                               {'penalty_thresh': p_t, 'zero_thresh': z_t, 'penalty_rate': p_r})
                    fold_maes.append(mean_absolute_error(y_true[v_idx], p_v))
                
                avg_mae = np.mean(fold_maes)
                if avg_mae < (best_cv_mae - inertia):
                    best_cv_mae, best_params = avg_mae, {'penalty_thresh': p_t, 'zero_thresh': z_t, 'penalty_rate': p_r}
                    print(f"  [Tuning] Found better params: {best_params} (MAE: {avg_mae:.4f})")
    return best_params

# ============================================================
# 4. MAIN PIPELINE (Entry Points)
# ============================================================

def fit_apply_post_processing(oof_preds, test_preds, y_true, method, **kwargs):
    """Main orchestrator for applying post-processing pipelines."""
    oof = np.maximum(np.asarray(oof_preds, dtype=float).copy(), 0)
    test = np.maximum(np.asarray(test_preds, dtype=float).copy(), 0)
    y_t = np.asarray(y_true)
    
    # Common requirements
    known = kwargs.get('known_values')
    rec_train = kwargs.get('recency_train')
    rec_test = kwargs.get('recency_test')
    ret_train = kwargs.get('returns_train')
    ret_test = kwargs.get('returns_test')

    if method == "raw": return oof, test

    # --- Simple Dispatch ---
    if method == "matthias_only":
        return apply_matthias_filter(oof, rec_train), apply_matthias_filter(test, rec_test)
    
    if method == "behavioral":
        return apply_behavioral_filter(oof, rec_train, ret_train), apply_behavioral_filter(test, rec_test, ret_test)
    
    if method == "snap_only":
        return apply_snapping(oof, known), apply_snapping(test, known)

    if method == "recency_only_cv" or method == "recency_only":
        params = tune_recency_filter_cv(y_t, oof, rec_train)
        return apply_matthias_filter(oof, rec_train, params), apply_matthias_filter(test, rec_test, params)

    if method == "force_zero_rate":
        q = kwargs.get('train_zero_rate', 0.0)
        return apply_zero_rate(oof, q), apply_zero_rate(test, q)

    if method == "zero_then_snap":
        q = kwargs.get('train_zero_rate', 0.0)
        oof = apply_zero_rate(oof, q)
        test = apply_zero_rate(test, q)
        return apply_snapping(oof, known), apply_snapping(test, known)

    if method == "piecewise_scale":
        f = tune_scaling_factor(y_t, oof)
        return apply_scaling(oof, f), apply_scaling(test, f)

    if method == "quantile_map":
        qt_p, qt_t = QuantileTransformer(output_distribution='uniform'), QuantileTransformer(output_distribution='uniform')
        qt_p.fit(oof.reshape(-1, 1)); qt_t.fit(y_t.reshape(-1, 1))
        return (
            qt_t.inverse_transform(qt_p.transform(oof.reshape(-1, 1))).ravel(),
            qt_t.inverse_transform(qt_p.transform(test.reshape(-1, 1))).ravel()
        )

    if method == "strong_combo":
        qt_p, qt_t = QuantileTransformer(output_distribution='uniform'), QuantileTransformer(output_distribution='uniform')
        qt_p.fit(oof.reshape(-1, 1)); qt_t.fit(y_t.reshape(-1, 1))
        oof = qt_t.inverse_transform(qt_p.transform(oof.reshape(-1, 1))).ravel()
        test = qt_t.inverse_transform(qt_p.transform(test.reshape(-1, 1))).ravel()
        q = tune_zero_rate(y_t, oof)
        oof, test = apply_zero_rate(oof, q), apply_zero_rate(test, q)
        f = tune_scaling_factor(y_t, oof)
        oof, test = apply_scaling(oof, f), apply_scaling(test, f)
        oof, test = apply_snapping(oof, known), apply_snapping(test, known)
        params = tune_recency_filter_cv(y_t, oof, rec_train)
        return apply_matthias_filter(oof, rec_train, params), apply_matthias_filter(test, rec_test, params)

    if method == "ultra_combo_cv" or method == "ultra_combo":
        # 1. Dist Mapping
        qt_p, qt_t = QuantileTransformer(output_distribution='uniform'), QuantileTransformer(output_distribution='uniform')
        qt_p.fit(oof.reshape(-1, 1)); qt_t.fit(y_t.reshape(-1, 1))
        oof = qt_t.inverse_transform(qt_p.transform(oof.reshape(-1, 1))).ravel()
        test = qt_t.inverse_transform(qt_p.transform(test.reshape(-1, 1))).ravel()
        
        # 2. Zero-rate & Scaling
        q = tune_zero_rate(y_t, oof)
        oof, test = apply_zero_rate(oof, q), apply_zero_rate(test, q)
        f = tune_scaling_factor(y_t, oof)
        oof, test = apply_scaling(oof, f), apply_scaling(test, f)
        
        # 3. Snap & Recency
        oof, test = apply_snapping(oof, known), apply_snapping(test, known)
        params = tune_recency_filter_cv(y_t, oof, rec_train)
        oof, test = apply_matthias_filter(oof, rec_train, params), apply_matthias_filter(test, rec_test, params)
        
        # 4. Clip
        return apply_clipping(oof, y_t), apply_clipping(test, y_t)

    # --- Search Orchestrator ---
    if method == "best":
        best_m, _ = find_best_method(oof, y_t, known, rec_train, ret_train)
        print(f"Auto-selected best method: {best_m}")
        return fit_apply_post_processing(oof_preds, test_preds, y_true, best_m, **kwargs)

    # Legacy grid search triggers
    if method.startswith("zeroq_") or method.startswith("scale_") or method.startswith("zeroqscale_"):
        return apply_simple_method(oof, test, method, y_t)

    raise ValueError(f"Unknown post-processing method: {method}")

def find_best_method(oof_preds, y_true, known=None, recency=None, returns=None):
    """Sweeps all methods to find the one minimizing OOF MAE."""
    methods = ["raw", "snap_only", "ultra_combo_cv", "matthias_only", "recency_only_cv", "behavioral"]
    results = []
    for m in methods:
        try:
            # Recursive call for the OOF side only
            p_oof, _ = fit_apply_post_processing(oof_preds, oof_preds, y_true, m, 
                                               known_values=known, recency_train=recency, recency_test=recency,
                                               returns_train=returns, returns_test=returns)
            mae = mean_absolute_error(y_true, p_oof)
            results.append({"method": m, "MAE": mae})
        except: continue
    
    df = pd.DataFrame(results).sort_values("MAE")
    print("\n--- Method Search Results ---\n", df.head(5).to_string(index=False))
    return df.iloc[0]['method'], df.iloc[0]['MAE']

def apply_simple_method(oof, test, method, y_true):
    """Helper for simple grid-based methods (e.g. scale_1.2)."""
    if method.startswith("zeroqscale_"):
        parts = method.split("_")
        q = float(parts[1])
        f = float(parts[2])
        oof = apply_zero_rate(oof, q)
        test = apply_zero_rate(test, q)
        return apply_scaling(oof, f), apply_scaling(test, f)
    if method.startswith("zeroq_"):
        q = float(method.split("_")[1])
        return apply_zero_rate(oof, q), apply_zero_rate(test, q)
    if method.startswith("scale_"):
        f = float(method.split("_")[1])
        return apply_scaling(oof, f), apply_scaling(test, f)
    return oof, test

# ============================================================
# 5. VISUALIZATION & LOGGING
# ============================================================

def evaluate_log_and_save(oof, test, y_true, test_ids, model_name, eda="standard", method="none"):
    """Unified evaluation orchestrator."""
    print(f"\n--- Evaluation: {model_name} (EDA: {eda}) ---")
    metrics = calculate_metrics(y_true, oof)
    for m, val in metrics.items(): print(f"  {m:<10}: {val:.4f}")
    
    # Plotting
    plot_diagnostics(y_true, oof, model_name)
    
    # Logging
    log_experiment(model_name, metrics, method, (oof == 0).mean(), eda)
    
    # Saving
    save_submissions(oof, test, y_true, test_ids, model_name)
    return metrics


def run_full_post_processing(model, X_train, y_true, y_pred, model_name, eda_used="standard", method="none"):
    """Basic full post-processing entry point for legacy model scripts."""
    print(f"\n--- Full post-processing: {model_name} (EDA: {eda_used}) ---")
    metrics = calculate_metrics(y_true, y_pred)
    for m, val in metrics.items(): print(f"  {m:<10}: {val:.4f}")
    
    plot_diagnostics(y_true, y_pred, model_name)
    log_experiment(model_name, metrics, method, (y_pred == 0).mean(), eda_used)

    if hasattr(model, 'feature_importances_') or hasattr(model, 'feature_importance'):
        if hasattr(X_train, 'columns'):
            plot_feature_importance(model, list(X_train.columns), model_name)
        else:
            plot_feature_importance(model, [], model_name)
    return metrics


def plot_diagnostics(y_true, y_pred, name):
    """Generates scatter and error distribution plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(y_true, y_pred, alpha=0.3)
    ax1.plot([0, max(y_true.max(), y_pred.max())], [0, max(y_true.max(), y_pred.max())], 'r--')
    ax1.set_title(f'Actual vs Predicted - {name}'); ax1.set_xlabel('Actual'); ax1.set_ylabel('Pred')
    
    sns.histplot(y_pred - y_true, bins=50, kde=True, ax=ax2)
    ax2.set_title(f'Error Distribution - {name}'); ax2.set_xlabel('Error')
    plt.show()

def log_experiment(name, metrics, method, z_rate, eda):
    """Logs results to submissions/experiment_log.csv."""
    path = os.path.join(os.path.dirname(__file__), 'submissions', 'experiment_log.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Model': name, 'EDA': eda, 
             'MAE': f"{metrics['MAE']:.4f}", 'Spearman': f"{metrics['Spearman']:.6f}", 
             'ZeroRate': f"{z_rate*100:.2f}%", 'PostProcess': method}
    df = pd.DataFrame([entry])
    df.to_csv(path, mode='a' if os.path.exists(path) else 'w', header=not os.path.exists(path), index=False)
    print(f"Logged to {path}")

def save_submissions(oof, test, y_true, test_ids, name):
    """Saves OOF and Test predictions to the submissions folder."""
    path = os.path.join(os.path.dirname(__file__), 'submissions')
    pd.DataFrame({'revenue': oof}, index=y_true.index).to_csv(os.path.join(path, f'{name}_oof.csv'))
    pd.DataFrame({'cust_id': test_ids, 'revenue': test}).to_csv(os.path.join(path, f'{name}_predictions.csv'), index=False)
    print(f"Saved predictions to {path}")

def plot_feature_importance(model, features, name, top_n=20):
    """Standardized importance plotter."""
    if hasattr(model, 'feature_importances_'): imp = model.feature_importances_
    elif hasattr(model, 'feature_importance'): imp = model.feature_importance()
    else: return
    df = pd.DataFrame({'Feature': features, 'Importance': imp}).sort_values('Importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, 8)); sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    plt.title(f'Importance - {name}'); plt.show()
