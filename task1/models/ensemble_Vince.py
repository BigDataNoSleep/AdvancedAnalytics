# %% [markdown]
# # Robust Ensemble + OOF-Selected Post-Processing
# This script combines multiple models (CatBoost, LightGBM, XGBoost) using convex weight optimization 
# and median blending, then applies automated post-processing search.

# %%
import os
import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# Add parent directory for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Select Data Source: Standard vs Advanced Features
# from eda_transactions import get_customer_model_data; EDA_TYPE = "standard"
from eda_transactions_advanced import get_advanced_customer_model_data as get_customer_model_data; EDA_TYPE = "advanced"
from post_processing import (
    calculate_metrics, 
    fit_apply_post_processing,
    evaluate_log_and_save
)

# ============================================================
# CONFIG
# ============================================================
MODEL_FILES = {
    "catboost": {
        "oof": "catboost_oof.csv",
        "comp": "catboost_baseline_predictions.csv",
    },
    "lightgbm": {
        "oof": "lightgbm_oof.csv",
        "comp": "lightgbm_tuning_predictions.csv",
    },
    "xgboost": {
        "oof": "xgboost_oof.csv",
        "comp": "xgboost_baseline_predictions.csv",
    }
}

SEARCH_MEDIAN_BLEND = True

# ============================================================
# PATHS
# ============================================================
def get_output_dir():
    base_dir = Path(__file__).resolve().parent
    if base_dir.name == "models":
        return base_dir.parent / "submissions"
    return base_dir / "submissions"

# ============================================================
# FILE HELPERS
# ============================================================
def load_single_oof(path):
    df = pd.read_csv(path)
    if "prediction" not in df.columns:
        raise ValueError(f"{path} must contain a 'prediction' column.")
    return df["prediction"].to_numpy(dtype=float)


def load_single_prediction_file(path, model_name):
    df = pd.read_csv(path)
    required_cols = {"cust_id", "prediction"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {required_cols}")

    df = df[["cust_id", "prediction"]].copy()
    df["cust_id"] = df["cust_id"].astype(str)
    df = df.rename(columns={"prediction": model_name})
    return df


def merge_prediction_frames(frames):
    merged = frames[0].copy()
    for df in frames[1:]:
        merged = merged.merge(df, on="cust_id", how="inner")

    if merged.empty:
        raise ValueError("Merged prediction frame is empty after alignment.")

    return merged.sort_values("cust_id").reset_index(drop=True)


# ============================================================
# ENSEMBLE HELPERS
# ============================================================
def powerset_nonempty(items):
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield list(combo)


def optimize_convex_weights(pred_matrix, y_true):
    n_models = pred_matrix.shape[1]
    init = np.full(n_models, 1.0 / n_models)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models

    def objective(w):
        pred = np.dot(pred_matrix, w)
        pred = np.maximum(pred, 0)
        return mean_absolute_error(y_true, pred)

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return init

    return result.x


def weighted_predict(pred_matrix, weights):
    return np.maximum(np.dot(pred_matrix, weights), 0)


def median_predict(pred_matrix):
    return np.maximum(np.median(pred_matrix, axis=1), 0)


def apply_best_ensemble(best_result, pred_frames_merged):
    subset = best_result["subset"]
    pred_matrix = pred_frames_merged[subset].to_numpy()

    if best_result["method"] == "weighted":
        final_pred = weighted_predict(pred_matrix, best_result["weights"])

    elif best_result["method"] == "median":
        final_pred = median_predict(pred_matrix)

    elif best_result["method"] == "weighted_plus_median":
        weighted_pred = weighted_predict(pred_matrix, best_result["weights"])
        median_pred = median_predict(pred_matrix)
        final_pred = (
            best_result["alpha"] * weighted_pred +
            (1.0 - best_result["alpha"]) * median_pred
        )
        final_pred = np.maximum(final_pred, 0)

    else:
        raise ValueError(f"Unknown method: {best_result['method']}")

    return np.maximum(final_pred, 0)


# ============================================================
# ENSEMBLE SEARCH ON OOF
# ============================================================
def find_best_ensemble(oof_dict, y_true):
    rows = []
    best_result = None

    model_names = list(oof_dict.keys())

    for subset in powerset_nonempty(model_names):
        pred_matrix = np.column_stack([oof_dict[m] for m in subset])

        # Weighted
        weights = optimize_convex_weights(pred_matrix, y_true)
        weighted_pred = weighted_predict(pred_matrix, weights)
        weighted_metrics = calculate_metrics(y_true, weighted_pred)

        rows.append({
            "subset": ",".join(subset),
            "method": "weighted",
            "weights": repr(dict(zip(subset, [round(w, 4) for w in weights]))),
            "alpha_weighted_vs_median": np.nan,
            "score_mae": weighted_metrics['MAE'],
            "score_spearman": weighted_metrics['Spearman']
        })

        candidate = {
            "subset": subset,
            "method": "weighted",
            "weights": weights,
            "alpha": None,
            "score_pred": weighted_pred,
            "score_mae": weighted_metrics['MAE'],
            "score_spearman": weighted_metrics['Spearman']
        }

        if best_result is None or candidate["score_mae"] < best_result["score_mae"]:
            best_result = candidate

        # Median
        if SEARCH_MEDIAN_BLEND and len(subset) >= 2:
            median_pred = median_predict(pred_matrix)
            median_metrics = calculate_metrics(y_true, median_pred)

            rows.append({
                "subset": ",".join(subset),
                "method": "median",
                "weights": "",
                "alpha_weighted_vs_median": np.nan,
                "score_mae": median_metrics['MAE'],
                "score_spearman": median_metrics['Spearman']
            })

            candidate = {
                "subset": subset,
                "method": "median",
                "weights": None,
                "alpha": None,
                "score_pred": median_pred,
                "score_mae": median_metrics['MAE'],
                "score_spearman": median_metrics['Spearman']
            }

            if candidate["score_mae"] < best_result["score_mae"]:
                best_result = candidate

            # Weighted + median
            best_alpha = 1.0
            best_blend_pred = weighted_pred
            best_blend_metrics = calculate_metrics(y_true, weighted_pred)

            for alpha in np.linspace(0.0, 1.0, 41):
                blend_pred = alpha * weighted_pred + (1.0 - alpha) * median_pred
                blend_pred = np.maximum(blend_pred, 0)
                blend_mae = mean_absolute_error(y_true, blend_pred)

                if blend_mae < best_blend_metrics['MAE']:
                    best_alpha = alpha
                    best_blend_pred = blend_pred
                    best_blend_metrics['MAE'] = blend_mae
                    # Spearman doesn't necessarily improve with MAE optimization
                    _, best_blend_metrics['Spearman'] = spearmanr(y_true, blend_pred)

            rows.append({
                "subset": ",".join(subset),
                "method": "weighted_plus_median",
                "weights": repr(dict(zip(subset, [round(w, 4) for w in weights]))),
                "alpha_weighted_vs_median": round(best_alpha, 4),
                "score_mae": best_blend_metrics['MAE'],
                "score_spearman": best_blend_metrics['Spearman']
            })

            candidate = {
                "subset": subset,
                "method": "weighted_plus_median",
                "weights": weights,
                "alpha": best_alpha,
                "score_pred": best_blend_pred,
                "score_mae": best_blend_metrics['MAE'],
                "score_spearman": best_blend_metrics['Spearman']
            }

            if candidate["score_mae"] < best_result["score_mae"]:
                best_result = candidate

    diagnostics = pd.DataFrame(rows).sort_values("score_mae").reset_index(drop=True)
    return best_result, diagnostics


def search_postprocess_on_oof(oof_pred, y_true, train_zero_rate, known_values, recency_train=None, recency_test=None):
    methods = ["raw", "force_zero_rate", "snap_only", "zero_then_snap", "piecewise_scale", "quantile_map", "strong_combo", "ultra_combo"]
    
    if recency_train is not None:
        methods.append("recency_only")

    low = max(0.45, train_zero_rate - 0.10)
    high = min(0.80, train_zero_rate + 0.10)

    for q in np.linspace(low, high, 15):
        methods.append(f"zeroq_{q:.4f}")

    for factor in np.linspace(0.8, 2.0, 25):
        methods.append(f"scale_{factor:.4f}")

    for q in np.linspace(low, high, 9):
        for factor in np.linspace(0.8, 2.0, 13):
            methods.append(f"zeroqscale_{q:.4f}_{factor:.4f}")

    rows = []
    best_method = None
    best_pred = None
    best_mae = np.inf

    for method in methods:
        pp, _ = fit_apply_post_processing(
            oof_pred, oof_pred, y_true, method,
            train_zero_rate=train_zero_rate,
            known_values=known_values,
            recency_train=recency_train,
            recency_test=recency_train
        )
        metrics = calculate_metrics(y_true, pp)

        rows.append({
            "postprocess": method,
            "score_mae": metrics['MAE'],
            "score_spearman": metrics['Spearman'],
            "zero_share": float((pp == 0).mean()),
            "positive_mean": float(pp[pp > 0].mean()) if np.any(pp > 0) else 0.0
        })

        if metrics['MAE'] < best_mae:
            best_mae = metrics['MAE']
            best_method = method
            best_pred = pp

    diagnostics = pd.DataFrame(rows).sort_values("score_mae").reset_index(drop=True)
    return best_method, best_pred, best_mae, diagnostics

# %% [markdown]
# ## 1. Config & Data Loading

output_dir = get_output_dir()
os.makedirs(output_dir, exist_ok=True)

print("Loading targets and model data...")
X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_customer_model_data()

y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
X_full = pd.concat([X_train, X_val]).reset_index(drop=True)

train_zero_rate = float((y_full == 0).mean())
known_values = pd.concat([y_train, y_val]).to_numpy(dtype=float)

print(f"OOF target length:   {len(y_full)}")
print(f"Competition length:  {len(test_unlabelled)}")
print(f"Train zero rate:     {train_zero_rate:.4f} ({train_zero_rate*100:.2f}%)")

# %% [markdown]
# ## 2. Load Predictions (OOF & Competition)

print("\nLoading OOF predictions...")
oof_dict = {}
available_models = []

for model_name, files in MODEL_FILES.items():
    oof_path = output_dir / files["oof"]
    if oof_path.exists():
        preds = load_single_oof(oof_path)
        if len(preds) != len(y_full):
            raise ValueError(f"Length mismatch for {model_name} OOF")
        oof_dict[model_name] = preds
        available_models.append(model_name)
    else:
        print(f"Skipping {model_name}: missing {oof_path.name}")

if not available_models:
    raise FileNotFoundError("No OOF files found.")

print(f"Integrated models: {available_models}")

# %% [markdown]
# ## 3. Ensemble Optimization

print("\nSearching best ensemble on OOF...")
best_ensemble_result, ensemble_diagnostics = find_best_ensemble(oof_dict, y_full.to_numpy())

print("\nBest OOF ensemble:")
print(f"  subset:    {best_ensemble_result['subset']}")
print(f"  method:    {best_ensemble_result['method']}")
print(f"  score_mae: {best_ensemble_result['score_mae']:.6f}")

print("\nTop 10 ensemble candidates:")
print(ensemble_diagnostics.head(10).to_string(index=False))

# %% [markdown]
# ## 4. Post-Processing Optimization

print("\nSearching best post-processing for ensemble...")
oof_best_raw = best_ensemble_result["score_pred"]

best_pp_name, best_oof_pp, best_pp_mae, pp_diag = search_postprocess_on_oof(
    oof_pred=oof_best_raw,
    y_true=y_full.to_numpy(),
    train_zero_rate=train_zero_rate,
    known_values=known_values,
    recency_train=X_full['recency_days'],
    recency_test=test_unlabelled['recency_days']
)

print("\nBest OOF postprocess:")
print(f"  method: {best_pp_name}")
print(f"  mae:    {best_pp_mae:.6f}")

print("\nTop 10 postprocess candidates:")
print(pp_diag.head(10)[["postprocess", "score_mae", "zero_share"]].to_string(index=False))

# %% [markdown]
# ## 5. Final Assembly & Submission

print("\nLoading competition predictions for chosen subset...")
comp_frames = []
for model_name in best_ensemble_result["subset"]:
    comp_path = output_dir / MODEL_FILES[model_name]["comp"]
    comp_frames.append(load_single_prediction_file(comp_path, model_name))

comp_merged = merge_prediction_frames(comp_frames)
comp_final_raw = apply_best_ensemble(best_ensemble_result, comp_merged)

# Apply best OOF postprocess identically across OOF and Test
comp_final = np.maximum(comp_final_raw, 0)
oof_final, comp_final = fit_apply_post_processing(
    oof_preds=oof_best_raw,
    test_preds=comp_final,
    y_true=y_full.to_numpy(),
    method=best_pp_name,
    train_zero_rate=train_zero_rate,
    known_values=known_values,
    recency_train=X_full['recency_days'],
    recency_test=test_unlabelled['recency_days']
)

# %% [markdown]
# ## 6. Diagnostics (The "After-Processing")

# Convert weights to dictionary for visualization
weights_dict = None
if best_ensemble_result["weights"] is not None:
    weights_dict = dict(zip(best_ensemble_result["subset"], best_ensemble_result["weights"]))

metrics = evaluate_log_and_save(
    oof_final,
    comp_final,
    y_full,
    comp_merged["cust_id"],
    model_name="Final_Ensemble",
    eda=EDA_TYPE,
    method=best_pp_name
)

print("\n=== ZERO PERCENTAGE SUMMARY ===")
print(f"Train target:         {train_zero_rate*100:.2f}% zeros")
print(f"Final predictions:    {(comp_final == 0).mean()*100:.2f}% zeros")
