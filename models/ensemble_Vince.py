#%%
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

# Add parent directory for eda_transactions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda_transactions import get_customer_model_data


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
# METRICS
# ============================================================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    spear, _ = spearmanr(y_true, y_pred)
    if np.isnan(spear):
        spear = 0.0
    return mae, spear


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
        weighted_mae, weighted_spear = evaluate(y_true, weighted_pred)

        rows.append({
            "subset": ",".join(subset),
            "method": "weighted",
            "weights": repr(dict(zip(subset, [round(w, 6) for w in weights]))),
            "alpha_weighted_vs_median": np.nan,
            "score_mae": weighted_mae,
            "score_spearman": weighted_spear
        })

        candidate = {
            "subset": subset,
            "method": "weighted",
            "weights": weights,
            "alpha": None,
            "score_pred": weighted_pred,
            "score_mae": weighted_mae,
            "score_spearman": weighted_spear
        }

        if best_result is None or candidate["score_mae"] < best_result["score_mae"]:
            best_result = candidate

        # Median
        if SEARCH_MEDIAN_BLEND and len(subset) >= 2:
            median_pred = median_predict(pred_matrix)
            median_mae, median_spear = evaluate(y_true, median_pred)

            rows.append({
                "subset": ",".join(subset),
                "method": "median",
                "weights": "",
                "alpha_weighted_vs_median": np.nan,
                "score_mae": median_mae,
                "score_spearman": median_spear
            })

            candidate = {
                "subset": subset,
                "method": "median",
                "weights": None,
                "alpha": None,
                "score_pred": median_pred,
                "score_mae": median_mae,
                "score_spearman": median_spear
            }

            if candidate["score_mae"] < best_result["score_mae"]:
                best_result = candidate

            # Weighted + median
            best_alpha = 1.0
            best_blend_pred = weighted_pred
            best_blend_mae, best_blend_spear = evaluate(y_true, weighted_pred)

            for alpha in np.linspace(0.0, 1.0, 41):
                blend_pred = alpha * weighted_pred + (1.0 - alpha) * median_pred
                blend_pred = np.maximum(blend_pred, 0)
                blend_mae, blend_spear = evaluate(y_true, blend_pred)

                if blend_mae < best_blend_mae:
                    best_alpha = alpha
                    best_blend_pred = blend_pred
                    best_blend_mae = blend_mae
                    best_blend_spear = blend_spear

            rows.append({
                "subset": ",".join(subset),
                "method": "weighted_plus_median",
                "weights": repr(dict(zip(subset, [round(w, 6) for w in weights]))),
                "alpha_weighted_vs_median": round(best_alpha, 4),
                "score_mae": best_blend_mae,
                "score_spearman": best_blend_spear
            })

            candidate = {
                "subset": subset,
                "method": "weighted_plus_median",
                "weights": weights,
                "alpha": best_alpha,
                "score_pred": best_blend_pred,
                "score_mae": best_blend_mae,
                "score_spearman": best_blend_spear
            }

            if candidate["score_mae"] < best_result["score_mae"]:
                best_result = candidate

    diagnostics = pd.DataFrame(rows).sort_values("score_mae").reset_index(drop=True)
    return best_result, diagnostics


# ============================================================
# POST-PROCESSING
# ============================================================
def force_zero_rate(preds, zero_rate):
    preds = np.asarray(preds, dtype=float).copy()
    preds = np.maximum(preds, 0)

    n_zero = int(round(len(preds) * zero_rate))
    if n_zero <= 0:
        return preds

    order = np.argsort(preds)
    preds[order[:n_zero]] = 0.0
    return preds


def snap_to_known_values(preds, known_values):
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
    preds = np.asarray(preds, dtype=float)
    return np.maximum(preds * factor, 0)


def apply_postprocess(preds, method_name, train_zero_rate, known_values):
    preds = np.asarray(preds, dtype=float).copy()
    preds = np.maximum(preds, 0)

    if method_name == "raw":
        return preds

    if method_name == "force_zero_rate":
        return force_zero_rate(preds, train_zero_rate)

    if method_name == "snap_only":
        out = snap_to_known_values(preds, known_values)
        out[out <= 0.01] = 0.0
        return out

    if method_name == "zero_then_snap":
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
        _, q, factor = method_name.split("_")
        out = force_zero_rate(preds, float(q))
        out = scale_predictions(out, float(factor))
        return out

    raise ValueError(f"Unknown postprocess method: {method_name}")


def search_postprocess_on_oof(oof_pred, y_true, train_zero_rate, known_values):
    methods = ["raw", "force_zero_rate", "snap_only", "zero_then_snap"]

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
        pp = apply_postprocess(oof_pred, method, train_zero_rate, known_values)
        mae, spear = evaluate(y_true, pp)

        rows.append({
            "postprocess": method,
            "score_mae": mae,
            "score_spearman": spear,
            "zero_share": float((pp == 0).mean()),
            "positive_mean": float(pp[pp > 0].mean()) if np.any(pp > 0) else 0.0
        })

        if mae < best_mae:
            best_mae = mae
            best_method = method
            best_pred = pp

    diagnostics = pd.DataFrame(rows).sort_values("score_mae").reset_index(drop=True)
    return best_method, best_pred, best_mae, diagnostics


# ============================================================
# PLOTTING
# ============================================================
def plot_distribution_pair(train_target, pred_before, pred_after, output_dir):
    train_target = pd.Series(train_target)
    pred_before = pd.Series(pred_before)
    pred_after = pd.Series(pred_after)

    train_pos = train_target[train_target > 0]
    before_pos = pred_before[pred_before > 0]
    after_pos = pred_after[pred_after > 0]

    plt.figure(figsize=(12, 6))
    plt.hist(train_pos, bins=60, density=True, alpha=0.45, edgecolor='black', label='Train target (>0)')
    plt.hist(before_pos, bins=60, density=True, alpha=0.45, edgecolor='black', label='Original ensemble (>0)')
    plt.title('Original distributions (positive only)')
    plt.xlabel('Revenue')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    path1 = output_dir / 'dist_original_positive.png'
    plt.savefig(path1, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(train_pos, bins=60, density=True, alpha=0.45, edgecolor='black', label='Train target (>0)')
    plt.hist(after_pos, bins=60, density=True, alpha=0.45, edgecolor='black', label='Adjusted predictions (>0)')
    plt.title('Adjusted distributions (positive only)')
    plt.xlabel('Revenue')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    path2 = output_dir / 'dist_adjusted_positive.png'
    plt.savefig(path2, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    if len(train_pos) > 0:
        plt.hist(np.log1p(train_pos), bins=60, density=True, alpha=0.40, edgecolor='black', label='Train target log1p')
    if len(before_pos) > 0:
        plt.hist(np.log1p(before_pos), bins=60, density=True, alpha=0.40, edgecolor='black', label='Original ensemble log1p')
    if len(after_pos) > 0:
        plt.hist(np.log1p(after_pos), bins=60, density=True, alpha=0.40, edgecolor='black', label='Adjusted predictions log1p')

    plt.title('Positive revenue distributions in log1p space')
    plt.xlabel('log1p(Revenue)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    path3 = output_dir / 'dist_log1p_before_after_train.png'
    plt.savefig(path3, dpi=150)
    plt.close()

    return path1, path2, path3


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 72)
    print("ROBUST ENSEMBLE + OOF-SELECTED POSTPROCESS")
    print("=" * 72)

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using submissions directory: {output_dir}")

    print("\nLoading targets from eda_transactions.py...")
    X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_customer_model_data()

    y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
    train_zero_rate = float((y_full == 0).mean())
    known_values = pd.concat([y_train, y_val]).to_numpy(dtype=float)

    print(f"OOF target length:   {len(y_full)}")
    print(f"Competition length:  {len(test_unlabelled)}")
    print(f"Train zero rate:     {train_zero_rate:.4f} ({train_zero_rate*100:.2f}%)")

    # --------------------------------------------------------
    # Load OOF predictions
    # --------------------------------------------------------
    print("\nLoading OOF predictions...")
    oof_dict = {}
    available_models = []

    for model_name, files in MODEL_FILES.items():
        oof_path = output_dir / files["oof"]
        if oof_path.exists():
            preds = load_single_oof(oof_path)
            if len(preds) != len(y_full):
                raise ValueError(
                    f"Length mismatch for {model_name} OOF: {len(preds)} vs {len(y_full)}"
                )
            oof_dict[model_name] = preds
            available_models.append(model_name)
        else:
            print(f"Skipping {model_name}: missing {oof_path.name}")

    if not available_models:
        raise FileNotFoundError("No OOF files found.")

    print(f"Available models: {available_models}")

    # --------------------------------------------------------
    # Search best ensemble on OOF
    # --------------------------------------------------------
    print("\nSearching best ensemble on OOF...")
    best_result, diagnostics = find_best_ensemble(oof_dict, y_full.to_numpy())

    print("\nBest OOF ensemble:")
    print(f"  subset:   {best_result['subset']}")
    print(f"  method:   {best_result['method']}")
    print(f"  score_mae:{best_result['score_mae']:.6f}")
    print(f"  spear:    {best_result['score_spearman']:.6f}")

    oof_best_raw = best_result["score_pred"]
    raw_mae, raw_spear = evaluate(y_full, oof_best_raw)

    print("\nRaw OOF diagnostics:")
    print(f"  OOF MAE     : {raw_mae:.6f}")
    print(f"  OOF Spearman: {raw_spear:.6f}")

    # --------------------------------------------------------
    # Search best post-processing on OOF
    # --------------------------------------------------------
    print("\nSearching best post-processing on OOF...")
    best_pp_name, best_oof_pp, best_pp_mae, pp_diag = search_postprocess_on_oof(
        oof_pred=oof_best_raw,
        y_true=y_full.to_numpy(),
        train_zero_rate=train_zero_rate,
        known_values=known_values
    )

    print("\nBest OOF postprocess:")
    print(f"  method: {best_pp_name}")
    print(f"  mae:    {best_pp_mae:.6f}")

    # --------------------------------------------------------
    # Load competition predictions for chosen subset
    # --------------------------------------------------------
    print("\nLoading competition predictions for chosen subset...")
    comp_frames = []
    for model_name in best_result["subset"]:
        comp_path = output_dir / MODEL_FILES[model_name]["comp"]
        if not comp_path.exists():
            raise FileNotFoundError(f"Missing competition file: {comp_path}")
        comp_frames.append(load_single_prediction_file(comp_path, model_name))

    comp_merged = merge_prediction_frames(comp_frames)
    comp_final_raw = apply_best_ensemble(best_result, comp_merged)

    # Apply best OOF postprocess to competition predictions
    comp_final = apply_postprocess(
        preds=comp_final_raw,
        method_name=best_pp_name,
        train_zero_rate=train_zero_rate,
        known_values=known_values
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    final_submission = pd.DataFrame({
        "cust_id": comp_merged["cust_id"],
        "prediction": comp_final
    })

    final_path = output_dir / "ensemble_final_predictions.csv"
    final_submission.to_csv(final_path, index=False)

    diag_path = output_dir / "ensemble_final_diagnostics.csv"
    diagnostics.to_csv(diag_path, index=False)

    pp_diag_path = output_dir / "ensemble_final_postprocess_diagnostics.csv"
    pp_diag.to_csv(pp_diag_path, index=False)

    print("\nSaved:")
    print(f"  {final_path}")
    print(f"  {diag_path}")
    print(f"  {pp_diag_path}")

    print("\nFinal prediction summary:")
    print(final_submission["prediction"].describe())
    print(f"Zero share: {(final_submission['prediction'] == 0).mean():.4f}")

    print("\n=== ZERO PERCENTAGE SUMMARY ===")
    print(f"Train target:         {train_zero_rate*100:.2f}% zeros")
    print(f"Raw predictions:      {(comp_final_raw == 0).mean()*100:.2f}% zeros")
    print(f"Adjusted predictions: {(comp_final == 0).mean()*100:.2f}% zeros")

    path1, path2, path3 = plot_distribution_pair(
        train_target=y_full.values,
        pred_before=comp_final_raw,
        pred_after=comp_final,
        output_dir=output_dir
    )

    print("\nSaved plots:")
    print(f"  {path1}")
    print(f"  {path2}")
    print(f"  {path3}")

    print("\nTop 10 ensemble candidates:")
    print(diagnostics.head(10).to_string(index=False))

    print("\nTop 10 postprocess candidates:")
    print(pp_diag.head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

# %%