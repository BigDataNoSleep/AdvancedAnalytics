# model_visualizations.py

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# Make imports work when script is inside /models or project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eda_transactions_advanced import get_advanced_customer_model_data


# ============================================================
# CONFIG
# ============================================================

OOF_FILES = {
    "LightGBM Advanced": "LightGBM_Advanced_oof.csv",
    "Hurdle LightGBM": "Hurdle_LightGBM_Advanced_oof.csv",
    "Tweedie LightGBM": "Tweedie_LightGBM_Advanced_oof.csv",
    "CatBoost Baseline": "catboost_oof.csv",
    "LightGBM Baseline": "lightgbm_oof.csv",
    "XGBoost Baseline": "xgboost_oof.csv",
}

# Resolve correct submissions directory path (handles both /models and root contexts)
_base_dir = Path(__file__).resolve().parent
SUBMISSIONS_DIR = _base_dir.parent / "submissions" if _base_dir.name == "models" else Path("submissions")
OUTPUT_DIR = Path("model_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD TRUE TARGET
# ============================================================

X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_advanced_customer_model_data()

X = pd.concat([X_train, X_val, X_test])
y_true = pd.concat([y_train, y_val, y_test]).reset_index(drop=True)


# ============================================================
# LOAD OOF PREDICTIONS
# ============================================================

preds = {}

for model_name, filename in OOF_FILES.items():
    path = SUBMISSIONS_DIR / filename

    if not path.exists():
        print(f"Skipping {model_name}: file not found at {path}")
        continue

    df = pd.read_csv(path)

    if "prediction" in df.columns:
        pred_col = "prediction"
    elif "revenue" in df.columns:
        pred_col = "revenue"
    else:
        print(f"Skipping {model_name}: no 'prediction' or 'revenue' column in {filename}")
        continue

    p = df[pred_col].to_numpy(dtype=float)
    preds[model_name] = np.maximum(p, 0)

if not preds:
    raise ValueError("No valid OOF prediction files found. Check OOF_FILES and submissions folder.")

# Align lengths of all predictions, y_true, and X to the minimum available length
# This ensures we can compare models evaluated on train+val with models evaluated on train+val+test
min_len = min(len(y_true), min(len(p) for p in preds.values()))

y_true = y_true.iloc[:min_len].reset_index(drop=True)
X = X.iloc[:min_len].reset_index(drop=True)

for model_name in list(preds.keys()):
    preds[model_name] = preds[model_name][:min_len]


# ============================================================
# METRICS TABLE
# ============================================================

metrics_rows = []

for model_name, p in preds.items():
    mae = mean_absolute_error(y_true, p)
    spearman, _ = spearmanr(y_true, p)
    zero_rate = np.mean(p == 0)

    metrics_rows.append({
        "Model": model_name,
        "MAE": mae,
        "Spearman": spearman,
        "ZeroRate": zero_rate,
    })

metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
metrics_df.to_csv(OUTPUT_DIR / "model_metrics_summary.csv", index=False)

print(metrics_df)


# ============================================================
# 1. MAE BARPLOT
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(metrics_df["Model"], metrics_df["MAE"])
plt.ylim(60, 65)
plt.ylabel("MAE")
plt.title("Model comparison by MAE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_mae_comparison.png", dpi=300)
plt.close()


# ============================================================
# 2. SPEARMAN BARPLOT
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(metrics_df["Model"], metrics_df["Spearman"])
plt.ylabel("Spearman correlation")
plt.ylim(0.35, 0.45)
plt.title("Model comparison by Spearman correlation")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_spearman_comparison.png", dpi=300)
plt.close()


# ============================================================
# 3. ZERO RATE VS MAE
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(metrics_df["ZeroRate"] * 100, metrics_df["MAE"])

for _, row in metrics_df.iterrows():
    plt.annotate(row["Model"], (row["ZeroRate"] * 100, row["MAE"]), fontsize=9)

plt.xlabel("Zero prediction rate (%)")
plt.ylabel("MAE")
plt.title("Zero-rate versus MAE")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "zero_rate_vs_mae.png", dpi=300)
plt.close()


# ============================================================
# 4. PREDICTED VS ACTUAL, LOG SCALE
# ============================================================

for model_name, p in preds.items():
    plt.figure(figsize=(6, 6))
    
    # Randomly sample 1000 points to reduce noise
    sample_size = min(1000, len(y_true))
    np.random.seed(42)
    idx = np.random.choice(len(y_true), sample_size, replace=False)
    
    plt.scatter(np.log1p(y_true.iloc[idx]), np.log1p(p[idx]), alpha=0.25, s=8)

    # Calculate and display points at (0, 0)
    zero_zero_count = np.sum((y_true.iloc[idx] == 0) & (p[idx] == 0))
    plt.text(0, 0, str(zero_zero_count), color='red', fontsize=12, fontweight='bold', ha='left', va='bottom')

    # Calculate and display points at (0, x) where x != 0
    zero_x_count = np.sum((y_true.iloc[idx] == 0) & (p[idx] != 0))
    plt.text(0, 7, str(zero_x_count), color='red', fontsize=12, fontweight='bold', ha='left', va='top')

    # Calculate and display points at (y, 0) where y != 0
    y_zero_count = np.sum((y_true.iloc[idx] != 0) & (p[idx] == 0))
    plt.text(7, 0, str(y_zero_count), color='red', fontsize=12, fontweight='bold', ha='right', va='bottom')

    max_val = max(np.log1p(y_true).max(), np.log1p(p).max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")

    plt.xlabel("Actual revenue, log(1 + y)")
    plt.ylabel("Predicted revenue, log(1 + prediction)")
    plt.title(f"Predicted vs actual revenue: {model_name}")
    plt.tight_layout()

    filename = f"predicted_vs_actual_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


# ============================================================
# 5. RESIDUAL BOXPLOT BY MODEL
# ============================================================

residual_data = []
labels = []

for model_name, p in preds.items():
    residuals = y_true.to_numpy() - p
    residual_data.append(residuals)
    labels.append(model_name)

plt.figure(figsize=(9, 5))
plt.boxplot(residual_data, tick_labels=labels, showfliers=False)
plt.axhline(0, linestyle="--")
plt.ylabel("Residual: actual - predicted")
plt.title("Residual distribution by model")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_boxplot_by_model.png", dpi=300)
plt.close()


# ============================================================
# 6. REVENUE BIN CALIBRATION PLOT
# ============================================================

for model_name, p in preds.items():
    df_cal = pd.DataFrame({
        "y_true": y_true.to_numpy(),
        "pred": p
    })

    df_cal["bin"] = pd.qcut(df_cal["pred"].rank(method="first"), q=10, labels=False)

    grouped = df_cal.groupby("bin").agg(
        avg_actual=("y_true", "mean"),
        avg_predicted=("pred", "mean")
    ).reset_index()

    plt.figure(figsize=(7, 5))
    plt.plot(grouped["bin"] + 1, grouped["avg_actual"], marker="o", label="Actual average revenue")
    plt.plot(grouped["bin"] + 1, grouped["avg_predicted"], marker="o", label="Predicted average revenue")

    plt.xlabel("Prediction decile, low to high")
    plt.ylabel("Average revenue")
    plt.title(f"Revenue calibration by prediction decile: {model_name}")
    plt.legend()
    plt.tight_layout()

    filename = f"calibration_deciles_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


# ============================================================
# 7. MAE BY RECENCY BIN
# ============================================================

if "recency_days" in X.columns:
    for model_name, p in preds.items():
        df_rec = pd.DataFrame({
            "recency_days": X["recency_days"].to_numpy(),
            "y_true": y_true.to_numpy(),
            "pred": p
        })

        df_rec["abs_error"] = np.abs(df_rec["y_true"] - df_rec["pred"])
        df_rec["recency_bin"] = pd.cut(
            df_rec["recency_days"],
            bins=[-1, 30, 90, 180, 365, 500, 10_000],
            labels=["0-30", "31-90", "91-180", "181-365", "366-500", "500+"]
        )

        grouped = df_rec.groupby("recency_bin", observed=False)["abs_error"].mean().reset_index()

        plt.figure(figsize=(7, 5))
        plt.bar(grouped["recency_bin"].astype(str), grouped["abs_error"])
        plt.xlabel("Recency bin, days since last purchase")
        plt.ylabel("MAE")
        plt.title(f"Prediction error by recency: {model_name}")
        plt.tight_layout()

        filename = f"mae_by_recency_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close()


# ============================================================
# 8. PREDICTION CORRELATION HEATMAP
# ============================================================

pred_matrix = pd.DataFrame(preds)
corr = pred_matrix.corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right")
plt.yticks(range(len(corr.index)), corr.index)

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Correlation between model predictions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_prediction_correlation_heatmap.png", dpi=300)
plt.close()


print(f"\nSaved all plots to: {OUTPUT_DIR.resolve()}")
