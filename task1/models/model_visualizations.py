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

_base_dir = Path(__file__).resolve().parent
SUBMISSIONS_DIR = _base_dir.parent / "submissions" if _base_dir.name == "models" else Path("submissions")
OUTPUT_DIR = Path("model_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_name(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def get_prediction_column(df):
    if "prediction" in df.columns:
        return "prediction"
    if "revenue" in df.columns:
        return "revenue"
    return None


def get_model_base_df(X_part, y_part):
    """
    Creates a safe customer-level reference table.
    Assumes cust_id is stored in the index, which is how your modelling data appears to work.
    """
    return pd.DataFrame({
        "cust_id": X_part.index.astype(str),
        "y_true": y_part.to_numpy()
    }).reset_index(drop=True)


def load_oof_with_alignment(model_name, filename, reference_df):
    path = SUBMISSIONS_DIR / filename

    if not path.exists():
        print(f"Skipping {model_name}: file not found at {path}")
        return None

    df = pd.read_csv(path)
    pred_col = get_prediction_column(df)

    if pred_col is None:
        print(f"Skipping {model_name}: no prediction/revenue column found.")
        return None

    if "cust_id" not in df.columns:
        print(f"Skipping {model_name}: no cust_id column, so safe alignment is impossible.")
        return None

    df = df[["cust_id", pred_col]].copy()
    df["cust_id"] = df["cust_id"].astype(str)
    df = df.rename(columns={pred_col: "pred"})

    if df["cust_id"].duplicated().any():
        print(f"Skipping {model_name}: duplicate cust_id values found.")
        return None

    merged = reference_df.merge(df, on="cust_id", how="inner")
    merged["pred"] = np.maximum(merged["pred"].astype(float), 0)

    missing = len(reference_df) - len(merged)

    print(f"\n{model_name}")
    print(f"OOF rows: {len(df)}")
    print(f"Aligned rows: {len(merged)}")
    print(f"Missing from full reference: {missing}")

    if len(merged) == 0:
        print(f"Skipping {model_name}: no overlapping cust_id values.")
        return None

    return merged


# ============================================================
# LOAD TRUE TARGET AND FEATURES
# ============================================================

X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_advanced_customer_model_data()

X_full = pd.concat([X_train, X_val, X_test])
y_full = pd.concat([y_train, y_val, y_test])

reference_df = get_model_base_df(X_full, y_full)

# Add recency if available
if "recency_days" in X_full.columns:
    recency_df = pd.DataFrame({
        "cust_id": X_full.index.astype(str),
        "recency_days": X_full["recency_days"].to_numpy()
    })
    reference_df = reference_df.merge(recency_df, on="cust_id", how="left")


# ============================================================
# LOAD AND ALIGN OOF PREDICTIONS BY cust_id
# ============================================================

model_data = {}

print("\n" + "=" * 60)
print("LOADING AND ALIGNING OOF FILES BY cust_id")
print("=" * 60)

for model_name, filename in OOF_FILES.items():
    aligned = load_oof_with_alignment(model_name, filename, reference_df)

    if aligned is not None:
        model_data[model_name] = aligned

if not model_data:
    raise ValueError("No valid aligned OOF files found.")

print("\nModels kept for visualization:")
for model_name, df in model_data.items():
    print(f"- {model_name}: {len(df)} aligned rows")


# ============================================================
# METRICS TABLE
# ============================================================

metrics_rows = []

for model_name, df in model_data.items():
    y_true = df["y_true"].to_numpy()
    pred = df["pred"].to_numpy()

    mae = mean_absolute_error(y_true, pred)
    spearman, _ = spearmanr(y_true, pred)
    zero_rate = np.mean(pred == 0)

    metrics_rows.append({
        "Model": model_name,
        "Rows": len(df),
        "MAE": mae,
        "Spearman": 0.0 if np.isnan(spearman) else spearman,
        "ZeroRate": zero_rate,
    })

metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
metrics_df.to_csv(OUTPUT_DIR / "model_metrics_summary.csv", index=False)

print("\nMetrics summary:")
print(metrics_df)


# ============================================================
# 1. MAE BARPLOT
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(metrics_df["Model"], metrics_df["MAE"])
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

for model_name, df in model_data.items():
    y_true = df["y_true"].to_numpy()
    pred = df["pred"].to_numpy()

    plt.figure(figsize=(6, 6))

    sample_size = min(1000, len(df))
    np.random.seed(42)
    idx = np.random.choice(len(df), sample_size, replace=False)

    plt.scatter(np.log1p(y_true[idx]), np.log1p(pred[idx]), alpha=0.25, s=8)

    zero_zero_count = np.sum((y_true[idx] == 0) & (pred[idx] == 0))
    zero_x_count = np.sum((y_true[idx] == 0) & (pred[idx] != 0))
    y_zero_count = np.sum((y_true[idx] != 0) & (pred[idx] == 0))

    plt.text(0, 0, str(zero_zero_count), color="red", fontsize=12, fontweight="bold", ha="left", va="bottom")
    plt.text(0, 7, str(zero_x_count), color="red", fontsize=12, fontweight="bold", ha="left", va="top")
    plt.text(7, 0, str(y_zero_count), color="red", fontsize=12, fontweight="bold", ha="right", va="bottom")

    max_val = max(np.log1p(y_true).max(), np.log1p(pred).max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")

    plt.xlabel("Actual revenue, log(1 + y)")
    plt.ylabel("Predicted revenue, log(1 + prediction)")
    plt.title(f"Predicted vs actual revenue: {model_name}")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"predicted_vs_actual_{clean_name(model_name)}.png", dpi=300)
    plt.close()


# ============================================================
# 5. RESIDUAL BOXPLOT BY MODEL
# ============================================================

residual_data = []
labels = []

for model_name, df in model_data.items():
    residuals = df["y_true"].to_numpy() - df["pred"].to_numpy()
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
# 6. REVENUE BIN CALIBRATION PLOTS
# ============================================================

for plot_type in ["linear", "linear_zoomed", "log", "log_zoomed"]:
    plt.figure(figsize=(9, 6))
    actual_plotted = False

    for model_name, df in model_data.items():
        df_cal = df[["y_true", "pred"]].copy()

        df_cal["bin"] = pd.qcut(
            df_cal["pred"].rank(method="first"),
            q=10,
            labels=False
        )

        df_cal["log_y_true"] = np.log1p(df_cal["y_true"])
        df_cal["log_pred"] = np.log1p(df_cal["pred"])

        grouped = df_cal.groupby("bin").agg(
            avg_actual=("y_true", "mean"),
            avg_predicted=("pred", "mean"),
            avg_log_actual=("log_y_true", "mean"),
            avg_log_predicted=("log_pred", "mean")
        ).reset_index()

        x = grouped["bin"] + 1

        if plot_type.startswith("linear"):
            y_act = grouped["avg_actual"]
            y_pred = grouped["avg_predicted"]
            y_label = "Average revenue"
            title = "Revenue calibration by prediction decile (Linear)"
        else:
            y_act = grouped["avg_log_actual"]
            y_pred = grouped["avg_log_predicted"]
            y_label = "Average revenue, log(1 + y)"
            title = "Revenue calibration by prediction decile (Log Scale)"

        if not actual_plotted:
            plt.plot(x, y_act, label="Actual average revenue", linewidth=2, color="black", linestyle="--")
            actual_plotted = True

        plt.plot(x, y_pred, label=f"Predicted: {model_name}", linewidth=1)

    plt.xlabel("Prediction decile, low to high")
    plt.ylabel(y_label)

    if "zoomed" in plot_type:
        plt.xlim(6, 10)
        title += " - Zoomed"
        if plot_type == "linear_zoomed":
            plt.ylim(bottom=np.exp(6), top=np.exp(10))
        elif plot_type == "log_zoomed":
            plt.ylim(bottom=6, top=10)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"calibration_deciles_all_models_{plot_type}.png", dpi=300)
    plt.close()


# 7. COMMON-SAMPLE CALIBRATION PLOTS
# ============================================================
# This makes calibration plots comparable across models.
# Only models with cust_id are used, and only customers present in all kept models are used.

common_ids = None

for df in model_data.values():
    ids = set(df["cust_id"])
    common_ids = ids if common_ids is None else common_ids.intersection(ids)

common_ids = set(common_ids)

print(f"\nCommon customer sample across kept models: {len(common_ids)} customers")

plt.figure(figsize=(9, 6))
actual_plotted_common = False

for model_name, df in model_data.items():
    df_common = df[df["cust_id"].isin(common_ids)].copy()

    df_common["bin"] = pd.qcut(
        df_common["pred"].rank(method="first"),
        q=10,
        labels=False
    )

    grouped = df_common.groupby("bin").agg(
        avg_actual=("y_true", "mean"),
        avg_predicted=("pred", "mean")
    ).reset_index()

    if not actual_plotted_common:
        plt.plot(grouped["bin"] + 1, grouped["avg_actual"], label="Actual average revenue", linewidth=2, color="black", linestyle="--")
        actual_plotted_common = True

    plt.plot(grouped["bin"] + 1, grouped["avg_predicted"], label=f"Predicted: {model_name}", linewidth=1)

plt.xlabel("Prediction decile, low to high")
plt.ylabel("Average revenue")
plt.title("Common-sample calibration (All Models)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "common_sample_calibration_all_models.png", dpi=300)
plt.close()

# ============================================================
# 8. MAE BY RECENCY BIN
# ============================================================

for model_name, df in model_data.items():
    if "recency_days" not in df.columns:
        continue

    df_rec = df.copy()
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

    plt.savefig(OUTPUT_DIR / f"mae_by_recency_{clean_name(model_name)}.png", dpi=300)
    plt.close()


# ============================================================
# 9. PREDICTION CORRELATION HEATMAP ON COMMON CUSTOMERS
# ============================================================

if len(model_data) >= 2 and len(common_ids) > 0:
    pred_frames = []

    for model_name, df in model_data.items():
        temp = df[df["cust_id"].isin(common_ids)][["cust_id", "pred"]].copy()
        temp = temp.rename(columns={"pred": model_name})
        pred_frames.append(temp)

    pred_matrix = pred_frames[0]

    for temp in pred_frames[1:]:
        pred_matrix = pred_matrix.merge(temp, on="cust_id", how="inner")

    corr = pred_matrix.drop(columns=["cust_id"]).corr()

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