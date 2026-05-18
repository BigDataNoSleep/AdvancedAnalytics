# model_interpretation.py

import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import lightgbm as lgb
from lightgbm import LGBMRegressor

# Make imports work from project root or /models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eda_transactions_advanced import get_advanced_customer_model_data


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
OUTPUT_DIR = Path("model_interpretation")
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_N_FEATURES = 25
PERMUTATION_SAMPLE_SIZE = 12000
PCA_TSNE_SAMPLE_SIZE = 5000


# ============================================================
# HELPERS
# ============================================================

def clean_feature_matrix(X):
    X_clean = X.copy()

    for col in X_clean.columns:
        if X_clean[col].dtype == "object" or str(X_clean[col].dtype) == "category":
            X_clean[col] = X_clean[col].astype("category").cat.codes

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(0)

    return X_clean


def save_barplot(df, x_col, y_col, title, filename, top_n=25):
    plot_df = df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df[y_col], plot_df[x_col])
    plt.xlabel(x_col)
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def revenue_segment(y):
    return pd.cut(
        y,
        bins=[-1, 0, 50, 150, 500, np.inf],
        labels=["Zero", "Low", "Medium", "High", "Very high"]
    )


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading advanced customer modelling data...")

X_train, X_val, X_test, y_train, y_val, y_test, test_unlabelled = get_advanced_customer_model_data()

X = pd.concat([X_train, X_val, X_test])
y = pd.concat([y_train, y_val, y_test])

X = clean_feature_matrix(X)
y = y.reset_index(drop=True)
X = X.reset_index(drop=True)

print(f"Rows: {X.shape[0]}")
print(f"Features: {X.shape[1]}")


# ============================================================
# 2. TRAIN INTERPRETATION MODEL
# ============================================================

print("\nTraining LightGBM interpretation model...")

X_tr, X_hold, y_tr, y_hold = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

model = LGBMRegressor(
    objective="mae",
    metric="mae",
    boosting_type="goss",
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=70,
    min_child_samples=200,
    colsample_bytree=0.7,
    reg_alpha=10.0,
    reg_lambda=10.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_hold, y_hold)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(300, verbose=False)]
)

pred_hold = np.maximum(model.predict(X_hold), 0)
mae = mean_absolute_error(y_hold, pred_hold)

print(f"Holdout MAE: {mae:.4f}")


# ============================================================
# 3. LIGHTGBM FEATURE IMPORTANCE
# ============================================================

print("\nSaving LightGBM feature importance...")

importance_gain = model.booster_.feature_importance(importance_type="gain")
importance_split = model.booster_.feature_importance(importance_type="split")

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_gain": importance_gain,
    "importance_split": importance_split
})

importance_df["importance_gain_normalized"] = (
    importance_df["importance_gain"] / importance_df["importance_gain"].sum()
)

importance_df = importance_df.sort_values("importance_gain", ascending=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance_lightgbm.csv", index=False)

save_barplot(
    importance_df,
    x_col="importance_gain",
    y_col="feature",
    title="Top features by LightGBM gain importance",
    filename="feature_importance_gain.png",
    top_n=TOP_N_FEATURES
)

save_barplot(
    importance_df.sort_values("importance_split", ascending=False),
    x_col="importance_split",
    y_col="feature",
    title="Top features by LightGBM split importance",
    filename="feature_importance_split.png",
    top_n=TOP_N_FEATURES
)


# ============================================================
# 4. PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance...")

sample_n = min(PERMUTATION_SAMPLE_SIZE, len(X_hold))
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_hold), size=sample_n, replace=False)

X_perm = X_hold.iloc[sample_idx]
y_perm = y_hold.iloc[sample_idx]

perm = permutation_importance(
    model,
    X_perm,
    y_perm,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": X.columns,
    "permutation_importance_mean": perm.importances_mean,
    "permutation_importance_std": perm.importances_std
}).sort_values("permutation_importance_mean", ascending=False)

perm_df.to_csv(OUTPUT_DIR / "feature_importance_permutation.csv", index=False)

save_barplot(
    perm_df,
    x_col="permutation_importance_mean",
    y_col="feature",
    title="Top features by permutation importance",
    filename="feature_importance_permutation.png",
    top_n=TOP_N_FEATURES
)


# ============================================================
# 5. SHAP IMPORTANCE, IF INSTALLED
# ============================================================

try:
    import shap

    print("\nCalculating SHAP values...")

    shap_n = min(4000, len(X_hold))
    shap_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_hold), size=shap_n, replace=False)
    X_shap = X_hold.iloc[shap_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    shap_abs = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": shap_abs
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(OUTPUT_DIR / "feature_importance_shap.csv", index=False)

    save_barplot(
        shap_df,
        x_col="mean_abs_shap",
        y_col="feature",
        title="Top features by mean absolute SHAP value",
        filename="feature_importance_shap.png",
        top_n=TOP_N_FEATURES
    )

except Exception as e:
    print(f"Skipping SHAP because it is not available or failed: {e}")


# ============================================================
# 6. PCA ANALYSIS
# ============================================================

print("\nRunning PCA analysis...")

sample_n = min(PCA_TSNE_SAMPLE_SIZE, len(X))
idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=sample_n, replace=False)

X_sample = X.iloc[idx].copy()
y_sample = y.iloc[idx].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

pca = PCA(n_components=10, random_state=RANDOM_STATE)
pca_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": pca_components[:, 0],
    "PC2": pca_components[:, 1],
    "revenue": y_sample.to_numpy(),
    "segment": revenue_segment(y_sample).astype(str)
})

pca_df.to_csv(OUTPUT_DIR / "pca_customer_segments.csv", index=False)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=np.log1p(pca_df["revenue"]),
    alpha=0.45,
    s=10
)
plt.colorbar(scatter, label="log(1 + actual revenue)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.title("PCA of customer feature space")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_customer_feature_space.png", dpi=300)
plt.close()

# PCA loadings
loadings = pd.DataFrame({
    "feature": X.columns,
    "PC1_loading": pca.components_[0],
    "PC2_loading": pca.components_[1],
})

loadings["abs_PC1_loading"] = loadings["PC1_loading"].abs()
loadings["abs_PC2_loading"] = loadings["PC2_loading"].abs()

loadings.to_csv(OUTPUT_DIR / "pca_feature_loadings.csv", index=False)

pc1_top = loadings.sort_values("abs_PC1_loading", ascending=False).head(TOP_N_FEATURES)
pc2_top = loadings.sort_values("abs_PC2_loading", ascending=False).head(TOP_N_FEATURES)

save_barplot(
    pc1_top.rename(columns={"abs_PC1_loading": "absolute_loading"}),
    x_col="absolute_loading",
    y_col="feature",
    title="Top features driving PCA component 1",
    filename="pca_pc1_top_loadings.png",
    top_n=TOP_N_FEATURES
)

save_barplot(
    pc2_top.rename(columns={"abs_PC2_loading": "absolute_loading"}),
    x_col="absolute_loading",
    y_col="feature",
    title="Top features driving PCA component 2",
    filename="pca_pc2_top_loadings.png",
    top_n=TOP_N_FEATURES
)


# ============================================================
# 7. t-SNE VISUALISATION
# ============================================================

print("\nRunning t-SNE visualisation...")

# Use PCA first to reduce noise and speed up t-SNE
pca_30 = PCA(n_components=min(30, X_scaled.shape[1]), random_state=RANDOM_STATE)
X_pca_30 = pca_30.fit_transform(X_scaled)

tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate="auto",
    init="pca",
    random_state=RANDOM_STATE
)

tsne_components = tsne.fit_transform(X_pca_30)

tsne_df = pd.DataFrame({
    "TSNE1": tsne_components[:, 0],
    "TSNE2": tsne_components[:, 1],
    "revenue": y_sample.to_numpy(),
    "segment": revenue_segment(y_sample).astype(str)
})

tsne_df.to_csv(OUTPUT_DIR / "tsne_customer_segments.csv", index=False)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    tsne_df["TSNE1"],
    tsne_df["TSNE2"],
    c=np.log1p(tsne_df["revenue"]),
    alpha=0.45,
    s=10
)
plt.colorbar(scatter, label="log(1 + actual revenue)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE customer segmentation by actual revenue")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tsne_customer_segments.png", dpi=300)
plt.close()

# ============================================================
# 7B. t-SNE FEATURE OVERLAY PLOTS
# ============================================================

print("\nCreating t-SNE feature overlay plots...")

important_features_for_tsne = [
    "n_orders",
    "recency_days",
    "total_spent",
    "customer_lifetime_days",
    "unique_months",
    "favorite_brand_freq",
    "revenue_std",
    "recency_tenure_ratio"
]

for feature in important_features_for_tsne:
    if feature not in X_sample.columns:
        print(f"Skipping {feature}: not found in X_sample")
        continue

    values = X_sample[feature].to_numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        tsne_df["TSNE1"],
        tsne_df["TSNE2"],
        c=values,
        alpha=0.45,
        s=10
    )
    plt.colorbar(scatter, label=feature)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"t-SNE coloured by {feature}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"tsne_by_{feature}.png", dpi=300)
    plt.close()

# ============================================================
# 8. BUSINESS INTERPRETATION TABLE
# ============================================================

print("\nCreating business interpretation table...")

top_features = importance_df.head(20)["feature"].tolist()

business_notes = []

for feature in top_features:
    f = feature.lower()

    if "recency" in f:
        interpretation = "Recent purchasing activity is important. Customers who bought recently are more likely to generate future revenue."
    elif "total_spent" in f or "revenue" in f or "spent" in f:
        interpretation = "Historical spending level is a strong signal of future customer value."
    elif "order" in f or "frequency" in f or "lifetime" in f:
        interpretation = "Purchase frequency and customer relationship length help separate one-time buyers from repeat customers."
    elif "return" in f:
        interpretation = "Return behaviour influences expected future revenue and may indicate lower net customer value."
    elif "discount" in f:
        interpretation = "Discount sensitivity helps identify bargain-driven customers with different future spending behaviour."
    elif "brand" in f:
        interpretation = "Brand preference and brand loyalty contain useful customer preference information."
    elif "product" in f or "category" in f or "color" in f or "size" in f:
        interpretation = "Product diversity and preferences help describe the customer profile."
    else:
        interpretation = "This feature contributes predictive signal through non-linear interactions with other customer behaviour variables."

    business_notes.append({
        "feature": feature,
        "business_interpretation": interpretation
    })

business_df = pd.DataFrame(business_notes)
business_df.to_csv(OUTPUT_DIR / "business_interpretation_top_features.csv", index=False)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL INTERPRETATION SUMMARY")
print("=" * 70)

print(f"Holdout MAE of interpretation model: {mae:.4f}")

print("\nTop 10 LightGBM gain features:")
print(importance_df[["feature", "importance_gain"]].head(10).to_string(index=False))

print("\nTop 10 permutation importance features:")
print(perm_df[["feature", "permutation_importance_mean"]].head(10).to_string(index=False))

print("\nPCA explained variance:")
print(f"PC1: {pca.explained_variance_ratio_[0] * 100:.2f}%")
print(f"PC2: {pca.explained_variance_ratio_[1] * 100:.2f}%")
for i in range(2, 11):
    print(f"Total first {i} PCs: {pca.explained_variance_ratio_[:i].sum() * 100:.2f}%")

print(f"\nSaved all interpretation outputs to: {OUTPUT_DIR.resolve()}")