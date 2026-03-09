# %% [markdown]
# Stronger customer-level feature engineering for return prediction
# Main upgrades:
# - time split
# - customer aggregations
# - RFM + recent activity windows
# - purchase interval features
# - diversity / activity-rate features
# - improved leakage-safe encodings with smoothing
# - richer customer-level modeling tables

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Load Data

# %%
transactions_path = "data/transactions_2016_2017.csv"
customer_path = "data/customer_clv_train.csv"

data_dir = Path("/mnt/data")

def _resolve_csv(preferred_path: str, glob_patterns: list[str]) -> str:
    p = Path(preferred_path)
    if p.exists():
        return str(p)

    candidates = []
    for pat in glob_patterns:
        candidates.extend(sorted(data_dir.glob(pat)))

    if len(candidates) == 0:
        print("Files available in /mnt/data:")
        print([x.name for x in sorted(data_dir.glob("*"))])
        raise FileNotFoundError(f"Could not resolve file for: {preferred_path}")

    resolved = str(candidates[0])
    print(f"resolved '{preferred_path}' -> '{resolved}'")
    return resolved

transactions_path = _resolve_csv(
    transactions_path,
    glob_patterns=["*transactions*2016*2017*.csv", "*transactions*.csv"]
)
customer_path = _resolve_csv(
    customer_path,
    glob_patterns=["*customer*clv*train*.csv", "*customer*.csv"]
)

tx_cols = pd.read_csv(transactions_path, nrows=0).columns
date_cols = [c for c in ["order_date", "pack_date"] if c in tx_cols]

df_transactions = pd.read_csv(
    transactions_path,
    parse_dates=date_cols if len(date_cols) > 0 else None
)
df_customer = pd.read_csv(customer_path)

print("Loaded transactions shape:", df_transactions.shape)
print("Loaded customer shape:", df_customer.shape)

# %%
df = pd.merge(df_transactions, df_customer, on="cust_id", how="left")
print("Merged shape:", df.shape)
print(df.head())

# %% [markdown]
# ## Basic cleaning

# %%
def _to_binary_flag(s: pd.Series) -> pd.Series:
    true_vals = {1, True, "1", "true", "True", "yes", "Yes", "Y", "y"}
    false_vals = {0, False, "0", "false", "False", "no", "No", "N", "n"}
    out = s.copy()

    out = out.map(lambda x: 1 if x in true_vals else (0 if x in false_vals else np.nan))
    return out.fillna(0).astype(int)

# Numeric coercion
for col in ["prod_size", "sale_revenue", "sale_discount_applied"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Binary-ish fields
for col in ["prod_web_only", "prod_insole", "prod_outlet"]:
    if col in df.columns:
        df[col] = _to_binary_flag(df[col])

# Returned flag
if "returned_to_shop_id" in df.columns:
    df["returned"] = df["returned_to_shop_id"].notna().astype(int)
else:
    df["returned"] = 0

# Packing delay
if "order_date" in df.columns and "pack_date" in df.columns:
    if np.issubdtype(df["order_date"].dtype, np.datetime64) and np.issubdtype(df["pack_date"].dtype, np.datetime64):
        df["days_to_pack"] = (df["pack_date"] - df["order_date"]).dt.days

# Keep positive spend proxy for cleaner customer aggregates
if "sale_revenue" in df.columns:
    df["sale_revenue_pos"] = df["sale_revenue"].clip(lower=0)
else:
    df["sale_revenue_pos"] = 0.0


# %%
df
# %% [markdown]
# ## Date features

# %%
if "order_date" not in df.columns:
    raise ValueError("order_date is required for this pipeline.")

df = df.sort_values(["cust_id", "order_date"]).copy()

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_dayofweek"] = df["order_date"].dt.dayofweek
df["order_day"] = df["order_date"].dt.day
df["order_quarter"] = df["order_date"].dt.quarter
df["order_weekofyear"] = df["order_date"].dt.isocalendar().week.astype(int)

# cyclical
df["order_month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12)
df["order_month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12)
df["order_dow_sin"] = np.sin(2 * np.pi * df["order_dayofweek"] / 7)
df["order_dow_cos"] = np.cos(2 * np.pi * df["order_dayofweek"] / 7)

# purchase interval features at transaction level
df["prev_order_date"] = df.groupby("cust_id")["order_date"].shift(1)
df["days_since_prev_order"] = (df["order_date"] - df["prev_order_date"]).dt.days

# %% [markdown]
# ## Discount / price behavior features

# %%
if "sale_discount_applied" in df.columns:
    df["discount_abs"] = df["sale_discount_applied"].abs()
    df["is_discounted"] = (df["discount_abs"].fillna(0) > 0).astype(int)
else:
    df["discount_abs"] = 0.0
    df["is_discounted"] = 0

base = df["sale_revenue_pos"].abs() + df["discount_abs"].abs()
df["discount_ratio"] = np.where(base > 0, df["discount_abs"] / base, 0.0)

# customer spent after discount vs. discount amount
df["net_value_proxy"] = df["sale_revenue_pos"]
df["gross_value_proxy"] = df["sale_revenue_pos"] + df["discount_abs"]

# %% [markdown]
# ## Time-based split

# %%
df = df.sort_values("order_date").copy()
split_date = df["order_date"].quantile(0.8)

df_train_tx = df[df["order_date"] <= split_date].copy()
df_val_tx = df[df["order_date"] > split_date].copy()

print("Time split date:", split_date)
print("df_train_tx:", df_train_tx.shape)
print("df_val_tx  :", df_val_tx.shape)

# %% [markdown]
# ## Helper functions

# %%
def _safe_fill_cat(s: pd.Series) -> pd.Series:
    return s.astype("object").fillna("Unknown")

def build_freq_map(df_fit: pd.DataFrame, col: str) -> dict:
    vc = _safe_fill_cat(df_fit[col]).value_counts(dropna=False)
    return (vc / len(df_fit)).to_dict()

def apply_freq_map(df_apply: pd.DataFrame, col: str, fmap: dict, new_col: str) -> pd.DataFrame:
    df_apply[new_col] = _safe_fill_cat(df_apply[col]).map(fmap).fillna(0.0).astype(float)
    return df_apply

def build_smoothed_target_mean_map(
    df_fit: pd.DataFrame,
    col: str,
    target_col: str,
    smoothing: float = 20.0
) -> tuple[dict, float]:
    temp = df_fit[[col, target_col]].copy()
    temp[col] = _safe_fill_cat(temp[col])

    global_mean = float(temp[target_col].mean())

    stats = temp.groupby(col)[target_col].agg(["mean", "count"]).reset_index()
    stats["smoothed"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean) /
        (stats["count"] + smoothing)
    )

    mapping = dict(zip(stats[col], stats["smoothed"]))
    return mapping, global_mean

def add_target_mean_feature(
    df_apply: pd.DataFrame,
    col: str,
    mapping: dict,
    global_mean: float,
    new_col: str
) -> pd.DataFrame:
    df_apply[new_col] = _safe_fill_cat(df_apply[col]).map(mapping).fillna(global_mean).astype(float)
    return df_apply

def _mode_or_unknown(x: pd.Series) -> str:
    x = _safe_fill_cat(x)
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else "Unknown"

# %% [markdown]
# ## Stronger customer-level feature builder

# %%
def build_customer_features(df_tx: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    df_tx = df_tx.sort_values(["cust_id", "order_date"]).copy()
    g = df_tx.groupby("cust_id", dropna=False)

    out = pd.DataFrame(index=df_tx["cust_id"].dropna().unique())
    out.index.name = "cust_id"

    # ----------------------------
    # Core counts and spend
    # ----------------------------
    out["cust_num_transactions"] = g.size()

    if "sale_revenue" in df_tx.columns:
        out["cust_txn_revenue_sum"] = g["sale_revenue"].sum()
        out["cust_txn_revenue_mean"] = g["sale_revenue"].mean()
        out["cust_txn_revenue_median"] = g["sale_revenue"].median()
        out["cust_txn_revenue_max"] = g["sale_revenue"].max()
        out["cust_txn_revenue_min"] = g["sale_revenue"].min()
        out["cust_txn_revenue_std"] = g["sale_revenue"].std()
        out["cust_txn_revenue_pos_sum"] = g["sale_revenue_pos"].sum()

    if "sale_discount_applied" in df_tx.columns:
        out["cust_discount_mean"] = g["sale_discount_applied"].mean()
        out["cust_discount_median"] = g["sale_discount_applied"].median()
        out["cust_discount_min"] = g["sale_discount_applied"].min()
        out["cust_discount_max"] = g["sale_discount_applied"].max()
        out["cust_discount_std"] = g["sale_discount_applied"].std()

    # ----------------------------
    # Product diversity
    # ----------------------------
    for c in ["prod_id", "prod_brand", "prod_color", "prod_material", "prod_type_1", "prod_type_3", "prod_type_4", "prod_type_5"]:
        if c in df_tx.columns:
            out[f"{c}_nunique"] = g[c].nunique()

    if "prod_id" in df_tx.columns:
        out["cust_num_unique_products"] = g["prod_id"].nunique()
    if "prod_brand" in df_tx.columns:
        out["cust_num_unique_brands"] = g["prod_brand"].nunique()

    # ----------------------------
    # RFM + tenure
    # ----------------------------
    first_purchase = g["order_date"].min()
    last_purchase = g["order_date"].max()

    out["recency_days"] = (snapshot_date - last_purchase).dt.days
    out["customer_tenure_days"] = (snapshot_date - first_purchase).dt.days
    out["frequency"] = g.size()

    if "sale_revenue" in df_tx.columns:
        out["monetary"] = g["sale_revenue"].sum()

    # ----------------------------
    # Purchase interval behavior
    # ----------------------------
    if "days_since_prev_order" in df_tx.columns:
        out["avg_days_between_orders"] = g["days_since_prev_order"].mean()
        out["std_days_between_orders"] = g["days_since_prev_order"].std()
        out["min_days_between_orders"] = g["days_since_prev_order"].min()
        out["max_days_between_orders"] = g["days_since_prev_order"].max()

    # ----------------------------
    # Recent activity windows
    # ----------------------------
    for w in [30, 90, 180]:
        start_date = snapshot_date - pd.Timedelta(days=w)
        recent = df_tx[df_tx["order_date"] >= start_date].copy()
        gr = recent.groupby("cust_id", dropna=False)

        out[f"txn_count_last_{w}d"] = gr.size()
        if "sale_revenue_pos" in recent.columns:
            out[f"revenue_sum_last_{w}d"] = gr["sale_revenue_pos"].sum()
        if "returned" in recent.columns:
            out[f"return_count_last_{w}d"] = gr["returned"].sum()
        if "is_discounted" in recent.columns:
            out[f"discounted_share_last_{w}d"] = gr["is_discounted"].mean()

    # ----------------------------
    # Discount behavior
    # ----------------------------
    out["cust_discount_ratio_mean"] = g["discount_ratio"].mean()
    out["cust_discount_ratio_max"] = g["discount_ratio"].max()
    out["cust_discounted_share"] = g["is_discounted"].mean()

    # ----------------------------
    # Returns behavior
    # ----------------------------
    out["return_rate"] = g["returned"].mean()
    out["num_returns"] = g["returned"].sum()

    # ----------------------------
    # Fulfilment behavior
    # ----------------------------
    if "days_to_pack" in df_tx.columns:
        out["avg_days_to_pack"] = g["days_to_pack"].mean()
        out["max_days_to_pack"] = g["days_to_pack"].max()
        out["std_days_to_pack"] = g["days_to_pack"].std()

    # ----------------------------
    # Seasonality means
    # ----------------------------
    for c in ["order_month_sin", "order_month_cos", "order_dow_sin", "order_dow_cos"]:
        out[f"{c}_mean"] = g[c].mean()

    # ----------------------------
    # Favorite categories / product preferences
    # ----------------------------
    for c in ["prod_type_1", "prod_type_3", "prod_type_4", "prod_type_5", "prod_brand", "prod_color", "prod_material", "prod_clasp", "prod_heel"]:
        if c in df_tx.columns:
            out[f"top_{c}"] = g[c].agg(_mode_or_unknown)

    # ----------------------------
    # Numeric product summaries
    # ----------------------------
    for c in ["prod_size", "prod_web_only", "prod_insole", "prod_outlet"]:
        if c in df_tx.columns:
            out[f"{c}_mean"] = g[c].mean()
            out[f"{c}_std"] = g[c].std()

    # ----------------------------
    # Rate features
    # ----------------------------
    tenure_months = (out["customer_tenure_days"].clip(lower=1) / 30.0)
    out["txn_per_month"] = out["cust_num_transactions"] / tenure_months
    out["revenue_per_month"] = out["cust_txn_revenue_pos_sum"].fillna(0) / tenure_months
    out["products_per_txn"] = out["cust_num_unique_products"].fillna(0) / out["cust_num_transactions"].clip(lower=1)
    out["brands_per_txn"] = out["cust_num_unique_brands"].fillna(0) / out["cust_num_transactions"].clip(lower=1)

    # Ratios from recent windows
    for w in [30, 90, 180]:
        if f"txn_count_last_{w}d" in out.columns:
            out[f"txn_share_last_{w}d"] = out[f"txn_count_last_{w}d"].fillna(0) / out["cust_num_transactions"].clip(lower=1)
        if f"revenue_sum_last_{w}d" in out.columns:
            out[f"revenue_share_last_{w}d"] = out[f"revenue_sum_last_{w}d"].fillna(0) / out["cust_txn_revenue_pos_sum"].clip(lower=1)

    # ----------------------------
    # Log transforms (often useful)
    # ----------------------------
    for c in [
        "recency_days",
        "customer_tenure_days",
        "cust_num_transactions",
        "cust_txn_revenue_pos_sum",
        "cust_num_unique_products",
        "cust_num_unique_brands",
        "txn_count_last_30d",
        "txn_count_last_90d",
        "txn_count_last_180d",
        "revenue_sum_last_30d",
        "revenue_sum_last_90d",
        "revenue_sum_last_180d",
    ]:
        if c in out.columns:
            out[f"log1p_{c}"] = np.log1p(out[c].fillna(0).clip(lower=0))

    return out.reset_index()

# %%
snapshot_train = df_train_tx["order_date"].max()
snapshot_val = df_val_tx["order_date"].max()

cust_train_feat = build_customer_features(df_train_tx, snapshot_train)
cust_val_feat = build_customer_features(df_val_tx, snapshot_val)

print("cust_train_feat:", cust_train_feat.shape)
print("cust_val_feat  :", cust_val_feat.shape)

# %% [markdown]
# ## Merge customer target data

# %%
df_customer_unique = df_customer.drop_duplicates(subset=["cust_id"]).copy()

df_train = cust_train_feat.merge(df_customer_unique, on="cust_id", how="left")
df_val = cust_val_feat.merge(df_customer_unique, on="cust_id", how="left")

print("df_train:", df_train.shape)
print("df_val  :", df_val.shape)

target_col = "revenue_2018_2019" if "revenue_2018_2019" in df_train.columns else None
print("Target column:", target_col)

# %% [markdown]
# ## Leakage-safe encodings fitted on training only

# %%
cat_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ["cust_id", target_col]]

print("Customer-level categorical columns:", cat_cols)

LOW_CARD_THRESHOLD = 8
low_card_cols = [c for c in cat_cols if df_train[c].nunique(dropna=True) <= LOW_CARD_THRESHOLD]
high_card_cols = [c for c in cat_cols if c not in low_card_cols]

print("low_card_cols :", low_card_cols)
print("high_card_cols:", high_card_cols)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

if len(low_card_cols) > 0:
    ohe.fit(df_train[low_card_cols].apply(_safe_fill_cat))
    ohe_feature_names = ohe.get_feature_names_out(low_card_cols).tolist()
else:
    ohe_feature_names = []

freq_maps = {}
for c in high_card_cols:
    freq_maps[c] = build_freq_map(df_train, c)

target_mean_cols = [c for c in high_card_cols if c in [
    "top_prod_brand",
    "top_prod_type_1",
    "top_prod_type_3",
    "top_prod_color",
    "top_prod_material",
    "top_prod_heel"
]]

target_mean_maps = {}
global_target_means = {}

if target_col is not None:
    for c in target_mean_cols:
        mapping, global_mean = build_smoothed_target_mean_map(
            df_fit=df_train,
            col=c,
            target_col=target_col,
            smoothing=25.0
        )
        target_mean_maps[c] = mapping
        global_target_means[c] = global_mean

def transform_customer_set(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()

    # frequency encoding
    for c in high_card_cols:
        out = apply_freq_map(out, c, freq_maps[c], f"{c}_rel_freq")

    # smoothed target mean encoding
    if target_col is not None:
        for c in target_mean_cols:
            out = add_target_mean_feature(
                out,
                c,
                target_mean_maps[c],
                global_target_means[c],
                f"{c}_target_mean"
            )

    # one-hot for very low-cardinality cols
    if len(low_card_cols) > 0:
        ohe_arr = ohe.transform(out[low_card_cols].apply(_safe_fill_cat))
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_feature_names, index=out.index)
        out = pd.concat([out, ohe_df], axis=1)

    return out

df_train_fe = transform_customer_set(df_train)
df_val_fe = transform_customer_set(df_val)

# %% [markdown]
# ## Build final X / y

# %%
drop_cols = ["cust_id"]

if target_col is not None:
    drop_cols.append(target_col)

drop_cols.extend(low_card_cols)
drop_cols.extend(high_card_cols)
drop_cols = list(set(drop_cols))

X_train = df_train_fe.drop(columns=[c for c in drop_cols if c in df_train_fe.columns], errors="ignore")
X_val = df_val_fe.drop(columns=[c for c in drop_cols if c in df_val_fe.columns], errors="ignore")

X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0.0)

if target_col is not None:
    y_train = df_train_fe[target_col]
    y_val = df_val_fe[target_col]
else:
    y_train, y_val = None, None

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
if y_train is not None:
    print("y_train:", y_train.shape)
    print("y_val  :", y_val.shape)

print("\nMissing values in X_train (top 20):")
mv_train = X_train.isnull().sum().sort_values(ascending=False)
print(mv_train[mv_train > 0].head(20))

print("\nMissing values in X_val (top 20):")
mv_val = X_val.isnull().sum().sort_values(ascending=False)
print(mv_val[mv_val > 0].head(20))

# %% [markdown]
# ## Save modeling datasets

# %%
output_dir = "model_data_customer_level_v2"
os.makedirs(output_dir, exist_ok=True)

X_train_save = X_train.copy()
X_val_save = X_val.copy()

X_train_save.insert(0, "cust_id", df_train.loc[X_train.index, "cust_id"].values)
X_val_save.insert(0, "cust_id", df_val.loc[X_val.index, "cust_id"].values)

X_train_save.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_val_save.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)

if y_train is not None:
    y_train.reset_index(drop=True).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.reset_index(drop=True).to_csv(os.path.join(output_dir, "y_val.csv"), index=False)

df_train_model = X_train_save.copy()
df_val_model = X_val_save.copy()

if y_train is not None:
    df_train_model[target_col] = y_train.reset_index(drop=True).values
    df_val_model[target_col] = y_val.reset_index(drop=True).values

df_train_model.to_csv(os.path.join(output_dir, "df_train_model.csv"), index=False)
df_val_model.to_csv(os.path.join(output_dir, "df_val_model.csv"), index=False)

print("Saved files to:", output_dir)
# %%
