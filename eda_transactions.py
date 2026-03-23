# %% [markdown]
# Exploratory Data Analysis: Transactions 2016-2017
# This script performs EDA on the transaction dataset.

# ============================================================
# Imports
# ============================================================
# Standard data analysis libraries used throughout the script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Set plotting style
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Load Data
# We are loading the transaction data from the CSV file.
transactions_path = "/Users/vincecoppens/Documents/Courses/Big Data/AdvancedAnalytics/data/transactions_2016_2017.csv"
customer_path = "/Users/vincecoppens/Documents/Courses/Big Data/AdvancedAnalytics/data/customer_clv_train.csv"

# Load transaction and customer datasets
# order_date and pack_date are parsed as datetime for later time-based features

df_transactions = pd.read_csv(transactions_path, parse_dates=['order_date', 'pack_date'])
df_customer = pd.read_csv(customer_path)

# %%
# Merge transaction data with customer-level information using cust_id
df = pd.merge(df_transactions, df_customer, on='cust_id', how='left')

# Display the first few rows and info
print(f"Merged Dataset shape: {df.shape}")
df

# %% [markdown]
# ## Data Types
# Inspect data types and identify potential missing values
# Show data types
print("Data Types:")
print(df.dtypes)

# %%
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

# %%
# Data Cleaning and Type Conversion
# Convert 'prod_size' to numeric, non-numeric values like 'XS' will become NaN
df['prod_size'] = pd.to_numeric(df['prod_size'], errors='coerce')

# Convert boolean-like columns
for col in ['prod_web_only', 'prod_insole', 'prod_outlet']:
    # Fill missing values with False before converting to boolean to avoid NaN -> True coercion
    df[col] = df[col].fillna(False).astype(bool)

# %%
# Quick overview of feature cardinality (useful for detecting high-cardinality columns)
print("Unique values per column:")
print(df.nunique())

# %%
# Inspect unique values for specific columns to check for inconsistencies
columns_to_inspect = [
    'returned_to_shop_id', 'prod_size', 'prod_web_only', 'prod_season',
    'prod_brand', 'prod_color', 'prod_type_1', 'prod_type_3',
    'prod_type_4', 'prod_type_5', 'prod_heel', 'prod_material',
    'prod_insole', 'prod_print', 'prod_comfort_sole',
    'prod_comfort_wear', 'prod_clasp', 'prod_outlet'
]

print("Unique values (after cleaning):")
for col in columns_to_inspect:
    unique_vals = df[col].unique()
    # Sort if possible, otherwise just print
    try:
        sorted_vals = sorted(unique_vals)
    except:
        sorted_vals = unique_vals
    
    print(f"'{col}': {sorted_vals[:10]} ... (Total: {len(unique_vals)})")
    print(f"Type: {df[col].dtype}")
    print("-" * 30)

# %%
# ============================================================
# Column groups
# ============================================================
# Columns are grouped by type to simplify preprocessing steps
id_cols = ['order_id', 'cust_id', 'prod_id']
numeric_cols = ['prod_size']
bool_cols = ['prod_web_only', 'prod_insole', 'prod_outlet']
single_cols = ['prod_season', 'prod_color', 'prod_type_1', 'prod_heel']
multi_cols = [
    'prod_type_3', 'prod_type_4', 'prod_type_5', 'prod_material', 
    'prod_print', 'prod_comfort_sole', 'prod_comfort_wear', 'prod_clasp'
]
brand_cols = ['prod_brand']
drop_cols = ['prod_title']

# %%
# ============================================================
# Basic text cleaning for categorical columns
# ============================================================
# Standardizes casing, spacing, and separator formatting
cat_cols = single_cols + multi_cols + brand_cols

for col in cat_cols:
    df[col] = (
        df[col]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(" and ", ",", regex=False)
        .str.replace(r"\s*,\s*", ",", regex=True)
        .str.strip(",")
    )
    print(df[col].nunique())
    print(df[col].value_counts().head(20))

# %%
# ============================================================
# Multi-label cleaning
# ============================================================
# Removes duplicate tokens and standardizes ordering within
# comma-separated categorical fields
def clean_multilabel(cell):
    if pd.isna(cell):
        return cell

    tokens = [t.strip() for t in cell.split(",") if t.strip()]
    tokens = sorted(set(tokens))
    return ",".join(tokens)

for col in multi_cols:
    df[col] = df[col].apply(clean_multilabel)
    print(df[col].nunique())
    print(df[col].value_counts().head(20))

# %%
# Convert comma-separated multi-label fields into Python lists
# This representation will later allow proper multi-label encoding
def to_list(x):
    if pd.isna(x) or x == "":
        return []
    return [t.strip() for t in x.split(",")]

for col in multi_cols:
    df[col] = df[col].apply(to_list)

# %%
type(df["prod_clasp"].iloc[0])

# %%
# ============================================================
# Token frequency inspection
# ============================================================
# Helps identify rare categories before grouping them into "other"

def token_counts(series):
    counter = Counter()
    for row in series:
        counter.update(row)
    return counter

for col in multi_cols:
    counts = token_counts(df[col])
    
    print("=" * 50)
    print(f"{col}")
    print(f"unique tokens: {len(counts)}")
    print("\nTop 20 tokens:")
    
    for token, count in counts.most_common(20):
        print(f"{token:<25} {count}")
        
    print("\n")

# %%
# ============================================================
# Rare token grouping
# ============================================================
# Tokens appearing fewer than MIN_COUNT times are grouped into "other"
MIN_COUNT = 300


for col in multi_cols:

    counter = Counter()
    for row in df[col]:
        counter.update(row)

    keep_tokens = {t for t, c in counter.items() if c >= MIN_COUNT}

    def replace_rare(tokens):
        new_tokens = []
        for t in tokens:
            if t in keep_tokens:
                new_tokens.append(t)
            else:
                new_tokens.append("other")

        return list(set(new_tokens))  # avoid duplicates

    df[col] = df[col].apply(replace_rare)

# %%
# ============================================================
# Rare brand grouping
# ============================================================
# Reduces very high brand cardinality by grouping infrequent brands
MIN_BRAND_COUNT = 1000

brand_counts = df["prod_brand"].value_counts()

keep_brands = brand_counts[brand_counts >= MIN_BRAND_COUNT].index

df["prod_brand"] = df["prod_brand"].apply(
    lambda x: x if x in keep_brands else "other_brand"
)

print(df["prod_brand"].nunique())
print(df["prod_brand"].value_counts().head(20))


# %%
# ============================================================
# Missing Value Handling
# ============================================================
# NOTE:
# Earlier in the analysis many categorical columns appeared to
# contain large numbers of missing values. However, during the
# preprocessing step we converted multi-label columns into lists
# using:
#
#     NaN -> []
#
# An empty list represents "no attribute present" rather than a
# missing value. Because pandas does not treat [] as NaN, these
# columns no longer appear in df.isna().
#
# Example interpretation:
# prod_print = []        -> product has no print
# prod_clasp = []        -> no clasp information / not applicable
#
# This behaviour is intentional and correct for multi-label data.


# Inspect remaining missing values
missing = df.isna().sum().sort_values(ascending=False)
print(missing[missing > 0])

# ------------------------------------------------------------
# returned_to_shop_id: Missing values likely indicate that the item was never returned.
df["returned_to_shop_id"] = df["returned_to_shop_id"].fillna("none")


# ------------------------------------------------------------
# prod_heel: Many products (e.g. sneakers, sandals) do not have a heel.
# Missing values are therefore interpreted as "no heel".
df["prod_heel"] = df["prod_heel"].fillna("no heel")


# ------------------------------------------------------------
# prod_size: Only one observation has a missing size.
# Instead of imputing (which could introduce leakage),
# we simply remove that row.
df = df[df["prod_size"].notna()]

# ------------------------------------------------------------
# IMPORTANT: revenue_2018_2019
# ------------------------------------------------------------
# Missing values in this column correspond to the rows for which
# we must predict the revenue (test set).
# Therefore this column is intentionally NOT modified.


# Verify remaining missing values
missing_after = df.isna().sum().sort_values(ascending=False)
print("\nRemaining missing values:")
print(missing_after[missing_after > 0])

# %%
# ============================================================
# Duplicate check (list-safe)
# ============================================================

# Pandas cannot hash lists, so duplicated() fails on columns
# that contain lists (our multi-label columns).
#
# We temporarily convert lists to tuples for the purpose of
# detecting duplicates.

df_check = df.copy()

for col in multi_cols:
    df_check[col] = df_check[col].apply(tuple)

duplicates = df_check.duplicated().sum()

print("Number of duplicate rows:", duplicates)
print("Percentage of duplicates:", round(duplicates / len(df), 4) * 100, "%")
print("Removeing duplicates...")
df = df.loc[~df_check.duplicated()]

# %% [markdown]
# ============================================================
# Competition split based on target availability
# ============================================================

# Transactions from customers with known revenue (train)
df_train = df[df['revenue_2018_2019'].notna()].copy()

# Transactions from customers with unknown revenue (competition test)
df_test = df[df['revenue_2018_2019'].isna()].copy()

print("Train transactions:", df_train.shape)
print("Competition test transactions:", df_test.shape)

print("Train customers:", df_train['cust_id'].nunique())
print("Test customers:", df_test['cust_id'].nunique())

# %% [markdown]
# ============================================================
# Reusable customer feature engineering function
# ============================================================

def build_customer_features(df_input):
    df_local = df_input.copy()

    # Transaction level features
    df_local['days_to_pack'] = (df_local['pack_date'] - df_local['order_date']).dt.days
    df_local['returned'] = df_local['returned_to_shop_id'] != 'none'

    # Activity cohort
    activity_features = df_local.groupby('cust_id').agg(
        n_orders=('order_date','nunique'),
        n_products=('prod_id','nunique'),
        n_brands=('prod_brand','nunique'),
        n_categories=('prod_type_1','nunique'),
        n_colors=('prod_color','nunique')
    )

    # Financial cohort
    financial_features = df_local.groupby('cust_id').agg(
        total_spent=('sale_revenue','sum'),
        avg_order_value=('sale_revenue','mean'),
        max_order_value=('sale_revenue','max'),
        revenue_std=('sale_revenue','std'),
        revenue_median=('sale_revenue','median')
    )

    # Discount cohort
    discount_features = df_local.groupby('cust_id').agg(
        avg_discount=('sale_discount_applied','mean'),
        max_discount=('sale_discount_applied','min'),
        discount_std=('sale_discount_applied','std')
    )

    # Return cohort
    return_features = df_local.groupby('cust_id').agg(
        return_rate=('returned','mean'),
        n_returns=('returned','sum'),
        returned_flag=('returned','max')
    )

    # Logistics
    logistics_features = df_local.groupby('cust_id').agg(
        avg_days_to_pack=('days_to_pack','mean')
    )

    # Time cohort
    time_features = df_local.groupby('cust_id').agg(
        first_order_date=('order_date','min'),
        last_order_date=('order_date','max')
    )

    customer_features = (
        activity_features
        .join(financial_features)
        .join(discount_features)
        .join(return_features)
        .join(logistics_features)
        .join(time_features)
        .reset_index()
    )

    # Time derived
    max_date = df_local['order_date'].max()

    customer_features['customer_lifetime_days'] = (
        customer_features['last_order_date'] -
        customer_features['first_order_date']
    ).dt.days

    customer_features['recency_days'] = (
        max_date - customer_features['last_order_date']
    ).dt.days

    customer_features['orders_per_day_raw'] = (
        customer_features['n_orders'] /
        (customer_features['customer_lifetime_days'] + 1)
    )

    customer_features['orders_per_day'] = (
        customer_features['n_orders'] /
        (customer_features['customer_lifetime_days'] + 30)
    )

    # Brand behaviour
    brand_counts = (
        df_local.groupby(['cust_id','prod_brand'])
        .size()
        .reset_index(name='brand_orders')
    )

    brand_counts_sorted = brand_counts.sort_values(
        ['cust_id','brand_orders'],
        ascending=[True, False]
    )

    top_brand = brand_counts_sorted.drop_duplicates('cust_id')

    top_brand = top_brand.rename(columns={
        'prod_brand':'favorite_brand',
        'brand_orders':'favorite_brand_orders'
    })

    customer_features = customer_features.merge(
        top_brand[['cust_id','favorite_brand','favorite_brand_orders']],
        on='cust_id',
        how='left'
    )

    customer_features['favorite_brand_share'] = (
        customer_features['favorite_brand_orders'] /
        customer_features['n_products']
    )

    # Advanced behavioural signals
    last_order_value = (
        df_local.sort_values('order_date')
        .groupby('cust_id')['sale_revenue']
        .last()
        .rename('last_order_value')
    )

    first_order_value = (
        df_local.sort_values('order_date')
        .groupby('cust_id')['sale_revenue']
        .first()
        .rename('first_order_value')
    )

    customer_features = customer_features.merge(last_order_value, on='cust_id')
    customer_features = customer_features.merge(first_order_value, on='cust_id')

    # Order spacing
    df_local = df_local.sort_values(['cust_id','order_date'])
    df_local['prev_order_date'] = df_local.groupby('cust_id')['order_date'].shift(1)
    df_local['days_between_orders'] = (
        df_local['order_date'] - df_local['prev_order_date']
    ).dt.days

    avg_days_between = (
        df_local.groupby('cust_id')['days_between_orders']
        .mean()
        .rename('avg_days_between_orders')
    )

    customer_features = customer_features.merge(avg_days_between, on='cust_id', how='left')
    customer_features['avg_days_between_orders'] = customer_features['avg_days_between_orders'].fillna(0)

    # Discount depth
    total_discount = (
        df_local.groupby('cust_id')['sale_discount_applied']
        .sum()
        .rename('total_discount')
    )

    customer_features = customer_features.merge(total_discount, on='cust_id')
    
    # Outlet behaviour
    outlet_rate = (
        df_local.groupby('cust_id')['prod_outlet']
        .mean()
        .rename('outlet_rate')
    )
    customer_features = customer_features.merge(outlet_rate, on='cust_id')

    customer_features['discount_ratio'] = (
        customer_features['total_discount'] /
        (customer_features['total_spent'] + 1e-9)
    )

    # Diversity ratios
    customer_features['product_diversity_ratio'] = (
        customer_features['n_products'] /
        customer_features['n_orders']
    )

    customer_features['brand_diversity_ratio'] = (
        customer_features['n_brands'] /
        customer_features['n_orders']
    )

    customer_features['returns_per_order'] = (
        customer_features['n_returns'] /
        customer_features['n_orders']
    )

    # Recent behaviour
    recent_cutoff = max_date - pd.Timedelta(days=90)

    recent_transactions = df_local[df_local['order_date'] >= recent_cutoff]

    recent_orders = (
        recent_transactions.groupby('cust_id')['order_date']
        .nunique()
        .rename('orders_last_90d')
    )

    recent_revenue = (
        recent_transactions.groupby('cust_id')['sale_revenue']
        .sum()
        .rename('revenue_last_90d')
    )

    customer_features = customer_features.merge(recent_orders, on='cust_id', how='left')
    customer_features = customer_features.merge(recent_revenue, on='cust_id', how='left')

    customer_features['orders_last_90d'] = customer_features['orders_last_90d'].fillna(0)
    customer_features['revenue_last_90d'] = customer_features['revenue_last_90d'].fillna(0)

    customer_features['recent_active_flag'] = (
        customer_features['recency_days'] <= 90
    ).astype(int)

    # Stability fixes
    customer_features['revenue_std'] = customer_features['revenue_std'].fillna(0)
    customer_features['discount_std'] = customer_features['discount_std'].fillna(0)

    # Log versions of skewed features (keep originals for tree models)
    customer_features['log_total_spent'] = np.log1p(customer_features['total_spent'].clip(lower=0))
    customer_features['log_revenue_last_90d'] = np.log1p(customer_features['revenue_last_90d'].clip(lower=0))
    customer_features['log_avg_order_value'] = np.log1p(customer_features['avg_order_value'].clip(lower=0))

    # Binary activity signals (often strong predictors)
    customer_features['has_recent_orders'] = (
        customer_features['orders_last_90d'] > 0
    ).astype(int)

    customer_features['has_returns'] = (
        customer_features['n_returns'] > 0
    ).astype(int)

    # Spending trend (growth signal)
    customer_features['spending_trend'] = (
        customer_features['last_order_value'] -
        customer_features['first_order_value']
    )

    # Ratio of recent revenue vs historical
    customer_features['recent_revenue_ratio'] = (
        customer_features['revenue_last_90d'] /
        (customer_features['total_spent'] + 1)
    )

    numeric_cols = customer_features.select_dtypes(include=[np.number]).columns
    customer_features[numeric_cols] = customer_features[numeric_cols].fillna(0)

    return customer_features

# ============================================================
# CUSTOMER FEATURE ENGINEERING
# Structured in clear behavioural cohorts
# ============================================================

# Build customer datasets consistently
customer_features = build_customer_features(df_train)
customer_features_test = build_customer_features(df_test)

print("Train customer features:", customer_features.shape)
print("Test customer features:", customer_features_test.shape)

# ------------------------------------------------------------
# TARGET MERGE — Add future revenue to customer feature table
# ------------------------------------------------------------

customer_target = (
    df_train[['cust_id','revenue_2018_2019']]
    .drop_duplicates()
)

customer_features = customer_features.merge(
    customer_target,
    on='cust_id',
    how='left'
)


# %%
print(customer_features.head())
# %%
# ============================================================
# CUSTOMER LEVEL EDA
# Understanding feature distributions and target behaviour
# ============================================================

print("Customer feature table shape:", customer_features.shape)

# ------------------------------------------------------------
# NUMERIC SUMMARY TABLES (easier interpretation than plots)
# ------------------------------------------------------------

summary_features = [
    'revenue_2018_2019',
    'total_spent',
    'n_orders',
    'recency_days',
    'orders_per_day',
    'avg_order_value',
    'revenue_last_90d'
]

numeric_summary = customer_features[summary_features].describe().T
numeric_summary['skew'] = customer_features[summary_features].skew()
numeric_summary['missing'] = customer_features[summary_features].isna().sum()

print("\nNumeric summary table:")
print(numeric_summary)

# Zero counts (important for CLV interpretation)
zero_stats = (customer_features[summary_features] == 0).mean()
print("\nFraction of zeros per feature:")
print(zero_stats)

# ------------------------------------------------------------
# Create folder for plots (so they can be uploaded easily)
# ------------------------------------------------------------

plot_dir = "eda_plots"
os.makedirs(plot_dir, exist_ok=True)

print(f"Plots will be saved in: {plot_dir}")

# ------------------------------------------------------------
# Basic overview
# ------------------------------------------------------------

print("\nTarget summary:")
print(customer_features['revenue_2018_2019'].describe())

print("\nZero revenue customers:")
print((customer_features['revenue_2018_2019'] == 0).mean())

# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(customer_features['revenue_2018_2019'], bins=50)
plt.title("Future revenue distribution")
plt.savefig(f"{plot_dir}/target_distribution.png", bbox_inches='tight')
plt.show()

# Log distribution (important because CLV is skewed)

plt.figure(figsize=(8,5))
sns.histplot(np.log1p(customer_features['revenue_2018_2019']), bins=50)
plt.title("Log future revenue distribution")
plt.savefig(f"{plot_dir}/target_log_distribution.png", bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# Key feature distributions
# ------------------------------------------------------------

key_features = [
    'total_spent',
    'n_orders',
    'recency_days',
    'orders_per_day',
    'avg_order_value',
    'revenue_last_90d'
]

for col in key_features:

    plt.figure(figsize=(6,4))
    sns.histplot(customer_features[col], bins=50)
    plt.title(col)
    plt.savefig(f"{plot_dir}/{col}_distribution.png", bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------
# Correlation with target
# ------------------------------------------------------------

numeric_features = customer_features.select_dtypes(include=[np.number])

target_corr = numeric_features.corr()['revenue_2018_2019'] \
    .sort_values(ascending=False)

print("\nTop correlations with target:")
print(target_corr.head(15))

print("\nLowest correlations:")
print(target_corr.tail(15))

# ------------------------------------------------------------
# Correlation heatmap (top features only)
# ------------------------------------------------------------

top_features = target_corr.abs().sort_values(ascending=False).head(15).index

plt.figure(figsize=(10,8))
sns.heatmap(
    customer_features[top_features].corr(),
    cmap='coolwarm',
    center=0
)
plt.title("Top feature correlations")
plt.show()

# ------------------------------------------------------------
# Outlier inspection
# ------------------------------------------------------------

outlier_cols = [
    'total_spent',
    'avg_order_value',
    'revenue_last_90d',
    'n_orders'
]

for col in outlier_cols:

    plt.figure(figsize=(6,4))
    sns.boxplot(x=customer_features[col])
    plt.title(col)
    plt.savefig(f"{plot_dir}/{col}_boxplot.png", bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------
# Behaviour vs target plots
# ------------------------------------------------------------

important_features = [
    'recency_days',
    'total_spent',
    'orders_per_day',
    'return_rate'
]

for col in important_features:

    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x=customer_features[col],
        y=customer_features['revenue_2018_2019'],
        alpha=0.3
    )
    plt.title(f"{col} vs future revenue")
    plt.savefig(f"{plot_dir}/{col}_vs_target.png", bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------
# Missing values check
# ------------------------------------------------------------

print("\nMissing values in customer features:")
print(customer_features.isna().sum().sort_values(ascending=False).head(20))

# ------------------------------------------------------------
# Feature variance check
# ------------------------------------------------------------

low_variance = numeric_features.var().sort_values()


print("\nLowest variance features:")
print(low_variance.head(10))

print("\nHighest variance features:")
print(low_variance.tail(10))

# %%
# ============================================================
# FINAL PRE‑MODELLING PREPARATION
# Remove irrelevant columns and encode categorical features
# ============================================================

# ------------------------------------------------------------
# Remove columns not suitable for modelling
# ------------------------------------------------------------

cols_to_drop = [
    'first_order_date',   # raw dates not suitable for models
    'last_order_date'
]

customer_model_df = customer_features.set_index('cust_id').drop(columns=cols_to_drop).copy()

print("Set cust_id as index and removed raw date columns")

# ------------------------------------------------------------
# Encode favorite brand (keep original for flexibility)
# ------------------------------------------------------------

customer_model_df['favorite_brand_encoded'] = (
    customer_model_df['favorite_brand']
    .astype('category')
    .cat.codes
)

print("Favorite brand encoded")

# Optional: also create frequency encoding (often stronger than label encoding)
brand_freq = customer_model_df['favorite_brand'].value_counts(normalize=True)

customer_model_df['favorite_brand_freq'] = (
    customer_model_df['favorite_brand'].map(brand_freq)
)
customer_model_df = customer_model_df.drop(columns=['favorite_brand'])

# Fill any missing values created by encoding
customer_model_df = customer_model_df.fillna(0)

print("Final modelling dataset shape:", customer_model_df.shape)
print(customer_model_df.head())

# ------------------------------------------------------------
# Apply same preprocessing to competition test set
# ------------------------------------------------------------

customer_model_df_test = customer_features_test.set_index('cust_id').drop(columns=cols_to_drop).copy()

# Apply same brand encoding
customer_model_df_test['favorite_brand_encoded'] = (
    customer_model_df_test['favorite_brand']
    .astype('category')
    .cat.codes
)

# Apply frequency encoding using TRAIN mapping (important: no refit)
customer_model_df_test['favorite_brand_freq'] = (
    customer_model_df_test['favorite_brand'].map(brand_freq)
)
customer_model_df_test = customer_model_df_test.drop(columns=['favorite_brand'])

# Fill any missing values
customer_model_df_test = customer_model_df_test.fillna(0)

print("Final test modelling dataset shape:", customer_model_df_test.shape)
print(customer_model_df_test.head())

# %%
# ============================================================
# TRAIN / VALIDATION / LOCAL TEST SPLIT (from competition train)
# ============================================================

from sklearn.model_selection import train_test_split

# Separate features and target
X = customer_model_df.drop(columns=['revenue_2018_2019'])
y = customer_model_df['revenue_2018_2019']

# First split: train vs temp (80 / 20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Second split: validation vs test (10 / 10)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Local test shape:", X_test.shape)

# %%
# ============================================================
# FEATURE DOCUMENTATION (customer modelling dataset)
# ============================================================

feature_descriptions = {
    'n_orders': 'Number of unique orders placed by customer',
    'n_products': 'Number of unique products purchased',
    'n_brands': 'Number of unique brands purchased',
    'n_categories': 'Number of unique product categories purchased',
    'n_colors': 'Number of unique product colours purchased',
    'total_spent': 'Total historical revenue from customer',
    'avg_order_value': 'Average revenue per order',
    'max_order_value': 'Maximum order value observed',
    'revenue_std': 'Standard deviation of order revenue',
    'revenue_median': 'Median order revenue',
    'avg_discount': 'Average discount received',
    'max_discount': 'Maximum discount received',
    'discount_std': 'Variation in discounts received',
    'return_rate': 'Fraction of orders returned',
    'n_returns': 'Total number of returned items',
    'returned_flag': 'Binary indicator if customer ever returned',
    'avg_days_to_pack': 'Average logistics processing time',
    'customer_lifetime_days': 'Days between first and last purchase',
    'recency_days': 'Days since last purchase',
    'orders_per_day': 'Stabilised purchase frequency',
    'favorite_brand_orders': 'Orders from favourite brand',
    'favorite_brand_share': 'Share of purchases from favourite brand',
    'last_order_value': 'Revenue of most recent order',
    'first_order_value': 'Revenue of first order',
    'avg_days_between_orders': 'Average spacing between purchases',
    'total_discount': 'Total discounts received',
    'discount_ratio': 'Discount relative to total spend',
    'outlet_rate': 'Fraction of purchases from outlet items',
    'product_diversity_ratio': 'Products per order ratio',
    'brand_diversity_ratio': 'Brands per order ratio',
    'returns_per_order': 'Returns relative to orders',
    'orders_last_90d': 'Orders in last 90 days',
    'revenue_last_90d': 'Revenue in last 90 days',
    'recent_active_flag': 'Customer active in last 90 days',
    'log_total_spent': 'Log transformed total spend',
    'log_revenue_last_90d': 'Log transformed recent revenue',
    'log_avg_order_value': 'Log transformed average order value',
    'has_recent_orders': 'Binary indicator of recent orders',
    'has_returns': 'Binary indicator of returns',
    'spending_trend': 'Difference between last and first order value',
    'recent_revenue_ratio': 'Recent revenue relative to historical spend',
    'favorite_brand_encoded': 'Label encoded favourite brand',
    'favorite_brand_freq': 'Frequency encoding of favourite brand',
    'revenue_2018_2019': 'Target: future customer revenue'
}

print("\nFeature documentation:")
for k,v in feature_descriptions.items():
    if k in customer_model_df.columns:
        print(f"{k:<30} : {v}")

def get_customer_model_data():

    return (
        X_train.copy(),
        X_val.copy(),
        X_test.copy(),
        y_train.copy(),
        y_val.copy(),
        y_test.copy(),
        customer_model_df_test.copy()
    )