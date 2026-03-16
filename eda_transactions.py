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
# Transaction-level feature preparation (needed for aggregation)
# ============================================================

df_train['days_to_pack'] = (df_train['pack_date'] - df_train['order_date']).dt.days

df_train['returned'] = df_train['returned_to_shop_id'] != 'none'

# %%
# ============================================================
# Customer feature engineering (rich behavioural aggregation)
# We summarise transaction behaviour into customer-level signals
# ============================================================

customer_features = df_train.groupby('cust_id').agg(
    # =========================
    # Activity features
    # =========================
    n_orders = ('order_date','nunique'),                 # how often customer buys
    n_products = ('prod_id','nunique'),                  # product diversity
    n_brands = ('prod_brand','nunique'),                 # brand diversity
    n_categories = ('prod_type_1','nunique'),            # men/women/kids diversity
    n_colors = ('prod_color','nunique'),                 # style diversity

    # =========================
    # Financial behaviour
    # =========================
    total_spent = ('sale_revenue','sum'),                # total historical value
    avg_order_value = ('sale_revenue','mean'),           # typical spend
    max_order_value = ('sale_revenue','max'),            # premium behaviour
    revenue_std = ('sale_revenue','std'),                # spending variability
    revenue_median = ('sale_revenue','median'),          # robust central spend

    # =========================
    # Discount behaviour
    # =========================
    avg_discount = ('sale_discount_applied','mean'),     # discount sensitivity
    max_discount = ('sale_discount_applied','min'),      # strongest discount taken
    discount_std = ('sale_discount_applied','std'),      # discount variability

    # =========================
    # Return behaviour
    # =========================
    return_rate = ('returned','mean'),                   # fraction of returns
    n_returns = ('returned','sum'),                      # total returns
    returned_flag = ('returned','max'),                  # ever returned (0/1)

    # =========================
    # Logistics behaviour
    # =========================
    avg_days_to_pack = ('days_to_pack','mean'),          # avg processing delay

    # =========================
    # Time behaviour (very important for CLV)
    # =========================
    first_order_date = ('order_date','min'),
    last_order_date = ('order_date','max')
)

customer_features = customer_features.reset_index()

# ============================================================
# Derived time features
# ============================================================

# Customer lifetime (activity window)
customer_features['customer_lifetime_days'] = (
    customer_features['last_order_date'] -
    customer_features['first_order_date']
).dt.days

# Recency (days since last purchase relative to dataset end)
max_date = df_train['order_date'].max()
customer_features['recency_days'] = (
    max_date - customer_features['last_order_date']
).dt.days

# Purchase frequency
customer_features['orders_per_day'] = (
    customer_features['n_orders'] /
    (customer_features['customer_lifetime_days'] + 1)
)

print("Customer feature table shape:", customer_features.shape)
customer_features.head()

# ============================================================
# Brand preference features (captures loyalty vs exploration)
# ============================================================

# Count how many times each customer buys each brand
brand_counts = (
    df_train.groupby(['cust_id','prod_brand'])
    .size()
    .reset_index(name='brand_orders')
)

# Find most purchased brand per customer
brand_counts_sorted = brand_counts.sort_values(
    ['cust_id','brand_orders'],
    ascending=[True, False]
)

top_brand = brand_counts_sorted.drop_duplicates('cust_id')

top_brand = top_brand.rename(columns={
    'prod_brand':'favorite_brand',
    'brand_orders':'favorite_brand_orders'
})

# Merge into customer feature table
customer_features = customer_features.merge(
    top_brand[['cust_id','favorite_brand','favorite_brand_orders']],
    on='cust_id',
    how='left'
)

# Brand loyalty: share of orders from favourite brand
customer_features['favorite_brand_share'] = (
    customer_features['favorite_brand_orders'] /
    customer_features['n_orders']
)

# %%
customer_features
# %%
