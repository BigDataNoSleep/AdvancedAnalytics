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
# Till here i (Vince) worked on monday 9 march. I did some basic cleaning and EDA.
#  I will continue on tuesday 10 march with more feature engineering and visualizations.

# %%
# ============================================================
# Feature engineering
# ============================================================
# Calculate packing delay (days between order and packing)
df['days_to_pack'] = (df['pack_date'] - df['order_date']).dt.days
df

# ------------------------------------------------------------
# Data consistency checks
# ------------------------------------------------------------

# Check for impossible dates (pack_date before order_date)
invalid_pack_dates = (df['pack_date'] < df['order_date']).sum()
print("Transactions with pack_date earlier than order_date:", invalid_pack_dates)

# Check distribution of discount values
print("\nDiscount statistics:")
print(df['sale_discount_applied'].describe())

# Check for extremely large revenue values
print("\nRevenue upper quantiles:")
print(df['sale_revenue'].quantile([0.99, 0.999]))
# %%
# Binary indicator showing whether the product was returned
#
# 'returned_to_shop_id' was previously filled with "none" for non-returned items
# Therefore a return occurred only when the value is different from "none"
df['returned'] = df['returned_to_shop_id'] != "none"
df

# %% [markdown]
# ## Brand Frequency
brand_counts = df['prod_brand'].value_counts()
top_brands_counts = brand_counts[brand_counts > 1500]
top_brands_names = top_brands_counts.index

print(f"Number of unique brands: {df['prod_brand'].nunique()}")
print(f"Total items in table: {len(df)}")
print(f"Items from brands with >1500 occurrences: {top_brands_counts.sum()} ({(top_brands_counts.sum() / len(df)) * 100:.2f}% of total)")

plt.figure(figsize=(12, 6))
sns.countplot(data=df[df['prod_brand'].isin(top_brands_names)], x='prod_brand', order=top_brands_names)
plt.title('Frequency of Brands (appearing > 1500 times)')
plt.xticks(rotation=45)
plt.show()

# # %%
# # Relative market share per brand (based on transaction frequency)
# df['brand_market_share'] = df['prod_brand'].map(df['prod_brand'].value_counts(normalize=True))

# # Average revenue per brand
# df['brand_avg_revenue'] = df.groupby('prod_brand')['revenue_2018_2019'].transform('mean')
# df


 
# encoden: prod season, prod_color, prod_type1, prod_heel
# prod_clasp: to be encoded but pretty fucked up
# prod_print changing to print? Yes/No

#Prod_type 3, 4 and 5 + prod_material + prod comfort wear + prod comfort sole???? (To be removed?)

# # %%
# # Drop 'prod_type_4' column
# df.drop(columns=['prod_type_4'], inplace=True)
# prod_id will be removed later
# prod_title to be removed


# %%

# %% [markdown]
# ## Basic Stats
# Summary statistics for numerical columns

# %%
# Summary statistics for numerical columns
df.describe()

# %% [markdown]
# ## Visual Analytics
# ### Distribution (Histogram)
# Visualize distribution of transaction revenue

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df['sale_revenue'], bins=50, kde=True)
plt.title('Distribution of Sale Revenue')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### Outliers (Boxplot)
# Boxplots help detect extreme values in revenue and discount variables

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['sale_revenue'])
plt.title('Boxplot of Sale Revenue')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['sale_discount_applied'])
plt.title('Boxplot of Discount Applied')
plt.show()

# %% [markdown]
# ### Relationships (Scatter Plot)
# Scatter plot to explore relationship between discounts and revenue

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sale_discount_applied', y='sale_revenue', alpha=0.5)
plt.title('Sale Revenue vs. Discount Applied')
plt.show()

# %% [markdown]
# ### Correlation Plot
# Correlation heatmap for numerical variables
# Helps identify strongly related predictors

# %%
# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
