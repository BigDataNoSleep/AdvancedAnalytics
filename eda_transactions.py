# %% [markdown]
# Exploratory Data Analysis: Transactions 2016-2017
# This script performs EDA on the transaction dataset.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Load Data
# We are loading the transaction data from the CSV file.
# Note: The file does not appear to have headers in the first line.

# %%
transactions_path = "/Users/vincecoppens/Documents/Courses/Big Data/AdvancedAnalytics/data/transactions_2016_2017.csv"
customer_path = "/Users/vincecoppens/Documents/Courses/Big Data/AdvancedAnalytics/data/customer_clv_train.csv"

# Load the data
df_transactions = pd.read_csv(transactions_path, parse_dates=['order_date', 'pack_date'])
df_customer = pd.read_csv(customer_path)


# %%
# Merge the datasets on 'cust_id'
df = pd.merge(df_transactions, df_customer, on='cust_id', how='left')

# Display the first few rows and info
print(f"Merged Dataset shape: {df.shape}")
df
# %%

# df_unlabeled = df[df['revenue_2018_2019'].isna()].copy()
# df_unlabeled.to_csv('test_transactions.csv', index=False)

# df_labeled = df[df['revenue_2018_2019'].notna()].copy()
# df = df_labeled


# %%

# from sklearn.model_selection import train_test_split

# # Split the merged dataframe into training (80%) and testing (20%) sets
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# print(f"Training set: {df_train.shape}")
# print(f"Testing set: {df_test.shape}")

# %% [markdown]
# ## Data Types
# Let's check the data types and missing values.

# %%
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
    df[col] = df[col].astype(bool)

# %%
# Number of unique values per feature
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
df['days_to_pack'] = (df['pack_date'] - df['order_date']).dt.days
df
# %%
df['returned'] = df['returned_to_shop_id'].notna()
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
# Summary statistics for numerical columns.

# %%
df.describe()

# %% [markdown]
# ## Visual Analytics
# ### Distribution (Histogram)
# Let's look at the distribution of `sale_revenue`.

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df['sale_revenue'], bins=50, kde=True)
plt.title('Distribution of Sale Revenue')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### Outliers (Boxplot)
# Visualizing revenue and discount applied.

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
# Relationship between `sale_revenue` and `sale_discount_applied`.

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sale_discount_applied', y='sale_revenue', alpha=0.5)
plt.title('Sale Revenue vs. Discount Applied')
plt.show()

# %% [markdown]
# ### Correlation Plot
# Heatmap of correlations between numerical features.

# %%
# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
