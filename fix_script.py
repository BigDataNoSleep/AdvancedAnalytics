import re

with open('eda_transactions.py', 'r') as f:
    text = f.read()

# 1. Update the `build_customer_features` function end.
find_function_end = """    # Recent behaviour
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

    numeric_cols = customer_features.select_dtypes(include=[np.number]).columns
    customer_features[numeric_cols] = customer_features[numeric_cols].fillna(0)

    return customer_features"""

replacement_function_end = """    # Advanced behavioural signals
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

    import numpy as np
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

    return customer_features"""

text = text.replace(find_function_end, replacement_function_end)

# 2. Extract and remove the duplicated global logic. 
# We look for the start pattern:
start_pattern = r"# Create transaction level features needed for standalone cohort aggregations below.*?df_test\['returned'\] = df_test\['returned_to_shop_id'\] != 'none'"
text = re.sub(start_pattern, "", text, flags=re.DOTALL)

# Delete all the standalone global cohort combinations between test size print, down to target merge
delete_pattern = r"(print\(\"Test customer features:\", customer_features_test\.shape\)).*?(# ------------------------------------------------------------\n# TARGET MERGE — Add future revenue to customer feature table)"
text = re.sub(delete_pattern, r"\1\n\n\2", text, flags=re.DOTALL)

with open('eda_transactions.py', 'w') as f:
    f.write(text)

print("Modification complete.")
