"""
Shared feature configuration for the pipeline.
All model scripts import feature lists from here.
Features are validated against the actual DataFrame columns at runtime.
"""

# All possible numeric features used for model training
ALL_MODEL_FEATURES = [
    "sales", "price", "promotion_flag", "promotion_discount", "is_holiday",
    "day_of_week", "week_of_year", "month", "is_weekend", "competitor_price",
    "stock_level", "web_search_trend", "macroeconomic_index",
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "price_x_promotion"
]

# Features to normalize with MinMaxScaler
ALL_NORMALIZE_FEATURES = [
    "sales", "price", "promotion_discount", "competitor_price",
    "stock_level", "web_search_trend", "macroeconomic_index",
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"
]

# Features for the LSTM model (lightweight subset)
LSTM_FEATURES = ["sales", "price", "rolling_mean_7", "lag_7"]

# Categorical columns used for grouping
CATEGORICAL_COLS = ["sku_id", "distribution_center", "category"]


def get_available_features(df, feature_list):
    """Return only features that exist in the DataFrame."""
    return [f for f in feature_list if f in df.columns]
