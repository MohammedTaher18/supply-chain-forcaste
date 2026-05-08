__all__ = []

import pandas as pd
import numpy as np


def add_lag_features(df, lags=[7, 14, 30]):
    df = df.sort_values(["sku_id", "distribution_center", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["sku_id", "distribution_center"])["sales"].shift(lag)
    return df


def add_rolling_features(df, windows=[7, 14, 30]):
    df = df.sort_values(["sku_id", "distribution_center", "date"])
    for window in windows:
        df[f"rolling_mean_{window}"] = df.groupby(["sku_id", "distribution_center"])["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = df.groupby(["sku_id", "distribution_center"])["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f"rolling_min_{window}"] = df.groupby(["sku_id", "distribution_center"])["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).min()
        )
        df[f"rolling_max_{window}"] = df.groupby(["sku_id", "distribution_center"])["sales"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
    return df


def add_date_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["year"] = df["date"].dt.year
    return df


def add_holiday_features(df):
    indian_holidays = [
        "01-26", "08-15", "10-02", "10-12", "10-15", "11-01", "11-04", "11-05", "12-25"
    ]
    global_holidays = [
        "01-01", "02-14", "07-04", "12-24", "12-31"
    ]
    
    df["is_indian_holiday"] = 0
    df["is_global_holiday"] = 0
    
    for holiday in indian_holidays:
        mask = df["date"].dt.strftime("%m-%d") == holiday
        df.loc[mask, "is_indian_holiday"] = 1
    
    for holiday in global_holidays:
        mask = df["date"].dt.strftime("%m-%d") == holiday
        df.loc[mask, "is_global_holiday"] = 1
    
    return df


def add_promotion_interaction_features(df):
    if "price" in df.columns and "promotion_flag" in df.columns:
        df["price_x_promotion"] = df["price"] * df["promotion_flag"]
    if "price" in df.columns and "promotion_discount" in df.columns:
        df["discount_amount"] = df["price"] * df["promotion_discount"] / 100
    return df


def engineer_all_features(df):
    df = add_date_features(df)
    df = add_holiday_features(df)
    df = add_promotion_interaction_features(df)
    return df


if __name__ == "__main__":
    import os
    import sys
    
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    file_path = os.path.join(DATA_DIR, "fmcg_sales.csv")
    
    print("Loading data for feature engineering...")
    df = pd.read_csv(file_path, parse_dates=["date"])
    
    print("Adding basic features...")
    df = engineer_all_features(df)
    
    if "--upload" in sys.argv or "sales" in df.columns:
        print("Generating lag and rolling features (this may take a moment)...")
        # Ensure it's sorted properly
        df = df.sort_values(["sku_id", "distribution_center", "date"]).reset_index(drop=True)
        
        # Add lags and rolling features
        df = add_lag_features(df)
        df = add_rolling_features(df)
        
        # Drop rows with NaN from lagging to keep data clean
        df = df.dropna(subset=["lag_30"]).reset_index(drop=True)
        
        print(f"Engineered features. Saving {len(df)} rows back to {file_path}")
        df.to_csv(file_path, index=False)
    else:
        print(f"Engineered basic features. New columns: {list(df.columns)}")