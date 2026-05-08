import os
import sys
__all__ = []

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from feature_config import ALL_NORMALIZE_FEATURES, get_available_features

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    sales_path = os.path.join(DATA_DIR, "fmcg_sales.csv")
    products_path = os.path.join(DATA_DIR, "product_metadata.csv")
    
    df = pd.read_csv(sales_path, parse_dates=["date"])
    products = pd.read_csv(products_path)
    
    print(f"Loaded sales data: {len(df):,} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique SKUs: {df['sku_id'].nunique()}")
    
    return df, products


def handle_missing_values(df):
    print("Handling missing values...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in ["rolling_mean_7", "rolling_mean_14", "rolling_mean_30"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    df["sales"] = df["sales"].fillna(0)
    
    for col in ["lag_7", "lag_14", "lag_30"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)
    
    return df


def detect_outliers(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return lower, upper


def cap_outliers(df, columns=["sales", "price", "stock_level"]):
    print("Capping outliers using IQR...")
    for col in columns:
        if col in df.columns:
            lower, upper = detect_outliers(df, col)
            capped = df[col].clip(lower, upper)
            df[col] = capped
    return df


def time_split(df, train_ratio=0.70, val_ratio=0.15):
    df = df.sort_values("date")
    dates = df["date"].unique()
    dates = np.sort(dates)
    
    n = len(dates)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])
    
    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"].isin(val_dates)]
    test_df = df[df["date"].isin(test_dates)]
    
    print(f"Train: {len(train_df):,} ({train_ratio*100:.0f}%)")
    print(f"Val: {len(val_df):,} ({val_ratio*100:.0f}%)")
    print(f"Test: {len(test_df):,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_df, val_df, test_df


def create_time_idx(df):
    min_date = df["date"].min()
    df["time_idx"] = (df["date"] - min_date).dt.days
    return df


def normalize_features(train_df, val_df, test_df, feature_cols):
    print("Normalizing features using MinMaxScaler...")
    
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    return train_scaled, val_scaled, test_scaled, scaler


def prepare_tft_datasets(train_df, val_df, test_df):
    print("Preparing data for TFT...")
    
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df = df.sort_values(["sku_id", "distribution_center", "date"]).reset_index(drop=True)
    
    static_categoricals = ["sku_id", "distribution_center", "category"]
    
    known_reals = ["price", "promotion_discount", "is_holiday", "day_of_week", "week_of_year", 
                 "competitor_price", "month", "is_weekend"]
    
    unknown_reals = ["sales", "stock_level", "web_search_trend", "lag_7", "lag_14", "lag_30",
                   "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"]
    
    all_features = static_categoricals + known_reals + unknown_reals
    
    return df, static_categoricals, known_reals, unknown_reals, all_features


def run_pipeline():
    print("=" * 50)
    print("Running Data Pipeline")
    print("=" * 50)
    
    df, products = load_data()
    
    df = handle_missing_values(df)
    
    df = cap_outliers(df)
    
    df = create_time_idx(df)
    
    train_df, val_df, test_df = time_split(df)
    
    numeric_features = get_available_features(df, ALL_NORMALIZE_FEATURES)
    
    train_df, val_df, test_df, scaler = normalize_features(train_df, val_df, test_df, numeric_features)
    
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    print(f"\nSaved processed datasets to {PROCESSED_DIR}")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()