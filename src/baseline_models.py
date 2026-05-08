import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from feature_config import ALL_MODEL_FEATURES, get_available_features

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_df, val_df, test_df


def calculate_wmape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred).sum()
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wmape = calculate_wmape(y_true, y_pred)
    print(f"{model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, WMAPE={wmape:.2f}%")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "WMAPE": wmape}


# Will be dynamically set based on train_df
feature_cols = []


class BaselineModels:
    def __init__(self):
        self.models = {}
        self.metrics = []
        self.arima_models = {}
    
    def prepare_sklearn_data(self, train_df, test_df):
        global feature_cols
        if not feature_cols:
            feature_cols = get_available_features(train_df, ALL_MODEL_FEATURES)
            
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["sales"].values
        
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df["sales"].values
        
        return X_train, y_train, X_test, y_test
    
    def naive_baseline(self, test_df):
        print("\n--- Naive Baseline ---")
        predictions = test_df["lag_7"].fillna(test_df["sales"].mean()).values
        actuals = test_df["sales"].values
        metrics = calculate_metrics(actuals, predictions, "Naive")
        self.metrics.append(metrics)
        return predictions
    
    def train_xgboost(self, train_df, test_df):
        print("\n--- Training XGBoost ---")
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost not installed. Skipping...")
            return None
        
        X_train, y_train, X_test, y_test = self.prepare_sklearn_data(train_df, test_df)
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        print("Training XGBoost...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0)
        
        metrics = calculate_metrics(y_test, predictions, "XGBoost")
        self.metrics.append(metrics)
        
        self.models["xgboost"] = model
        dump(model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
        
        return predictions
    
    def train_lightgbm(self, train_df, test_df):
        print("\n--- Training LightGBM ---")
        try:
            import lightgbm as lgb
        except ImportError:
            print("LightGBM not installed. Skipping...")
            return None
        
        X_train, y_train, X_test, y_test = self.prepare_sklearn_data(train_df, test_df)
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        print("Training LightGBM...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0)
        
        metrics = calculate_metrics(y_test, predictions, "LightGBM")
        self.metrics.append(metrics)
        
        self.models["lightgbm"] = model
        dump(model, os.path.join(MODELS_DIR, "lightgbm_model.joblib"))
        
        return predictions
    
    def train_arima(self, train_df, test_df):
        print("\n--- Training ARIMA (per-SKU) ---")
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            print("Statsmodels not installed. Skipping ARIMA...")
            return None
        
        results_df = test_df.copy()
        results_df["arima_pred"] = 0.0
        
        y_test = test_df["sales"].values
        predictions = []
        
        sku_groups = train_df.groupby("sku_id")
        
        sample_skus = list(sku_groups.groups.keys())[:10]
        
        print(f"Training ARIMA for {len(sample_skus)} SKUs (sample)...")
        
        for sku in sample_skus:
            try:
                sku_train = train_df[train_df["sku_id"] == sku].sort_values("date")
                series = sku_train["sales"].values[-90:]
                
                if len(series) < 20:
                    predictions.extend([np.mean(series)] * len(series))
                    continue
                
                model = ARIMA(series, order=(5, 1, 0))
                fitted = model.fit()
                
                forecast = fitted.forecast(steps=min(30, len(series)))
                predictions.extend(forecast.tolist())
                
            except Exception as e:
                predictions.extend([np.mean(series)] * 30)
        
        missing = len(y_test) - len(predictions)
        if missing > 0:
            predictions.extend([np.mean(y_test)] * missing)
        
        predictions = np.array(predictions[:len(y_test)])
        predictions = np.maximum(predictions, 0)
        
        metrics = calculate_metrics(y_test, predictions, "ARIMA")
        self.metrics.append(metrics)
        
        return predictions
    
    def run_all(self, train_df, test_df):
        print("=" * 50)
        print("Training Baseline Models")
        print("=" * 50)
        
        self.naive_baseline(test_df)
        self.train_xgboost(train_df, test_df)
        self.train_lightgbm(train_df, test_df)
        self.train_arima(train_df, test_df)
        
        print("\n" + "=" * 50)
        print("Baseline Model Results")
        print("=" * 50)
        
        metrics_df = pd.DataFrame(self.metrics)
        print(metrics_df.to_string(index=False))
        
        metrics_df.to_csv(os.path.join(MODELS_DIR, "baseline_metrics.csv"), index=False)
        
        return self.metrics


if __name__ == "__main__":
    train_df, val_df, test_df = load_data()
    
    baseline = BaselineModels()
    baseline.run_all(train_df, test_df)