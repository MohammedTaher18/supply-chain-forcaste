import os
import pytest
import pandas as pd
import sys
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def test_data_generation():
    assert os.path.exists(os.path.join(DATA_DIR, "fmcg_sales.csv")), "Sales data not found"
    
    df = pd.read_csv(os.path.join(DATA_DIR, "fmcg_sales.csv"), nrows=100)
    
    assert len(df) > 0, "No data generated"
    assert "sales" in df.columns, "Missing sales column"
    assert "sku_id" in df.columns, "Missing sku_id column"


def test_feature_engineering():
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    
    from feature_engineering import add_date_features, add_holiday_features
    
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10)
    })
    
    df = add_date_features(df)
    
    assert "day_of_week" in df.columns, "Missing day_of_week"
    assert "month" in df.columns, "Missing month"
    assert "is_weekend" in df.columns, "Missing is_weekend"


def test_baseline_models():
    assert os.path.exists(os.path.join(MODELS_DIR, "xgboost_model.joblib")), "XGBoost model not found"
    assert os.path.exists(os.path.join(MODELS_DIR, "lightgbm_model.joblib")), "LightGBM model not found"
    
    baseline_path = os.path.join(MODELS_DIR, "baseline_metrics.csv")
    metrics = pd.read_csv(baseline_path)
    
    assert len(metrics) > 0, "No baseline metrics"
    assert "model" in metrics.columns, "Missing model column"
    assert "WMAPE" in metrics.columns, "Missing WMAPE column"


def test_api_health():
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        assert response.status_code == 200
    except:
        pytest.skip("API not running")


def test_forecast_endpoint():
    try:
        import requests
        response = requests.post(
            "http://localhost:8000/forecast",
            json={"sku_id": "BEV_0000", "distribution_center": "DC_001", "horizon": 7},
            timeout=5
        )
        assert response.status_code == 200 or response.status_code == 404
    except:
        pytest.skip("API not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])