# Supply Chain Demand Forecasting — Complete Project Walkthrough

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Layer](#3-data-layer)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [ML Models](#6-ml-models)
7. [Model Explainability](#7-model-explainability)
8. [MLOps Pipeline](#8-mlops-pipeline)
9. [Drift Monitoring](#9-drift-monitoring)
10. [FastAPI Backend](#10-fastapi-backend)
11. [Streamlit Dashboard](#11-streamlit-dashboard)
12. [Testing](#12-testing)
13. [Deployment (Docker)](#13-deployment-docker)
14. [End-to-End Execution Order](#14-end-to-end-execution-order)

---

## 1. Project Overview

This is an **end-to-end ML system** for demand forecasting in FMCG (Fast-Moving Consumer Goods) retail supply chains. It predicts future product sales across multiple SKUs and distribution centers using gradient boosting and deep learning models.

**Key capabilities:**
- Synthetic data generation simulating real-world FMCG sales patterns
- Time-series feature engineering (lags, rolling stats, holidays)
- Multiple model training and comparison (Naive, XGBoost, LightGBM, ARIMA, LSTM, TFT)
- SHAP-based model explainability
- Data drift monitoring
- REST API for on-demand forecasting
- Interactive Streamlit dashboard for visualization
- Docker containerization for deployment

**Tech Stack:** Python, Pandas, Scikit-learn, LightGBM, XGBoost, PyTorch, SHAP, FastAPI, Streamlit, Docker

---

## 2. Repository Structure

```
supply_chain_forecast/
├── .env.example                  # Environment variable template
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Multi-service Docker setup
├── requirements.txt              # Python dependencies (27 packages)
├── README.md                     # Project README
│
├── data/
│   ├── generate_data.py          # Synthetic data generator
│   ├── raw/
│   │   ├── fmcg_sales.csv        # Raw sales data (~55 MB, ~365K rows)
│   │   └── product_metadata.csv  # Product catalog (50 products)
│   └── processed/
│       ├── train.csv             # Training set (~74 MB, 70% of data)
│       ├── val.csv               # Validation set (~16 MB, 15% of data)
│       └── test.csv              # Test set (~16 MB, 15% of data)
│
├── src/
│   ├── data_pipeline.py          # Data cleaning, splitting, normalization
│   ├── feature_engineering.py    # Lag, rolling, date, holiday features
│   ├── baseline_models.py        # Naive, XGBoost, LightGBM, ARIMA
│   ├── lstm_model.py             # PyTorch LSTM model
│   ├── tft_model.py              # "TFT" model (actually LightGBM)
│   ├── explainability.py         # SHAP value analysis
│   └── mlops_pipeline.py         # Model monitoring and hyperparams
│
├── api/
│   ├── main.py                   # FastAPI server with endpoints
│   └── schemas.py                # Pydantic request/response models
│
├── dashboard/
│   └── app.py                    # Streamlit multi-page dashboard
│
├── monitoring/
│   └── drift_monitor.py          # Data drift detection
│
├── tests/
│   └── test_pipeline.py          # Pytest test suite
│
├── models/                       # Saved model artifacts
│   ├── scaler.joblib             # MinMaxScaler (fitted on train)
│   ├── xgboost_model.joblib      # Trained XGBoost (~493 KB)
│   ├── lightgbm_model.joblib     # Trained LightGBM (~316 KB)
│   ├── tft_model.joblib          # Trained "TFT" LightGBM (~648 KB)
│   ├── lstm_model.pt             # Trained LSTM weights (~23 KB)
│   ├── baseline_metrics.csv      # Baseline model scores
│   ├── tft_metrics.csv           # TFT model scores
│   └── lstm_metrics.csv          # LSTM model scores
│
├── outputs/
│   ├── tft_feature_importance.png
│   ├── explainability/
│   │   ├── shap_feature_importance.png
│   │   └── shap_beeswarm.png
│   └── monitoring/
│       └── drift_report.txt
│
├── config/                       # (Generated at runtime)
│   └── best_params.json          # Best hyperparameters
│
└── notebooks/                    # (Empty — for Jupyter exploration)
```

---

## 3. Data Layer

### 3.1 Synthetic Data Generation — `data/generate_data.py`

Since this project uses **simulated data**, `generate_data.py` creates a realistic FMCG sales dataset from scratch.

**What it generates:**

| Parameter | Value |
|---|---|
| SKUs | 50 products across 10 categories |
| Distribution Centers | 10 (DC_001 to DC_010) |
| Date Range | Jan 1, 2023 → Dec 31, 2024 (731 days) |
| Total Rows | ~365,000 (50 SKUs × 10 DCs × 731 days) |

**Categories:** Beverages, Snacks, Dairy, Bakery, Personal Care, Home Care, Frozen Foods, Confectionery, Breakfast Cereal, Condiments

**How demand is simulated:**

1. **Base demand** — drawn from a lognormal distribution (`np.random.lognormal(3.5, 0.8)`) per SKU
2. **DC factor** — random multiplier (0.5–1.5) per distribution center
3. **Seasonality** — monthly multipliers:
   - December: ×1.3 (holiday season)
   - October: ×1.4 (Diwali/festive)
   - April–June: ×1.15 (summer)
4. **Weekend boost** — ×1.1 on Saturdays/Sundays
5. **Holiday boost** — ×1.25 on Indian national holidays (Republic Day, Independence Day, Gandhi Jayanti, Diwali, Christmas, etc.)
6. **Promotions** — 8% random chance; discount between 5–25%
7. **Noise** — Gaussian noise (`normalvariate(1.0, 0.15)`)

**Columns generated (per row):**

| Column | Description |
|---|---|
| `date` | Sale date |
| `sku_id` | e.g., `BEV_0000`, `SNK_0001` |
| `distribution_center` | e.g., `DC_001` |
| `category` | Product category |
| `sales` | Units sold (target variable) |
| `price` | Effective selling price |
| `promotion_flag` | 0 or 1 |
| `promotion_discount` | Discount % (0 if no promo) |
| `is_holiday` | 0 or 1 |
| `day_of_week` | 0=Mon … 6=Sun |
| `week_of_year` | ISO week number |
| `month` | 1–12 |
| `is_weekend` | 0 or 1 |
| `competitor_price` | 85–120% of effective price |
| `stock_level` | 2–5× current demand |
| `web_search_trend` | ~N(50, 10) |
| `macroeconomic_index` | Linear trend starting at 100 |

**Post-processing (also in generate_data.py):**
- Lag features: `lag_7`, `lag_14`, `lag_30` (shifted sales)
- Rolling means: `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`
- Interaction: `price_x_promotion`
- Rows with NaN (from lag computation) are dropped

**Output:** `data/raw/fmcg_sales.csv` (~55 MB) and `data/raw/product_metadata.csv`

### 3.2 Product Metadata — `data/raw/product_metadata.csv`

A simple 50-row catalog with: `sku_id`, `category`, `product_name`, `base_price`, `avg_price`

---

## 4. Data Pipeline — `src/data_pipeline.py`

This is the **central preprocessing script** that transforms raw data into model-ready train/val/test splits.

**Pipeline steps (executed via `run_pipeline()`):**

### Step 1: Load Data
- Reads `fmcg_sales.csv` and `product_metadata.csv` from `data/raw/`
- Parses dates

### Step 2: Handle Missing Values
- Rolling mean columns → filled with column median
- `sales` → filled with 0
- Lag columns → forward-filled, then 0

### Step 3: Cap Outliers (IQR method)
- For `sales`, `price`, `stock_level`:
  - Computes Q1, Q3, IQR
  - Clips values to [Q1 − 1.5×IQR, Q3 + 1.5×IQR]

### Step 4: Create Time Index
- Adds `time_idx` column = number of days since the earliest date (used by temporal models)

### Step 5: Time-Based Train/Val/Test Split
- **70% train** / **15% validation** / **15% test**
- Split by date (not random) to prevent data leakage

### Step 6: Normalize Features (MinMaxScaler)
- Fits scaler on **training data only**
- Transforms train, val, and test using the same scaler
- Features normalized: `sales`, `price`, `promotion_discount`, `competitor_price`, `stock_level`, `web_search_trend`, `macroeconomic_index`, all lags, all rolling means
- Saves scaler to `models/scaler.joblib`

### Step 7: Save Outputs
- `data/processed/train.csv`, `val.csv`, `test.csv`

**Run:** `python src/data_pipeline.py`

---

## 5. Feature Engineering — `src/feature_engineering.py`

A standalone module of reusable feature engineering functions. These are the **same features** already created in `generate_data.py`, but this module exists so you can apply them to new/external datasets.

| Function | Features Created |
|---|---|
| `add_lag_features()` | `lag_7`, `lag_14`, `lag_30` — shifted sales per SKU+DC |
| `add_rolling_features()` | `rolling_mean_7/14/30`, `rolling_std_7/14/30`, `rolling_min_7/14/30`, `rolling_max_7/14/30` |
| `add_date_features()` | `day_of_week`, `week_of_year`, `month`, `quarter`, `is_weekend`, `year` |
| `add_holiday_features()` | `is_indian_holiday`, `is_global_holiday` |
| `add_promotion_interaction_features()` | `price_x_promotion`, `discount_amount` |
| `engineer_all_features()` | Calls date + holiday + promotion functions |

**Run:** `python src/feature_engineering.py`

---

## 6. ML Models

All models read from `data/processed/` and save artifacts to `models/`.

### 6.1 Baseline Models — `src/baseline_models.py`

The `BaselineModels` class trains and evaluates 4 models:

#### Naive Baseline
- Prediction = `lag_7` value (sales from 7 days ago)
- Falls back to mean if lag is missing
- **No training** — just a reference point

#### XGBoost
- `XGBRegressor` with 100 estimators, max_depth=6, lr=0.1
- Uses 18 features (price, promotions, lags, rolling stats, etc.)
- Saves to `models/xgboost_model.joblib`

#### LightGBM
- `LGBMRegressor` with 100 estimators, max_depth=6, lr=0.1
- Same 18 features as XGBoost
- Saves to `models/lightgbm_model.joblib`

#### ARIMA
- Per-SKU `ARIMA(5,1,0)` model
- Only trains on 10 sample SKUs (for speed)
- Uses last 90 days of training data per SKU
- Forecasts 30 steps ahead

**All predictions are clipped to ≥ 0** (no negative sales).

**Metrics computed:** MAE, RMSE, WMAPE (Weighted Mean Absolute Percentage Error)

**Output:** `models/baseline_metrics.csv`

**Run:** `python src/baseline_models.py`

### 6.2 LSTM Model — `src/lstm_model.py`

A PyTorch sequence-to-one LSTM for time series.

**Architecture:**
```
Input (seq_len=14, features=4) → LSTM(hidden=32, layers=1) → Linear(32→1) → Output
```

**How it works:**
1. Uses 4 features: `sales`, `price`, `rolling_mean_7`, `lag_7`
2. Creates sliding windows of length 14 (2 weeks)
3. Predicts the next day's sales from each window
4. Trains for 5 epochs with Adam optimizer (lr=0.01), MSE loss, batch_size=256
5. Runs on CPU

**Saves:** `models/lstm_model.pt` (state dict), `models/lstm_metrics.csv`

**Run:** `python src/lstm_model.py`

### 6.3 TFT Model — `src/tft_model.py`

> **Important:** Despite the name "Temporal Fusion Transformer", this is actually a **LightGBM** gradient boosting model. The naming is for project labeling purposes.

**Configuration:**
- `LGBMRegressor` with: 200 estimators, 31 leaves, lr=0.05, feature_fraction=0.9, bagging=0.8
- Early stopping after 50 rounds without improvement
- Uses all 19 features

**What it does:**
1. Trains LightGBM on the full feature set
2. Predicts on test set (clipped to ≥ 0)
3. Computes MAE, RMSE, WMAPE
4. Saves model to `models/tft_model.joblib`
5. Generates and saves a feature importance bar chart to `outputs/tft_feature_importance.png`

**Run:** `python src/tft_model.py`

### 6.4 Model Performance Summary

| Model | WMAPE | MAE | RMSE |
|---|---|---|---|
| Naive | 90.86% | 0.345 | 0.429 |
| ARIMA | 61.10% | 0.232 | 0.284 |
| LSTM | 58.81% | 0.223 | 0.288 |
| LightGBM | 9.02% | 0.034 | 0.052 |
| XGBoost | 8.93% | 0.034 | 0.051 |
| **TFT** | **0.18%** | **0.001** | **0.001** |

---

## 7. Model Explainability — `src/explainability.py`

Uses **SHAP (SHapley Additive exPlanations)** to interpret the TFT model.

**How it works:**
1. Loads `tft_model.joblib` (the LightGBM model)
2. Samples up to 1,000 rows from `test.csv`
3. Creates a `TreeExplainer` (optimized for tree-based models)
4. Computes SHAP values for all 19 features

**Outputs (saved to `outputs/explainability/`):**
- `shap_feature_importance.png` — bar chart of mean |SHAP| per feature
- `shap_beeswarm.png` — beeswarm plot showing feature impact distribution

**Run:** `python src/explainability.py`

---

## 8. MLOps Pipeline — `src/mlops_pipeline.py`

Handles model monitoring, retraining triggers, and hyperparameter management.

**What it does:**
1. Checks if **MLflow** is available (optional dependency)
2. Loads test data and evaluates latest model metrics
3. Compares WMAPE against a **retrain threshold** (15%)
   - If WMAPE > 15% → prints retraining alert
   - Otherwise → model is considered healthy
4. Saves best hyperparameters to `config/best_params.json`:
   ```json
   {
     "num_leaves": 31,
     "learning_rate": 0.05,
     "feature_fraction": 0.9,
     "bagging_fraction": 0.8,
     "n_estimators": 200,
     "max_depth": -1
   }
   ```

**Note:** Currently uses hardcoded metrics (WMAPE=0.18%). In production, this would read from actual model evaluation.

**Run:** `python src/mlops_pipeline.py`

---

## 9. Drift Monitoring — `monitoring/drift_monitor.py`

Detects **data drift** between training data and recent production data.

**Method:** Z-score based drift detection
- For each feature: `drift_score = |mean_recent - mean_train| / std_train`
- Monitors: `sales`, `price`, `stock_level`
- Uses the last 30 rows of test data as "recent" data
- **Threshold:** drift_score > 0.2 → Alert

**Output:** `outputs/monitoring/drift_report.txt`

**Run:** `python monitoring/drift_monitor.py`

---

## 10. FastAPI Backend — `api/main.py`

A REST API serving model predictions and metrics.

**Server:** Uvicorn on `0.0.0.0:8000`

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "healthy", "timestamp": "..."}` |
| `POST` | `/forecast` | Generates demand forecast for a SKU+DC |
| `GET` | `/model/metrics` | Returns latest model metrics + baseline comparison |
| `GET` | `/model/feature_importance` | Returns top 10 feature importances |
| `POST` | `/retrain` | Placeholder — returns success message |

### `/forecast` Endpoint (Detail)

**Request body:**
```json
{
  "sku_id": "BEV_0000",
  "distribution_center": "DC_001",
  "horizon": 7
}
```

**How it works:**
1. Loads `tft_model.joblib`
2. Filters test.csv for the requested SKU+DC (last 30 rows)
3. Runs model prediction on the 19 features
4. Clips predictions to ≥ 0
5. Computes 95% confidence interval (±1.96 × std)

**Response:**
```json
{
  "sku_id": "BEV_0000",
  "distribution_center": "DC_001",
  "horizon": 7,
  "forecasts": [12.5, 13.1, ...],
  "lower_bound": [8.2, 8.8, ...],
  "upper_bound": [16.8, 17.4, ...]
}
```

### `/model/metrics` Endpoint
- Reads `models/tft_metrics.csv` if it exists, otherwise returns hardcoded defaults
- Also reads `models/baseline_metrics.csv` for comparison data

### Schemas — `api/schemas.py`
Pydantic models: `ForecastRequest`, `ForecastResponse`, `MetricsResponse`, `FeatureImportanceResponse`, `RetrainResponse`

**Run:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`

---

## 11. Streamlit Dashboard — `dashboard/app.py`

A multi-page interactive dashboard with 5 views.

**Data source:** Reads `data/processed/test.csv` and fetches metrics from the FastAPI at `http://localhost:8000`

### Page 1: Forecast View
- Dropdowns to select SKU and Distribution Center
- Slider for forecast horizon (1–30 days)
- Plotly line chart showing:
  - Actual sales (solid blue line)
  - 7-day moving average (dashed orange line)
- Metric cards: WMAPE, MAE, RMSE

### Page 2: SKU Analytics
- Aggregates sales by SKU (sum, mean, std, avg stock)
- Bar chart of top 10 SKUs by total sales
- **Stock risk analysis:** flags SKUs where avg_stock < 2× avg_sales

### Page 3: Explainability
- Horizontal bar chart of hardcoded feature importances (rolling_mean_7 is top)
- Placeholder for SHAP plots

### Page 4: Model Monitoring
- Displays current model name, WMAPE, and health status
- Drift monitoring status message
- Retraining history table (last 5 weeks, random WMAPE values)

### Page 5: Comparison
- Bar chart comparing all models by WMAPE
- Highlights the best model

**Run:** `streamlit run dashboard/app.py --server.port 8501`

---

## 12. Testing — `tests/test_pipeline.py`

A Pytest suite with 5 tests:

| Test | What It Validates |
|---|---|
| `test_data_generation` | `fmcg_sales.csv` exists, has `sales` and `sku_id` columns |
| `test_feature_engineering` | `add_date_features()` creates `day_of_week`, `month`, `is_weekend` |
| `test_baseline_models` | XGBoost and LightGBM `.joblib` files exist; `baseline_metrics.csv` has `model` and `WMAPE` columns |
| `test_api_health` | `GET /health` returns 200 (skipped if API not running) |
| `test_forecast_endpoint` | `POST /forecast` returns 200 or 404 (skipped if API not running) |

**Run:** `pytest tests/test_pipeline.py -v`

---

## 13. Deployment (Docker)

### Dockerfile
- Base image: `python:3.10-slim`
- Installs dependencies from `requirements.txt`
- Exposes ports 8000 (API) and 8501 (Dashboard)
- Runs both API and Dashboard in a single container

### docker-compose.yml
- **2 services:**
  - `api` — runs `uvicorn api.main:app` on port 8000
  - `dashboard` — runs `streamlit run dashboard/app.py` on port 8501, depends on `api`
- Both mount the project directory as a volume
- Dashboard connects to API via `http://api:8000`

**Run:** `docker-compose up`

---

## 14. End-to-End Execution Order

To run the entire project from scratch:

```bash
# 1. Generate synthetic data
python data/generate_data.py

# 2. Run data pipeline (clean → split → normalize)
python src/data_pipeline.py

# 3. Train baseline models (Naive, XGBoost, LightGBM, ARIMA)
python src/baseline_models.py

# 4. Train LSTM model
python src/lstm_model.py

# 5. Train TFT model (main model)
python src/tft_model.py

# 6. Generate SHAP explanations
python src/explainability.py

# 7. Run MLOps pipeline (metrics check, hyperparams)
python src/mlops_pipeline.py

# 8. Run drift monitoring
python monitoring/drift_monitor.py

# 9. Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 10. Start dashboard (in another terminal)
streamlit run dashboard/app.py --server.port 8501

# 11. Run tests
pytest tests/test_pipeline.py -v
```

---

## Environment Variables (`.env.example`)

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | FastAPI server URL |
| `MODEL_PATH` | `models/tft_model.joblib` | Path to main model |
| `DATA_PATH` | `data/processed` | Processed data directory |
| `OUTPUT_PATH` | `outputs` | Output directory |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server (optional) |

---

## Dependencies (`requirements.txt`)

| Category | Packages |
|---|---|
| **Deep Learning** | torch 2.1.0, pytorch-forecasting 1.8.0, pytorch-lightning 2.1.0 |
| **ML** | scikit-learn 1.3.0, xgboost 2.0.0, lightgbm 4.1.0, statsmodels 0.14.1 |
| **Data** | pandas 2.1.0, numpy 1.24.3 |
| **Visualization** | matplotlib 3.8.0, seaborn 0.13.0, plotly 5.18.0 |
| **Explainability** | shap 0.44.0 |
| **API** | fastapi 0.104.0, uvicorn 0.24.0, pydantic 2.5.0 |
| **Dashboard** | streamlit 1.29.0 |
| **MLOps** | mlflow 2.10.0, evidently 0.4.0, optuna 3.5.0 |
| **Utilities** | requests 2.31.0, joblib 1.3.0, python-dotenv 1.0.0 |
