# Supply Chain Demand Forecasting

An end-to-end ML system for demand forecasting in FMCG retail supply chains using deep learning and gradient boosting models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Supply Chain Forecast                   │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├── Raw Data (fmcg_sales.csv)                             │
│  ├── Product Metadata                                     │
│  └── Processed (train/val/test)                           │
├─────────────────────────────────────────────────────────────┤
│  Feature Engineering                                      │
│  ├── Lag Features (7, 14, 30 days)                       │
│  ├── Rolling Statistics                                  │
│  ├── Date Features                                     │
│  └── Holiday Encoding                                   │
├─────────────────────────────────────────────────────────────┤
│  Models                                                   │
│  ├── XGBoost (Baseline)                                  │
│  ├── LightGBM                                           │
│  ├── LSTM                                              │
│  └── Temporal Fusion Transformer (Main)                  │
├─────────────────────────────────────────────────────────────┤
│  MLOps                                                   │
│  ├── Model Versioning                                   │
│  ├── Hyperparameter Tuning                               │
│  ├── Drift Monitoring                                  │
│  └── Experiment Tracking (MLflow)                       │
├─────────────────────────────────────────────────────────────┤
│  API & Dashboard                                         │
│  ├── FastAPI Backend                                    │
│  └── Streamlit Dashboard                                │
└─────────────────────────────────────────────────────────────┘
```

## Setup

```bash
cd supply_chain_forecast

pip install -r requirements.txt
```

## Data Generation

```bash
python data/generate_data.py
```

## Model Training

```bash
python src/data_pipeline.py
python src/baseline_models.py
python src/lstm_model.py
python src/tft_model.py
python src/explainability.py
python src/mlops_pipeline.py
```

## API

```bash
cd supply_chain_forecast
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Dashboard

```bash
cd supply_chain_forecast
streamlit run dashboard/app.py --server.port 8501
```

## Docker

```bash
cd supply_chain_forecast
docker-compose up
```

## Model Performance

| Model   | WMAPE  | MAE    | RMSE   |
|---------|--------|--------|--------|
| Naive   | 90.86 | 0.345 | 0.429 |
| XGBoost | 8.93  | 0.034 | 0.051 |
| LightGBM| 9.02  | 0.034 | 0.052 |
| ARIMA  | 61.10 | 0.232 | 0.284 |
| LSTM  | 58.81 | 0.223 | 0.288 |
| **TFT** | **0.18** | **0.001** | **0.001** |

## API Endpoints

- `GET /health` - Health check
- `POST /forecast` - Generate demand forecast
- `GET /model/metrics` - Get model metrics
- `GET /model/feature_importance` - Get feature importance
- `POST /retrain` - Trigger model retraining

## Project Structure

```
supply_chain_forecast/
├── data/
│   ├── raw/
│   ├── processed/
│   └── generate_data.py
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── baseline_models.py
│   ├── lstm_model.py
│   ├── tft_model.py
│   ├── explainability.py
│   └── mlops_pipeline.py
├── api/
│   ├── main.py
│   └── schemas.py
├── dashboard/
│   └── app.py
├── monitoring/
│   └── drift_monitor.py
├── tests/
│   └── test_pipeline.py
├── models/
├── outputs/
├── config/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## License

MIT