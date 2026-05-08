import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from feature_config import get_available_features

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def calculate_wmape(y_true, y_pred):
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true).sum()
    return (num / den) * 100 if den > 0 else 0.0


def run_tft(train_df, test_df, epochs=10):
    print("=" * 50)
    print("Training Temporal Fusion Transformer (PyTorch Forecasting)")
    print("=" * 50)
    
    # Define features based on available columns
    static_categoricals = get_available_features(train_df, ["sku_id", "distribution_center", "category"])
    
    # Cast static categoricals to string/category so TimeSeriesDataSet can process them
    for col in static_categoricals:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
    time_varying_known_reals = get_available_features(train_df, ["price", "promotion_discount", "is_holiday", "is_weekend", "promotion_flag"])
    time_varying_unknown_reals = get_available_features(train_df, ["lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"])
    
    # Create time_idx if not exists
    if "time_idx" not in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"])
        min_date = train_df["date"].min()
        train_df["time_idx"] = (train_df["date"] - min_date).dt.days
        
        test_df["date"] = pd.to_datetime(test_df["date"])
        test_df["time_idx"] = (test_df["date"] - min_date).dt.days
    
    # Make sure time_idx is int
    train_df["time_idx"] = train_df["time_idx"].astype(int)
    test_df["time_idx"] = test_df["time_idx"].astype(int)
    
    max_encoder_length = 30
    max_prediction_length = 7
    
    print("Creating TimeSeriesDataSet...")
    
    # PyTorch Forecasting requires target to be float
    train_df["sales"] = train_df["sales"].astype(float)
    test_df["sales"] = test_df["sales"].astype(float)

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="sales",
        group_ids=["sku_id", "distribution_center"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["sku_id", "distribution_center"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    # Create validation set
    validation = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)
    
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    # Define model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )
    
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
    
    # Setup trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        enable_model_summary=True,
        callbacks=[early_stop_callback],
        logger=CSVLogger(save_dir=OUTPUT_DIR, name="tft_logs")
    )
    
    print(f"Training for {epochs} epochs...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Evaluate
    print("Evaluating...")
    
    # Predict returns a tensor if return_x=False, but we want the median forecast (idx 3 usually in QuantileLoss, or just use mode="prediction")
    predictions = tft.predict(val_dataloader, mode="prediction")
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    
    actuals_np = actuals.numpy().flatten()
    preds_np = predictions.numpy().flatten()
    
    # Truncate to min length in case of mismatches
    min_len = min(len(actuals_np), len(preds_np))
    actuals_np = actuals_np[:min_len]
    preds_np = preds_np[:min_len]
    
    # Filter out NaNs if any
    valid_idx = ~np.isnan(actuals_np) & ~np.isnan(preds_np)
    y_test = actuals_np[valid_idx]
    pred = np.maximum(preds_np[valid_idx], 0)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    wmape = calculate_wmape(y_test, pred)
    
    print(f"\nTFT Results: MAE={mae:.4f}, RMSE={rmse:.4f}, WMAPE={wmape:.2f}%")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "tft_model.pt")
    torch.save(tft.state_dict(), model_path)
    print(f"Saved PyTorch model to {model_path}")
    
    # Save metrics
    pd.DataFrame([{"model": "TFT", "MAE": mae, "RMSE": rmse, "WMAPE": wmape}]).to_csv(
        os.path.join(MODELS_DIR, "tft_metrics.csv"), index=False)
    
    # Feature importance plot
    raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)
    interpretation = tft.interpret_output(raw_predictions, reduction="sum")
    
    # Save interpretation plots
    fig, ax = plt.subplots(figsize=(10, 6))
    tft.plot_interpretation(interpretation)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tft_feature_importance.png"), dpi=100)
    plt.close('all')
    
    print(f"Saved feature importance plot to {OUTPUT_DIR}/tft_feature_importance.png")
    
    return {"model": "TFT", "MAE": mae, "RMSE": rmse, "WMAPE": wmape}


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    run_tft(train_df, test_df, epochs=10)