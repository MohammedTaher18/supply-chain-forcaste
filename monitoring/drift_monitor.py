import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "monitoring")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_drift_score(train_data, recent_data, column):
    train_mean = train_data[column].mean()
    train_std = train_data[column].std()
    recent_mean = recent_data[column].mean()
    
    if train_std == 0:
        drift_score = 0
    else:
        drift_score = abs(recent_mean - train_mean) / train_std
    
    return drift_score


def run_drift_monitor():
    print("=" * 50)
    print("Drift Monitoring")
    print("=" * 50)
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    most_recent = test_df.tail(30)
    
    features = ["sales", "price", "stock_level"]
    
    drift_scores = {}
    for feature in features:
        drift = calculate_drift_score(train_df, most_recent, feature)
        drift_scores[feature] = drift
    
    overall_drift = np.mean(list(drift_scores.values()))
    
    print(f"\nDrift Scores:")
    for feature, score in drift_scores.items():
        status = "Alert" if score > 0.2 else "OK"
        print(f"  {feature}: {score:.4f} ({status})")
    
    print(f"\nOverall Drift Score: {overall_drift:.4f}")
    
    if overall_drift > 0.2:
        print("ALERT: Drift detected! Consider retraining.")
    else:
        print("Status: No significant drift detected")
    
    with open(os.path.join(OUTPUT_DIR, "drift_report.txt"), "w") as f:
        f.write("Drift Monitoring Report\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Overall Drift Score: {overall_drift:.4f}\n")
        f.write(f"Status: {'Alert' if overall_drift > 0.2 else 'OK'}\n\n")
        f.write("Feature Drift Scores:\n")
        for feature, score in drift_scores.items():
            f.write(f"  {feature}: {score:.4f}\n")
    
    print(f"\nSaved report to {OUTPUT_DIR}/drift_report.txt")
    
    return drift_scores


if __name__ == "__main__":
    run_drift_monitor()