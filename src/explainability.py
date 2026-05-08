import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from feature_config import ALL_MODEL_FEATURES, get_available_features

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "explainability")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_shap_explainability():
    print("=" * 50)
    print("SHAP Explainability")
    print("=" * 50)
    
    try:
        import shap
    except ImportError:
        print("Installing SHAP...")
        import subprocess
        subprocess.run(["pip", "install", "shap", "--quiet"])
        import shap
    
    model = joblib.load(os.path.join(MODELS_DIR, "tft_model.joblib"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    features = get_available_features(test_df, ALL_MODEL_FEATURES)
    
    X_test = test_df[features].fillna(0).sample(n=min(1000, len(test_df)), random_state=42)
    
    print("Computing SHAP values...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance.png"), dpi=100)
    plt.close()
    print(f"Saved SHAP feature importance")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=100)
    plt.close()
    print(f"Saved SHAP beeswarm plot")
    
    print(f"\nSaved all SHAP plots to {OUTPUT_DIR}")
    
    return True


if __name__ == "__main__":
    run_shap_explainability()