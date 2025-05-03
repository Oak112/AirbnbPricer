#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Airbnb Rental Price Prediction - Plot Feature Importances for Best Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import os
import warnings
import shap # Import SHAP

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Input/output directories
model_dir = Path("models")
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

print("Loading specific model and test data for SHAP analysis...")

# --- Load Model ---
try:
    # Force load the specified model for Combined Features
    model_path = model_dir / "xgboost_combined_features.pkl" # Use Combined Features model
    model_name = "XGBoost"
    feature_set_name = "Combined Features" # Update feature set name

    # Load the model pipeline
    pipeline = joblib.load(model_path)
    print(f"Loaded model pipeline from {model_path}")

    # Get the preprocessor and the XGBoost model step
    preprocessor = pipeline.named_steps['preprocessor']
    xgb_model = pipeline.named_steps['model']

except Exception as e:
    print(f"ERROR: Could not load model pipeline: {e}")
    exit(1)

# --- Load Data ---
try:
    # Define the data directory
    processed_data_dir = Path("processed_data") 
    # Load the test data corresponding to the feature set used by the model
    # Assuming the model was trained using the standard split saved by save_models.py
    X_test = pd.read_csv(processed_data_dir / "X_test.csv")
    # Ensure columns match the 'Selected Features'/'Combined Features' set
    # (Based on previous context, Selected Features == Combined Features)
    print(f"Loaded X_test data with shape {X_test.shape}")

except FileNotFoundError:
     print(f"ERROR: X_test.csv not found in {processed_data_dir}. Please ensure test data is saved.")
     exit(1)
except Exception as e:
    print(f"ERROR: Could not load test data: {e}")
    exit(1)

# --- Preprocess Data ---
try:
    print("Preprocessing X_test data...")
    X_test_processed = preprocessor.transform(X_test)
    print(f"Shape of preprocessed X_test: {X_test_processed.shape}")

    # Get feature names after preprocessing
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_categorical_features = onehot_encoder.get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(encoded_categorical_features)
    print(f"Total features after preprocessing: {len(all_feature_names)}")

    # Convert sparse matrix to dense if necessary for SHAP
    if hasattr(X_test_processed, "toarray"): # Check if it's sparse
         print("Converting sparse matrix to dense array for SHAP.")
         X_test_processed_dense = X_test_processed.toarray()
    else:
         X_test_processed_dense = X_test_processed

    # Create a DataFrame for SHAP summary plot (optional but good for feature names)
    X_test_processed_df = pd.DataFrame(X_test_processed_dense, columns=all_feature_names)

except Exception as e:
    print(f"ERROR: Could not preprocess test data: {e}")
    exit(1)

# --- Calculate SHAP Values ---
try:
    print("Calculating SHAP values (this may take a while)...")
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_processed_df) # Use DataFrame here
    print("SHAP values calculated.")

except Exception as e:
    print(f"ERROR: Could not calculate SHAP values: {e}")
    # It might be due to the shap library not being installed.
    print("Ensure the 'shap' library is installed ('pip install shap').")
    exit(1)

# --- Plotting SHAP Summary ---
print("Generating SHAP summary plot...")

plt.figure()
shap.summary_plot(shap_values, X_test_processed_df, plot_type="bar", show=False)

# Adjust plot aesthetics if needed (summary_plot modifies the current figure)
f = plt.gcf() # Get current figure
f.set_figheight(10) # Adjust height
f.set_figwidth(12) # Adjust width
plt.title(f'SHAP Feature Importance\nModel: {model_name}, Feature Set: {feature_set_name}', fontsize=14)
plt.xlabel("mean(|SHAP value|) (Average impact on model output magnitude)", fontsize=12)
plt.tight_layout()

# Save the plot
plot_filename = report_dir / f"shap_importance_{model_name.replace(' ', '_')}_{feature_set_name.replace(' ', '_')}.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Saved SHAP feature importance plot to {plot_filename}")

# Optionally display the plot
# plt.show()

print("SHAP plot generation complete.") 