#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Airbnb Rental Price Prediction and Community Feature Analysis - Model Training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
import xgboost as xgb

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Output and input directories
output_dir = Path("processed_data")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

print("Loading enriched Airbnb data...")
# Load enriched listings data
try:
    enriched_listings = pd.read_csv(output_dir / "enriched_listings.csv")
    print(f"Loaded enriched data with shape: {enriched_listings.shape}")
except FileNotFoundError:
    print("ERROR: Enriched listings file not found. Please run neighborhood_features.py first.")
    exit(1)

# Step 1: Separate features into categories: property-level and neighborhood-level
print("\nCategorizing features...")

# Property-level features (that we might want to control for / downweight)
property_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'minimum_nights', 'maximum_nights', 'availability_365',
    'number_of_reviews', 'review_score_avg', 'months_since_last_review',
    'host_years_experience', 'nearest_subway_km',
    'calculated_host_listings_count'
]

# Categorical property features
categorical_property_features = [
    'room_type', 'property_type_cleaned'
]

# Neighborhood-level features (our main focus)
neighborhood_features = [
    'listing_count', 'listing_density',
    'avg_price', 'median_price', 'avg_review_score',
    'subway_station_count', 'subway_density',
    'crime_count', 'crime_per_1000_listings'
]

# Location features that may be used but aren't strictly property or neighborhood
location_features = [
    'latitude', 'longitude'
]

# Filter to include only those that are available in the dataset
available_property_features = [f for f in property_features if f in enriched_listings.columns]
available_categorical_property = [f for f in categorical_property_features if f in enriched_listings.columns]
available_neighborhood_features = [f for f in neighborhood_features if f in enriched_listings.columns]
available_location_features = [f for f in location_features if f in enriched_listings.columns]

# Print available features in each category
print(f"Available property features: {available_property_features}")
print(f"Available categorical property features: {available_categorical_property}")
print(f"Available neighborhood features: {available_neighborhood_features}")
print(f"Available location features: {available_location_features}")

# Step 2: Analyze collinearity between features
print("\nAnalyzing feature collinearity...")

# Combine all numeric features
all_numeric_features = available_property_features + available_neighborhood_features + available_location_features

# Calculate correlation matrix
correlation_matrix = enriched_listings[all_numeric_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(18, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of All Numeric Features')
plt.tight_layout()
plt.savefig(model_dir / 'feature_collinearity.png', dpi=300, bbox_inches='tight')
print(f"Saved feature collinearity heatmap to {model_dir / 'feature_collinearity.png'}")

# Identify highly correlated features (|corr| > 0.7)
high_corr_pairs = []
for i in range(len(all_numeric_features)):
    for j in range(i+1, len(all_numeric_features)):
        feat_i = all_numeric_features[i]
        feat_j = all_numeric_features[j]
        corr_val = abs(correlation_matrix.iloc[i, j])
        if corr_val > 0.7:
            high_corr_pairs.append((feat_i, feat_j, corr_val))

if high_corr_pairs:
    print("\nHighly correlated feature pairs (|corr| > 0.7):")
    for feat_i, feat_j, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
        print(f"{feat_i} and {feat_j}: {corr_val:.3f}")

# Step 3: Create different feature sets for model comparison
print("\nCreating feature sets for model comparison...")

# Set 1: Property features only (baseline)
property_features_set = available_property_features + available_categorical_property

# Set 2: Neighborhood features only
neighborhood_features_set = available_neighborhood_features 

# Set 3: Combined features (all)
combined_features_set = property_features_set + neighborhood_features_set + available_location_features

# Set 4: Selected features (removing highly correlated ones)
# Based on the correlation analysis, we'll manually select features that don't have high collinearity
# But we'll keep this as a placeholder for now
selected_features_set = combined_features_set.copy()

# Print feature sets
print(f"Property features set: {len(property_features_set)} features")
print(f"Neighborhood features set: {len(neighborhood_features_set)} features")
print(f"Combined features set: {len(combined_features_set)} features")
print(f"Selected features set: {len(selected_features_set)} features")

# Step 4: Prepare the target variable
print("\nPreparing target variable...")

# Check price distribution
plt.figure(figsize=(10, 6))
sns.histplot(enriched_listings['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig(model_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved price distribution plot to {model_dir / 'price_distribution.png'}")

# Log-transform the target for better model performance
enriched_listings['log_price'] = np.log1p(enriched_listings['price'])

# Plot log-transformed price
plt.figure(figsize=(10, 6))
sns.histplot(enriched_listings['log_price'], bins=50, kde=True)
plt.title('Log-Transformed Price Distribution')
plt.xlabel('Log(Price + 1)')
plt.ylabel('Frequency')
plt.savefig(model_dir / 'log_price_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved log-transformed price plot to {model_dir / 'log_price_distribution.png'}")

# Step 5: Split the data into train and test sets
print("\nSplitting data into train and test sets...")

# Define target
target = 'log_price'

# Split the data
X = enriched_listings.drop(['id', 'price', 'log_price', 'neighbourhood_cleansed', 'borough'], axis=1, errors='ignore')
y = enriched_listings[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Step 6: Define preprocessing pipeline
print("\nDefining preprocessing pipeline...")

# Identify categorical and numeric columns in the training data
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns: {len(numeric_cols)} columns")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 7: Train and evaluate models with different feature sets
print("\nTraining and evaluating models...")

# Function to train and evaluate a model with a specific feature set
def train_and_evaluate_model(model_name, model, feature_set, feature_set_name):
    print(f"\nTraining {model_name} with {feature_set_name}...")
    
    # Select features from the dataframe
    X_train_subset = X_train[feature_set].copy()
    X_test_subset = X_test[feature_set].copy()
    
    # Identify categorical and numeric columns in this subset
    cat_cols = X_train_subset.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train_subset.select_dtypes(include=['number']).columns.tolist()
    
    # Create subset-specific preprocessor
    subset_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', subset_preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train_subset, y_train)
    
    # Make predictions
    y_pred_train = pipeline.predict(X_train_subset)
    y_pred_test = pipeline.predict(X_test_subset)
    
    # Convert log predictions back to original scale
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train)
    y_pred_test_orig = np.expm1(y_pred_test)
    
    # Calculate metrics (log scale)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Calculate metrics (original scale)
    train_rmse_orig = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
    test_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    train_r2_orig = r2_score(y_train_orig, y_pred_train_orig)
    test_r2_orig = r2_score(y_test_orig, y_pred_test_orig)
    
    # Mean absolute error (original scale)
    train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
    
    # Print metrics
    print(f"Log scale metrics:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    print(f"Original scale metrics:")
    print(f"  Train RMSE: ${train_rmse_orig:.2f}")
    print(f"  Test RMSE: ${test_rmse_orig:.2f}")
    print(f"  Train MAE: ${train_mae:.2f}")
    print(f"  Test MAE: ${test_mae:.2f}")
    print(f"  Train R²: {train_r2_orig:.4f}")
    print(f"  Test R²: {test_r2_orig:.4f}")
    
    # Save model predictions for later analysis
    result = {
        'model_name': model_name,
        'feature_set': feature_set_name,
        'y_test': y_test_orig,
        'y_pred': y_pred_test_orig,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse_orig': train_rmse_orig,
        'test_rmse_orig': test_rmse_orig,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2_orig': train_r2_orig,
        'test_r2_orig': test_r2_orig,
        'pipeline': pipeline,
        'feature_set_list': feature_set
    }
    
    return result

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

# Define feature sets to evaluate
feature_sets = {
    'Property Features': property_features_set,
    'Neighborhood Features': neighborhood_features_set,
    'Combined Features': combined_features_set,
    'Selected Features': selected_features_set
}

# Train and evaluate each model with each feature set
results = []

for model_name, model in models.items():
    for feature_set_name, feature_set in feature_sets.items():
        # Skip if feature set is empty
        if not feature_set:
            continue
        
        result = train_and_evaluate_model(model_name, model, feature_set, feature_set_name)
        results.append(result)

# Step 8: Analyze and compare model results
print("\nComparing model results...")

# Create comparison dataframe
comparison_df = pd.DataFrame([
    {
        'Model': result['model_name'],
        'Feature Set': result['feature_set'],
        'Test RMSE': result['test_rmse'],
        'Test R²': result['test_r2'],
        'Test RMSE ($)': result['test_rmse_orig'],
        'Test MAE ($)': result['test_mae'],
        'Test R² (orig)': result['test_r2_orig']
    } for result in results
])

# Sort by Test R²
comparison_df = comparison_df.sort_values('Test R²', ascending=False)

# Save comparison to CSV
comparison_df.to_csv(model_dir / 'model_comparison.csv', index=False)
print(f"Saved model comparison to {model_dir / 'model_comparison.csv'}")

# Display the top 5 models
print("\nTop 5 models by Test R²:")
print(comparison_df.head(5).to_string(index=False))

# Step 9: Feature importance analysis for the best model
print("\nAnalyzing feature importances...")

# Find the best model that uses neighborhood features
best_neighborhood_model = None
best_r2 = -float('inf')

for result in results:
    if 'Neighborhood' in result['feature_set'] and result['test_r2'] > best_r2:
        best_neighborhood_model = result
        best_r2 = result['test_r2']

if best_neighborhood_model:
    print(f"\nBest model with neighborhood features: {best_neighborhood_model['model_name']} with {best_neighborhood_model['feature_set']}")
    print(f"Test R²: {best_neighborhood_model['test_r2']:.4f}")
    
    # Extract feature importances from the model
    pipeline = best_neighborhood_model['pipeline']
    
    # Get feature names after preprocessing
    cat_cols = X_train[best_neighborhood_model['feature_set_list']].select_dtypes(include=['object', 'category']).columns
    num_cols = X_train[best_neighborhood_model['feature_set_list']].select_dtypes(include=['number']).columns
    
    # Get one-hot encoded feature names
    if cat_cols.size > 0:
        cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols)
    else:
        encoded_cat_cols = []
    
    # Get all feature names after preprocessing
    all_feature_names = list(num_cols) + list(encoded_cat_cols)
    
    # Get feature importances if available
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        importances = pipeline.named_steps['model'].feature_importances_
        
        # Check if lengths match
        if len(all_feature_names) != len(importances):
            print(f"WARNING: Feature names length ({len(all_feature_names)}) doesn't match importances length ({len(importances)})")
            # Debug information
            print(f"Numeric columns: {len(num_cols)}")
            print(f"Encoded categorical columns: {len(encoded_cat_cols)}")
            
            # Fix the length mismatch by truncating the longer list
            min_len = min(len(all_feature_names), len(importances))
            all_feature_names = all_feature_names[:min_len]
            importances = importances[:min_len]
        
        # Create a DataFrame of feature importances
        feature_importances = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        
        # Save to CSV
        feature_importances.to_csv(model_dir / 'feature_importances.csv', index=False)
        print(f"Saved feature importances to {model_dir / 'feature_importances.csv'}")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
        plt.title(f'Top 20 Feature Importances - {best_neighborhood_model["model_name"]}')
        plt.tight_layout()
        plt.savefig(model_dir / 'feature_importances.png', dpi=300, bbox_inches='tight')
        print(f"Saved feature importances plot to {model_dir / 'feature_importances.png'}")
        
        # Identify most important neighborhood features
        neighborhood_importances = feature_importances[
            feature_importances['Feature'].isin(available_neighborhood_features)
        ]
        
        print("\nNeighborhood feature importances:")
        print(neighborhood_importances.to_string(index=False))
        
        # Plot neighborhood feature importances
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=neighborhood_importances)
        plt.title('Neighborhood Feature Importances')
        plt.tight_layout()
        plt.savefig(model_dir / 'neighborhood_importances.png', dpi=300, bbox_inches='tight')
        print(f"Saved neighborhood importances plot to {model_dir / 'neighborhood_importances.png'}")
    
    elif hasattr(pipeline.named_steps['model'], 'coef_'):
        # For linear models
        coefficients = pipeline.named_steps['model'].coef_
        
        # Create a DataFrame of coefficients
        feature_coefficients = pd.DataFrame({
            'Feature': all_feature_names,
            'Coefficient': coefficients
        })
        
        # Sort by absolute coefficient value
        feature_coefficients['Abs_Coefficient'] = feature_coefficients['Coefficient'].abs()
        feature_coefficients = feature_coefficients.sort_values('Abs_Coefficient', ascending=False)
        
        # Save to CSV
        feature_coefficients.to_csv(model_dir / 'feature_coefficients.csv', index=False)
        print(f"Saved feature coefficients to {model_dir / 'feature_coefficients.csv'}")
        
        # Plot feature coefficients
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_coefficients.head(20))
        plt.title(f'Top 20 Feature Coefficients - {best_neighborhood_model["model_name"]}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_dir / 'feature_coefficients.png', dpi=300, bbox_inches='tight')
        print(f"Saved feature coefficients plot to {model_dir / 'feature_coefficients.png'}")
        
        # Identify most important neighborhood features
        neighborhood_coefficients = feature_coefficients[
            feature_coefficients['Feature'].isin(available_neighborhood_features)
        ]
        
        print("\nNeighborhood feature coefficients:")
        print(neighborhood_coefficients.to_string(index=False))
        
        # Plot neighborhood feature coefficients
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Coefficient', y='Feature', data=neighborhood_coefficients)
        plt.title('Neighborhood Feature Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_dir / 'neighborhood_coefficients.png', dpi=300, bbox_inches='tight')
        print(f"Saved neighborhood coefficients plot to {model_dir / 'neighborhood_coefficients.png'}")
    
    # Scatter plot of predicted vs actual prices
    plt.figure(figsize=(10, 8))
    plt.scatter(best_neighborhood_model['y_test'], best_neighborhood_model['y_pred'], alpha=0.5)
    plt.plot([0, max(best_neighborhood_model['y_test'])], [0, max(best_neighborhood_model['y_test'])], 'r--')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig(model_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print(f"Saved actual vs predicted plot to {model_dir / 'actual_vs_predicted.png'}")
    
    # Residual plot
    residuals = best_neighborhood_model['y_pred'] - best_neighborhood_model['y_test']
    plt.figure(figsize=(10, 8))
    plt.scatter(best_neighborhood_model['y_pred'], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residual ($)')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(model_dir / 'residual_plot.png', dpi=300, bbox_inches='tight')
    print(f"Saved residual plot to {model_dir / 'residual_plot.png'}")
    
    # Save the best model
    best_model_path = model_dir / f"best_model_{best_neighborhood_model['model_name'].replace(' ', '_').lower()}.pkl"
    joblib.dump(pipeline, best_model_path)
    print(f"Saved best model to {best_model_path}")

# Step 10: Additional analysis comparing property vs. neighborhood features
print("\nComparing impact of property vs. neighborhood features...")

# Calculate average metrics for each feature set
feature_set_comparison = comparison_df.groupby('Feature Set').agg({
    'Test R²': 'mean',
    'Test RMSE ($)': 'mean',
    'Test MAE ($)': 'mean'
}).reset_index()

print("\nAverage performance by feature set:")
print(feature_set_comparison.sort_values('Test R²', ascending=False).to_string(index=False))

# Plot feature set comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Feature Set', y='Test R²', data=feature_set_comparison)
plt.title('Average R² by Feature Set')
plt.ylim(0, max(feature_set_comparison['Test R²']) * 1.1)  # Give some headroom
plt.tight_layout()
plt.savefig(model_dir / 'feature_set_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved feature set comparison to {model_dir / 'feature_set_comparison.png'}")

print("\nModel training and evaluation completed.") 