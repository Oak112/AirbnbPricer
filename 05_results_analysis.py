#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Airbnb Rental Price Prediction and Community Feature Analysis - Results Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
import os

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Output and input directories
output_dir = Path("processed_data")
model_dir = Path("models")
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

print("Analyzing model results and neighborhood impacts...")

# Step 1: Load model comparison results
print("\nLoading model comparison results...")
model_comparison = pd.read_csv(model_dir / "model_comparison.csv")

# Display best models
best_models = model_comparison.sort_values('Test R²', ascending=False).head(5)
print("\nTop 5 models by Test R²:")
print(best_models[['Model', 'Feature Set', 'Test R²', 'Test RMSE ($)', 'Test MAE ($)']].to_string(index=False))

# Step 2: Load feature importances
print("\nLoading feature importance results...")
try:
    # Load feature importances if available
    feature_importances = pd.read_csv(model_dir / "feature_importances.csv")
    print("\nTop 10 most important features:")
    print(feature_importances.head(10).to_string(index=False))
    
    # Filter for neighborhood features
    neighborhood_features = [
        'listing_count', 'listing_density',
        'avg_price', 'median_price', 'avg_review_score',
        'subway_station_count', 'subway_density',
        'crime_count', 'crime_per_1000_listings'
    ]
    
    neighborhood_importances = feature_importances[
        feature_importances['Feature'].isin(neighborhood_features)
    ].sort_values('Importance', ascending=False)
    
    print("\nNeighborhood feature importances:")
    print(neighborhood_importances.to_string(index=False))
    
except FileNotFoundError:
    # Check for coefficients instead
    try:
        feature_coefficients = pd.read_csv(model_dir / "feature_coefficients.csv")
        print("\nTop 10 features by coefficient magnitude:")
        top_coefs = feature_coefficients.sort_values('Abs_Coefficient', ascending=False).head(10)
        print(top_coefs[['Feature', 'Coefficient']].to_string(index=False))
        
        # Filter for neighborhood features
        neighborhood_features = [
            'listing_count', 'listing_density',
            'avg_price', 'median_price', 'avg_review_score',
            'subway_station_count', 'subway_density',
            'crime_count', 'crime_per_1000_listings'
        ]
        
        neighborhood_coefficients = feature_coefficients[
            feature_coefficients['Feature'].isin(neighborhood_features)
        ].sort_values('Abs_Coefficient', ascending=False)
        
        print("\nNeighborhood feature coefficients:")
        print(neighborhood_coefficients[['Feature', 'Coefficient']].to_string(index=False))
        
    except FileNotFoundError:
        print("No feature importance or coefficient files found.")

# Step 3: Compare different feature sets
print("\nComparing feature set performance...")
feature_set_comparison = model_comparison.groupby('Feature Set').agg({
    'Test R²': ['mean', 'max'],
    'Test RMSE ($)': 'mean',
    'Test MAE ($)': 'mean'
}).reset_index()

# Flatten the multi-level column names
feature_set_comparison.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in feature_set_comparison.columns]

print("\nFeature set performance:")
print(feature_set_comparison.sort_values('Test R²_mean', ascending=False).to_string(index=False))

# Step 4: Load the enriched listings data for additional analysis
print("\nLoading enriched listings data...")
enriched_listings = pd.read_csv(output_dir / "enriched_listings.csv")

# Step 5: Analyze neighborhood impacts
print("\nAnalyzing neighborhood impacts on price...")

# Group data by neighborhood and calculate average prices
neighborhood_prices = enriched_listings.groupby('neighbourhood_cleansed').agg({
    'price': ['mean', 'median', 'count'],
    'review_score_avg': 'mean',
    'listing_density': 'mean',
    'subway_station_count': 'mean',
    'subway_density': 'mean',
    'crime_count': 'mean',
    'crime_per_1000_listings': 'mean'
}).reset_index()

# Flatten the multi-level column names
neighborhood_prices.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in neighborhood_prices.columns]

# Sort by average price
top_neighborhoods = neighborhood_prices.sort_values('price_mean', ascending=False).head(10)
print("\nTop 10 neighborhoods by average price:")
print(top_neighborhoods[['neighbourhood_cleansed', 'price_mean', 'price_count']].to_string(index=False))

# Step 6: Create visualizations of neighborhood characteristics vs. price
print("\nCreating neighborhood analysis visualizations...")

# Neighborhoods with enough listings (more than 20)
popular_neighborhoods = neighborhood_prices[neighborhood_prices['price_count'] >= 20].copy()
popular_neighborhoods['price_mean'] = popular_neighborhoods['price_mean'].round(2)

# Check available columns
print(f"\nAvailable columns in popular_neighborhoods dataframe:")
print(popular_neighborhoods.columns.tolist())

# Plotting functions for neighborhood analysis
def plot_neighborhood_feature(neighborhood_df, x_feature, y_feature='price_mean', n_neighborhoods=15):
    """Create a bar chart of neighborhoods sorted by a given feature."""
    # Check if the requested feature exists
    if x_feature != 'neighbourhood_cleansed' and x_feature not in neighborhood_df.columns:
        print(f"Warning: Column '{x_feature}' not found in the dataframe. Skipping this plot.")
        return

    if y_feature not in neighborhood_df.columns:
        print(f"Warning: Column '{y_feature}' not found in the dataframe. Skipping this plot.")
        return
        
    if x_feature == 'neighbourhood_cleansed':
        # Sort by price if the x-axis is neighborhood
        sorted_df = neighborhood_df.sort_values(y_feature, ascending=False).head(n_neighborhoods)
    else:
        # Otherwise sort by the x-feature
        sorted_df = neighborhood_df.sort_values(x_feature, ascending=False).head(n_neighborhoods)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = sorted_df['neighbourhood_cleansed']
    y = sorted_df[y_feature]
    c = sorted_df[x_feature] if x_feature != 'neighbourhood_cleansed' else None
    
    if x_feature == 'neighbourhood_cleansed':
        # Simple bar chart for neighborhoods
        sns.barplot(x=sorted_df['neighbourhood_cleansed'], y=sorted_df[y_feature], ax=ax)
        plt.title(f'Top {n_neighborhoods} Neighborhoods by {y_feature}')
    else:
        # Create a color-coded bar chart
        bars = ax.bar(x, y, color='lightgray')
        
        # Color bars based on the value of the x_feature
        if c is not None:
            norm = plt.Normalize(c.min(), c.max())
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            
            for i, bar in enumerate(bars):
                bar.set_color(sm.to_rgba(c.iloc[i]))
            
            fig.colorbar(sm, ax=ax, label=x_feature)
        
        plt.title(f'Average Price by Neighborhood, Colored by {x_feature}')
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Neighborhood')
    plt.ylabel(y_feature)
    plt.tight_layout()
    
    filename = f"{y_feature}_by_{x_feature.replace(' ', '_')}.png"
    plt.savefig(report_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

# Create various neighborhood analysis plots
# 1. Top neighborhoods by price
plot_neighborhood_feature(popular_neighborhoods, 'neighbourhood_cleansed', 'price_mean', 15)

# After checking the actual column names in popular_neighborhoods
# Update the feature names used for plotting
# For example, if listing_density is not available, but listing_density_mean is:
neighborhood_features_map = {
    'listing_density': 'listing_density_mean',
    'subway_station_count': 'subway_station_count_mean',
    'subway_density': 'subway_density_mean',
    'crime_per_1000_listings': 'crime_per_1000_listings_mean'
}

# 2. Price vs. listing density
if 'listing_density_mean' in popular_neighborhoods.columns:
    plot_neighborhood_feature(popular_neighborhoods, 'listing_density_mean', 'price_mean', 15)
else:
    print("Skipping listing density plot as column not found")

# 3. Price vs. subway station count
if 'subway_station_count_mean' in popular_neighborhoods.columns:
    plot_neighborhood_feature(popular_neighborhoods, 'subway_station_count_mean', 'price_mean', 15)
else:
    print("Skipping subway station count plot as column not found")

# 4. Price vs. subway density
if 'subway_density_mean' in popular_neighborhoods.columns:
    plot_neighborhood_feature(popular_neighborhoods, 'subway_density_mean', 'price_mean', 15)
else:
    print("Skipping subway density plot as column not found")

# 5. Price vs. crime 
if 'crime_per_1000_listings_mean' in popular_neighborhoods.columns:
    plot_neighborhood_feature(popular_neighborhoods, 'crime_per_1000_listings_mean', 'price_mean', 15)
else:
    print("Skipping crime plot as column not found")

# Step 7: Create a correlation matrix for neighborhood-level features
print("\nCalculating correlation matrix for neighborhood features...")

# Get actual neighborhood features that exist in the DataFrame
available_features = []
for feature in ['listing_density_mean', 'subway_station_count_mean', 'subway_density_mean', 
                'crime_per_1000_listings_mean', 'price_mean']:
    if feature in popular_neighborhoods.columns:
        available_features.append(feature)
    else:
        print(f"Warning: Feature '{feature}' not found in DataFrame")

if len(available_features) > 1:
    # Create correlation matrix
    corr_matrix = popular_neighborhoods[available_features].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Neighborhood Features')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'neighborhood_correlation_matrix.png'))
    plt.close()
    print("Saved neighborhood correlation matrix to:", os.path.join(report_dir, 'neighborhood_correlation_matrix.png'))
else:
    print("Not enough features available to create a correlation matrix")

# Step 8: Generate a report summarizing the key neighborhood factors affecting price
print("\nGenerating neighborhood impact report...")

report_content = """# Neighborhood Impact on Airbnb Prices Report

## 1. Top Models and Performance

The following models performed best in predicting Airbnb prices:

"""
report_content += model_comparison.sort_values('Test R²', ascending=False).head(5).to_markdown() + "\n\n"

report_content += """
## 2. Impact of Neighborhood Features on Price Prediction

When comparing models with different feature sets:
"""

# 修复这部分代码，避免使用.loc进行索引访问，因为DataFrame已经重置了索引
# 首先检查是否存在相应的行
neighborhood_r2 = feature_set_comparison[feature_set_comparison['Feature Set'] == 'Neighborhood Features']['Test R²_mean'].values
combined_r2 = feature_set_comparison[feature_set_comparison['Feature Set'] == 'Combined Features']['Test R²_mean'].values

report_content += feature_set_comparison.to_markdown() + "\n\n"

report_content += """
**Finding:** """

if len(neighborhood_r2) > 0 and len(combined_r2) > 0:
    report_content += f"Neighborhood features alone can explain approximately {neighborhood_r2[0]:.2%} of price variation, while combined with property features, the model explains {combined_r2[0]:.2%}."
else:
    report_content += "The impact of different feature sets on price prediction varies as shown in the table above."

report_content += """

## 3. Most Important Neighborhood Features

The following neighborhood features were most important in predicting price:
"""

if len(neighborhood_importances) > 0:
    report_content += neighborhood_importances.head(5).to_markdown() + "\n\n"
else:
    report_content += "No significant neighborhood features were identified.\n\n"

report_content += """
## 4. Top-Priced Neighborhoods
"""

report_content += popular_neighborhoods.sort_values('price_mean', ascending=False).head(10)[['neighbourhood_cleansed', 'price_mean', 'price_count']].to_markdown() + "\n\n"

report_content += """
## 5. Neighborhood Characteristic Correlations
"""

if len(available_features) > 1:
    report_content += """
The following correlations were observed between neighborhood characteristics and prices:
"""
    for i, feature in enumerate(available_features):
        if feature != 'price_mean' and 'price_mean' in available_features:
            corr_value = corr_matrix.loc[feature, 'price_mean']
            report_content += f"- **{feature}** has a correlation of **{corr_value:.2f}** with price\n"
else:
    report_content += "Not enough features available to calculate correlations.\n"

report_content += """
## 6. Conclusion

Based on the analysis, community-level features do impact Airbnb pricing, but individual property features remain more predictive of price. The most significant neighborhood factors include...
"""

# Write the report to a file
with open(os.path.join(report_dir, 'neighborhood_impact_report.md'), 'w') as f:
    f.write(report_content)

print("Saved neighborhood impact report to:", os.path.join(report_dir, 'neighborhood_impact_report.md'))
print("\nAnalysis complete!") 