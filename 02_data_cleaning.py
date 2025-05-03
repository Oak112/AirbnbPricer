#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Airbnb Rental Price Prediction and Community Feature Analysis - Data Cleaning and Preprocessing
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import warnings
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Output directory
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

print("Loading Airbnb data...")

# Load listings data
listings_path = Path("data/Airbnb/listings.csv.gz")
listings = pd.read_csv(listings_path, compression='gzip')
print(f"Original data shape: {listings.shape}")

# Step 1: Handle missing values in the price column
print("\nProcessing price data...")

# Convert price column to float
if 'price' in listings.columns:
    # Remove rows where price is missing
    price_null_count = listings['price'].isnull().sum()
    print(f"Rows with missing price: {price_null_count}")
    
    if price_null_count > 0:
        listings = listings[listings['price'].notnull()]
        print(f"Shape after removing missing prices: {listings.shape}")
    
    # Convert price from string to float if needed
    if listings['price'].dtype == 'object':
        listings['price'] = listings['price'].replace(r'[\$,]', '', regex=True).astype(float)
    
    # Handle extreme values in price (remove top 1% of prices as outliers)
    upper_price = listings['price'].quantile(0.99)
    print(f"Removing properties with price > ${upper_price:.2f} (99th percentile)")
    listings = listings[listings['price'] <= upper_price]
    print(f"Shape after removing price outliers: {listings.shape}")

# Step 2: Process neighborhood information
print("\nProcessing neighborhood data...")

# Use neighborhood_cleansed as it's more complete
if 'neighbourhood_cleansed' in listings.columns:
    # Check for missing values
    null_neighborhoods = listings['neighbourhood_cleansed'].isnull().sum()
    print(f"Listings with missing neighborhood_cleansed: {null_neighborhoods}")
    
    if null_neighborhoods > 0:
        # Remove rows with missing neighborhood
        listings = listings[listings['neighbourhood_cleansed'].notnull()]
        print(f"Shape after removing missing neighborhoods: {listings.shape}")

# Step 3: Process property attributes
print("\nProcessing property attributes...")

# Convert bathrooms_text to numeric bathrooms
if 'bathrooms_text' in listings.columns and 'bathrooms' not in listings.columns:
    # Extract numeric values from bathrooms_text
    def extract_bathrooms(text):
        if pd.isnull(text):
            return np.nan
        # Extract numbers from the text
        match = re.search(r'(\d+(\.\d+)?)', str(text))
        if match:
            return float(match.group(1))
        elif 'half' in str(text).lower():
            return 0.5
        elif 'shared' in str(text).lower():
            return 0.5
        return np.nan
    
    listings['bathrooms'] = listings['bathrooms_text'].apply(extract_bathrooms)
    print(f"Created numeric bathrooms column from bathrooms_text")

# Clean and convert other numeric features
numeric_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                    'minimum_nights', 'maximum_nights', 'number_of_reviews']

for feature in numeric_features:
    if feature in listings.columns:
        # Replace missing values with median
        median_value = listings[feature].median()
        listings[feature].fillna(median_value, inplace=True)
        print(f"Filled missing values in {feature} with median: {median_value}")

# Step 4: Clean categorical features
print("\nProcessing categorical features...")

# Room type
if 'room_type' in listings.columns:
    # Check for missing values
    null_room_types = listings['room_type'].isnull().sum()
    print(f"Listings with missing room_type: {null_room_types}")
    
    if null_room_types > 0:
        # Fill with most common
        most_common = listings['room_type'].mode()[0]
        listings['room_type'].fillna(most_common, inplace=True)
        print(f"Filled missing room_type with most common: {most_common}")

# Property type
if 'property_type' in listings.columns:
    # Check for missing values
    null_property_types = listings['property_type'].isnull().sum()
    print(f"Listings with missing property_type: {null_property_types}")
    
    if null_property_types > 0:
        # Fill with most common
        most_common = listings['property_type'].mode()[0]
        listings['property_type'].fillna(most_common, inplace=True)
        print(f"Filled missing property_type with most common: {most_common}")
    
    # Check for too many property types and consolidate if needed
    property_type_counts = listings['property_type'].value_counts()
    print(f"Number of unique property types: {len(property_type_counts)}")
    
    if len(property_type_counts) > 10:
        # Keep only top 10 property types, group others as "Other"
        top_types = property_type_counts.head(10).index
        listings['property_type_cleaned'] = listings['property_type'].apply(
            lambda x: x if x in top_types else 'Other'
        )
        print(f"Consolidated property types to top 10 + 'Other'")
        
        # Check the new distribution
        print("Property type distribution after consolidation:")
        print(listings['property_type_cleaned'].value_counts())

# Step 5: Create some feature-engineered columns
print("\nCreating engineered features...")

# Create a host experience feature (years as host)
if 'host_since' in listings.columns:
    # Convert to datetime
    listings['host_since'] = pd.to_datetime(listings['host_since'], errors='coerce')
    
    # Calculate years as host
    current_date = pd.to_datetime('2025-04-15')  # Using the date from file names as reference
    listings['host_years_experience'] = ((current_date - listings['host_since']).dt.days / 365).round(1)
    
    # Fill missing values with median
    median_exp = listings['host_years_experience'].median()
    listings['host_years_experience'].fillna(median_exp, inplace=True)
    print(f"Created host_years_experience feature (median: {median_exp:.1f} years)")

# Create a review recency feature
if all(col in listings.columns for col in ['last_review', 'number_of_reviews']):
    # Convert to datetime
    listings['last_review'] = pd.to_datetime(listings['last_review'], errors='coerce')
    
    # Calculate months since last review (only for properties with reviews)
    has_reviews = listings['number_of_reviews'] > 0
    current_date = pd.to_datetime('2025-04-15')  # Using the date from file names as reference
    
    listings['months_since_last_review'] = np.nan
    listings.loc[has_reviews, 'months_since_last_review'] = ((current_date - listings.loc[has_reviews, 'last_review']).dt.days / 30).round(1)
    
    # Fill missing values with a high number (indicating no reviews)
    listings['months_since_last_review'].fillna(120, inplace=True)  # 10 years as arbitrary high value
    print(f"Created months_since_last_review feature")

# Create review score composite
review_scores = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 
                'review_scores_value']

if any(col in listings.columns for col in review_scores):
    # Check which ones exist
    available_scores = [col for col in review_scores if col in listings.columns]
    
    if available_scores:
        # Create a composite score (average of all available scores)
        listings['review_score_avg'] = listings[available_scores].mean(axis=1)
        
        # Fill missing values with median
        median_score = listings['review_score_avg'].median()
        listings['review_score_avg'].fillna(median_score, inplace=True)
        print(f"Created review_score_avg feature (median: {median_score:.2f})")

# Step 6: Load and prepare neighbourhood data
try:
    print("\nLoading and processing neighborhood data...")
    
    # Load neighborhood geometry
    neighborhood_geo_path = Path("data/Airbnb/neighbourhoods.geojson")
    neighborhood_geo = gpd.read_file(neighborhood_geo_path)
    print(f"Loaded neighborhood geometry: {neighborhood_geo.shape}")
    
    # Check if we need to merge with listings
    if 'neighbourhood_cleansed' in listings.columns and 'neighbourhood' in neighborhood_geo.columns:
        print("Neighborhoods in listings that don't match geometry data:")
        missing_neighborhoods = set(listings['neighbourhood_cleansed']) - set(neighborhood_geo['neighbourhood'])
        print(missing_neighborhoods)
        
        if missing_neighborhoods:
            print(f"WARNING: {len(missing_neighborhoods)} neighborhoods in listings don't match geometry data.")
            # Create a mapping for close matches if needed
            # For now, we'll just note the issue
        
        # Create a column indicating the borough/neighborhood_group
        neighborhood_map = neighborhood_geo.set_index('neighbourhood')['neighbourhood_group'].to_dict()
        listings['borough'] = listings['neighbourhood_cleansed'].map(neighborhood_map)
        print("Added borough information to listings")
        
        # Check for missing values
        null_boroughs = listings['borough'].isnull().sum()
        print(f"Listings with missing borough mapping: {null_boroughs}")
        
except Exception as e:
    print(f"Error processing neighborhood data: {e}")

# Step 7: Process latitude and longitude
print("\nProcessing geospatial data...")

if all(col in listings.columns for col in ['latitude', 'longitude']):
    # Check for missing or invalid coordinates
    invalid_coords = ((listings['latitude'].isnull()) | 
                      (listings['longitude'].isnull()) | 
                      (listings['latitude'] == 0) | 
                      (listings['longitude'] == 0)).sum()
    
    print(f"Listings with missing or invalid coordinates: {invalid_coords}")
    
    if invalid_coords > 0:
        # Remove rows with invalid coordinates
        listings = listings[(listings['latitude'].notnull()) & 
                           (listings['longitude'].notnull()) & 
                           (listings['latitude'] != 0) & 
                           (listings['longitude'] != 0)]
        print(f"Shape after removing invalid coordinates: {listings.shape}")

# Step 8: Create a final cleaned dataset with selected features
print("\nCreating final cleaned dataset...")

# Select the features we want to keep (focusing on features that could be useful for modeling)
selected_features = [
    # Identifiers and target
    'id', 'price',
    
    # Location features
    'neighbourhood_cleansed', 'borough', 'latitude', 'longitude',
    
    # Property attributes
    'property_type_cleaned' if 'property_type_cleaned' in listings.columns else 'property_type',
    'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
    
    # Reservation attributes
    'minimum_nights', 'maximum_nights', 'availability_365',
    
    # Review features
    'number_of_reviews', 'review_score_avg', 'months_since_last_review',
    
    # Host attributes
    'host_years_experience', 'host_is_superhost',
    'calculated_host_listings_count'
]

# Check which selected features actually exist in the DataFrame
available_features = [f for f in selected_features if f in listings.columns]
missing_features = set(selected_features) - set(available_features)

if missing_features:
    print(f"Note: The following selected features are not available: {missing_features}")

# Create the cleaned dataset
cleaned_listings = listings[available_features].copy()

# Check for any remaining missing values
missing_values = cleaned_listings.isnull().sum()
features_with_missing = missing_values[missing_values > 0]

if not features_with_missing.empty:
    print("\nRemaining missing values in cleaned dataset:")
    print(features_with_missing)
    
    # Fill missing values with appropriate methods
    # Numeric: median, Categorical: mode
    numeric_cols = cleaned_listings.select_dtypes(include=['number']).columns
    categorical_cols = cleaned_listings.select_dtypes(include=['object', 'category']).columns
    
    # Fill numeric columns with median
    for col in numeric_cols:
        if cleaned_listings[col].isnull().sum() > 0:
            cleaned_listings[col].fillna(cleaned_listings[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    for col in categorical_cols:
        if cleaned_listings[col].isnull().sum() > 0 and col != 'id':  # Skip id column
            cleaned_listings[col].fillna(cleaned_listings[col].mode()[0], inplace=True)
    
    # Check if all missing values are handled
    still_missing = cleaned_listings.isnull().sum().sum()
    print(f"Missing values after filling: {still_missing}")

# Step 9: Save the cleaned dataset
cleaned_file_path = output_dir / "cleaned_listings.csv"
cleaned_listings.to_csv(cleaned_file_path, index=False)
print(f"Saved cleaned dataset to {cleaned_file_path} with shape {cleaned_listings.shape}")

# Step 10: Provide some basic statistics of the cleaned dataset
print("\nBasic statistics of cleaned dataset:")
print(f"Number of listings: {len(cleaned_listings)}")
print(f"Average price: ${cleaned_listings['price'].mean():.2f}")
print(f"Median price: ${cleaned_listings['price'].median():.2f}")
print(f"Price range: ${cleaned_listings['price'].min():.2f} - ${cleaned_listings['price'].max():.2f}")

if 'neighbourhood_cleansed' in cleaned_listings.columns:
    neighborhood_counts = cleaned_listings['neighbourhood_cleansed'].value_counts()
    print(f"Number of neighborhoods: {len(neighborhood_counts)}")
    print(f"Neighborhood with most listings: {neighborhood_counts.idxmax()} ({neighborhood_counts.max()} listings)")

if 'borough' in cleaned_listings.columns:
    borough_counts = cleaned_listings['borough'].value_counts()
    print(f"Borough distribution:")
    for borough, count in borough_counts.items():
        percentage = (count / len(cleaned_listings)) * 100
        print(f"  {borough}: {count} listings ({percentage:.1f}%)")

# Create a histogram of prices in the cleaned dataset
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_listings['price'], bins=50, kde=True)
plt.title('Price Distribution in Cleaned Dataset')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig(output_dir / 'cleaned_price_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved price distribution plot to {output_dir / 'cleaned_price_distribution.png'}")

print("\nData cleaning and preprocessing completed.") 