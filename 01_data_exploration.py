#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Airbnb Rental Price Prediction and Community Feature Analysis - Data Exploration
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

print("Loading Airbnb data...")

# Load listings data
listings_path = Path("data/Airbnb/listings.csv.gz")
listings = pd.read_csv(listings_path, compression='gzip')

# Display basic information
print(f"\n=== Airbnb Listings Data Basic Information ===")
print(f"Data shape: {listings.shape}")
print(f"Columns: {', '.join(listings.columns)}")
print("\nFirst 5 rows:")
print(listings.head())

# Display basic statistics
print("\nData types:")
print(listings.dtypes)

print("\nNumeric column statistics:")
print(listings.describe())

# Check for missing values
print("\nMissing value statistics:")
missing_values = listings.isnull().sum()
missing_percent = (missing_values / len(listings)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage(%)': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percentage(%)', ascending=False))

# Price analysis
# First process the price column, remove $ and comma, and convert to float
if 'price' in listings.columns:
    # Check if price column is string type
    if listings['price'].dtype == 'object':
        listings['price'] = listings['price'].replace(r'[\$,]', '', regex=True).astype(float)
    
    print("\nPrice distribution:")
    print(listings['price'].describe())
    
    # Plot price distribution histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(listings['price'][listings['price'] < listings['price'].quantile(0.95)], 
                 bins=50, kde=True)
    plt.title('Airbnb Listing Price Distribution (excluding top 5% extreme values)')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
    print("Price distribution plot saved to price_distribution.png")

# Location analysis - visualize listing distributions on map
if all(col in listings.columns for col in ['latitude', 'longitude']):
    plt.figure(figsize=(12, 10))
    plt.scatter(listings['longitude'], listings['latitude'], alpha=0.3, s=10)
    plt.title('Airbnb Listings Geographical Distribution')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('listings_geographic_distribution.png', dpi=300, bbox_inches='tight')
    print("Geographic distribution plot saved to listings_geographic_distribution.png")

# Load neighbourhoods data
print("\nLoading neighborhood data...")
try:
    neighborhoods_path = Path("data/Airbnb/neighbourhoods.csv")
    neighborhoods = pd.read_csv(neighborhoods_path)
    print("\n=== Neighbourhoods Data Basic Information ===")
    print(f"Data shape: {neighborhoods.shape}")
    print(f"Columns: {', '.join(neighborhoods.columns)}")
    print(neighborhoods.head())
    
    # Load geographic boundary data
    neighborhoods_geo_path = Path("data/Airbnb/neighbourhoods.geojson")
    neighborhoods_geo = gpd.read_file(neighborhoods_geo_path)
    print("\n=== Neighbourhoods Geographic Data Information ===")
    print(f"Data shape: {neighborhoods_geo.shape}")
    print(neighborhoods_geo.head())
    
    # Simple map visualization
    plt.figure(figsize=(12, 10))
    neighborhoods_geo.plot(figsize=(12, 10), color='lightgrey', edgecolor='black')
    plt.title('NYC Neighborhoods Map')
    plt.savefig('nyc_neighborhoods.png', dpi=300, bbox_inches='tight')
    print("Neighborhood map saved to nyc_neighborhoods.png")
    
    # If we have both neighborhood data and listings with neighborhood info
    if 'neighbourhood_cleansed' in listings.columns and 'neighbourhood' in neighborhoods_geo.columns:
        # Count listings by neighborhood
        neighborhood_counts = listings['neighbourhood_cleansed'].value_counts()
        
        # Join with geo data
        neighborhood_geo_counts = neighborhoods_geo.merge(
            neighborhood_counts.reset_index(), 
            how='left',
            left_on='neighbourhood', 
            right_on='neighbourhood_cleansed'
        )
        
        # Fill NaN values with 0
        neighborhood_geo_counts['count'] = neighborhood_geo_counts[0].fillna(0)
        
        # Plot choropleth map
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        neighborhood_geo_counts.plot(
            column='count',
            ax=ax,
            legend=True,
            cmap='YlOrRd',
            legend_kwds={'label': "Number of Listings"}
        )
        plt.title('Number of Airbnb Listings by Neighborhood')
        plt.savefig('listings_by_neighborhood.png', dpi=300, bbox_inches='tight')
        print("Listings by neighborhood map saved to listings_by_neighborhood.png")
    
    # Count by neighborhood
    if 'neighbourhood_cleansed' in listings.columns:
        neighborhood_counts = listings['neighbourhood_cleansed'].value_counts()
        print("\nListing counts by neighborhood (top 10):")
        print(neighborhood_counts.head(10))
        
        # Plot bar chart of top 10 neighborhoods by listing count
        plt.figure(figsize=(12, 6))
        neighborhood_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Neighborhoods by Number of Airbnb Listings')
        plt.xlabel('Neighborhood')
        plt.ylabel('Number of Listings')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('neighborhood_listing_counts.png', dpi=300, bbox_inches='tight')
        print("Neighborhood listing counts plot saved to neighborhood_listing_counts.png")
        
        # Analyze average price by neighborhood (for top 20 neighborhoods by listing count)
        if 'price' in listings.columns:
            # Group by neighborhood and calculate mean price
            neighborhood_price = listings.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False)
            print("\nAverage price by neighborhood (top 10 most expensive):")
            print(neighborhood_price.head(10))
            
            # Plot top 10 most expensive neighborhoods
            plt.figure(figsize=(12, 6))
            neighborhood_price.head(10).plot(kind='bar')
            plt.title('Top 10 Most Expensive Neighborhoods (Average Listing Price)')
            plt.xlabel('Neighborhood')
            plt.ylabel('Average Price (USD)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('expensive_neighborhoods.png', dpi=300, bbox_inches='tight')
            print("Most expensive neighborhoods plot saved to expensive_neighborhoods.png")
            
            # Plot top 10 neighborhoods by listing count with their average prices
            top10_neighborhoods = neighborhood_counts.head(10).index
            avg_prices = listings[listings['neighbourhood_cleansed'].isin(top10_neighborhoods)]\
                        .groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 6))
            avg_prices.plot(kind='bar')
            plt.title('Average Prices in Top 10 Most Popular Neighborhoods')
            plt.xlabel('Neighborhood')
            plt.ylabel('Average Price (USD)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('popular_neighborhood_prices.png', dpi=300, bbox_inches='tight')
            print("Popular neighborhood prices plot saved to popular_neighborhood_prices.png")
        
except Exception as e:
    print(f"Error loading neighborhood data: {e}")

# View listing type distribution
if 'room_type' in listings.columns:
    room_type_counts = listings['room_type'].value_counts()
    print("\nListing type distribution:")
    print(room_type_counts)
    
    # Plot pie chart
    plt.figure(figsize=(10, 6))
    room_type_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Listing Type Distribution')
    plt.ylabel('')
    plt.savefig('room_type_distribution.png', dpi=300, bbox_inches='tight')
    print("Room type distribution plot saved to room_type_distribution.png")
    
    # Average price by room type
    if 'price' in listings.columns:
        room_price = listings.groupby('room_type')['price'].mean().sort_values(ascending=False)
        print("\nAverage price by room type:")
        print(room_price)
        
        plt.figure(figsize=(10, 6))
        room_price.plot(kind='bar')
        plt.title('Average Price by Room Type')
        plt.xlabel('Room Type')
        plt.ylabel('Average Price (USD)')
        plt.tight_layout()
        plt.savefig('room_type_prices.png', dpi=300, bbox_inches='tight')
        print("Room type prices plot saved to room_type_prices.png")

print("\nExploratory analysis completed.")

# Quick look at Demographics data
try:
    print("\nLoading Demographics data...")
    demographics_path = Path("data/Demographics/2020_Neighborhood_Tabulation_Areas__NTAs__20250415.csv")
    demographics = pd.read_csv(demographics_path)
    print("\n=== Demographics Data Basic Information ===")
    print(f"Data shape: {demographics.shape}")
    print(f"Columns: {', '.join(demographics.columns)}")
    print(demographics.head())
except Exception as e:
    print(f"Error loading Demographics data: {e}")

# Quick look at Transit data
try:
    print("\nLoading Transit data...")
    transit_path = Path("data/Transit/MTA_Subway_Stations_20250415.csv")
    transit = pd.read_csv(transit_path)
    print("\n=== Transit Data Basic Information ===")
    print(f"Data shape: {transit.shape}")
    print(f"Columns: {', '.join(transit.columns)}")
    print(transit.head())
except Exception as e:
    print(f"Error loading Transit data: {e}")

# Quick look at Crime data (only read first 1000 rows because file may be large)
try:
    print("\nLoading Crime data (partial)...")
    crime_path = Path("data/Crime/NYPD_Complaint_Data_Historic_20250415.csv")
    crime = pd.read_csv(crime_path, nrows=1000)  # Only read first 1000 rows for quick view
    print("\n=== Crime Data Basic Information (first 1000 rows only) ===")
    print(f"Data shape: {crime.shape}")
    print(f"Columns: {', '.join(crime.columns)}")
    print(crime.head())
except Exception as e:
    print(f"Error loading Crime data: {e}")

if __name__ == "__main__":
    print("Data exploration completed. Please check the generated plots and printed information.") 