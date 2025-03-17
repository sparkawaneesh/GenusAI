import pandas as pd
import numpy as np
import os
import streamlit as st

# Cache the data loading to improve performance
@st.cache_data
def load_real_estate_data():
    """
    Load real estate dataset for the application
    
    Returns:
        DataFrame: A pandas DataFrame containing real estate data
    """
    try:
        # Load data from local file
        data_path = os.path.join('data', 'sample_real_estate_data.csv')
        df = pd.read_csv(data_path)
        
        # Perform basic data cleaning
        df = clean_data(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return an empty dataframe with expected columns
        return pd.DataFrame({
            'property_id': [], 
            'price': [], 
            'bedrooms': [], 
            'bathrooms': [], 
            'sqft': [], 
            'year_built': [], 
            'lot_size': [], 
            'property_type': [], 
            'neighborhood': [], 
            'city': [], 
            'state': [], 
            'zip_code': [], 
            'latitude': [], 
            'longitude': [], 
            'days_on_market': [],
            'last_sold_date': [],
            'last_sold_price': [],
            'monthly_rent_estimate': []
        })

def clean_data(df):
    """
    Clean and preprocess the raw real estate data
    
    Args:
        df (DataFrame): Raw pandas DataFrame
        
    Returns:
        DataFrame: Cleaned pandas DataFrame
    """
    # Make a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Convert price columns to numeric
    for col in ['price', 'last_sold_price', 'monthly_rent_estimate']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Handle missing values
    for col in ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'lot_size']:
        if col in df_clean.columns:
            # Fill missing values with column median
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Convert date columns to datetime
    for col in ['last_sold_date']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    return df_clean

def get_available_cities(df):
    """
    Get a list of available cities in the dataset
    
    Args:
        df (DataFrame): Real estate DataFrame
        
    Returns:
        list: List of city names
    """
    if 'city' in df.columns:
        return sorted(df['city'].unique().tolist())
    return []

def get_available_property_types(df):
    """
    Get a list of available property types in the dataset
    
    Args:
        df (DataFrame): Real estate DataFrame
        
    Returns:
        list: List of property types
    """
    if 'property_type' in df.columns:
        return sorted(df['property_type'].unique().tolist())
    return []

def get_properties_by_filter(df, property_type=None, min_price=None, max_price=None, 
                            min_beds=None, max_beds=None, min_baths=None, 
                            max_baths=None, city=None):
    """
    Filter properties based on input criteria
    
    Args:
        df (DataFrame): Real estate DataFrame
        property_type (str, optional): Type of property
        min_price (float, optional): Minimum price
        max_price (float, optional): Maximum price
        min_beds (int, optional): Minimum number of bedrooms
        max_beds (int, optional): Maximum number of bedrooms
        min_baths (int, optional): Minimum number of bathrooms
        max_baths (int, optional): Maximum number of bathrooms
        city (str, optional): City name
        
    Returns:
        DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Apply filters if they exist
    if property_type and 'property_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['property_type'] == property_type]
        
    if min_price is not None and 'price' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['price'] >= min_price]
        
    if max_price is not None and 'price' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['price'] <= max_price]
        
    if min_beds is not None and 'bedrooms' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['bedrooms'] >= min_beds]
        
    if max_beds is not None and 'bedrooms' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['bedrooms'] <= max_beds]
        
    if min_baths is not None and 'bathrooms' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['bathrooms'] >= min_baths]
        
    if max_baths is not None and 'bathrooms' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['bathrooms'] <= max_baths]
        
    if city and 'city' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['city'] == city]
        
    return filtered_df
