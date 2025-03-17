import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st

@st.cache_data
def preprocess_data_for_ml(df):
    """
    Preprocess the real estate data for machine learning
    
    Args:
        df (DataFrame): Raw real estate data
        
    Returns:
        tuple: (X_processed, y, preprocessing_pipeline)
    """
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Identify numerical and categorical features
    numerical_features = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'lot_size', 
                         'latitude', 'longitude', 'days_on_market']
    categorical_features = ['property_type', 'neighborhood', 'city', 'state', 'zip_code']
    
    # Only include columns that are actually in the dataframe
    numerical_features = [f for f in numerical_features if f in data.columns]
    categorical_features = [f for f in categorical_features if f in data.columns]
    
    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Setup the target variable
    if 'price' in data.columns:
        y = data['price']
        X = data.drop('price', axis=1)
    else:
        # If price not available, just create X (no target)
        y = None
        X = data
    
    # Create a list of features that are actually used
    used_features = numerical_features + categorical_features
    X = X[used_features].copy()
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor, used_features

def preprocess_new_data(preprocessing_pipeline, data, used_features):
    """
    Preprocess new data using a fitted preprocessing pipeline
    
    Args:
        preprocessing_pipeline: Fitted preprocessing pipeline
        data (DataFrame): New data to transform
        used_features (list): List of features used in the preprocessing
        
    Returns:
        array: Transformed data ready for model prediction
    """
    # Ensure only used features are included
    data_subset = data[used_features].copy()
    
    # Transform the data
    processed_data = preprocessing_pipeline.transform(data_subset)
    
    return processed_data

def calculate_property_metrics(df):
    """
    Calculate additional property metrics useful for analysis
    
    Args:
        df (DataFrame): Real estate DataFrame
        
    Returns:
        DataFrame: DataFrame with additional calculated metrics
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate price per square foot
    if all(col in result_df.columns for col in ['price', 'sqft']):
        result_df['price_per_sqft'] = result_df['price'] / result_df['sqft']
    
    # Calculate property age
    if 'year_built' in result_df.columns:
        current_year = pd.Timestamp.now().year
        result_df['property_age'] = current_year - result_df['year_built']
    
    # Calculate price appreciation (if sold before)
    if all(col in result_df.columns for col in ['price', 'last_sold_price']):
        # Calculate appreciation percentage
        result_df['price_appreciation'] = ((result_df['price'] - result_df['last_sold_price']) / 
                                          result_df['last_sold_price']) * 100
        
        # Calculate annual appreciation rate (if last_sold_date is available)
        if 'last_sold_date' in result_df.columns:
            result_df['last_sold_date'] = pd.to_datetime(result_df['last_sold_date'])
            current_date = pd.Timestamp.now()
            
            # Years since last sale
            result_df['years_since_last_sale'] = ((current_date - result_df['last_sold_date']).dt.days / 365.25)
            
            # Calculate annual appreciation rate
            result_df['annual_appreciation_rate'] = result_df['price_appreciation'] / result_df['years_since_last_sale']
    
    # Calculate potential rental yield
    if all(col in result_df.columns for col in ['price', 'monthly_rent_estimate']):
        # Annual rental income
        result_df['annual_rental_income'] = result_df['monthly_rent_estimate'] * 12
        
        # Gross rental yield
        result_df['gross_rental_yield'] = (result_df['annual_rental_income'] / result_df['price']) * 100
    
    return result_df
