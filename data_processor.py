import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data.sample_data import generate_property_data

def load_data():
    """
    Load real estate data from source.
    If no external data is available, use generated sample data.
    """
    try:
        # Try to load data from an external source (this would be your actual data source)
        # For demonstration, we'll use generated sample data
        return generate_property_data()
    except Exception as e:
        print(f"Error loading external data: {e}")
        print("Using generated sample data instead.")
        return generate_property_data()

def preprocess_data(df):
    """
    Preprocess and clean the real estate data.
    
    Args:
        df (DataFrame): Raw data
    
    Returns:
        DataFrame: Processed data
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    if processed_df['price'].isna().sum() > 0:
        # For missing prices, impute with median price for that location and property type
        processed_df['price'] = processed_df.groupby(['location', 'property_type'])['price'].transform(
            lambda x: x.fillna(x.median())
        )
        
    # Fill remaining missing prices with overall median
    processed_df['price'] = processed_df['price'].fillna(processed_df['price'].median())
    
    # Handle other missing numerical values
    for col in ['bedrooms', 'bathrooms', 'sqft', 'year_built']:
        if col in processed_df.columns and processed_df[col].isna().sum() > 0:
            processed_df[col] = processed_df.groupby('property_type')[col].transform(
                lambda x: x.fillna(x.median())
            )
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # Calculate price per square foot
    if 'sqft' in processed_df.columns:
        processed_df['price_per_sqft'] = processed_df['price'] / processed_df['sqft']
    
    # Calculate property age
    if 'year_built' in processed_df.columns:
        current_year = pd.Timestamp.now().year
        processed_df['property_age'] = current_year - processed_df['year_built']
    
    # Add estimated monthly rent if not present (for investment calculations)
    if 'estimated_monthly_rent' not in processed_df.columns:
        # Estimate monthly rent as 0.8% of property value for properties < $500K
        # and 0.6% for properties >= $500K
        rent_factor = processed_df['price'].apply(lambda x: 0.008 if x < 500000 else 0.006)
        processed_df['estimated_monthly_rent'] = processed_df['price'] * rent_factor
    
    # Add estimated ROI if not present
    if 'estimated_roi' not in processed_df.columns:
        # Simple ROI calculation based on rent and price
        annual_rent = processed_df['estimated_monthly_rent'] * 12
        annual_expenses = processed_df['price'] * 0.02  # Assume 2% annual expenses (taxes, maintenance, etc.)
        processed_df['estimated_roi'] = (annual_rent - annual_expenses) / processed_df['price']
    
    # One-hot encode categorical variables for ML models
    #categorical_cols = ['property_type', 'location']
    #for col in categorical_cols:
    #    if col in processed_df.columns:
    #        dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
    #        processed_df = pd.concat([processed_df, dummies], axis=1)
    
    return processed_df

def prepare_features(df, target_col='price', feature_cols=None):
    """
    Prepare features for machine learning models.
    
    Args:
        df (DataFrame): Processed data
        target_col (str): Target column for prediction
        feature_cols (list): List of feature columns to use
    
    Returns:
        tuple: X (features), y (target), feature_names
    """
    # Default feature columns if not specified
    if feature_cols is None:
        feature_cols = ['bedrooms', 'bathrooms', 'sqft', 'property_age', 
                        'location', 'property_type']
    
    processed_df = df.copy()
    
    # Handle categorical features
    categorical_cols = [col for col in feature_cols if processed_df[col].dtype == 'object']
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    
    # Create encoded dataframe
    X_numeric = processed_df[numerical_cols].copy()
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(processed_df[categorical_cols], drop_first=True)
    
    # Combine encoded features
    X = pd.concat([X_numeric, X_encoded], axis=1)
    
    # Get target
    y = processed_df[target_col]
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, X.columns.tolist()

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (DataFrame): Features
        y (Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
