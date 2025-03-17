import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import streamlit as st
from utils.preprocessing import preprocess_data_for_ml, preprocess_new_data

class PropertyValuationModel:
    """Property Valuation Model using machine learning algorithms"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.used_features = None
        self.trained = False
        self.metrics = {}
        
    def train(self, df, test_size=0.2, random_state=42, model_type='xgboost'):
        """
        Train the property valuation model
        
        Args:
            df (DataFrame): Real estate data
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            model_type (str): Type of model to use ('rf', 'gb', or 'xgboost')
        
        Returns:
            dict: Model performance metrics
        """
        # Preprocess data
        X_processed, y, self.preprocessor, self.used_features = preprocess_data_for_ml(df)
        
        if y is None:
            st.error("Cannot train model: target variable (price) not found in dataset")
            return None
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )
        
        # Select and train the model
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_type == 'gb':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        else:  # default to xgboost
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.trained = True
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return self.metrics
    
    def predict(self, property_data):
        """
        Predict the value of a property
        
        Args:
            property_data (DataFrame): Property features
            
        Returns:
            float: Predicted property value
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Preprocess the input data
        processed_data = preprocess_new_data(
            self.preprocessor, property_data, self.used_features
        )
        
        # Make prediction
        prediction = self.model.predict(processed_data)
        
        return prediction[0]
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame: Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # For RF and GB models
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names (after preprocessing)
            feature_names = []
            
            # For numerical features (they maintain their original names)
            for feature in self.used_features:
                if feature in ['bedrooms', 'bathrooms', 'sqft', 'year_built', 
                               'lot_size', 'latitude', 'longitude', 'days_on_market']:
                    feature_names.append(feature)
            
            # Get the feature importance
            importance = self.model.feature_importances_
            
            # Create a dataframe with feature names and importance scores
            # Note: This is simplified and might not align perfectly with transformed features
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance[:len(feature_names)]  # May need adjustment for actual output shape
            })
            
            return imp_df.sort_values('Importance', ascending=False)
        
        return pd.DataFrame(columns=['Feature', 'Importance'])

def valuate_property_with_adjustments(base_value, adjustments):
    """
    Adjust property valuation based on specific factors
    
    Args:
        base_value (float): Base property value from model
        adjustments (dict): Dictionary of adjustment factors
        
    Returns:
        float: Adjusted property value
    """
    adjusted_value = base_value
    
    # Apply percentage adjustments
    for factor, percentage in adjustments.items():
        adjustment = base_value * (percentage / 100)
        adjusted_value += adjustment
    
    return adjusted_value

def estimate_renovation_impact(property_value, renovation_type):
    """
    Estimate the impact of renovations on property value
    
    Args:
        property_value (float): Current property value
        renovation_type (str): Type of renovation
        
    Returns:
        dict: Expected value increase and ROI
    """
    # Estimated ROI for different renovation types
    renovation_roi = {
        'kitchen': {'min': 5, 'max': 15, 'cost_percentage': 3},
        'bathroom': {'min': 3, 'max': 10, 'cost_percentage': 2},
        'exterior': {'min': 2, 'max': 5, 'cost_percentage': 1},
        'basement': {'min': 4, 'max': 8, 'cost_percentage': 2.5},
        'addition': {'min': 6, 'max': 12, 'cost_percentage': 7},
        'energy_efficiency': {'min': 1, 'max': 3, 'cost_percentage': 1.5},
        'roof': {'min': 1, 'max': 3, 'cost_percentage': 2},
        'landscaping': {'min': 1, 'max': 2, 'cost_percentage': 0.5}
    }
    
    if renovation_type not in renovation_roi:
        return {
            'value_increase_percentage': 0,
            'value_increase': 0,
            'renovation_cost': 0,
            'roi': 0
        }
    
    # Get the ROI range for the renovation type
    roi_data = renovation_roi[renovation_type]
    
    # Calculate value increase (random point in the range)
    value_increase_percentage = np.random.uniform(roi_data['min'], roi_data['max'])
    value_increase = property_value * (value_increase_percentage / 100)
    
    # Calculate renovation cost
    renovation_cost = property_value * (roi_data['cost_percentage'] / 100)
    
    # Calculate ROI
    roi = (value_increase / renovation_cost) * 100
    
    return {
        'value_increase_percentage': value_increase_percentage,
        'value_increase': value_increase,
        'renovation_cost': renovation_cost,
        'roi': roi
    }
