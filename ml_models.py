import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from data_processor import prepare_features, split_data

class PropertyValuationModel:
    """
    Machine learning model for property valuation.
    """
    
    def __init__(self):
        """Initialize the property valuation model."""
        self.model = None
        self.feature_names = None
        self.numerical_features = ['bedrooms', 'bathrooms', 'sqft', 'property_age']
        self.categorical_features = ['location', 'property_type']
        self.preprocessor = None
        self.mae = None
        self.mse = None
        self.r2 = None
    
    def train(self, df, target_col='price'):
        """
        Train the property valuation model.
        
        Args:
            df (DataFrame): Processed property data
            target_col (str): Target column for prediction
        
        Returns:
            self: Trained model instance
        """
        # Create a copy of the data
        train_df = df.copy()
        
        # Ensure all required columns exist
        for col in self.numerical_features:
            if col not in train_df.columns:
                if col == 'property_age' and 'year_built' in train_df.columns:
                    current_year = 2023
                    train_df['property_age'] = current_year - train_df['year_built']
                else:
                    raise ValueError(f"Required column {col} not found in the data")
        
        # Prepare numerical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Prepare categorical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Create feature names for transformed data
        self.feature_names = self.numerical_features.copy()
        for cat_feature in self.categorical_features:
            unique_values = train_df[cat_feature].unique()
            for value in unique_values:
                self.feature_names.append(f"{cat_feature}_{value}")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Split data
        available_features = self.numerical_features + self.categorical_features
        X = train_df[available_features]
        y = train_df[target_col]
        
        # Train the model
        pipeline.fit(X, y)
        self.model = pipeline
        
        # Calculate performance metrics
        y_pred = self.model.predict(X)
        self.mae = mean_absolute_error(y, y_pred)
        self.mse = mean_squared_error(y, y_pred)
        self.r2 = r2_score(y, y_pred)
        
        print(f"Model trained. MAE: {self.mae:.2f}, MSE: {self.mse:.2f}, R2: {self.r2:.2f}")
        
        return self
    
    def predict(self, features):
        """
        Predict property value based on features.
        
        Args:
            features (dict): Property features
        
        Returns:
            float: Predicted property value
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert dictionary to DataFrame
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Ensure all required features are present
        for feature in self.numerical_features + self.categorical_features:
            if feature not in features_df.columns:
                if feature == 'property_age' and 'year_built' in features_df.columns:
                    current_year = 2023
                    features_df['property_age'] = current_year - features_df['year_built']
                else:
                    raise ValueError(f"Required feature {feature} not found in input")
        
        # Make prediction
        prediction = self.model.predict(features_df[self.numerical_features + self.categorical_features])
        
        # Return prediction (single value or array)
        if len(prediction) == 1:
            return prediction[0]
        return prediction
    
    def get_confidence_interval(self, features, confidence=0.95):
        """
        Get confidence interval for prediction.
        
        Args:
            features (dict): Property features
            confidence (float): Confidence level (0-1)
        
        Returns:
            tuple: Lower and upper bounds of confidence interval
        """
        # Simple approximation of confidence interval based on MAE
        prediction = self.predict(features)
        z_score = 1.96  # 95% confidence interval
        
        # Adjust z-score for different confidence levels
        if confidence != 0.95:
            import scipy.stats as stats
            z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate bounds using MAE as approximation of standard error
        lower_bound = prediction - z_score * self.mae
        upper_bound = prediction + z_score * self.mae
        
        return (max(0, lower_bound), upper_bound)
    
    def find_similar_properties(self, df, features, n=5):
        """
        Find similar properties in the dataset.
        
        Args:
            df (DataFrame): Property data
            features (dict): Property features to match
            n (int): Number of similar properties to return
        
        Returns:
            DataFrame: Similar properties
        """
        # Convert features to DataFrame if it's a dictionary
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Calculate property_age if year_built exists but property_age doesn't
        if 'year_built' in df.columns and 'property_age' not in df.columns:
            current_year = pd.Timestamp.now().year
            df = df.copy()  # Create a copy to avoid modifying the original
            df['property_age'] = current_year - df['year_built']
        
        if 'year_built' in features_df.columns and 'property_age' not in features_df.columns:
            current_year = pd.Timestamp.now().year
            features_df['property_age'] = current_year - features_df['year_built']
        
        # Get numerical features that exist in both dataframes
        available_numerical_cols = [col for col in self.numerical_features 
                                   if col in df.columns and col in features_df.columns]
        
        if not available_numerical_cols:
            # Fall back to basic features if none of the preferred numerical features are available
            available_numerical_cols = [col for col in df.columns 
                                      if df[col].dtype in ['int64', 'float64'] 
                                      and col in features_df.columns][:3]  # Use the first 3 numeric columns
            
        if not available_numerical_cols:
            # If still no matching numerical columns, return random properties
            return df.sample(min(n, len(df)))
            
        # Scale numerical features
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[available_numerical_cols]),
            columns=available_numerical_cols,
            index=df.index
        )
        
        # Scale input features
        features_scaled = pd.DataFrame(
            scaler.transform(features_df[available_numerical_cols]),
            columns=available_numerical_cols
        )
        
        # Calculate Euclidean distance
        distances = np.sqrt(((df_scaled - features_scaled.iloc[0])**2).sum(axis=1))
        
        # Get indices of n most similar properties
        similar_indices = distances.argsort()[:n]
        
        # Return similar properties
        return df.iloc[similar_indices]
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
            dict: Feature names and their importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get feature importance from the model
        importance = self.model.named_steps['model'].feature_importances_
        
        # Get feature names from the preprocessor
        preprocessor = self.model.named_steps['preprocessor']
        feature_names = []
        
        # Get numerical feature names
        for name in self.numerical_features:
            feature_names.append(name)
        
        # Get one-hot encoded categorical feature names
        categorical_features_idx = preprocessor.transformers_[1][2]
        ohe = preprocessor.transformers_[1][1].named_steps['onehot']
        categorical_names = [f"{self.categorical_features[i]}_{val}" for i, vals in enumerate(ohe.categories_) for val in vals]
        feature_names.extend(categorical_names)
        
        # Create a dictionary of feature importance
        feature_importance = {}
        for name, imp in zip(feature_names, importance):
            feature_importance[name] = imp
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

class RentalPriceModel:
    """
    Machine learning model for rental price prediction.
    """
    
    def __init__(self):
        """Initialize the rental price model."""
        self.model = None
        self.feature_names = None
        self.numerical_features = ['bedrooms', 'bathrooms', 'sqft', 'property_age']
        self.categorical_features = ['location', 'property_type']
        self.preprocessor = None
    
    def train(self, df, target_col='estimated_monthly_rent'):
        """
        Train the rental price model.
        
        Args:
            df (DataFrame): Processed property data
            target_col (str): Target column for prediction
        
        Returns:
            self: Trained model instance
        """
        # Similar implementation to PropertyValuationModel, but with rental price as target
        # Create a copy of the data
        train_df = df.copy()
        
        # Prepare numerical features
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Prepare categorical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ])
        
        # Split data
        available_features = self.numerical_features + self.categorical_features
        X = train_df[available_features]
        y = train_df[target_col]
        
        # Train the model
        pipeline.fit(X, y)
        self.model = pipeline
        
        # Calculate performance metrics
        y_pred = self.model.predict(X)
        self.mae = mean_absolute_error(y, y_pred)
        self.mse = mean_squared_error(y, y_pred)
        self.r2 = r2_score(y, y_pred)
        
        print(f"Rental model trained. MAE: {self.mae:.2f}, MSE: {self.mse:.2f}, R2: {self.r2:.2f}")
        
        return self
    
    def predict(self, features):
        """
        Predict rental price based on features.
        
        Args:
            features (dict): Property features
        
        Returns:
            float: Predicted rental price
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert dictionary to DataFrame
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Make prediction
        prediction = self.model.predict(features_df[self.numerical_features + self.categorical_features])
        
        # Return prediction (single value or array)
        if len(prediction) == 1:
            return prediction[0]
        return prediction

class MarketTrendModel:
    """
    Model for market trend prediction and analysis.
    """
    
    def __init__(self):
        """Initialize the market trend model."""
        self.model = None
        self.location_trends = {}
    
    def train(self, df, time_periods=12):
        """
        Analyze market trends by location.
        
        Args:
            df (DataFrame): Property data with historical price information
            time_periods (int): Number of time periods to analyze
        
        Returns:
            self: Trained model instance
        """
        # Analyze price trends by location
        locations = df['location'].unique()
        
        for location in locations:
            location_data = df[df['location'] == location]
            
            # Calculate average price and growth rate
            avg_price = location_data['price'].mean()
            
            # If there's temporal data, calculate trend
            if 'date_listed' in location_data.columns:
                location_data = location_data.sort_values('date_listed')
                # Calculate price change over time
                location_data['price_pct_change'] = location_data['price'].pct_change()
                growth_rate = location_data['price_pct_change'].mean()
            else:
                # Generate synthetic growth rate based on property age and price
                # Newer properties tend to be more expensive
                growth_rate = 0.03  # Default 3% annual growth
            
            # Store location trend data
            self.location_trends[location] = {
                'avg_price': avg_price,
                'growth_rate': growth_rate,
                'forecast': [avg_price * (1 + growth_rate)**i for i in range(1, time_periods+1)]
            }
        
        return self
    
    def predict_trend(self, location, periods=12):
        """
        Predict market trend for a location.
        
        Args:
            location (str): Location to predict trend for
            periods (int): Number of future periods to predict
        
        Returns:
            list: Predicted prices for future periods
        """
        if not self.location_trends:
            raise ValueError("Model not trained. Call train() first.")
        
        if location not in self.location_trends:
            raise ValueError(f"Location {location} not found in training data")
        
        # Get trend data for location
        trend_data = self.location_trends[location]
        
        # Predict future prices
        current_price = trend_data['avg_price']
        growth_rate = trend_data['growth_rate']
        
        # Generate forecast
        forecast = [current_price * (1 + growth_rate)**i for i in range(1, periods+1)]
        
        return forecast
    
    def get_growth_rates(self):
        """
        Get growth rates for all locations.
        
        Returns:
            dict: Location growth rates
        """
        if not self.location_trends:
            raise ValueError("Model not trained. Call train() first.")
        
        return {location: data['growth_rate'] for location, data in self.location_trends.items()}
    
    def get_hotspots(self, threshold=0.05):
        """
        Identify market hotspots (locations with high growth).
        
        Args:
            threshold (float): Growth rate threshold for hotspots
        
        Returns:
            list: Hotspot locations
        """
        if not self.location_trends:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get locations with growth rate above threshold
        hotspots = [
            location for location, data in self.location_trends.items()
            if data['growth_rate'] > threshold
        ]
        
        return sorted(hotspots, key=lambda x: self.location_trends[x]['growth_rate'], reverse=True)
