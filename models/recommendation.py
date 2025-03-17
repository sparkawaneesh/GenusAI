import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class PropertyRecommendationEngine:
    """
    Recommendation engine for suggesting properties based on user preferences
    and similarity to other properties
    """
    
    def __init__(self, property_data=None):
        self.property_data = property_data
        self.similarity_matrix = None
        self.standardized_features = None
        self.feature_columns = None
    
    def set_property_data(self, property_data):
        """
        Set the property data and reset the recommendation engine
        
        Args:
            property_data (DataFrame): Property listing data
        """
        self.property_data = property_data
        self.similarity_matrix = None
        self.standardized_features = None
    
    def preprocess_data(self):
        """
        Preprocess the property data for recommendation calculations
        
        Returns:
            array: Standardized feature matrix
        """
        if self.property_data is None:
            raise ValueError("Property data not set")
        
        # Select numerical features for similarity calculation
        numerical_features = [
            'bedrooms', 'bathrooms', 'sqft', 'price', 'year_built',
            'lot_size', 'days_on_market'
        ]
        
        # Filter for features that actually exist in the dataframe
        self.feature_columns = [col for col in numerical_features if col in self.property_data.columns]
        
        if not self.feature_columns:
            raise ValueError("No usable numerical features found in the data")
        
        # Extract feature matrix
        feature_matrix = self.property_data[self.feature_columns].copy()
        
        # Handle missing values
        feature_matrix.fillna(feature_matrix.median(), inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        self.standardized_features = scaler.fit_transform(feature_matrix)
        
        return self.standardized_features
    
    def build_similarity_matrix(self):
        """
        Build property similarity matrix using cosine similarity
        
        Returns:
            array: Similarity matrix
        """
        if self.standardized_features is None:
            self.preprocess_data()
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(self.standardized_features)
        
        return self.similarity_matrix
    
    def get_similar_properties(self, property_index, n=5):
        """
        Get similar properties to a given property
        
        Args:
            property_index (int): Index of the reference property
            n (int): Number of similar properties to return
            
        Returns:
            DataFrame: Similar properties
        """
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        # Get similarity scores for the given property
        sim_scores = list(enumerate(self.similarity_matrix[property_index]))
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n similar properties (excluding the property itself)
        top_similar = sim_scores[1:n+1]
        
        # Get indices of similar properties
        similar_indices = [i[0] for i in top_similar]
        similarity_scores = [i[1] for i in top_similar]
        
        # Create a copy of the similar properties with similarity score
        similar_properties = self.property_data.iloc[similar_indices].copy()
        similar_properties['similarity_score'] = similarity_scores
        
        return similar_properties
    
    def recommend_by_preferences(self, preferences, n=5):
        """
        Recommend properties based on user preferences
        
        Args:
            preferences (dict): User preferences (e.g., min_price, max_price, bedrooms)
            n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Recommended properties
        """
        if self.property_data is None:
            raise ValueError("Property data not set")
        
        # Filter properties based on preferences
        filtered_properties = self.property_data.copy()
        
        # Apply filters
        if 'min_price' in preferences and preferences['min_price'] is not None:
            filtered_properties = filtered_properties[filtered_properties['price'] >= preferences['min_price']]
            
        if 'max_price' in preferences and preferences['max_price'] is not None:
            filtered_properties = filtered_properties[filtered_properties['price'] <= preferences['max_price']]
            
        if 'min_bedrooms' in preferences and preferences['min_bedrooms'] is not None:
            filtered_properties = filtered_properties[filtered_properties['bedrooms'] >= preferences['min_bedrooms']]
            
        if 'max_bedrooms' in preferences and preferences['max_bedrooms'] is not None:
            filtered_properties = filtered_properties[filtered_properties['bedrooms'] <= preferences['max_bedrooms']]
            
        if 'min_bathrooms' in preferences and preferences['min_bathrooms'] is not None:
            filtered_properties = filtered_properties[filtered_properties['bathrooms'] >= preferences['min_bathrooms']]
            
        if 'max_bathrooms' in preferences and preferences['max_bathrooms'] is not None:
            filtered_properties = filtered_properties[filtered_properties['bathrooms'] <= preferences['max_bathrooms']]
            
        if 'property_type' in preferences and preferences['property_type'] is not None:
            filtered_properties = filtered_properties[filtered_properties['property_type'] == preferences['property_type']]
            
        if 'neighborhood' in preferences and preferences['neighborhood'] is not None:
            filtered_properties = filtered_properties[filtered_properties['neighborhood'] == preferences['neighborhood']]
            
        if 'city' in preferences and preferences['city'] is not None:
            filtered_properties = filtered_properties[filtered_properties['city'] == preferences['city']]
        
        # Sort by relevance (can be customized based on investment goals)
        if 'investment_goal' in preferences:
            if preferences['investment_goal'] == 'rental_income':
                # Prioritize properties with high rental yield
                if 'gross_rental_yield' in filtered_properties.columns:
                    filtered_properties = filtered_properties.sort_values('gross_rental_yield', ascending=False)
            elif preferences['investment_goal'] == 'appreciation':
                # Prioritize properties in areas with high appreciation
                if 'annual_appreciation_rate' in filtered_properties.columns:
                    filtered_properties = filtered_properties.sort_values('annual_appreciation_rate', ascending=False)
            elif preferences['investment_goal'] == 'balanced':
                # Balance between rental yield and appreciation
                if all(col in filtered_properties.columns for col in ['gross_rental_yield', 'annual_appreciation_rate']):
                    filtered_properties['combined_score'] = (
                        filtered_properties['gross_rental_yield'] * 0.5 + 
                        filtered_properties['annual_appreciation_rate'] * 0.5
                    )
                    filtered_properties = filtered_properties.sort_values('combined_score', ascending=False)
        
        # Return top n recommendations
        return filtered_properties.head(n)
    
    def get_investment_recommendation(self, budget, investment_goal='balanced', risk_tolerance='medium'):
        """
        Get investment recommendations based on budget and goals
        
        Args:
            budget (float): Investment budget
            investment_goal (str): 'rental_income', 'appreciation', or 'balanced'
            risk_tolerance (str): 'low', 'medium', or 'high'
            
        Returns:
            DataFrame: Recommended investment properties
        """
        if self.property_data is None:
            raise ValueError("Property data not set")
        
        # Create a copy for filtering
        filtered_properties = self.property_data.copy()
        
        # Filter by budget (assuming 20% down payment)
        down_payment_percentage = 20
        max_price = budget * (100 / down_payment_percentage)
        filtered_properties = filtered_properties[filtered_properties['price'] <= max_price]
        
        # Calculate investment metrics if not already present
        if 'gross_rental_yield' not in filtered_properties.columns and 'monthly_rent_estimate' in filtered_properties.columns:
            filtered_properties['annual_rent'] = filtered_properties['monthly_rent_estimate'] * 12
            filtered_properties['gross_rental_yield'] = (filtered_properties['annual_rent'] / filtered_properties['price']) * 100
        
        # Scoring based on investment goal
        if investment_goal == 'rental_income':
            if 'gross_rental_yield' in filtered_properties.columns:
                # Higher weight for rental yield
                filtered_properties['investment_score'] = filtered_properties['gross_rental_yield']
        elif investment_goal == 'appreciation':
            if 'annual_appreciation_rate' in filtered_properties.columns:
                # Higher weight for appreciation
                filtered_properties['investment_score'] = filtered_properties['annual_appreciation_rate']
        else:  # balanced
            # Equal weight to rental yield and appreciation
            if all(col in filtered_properties.columns for col in ['gross_rental_yield', 'annual_appreciation_rate']):
                filtered_properties['investment_score'] = (
                    filtered_properties['gross_rental_yield'] * 0.5 + 
                    filtered_properties['annual_appreciation_rate'] * 0.5
                )
            elif 'gross_rental_yield' in filtered_properties.columns:
                filtered_properties['investment_score'] = filtered_properties['gross_rental_yield']
        
        # Risk adjustment
        if risk_tolerance == 'low':
            # Prefer properties in stable neighborhoods with consistent returns
            if 'property_age' in filtered_properties.columns:
                # Older properties in established areas tend to be more stable
                filtered_properties['risk_score'] = 1 / (filtered_properties['property_age'] + 1)
                filtered_properties['investment_score'] = filtered_properties['investment_score'] - filtered_properties['risk_score']
        elif risk_tolerance == 'high':
            # Prefer properties with higher potential returns, even if more volatile
            if 'property_age' in filtered_properties.columns:
                # Newer properties or those in developing areas might have higher appreciation potential
                filtered_properties['risk_score'] = 1 / (filtered_properties['property_age'] + 1)
                filtered_properties['investment_score'] = filtered_properties['investment_score'] + filtered_properties['risk_score']
        
        # Sort by investment score
        if 'investment_score' in filtered_properties.columns:
            filtered_properties = filtered_properties.sort_values('investment_score', ascending=False)
        
        # Return top recommendations
        return filtered_properties.head(5)

def calculate_similarity_score(property1, property2, feature_weights=None):
    """
    Calculate similarity score between two properties
    
    Args:
        property1 (Series): First property
        property2 (Series): Second property
        feature_weights (dict): Weights for different features
        
    Returns:
        float: Similarity score (0-1)
    """
    if feature_weights is None:
        feature_weights = {
            'price': 0.3,
            'bedrooms': 0.1,
            'bathrooms': 0.1,
            'sqft': 0.2,
            'property_type': 0.15,
            'neighborhood': 0.15
        }
    
    # Calculate similarity for each feature
    similarity = 0
    total_weight = 0
    
    for feature, weight in feature_weights.items():
        if feature not in property1 or feature not in property2:
            continue
            
        # Handle different types of features
        if feature in ['property_type', 'neighborhood', 'city']:
            # Categorical features - exact match or no match
            if property1[feature] == property2[feature]:
                similarity += weight
        else:
            # Numerical features - normalized difference
            max_value = max(property1[feature], property2[feature])
            min_value = min(property1[feature], property2[feature])
            
            if max_value == 0:
                feature_similarity = 1  # Both values are 0
            else:
                feature_similarity = 1 - ((max_value - min_value) / max_value)
                
            similarity += feature_similarity * weight
            
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        similarity = similarity / total_weight
    
    return similarity
