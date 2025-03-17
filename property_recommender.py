import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def recommend_properties(df, user_profile, n=10):
    """
    Recommend properties based on user profile and preferences.
    
    Args:
        df (DataFrame): Property data
        user_profile (dict): User profile with preferences
        n (int): Number of recommendations to return
    
    Returns:
        DataFrame: Recommended properties
    """
    # Create a copy of the data
    data = df.copy()
    
    # Extract user preferences
    goal = user_profile.get('goal', 'balanced')  # 'cash_flow', 'appreciation', or 'balanced'
    risk_tolerance = user_profile.get('risk_tolerance', 'medium')  # 'low', 'medium', or 'high'
    investment_horizon = user_profile.get('investment_horizon', 'Medium-term (4-7 years)')
    budget_min = user_profile.get('budget_min', 0)
    budget_max = user_profile.get('budget_max', float('inf'))
    preferred_locations = user_profile.get('preferred_locations', [])
    
    # Filter by budget
    filtered_data = data[(data['price'] >= budget_min) & (data['price'] <= budget_max)]
    
    # Filter by location if preferred locations are specified
    if preferred_locations:
        filtered_data = filtered_data[filtered_data['location'].isin(preferred_locations)]
    
    # If no properties match the filters, return empty DataFrame
    if filtered_data.empty:
        return pd.DataFrame()
    
    # Calculate match scores based on user preferences
    # The scoring logic will depend on the user's investment goal and preferences
    
    # Convert risk tolerance to numeric value
    risk_scores = {'Low': 0.25, 'Medium': 0.5, 'High': 0.75}
    risk_score = risk_scores.get(risk_tolerance, 0.5)
    
    # Convert investment horizon to numeric value
    horizon_mapping = {
        'Short-term (1-3 years)': 2,
        'Medium-term (4-7 years)': 5,
        'Long-term (8+ years)': 10
    }
    investment_years = horizon_mapping.get(investment_horizon, 5)
    
    # Calculate different investment scores
    filtered_data['cash_flow_score'] = calculate_cash_flow_score(filtered_data)
    filtered_data['appreciation_score'] = calculate_appreciation_score(filtered_data, investment_years)
    filtered_data['risk_score'] = calculate_risk_score(filtered_data)
    
    # Calculate overall match score based on user's goal
    if goal == 'Cash Flow':
        filtered_data['match_score'] = (
            filtered_data['cash_flow_score'] * 0.6 +
            filtered_data['appreciation_score'] * 0.2 +
            (1 - abs(filtered_data['risk_score'] - risk_score)) * 0.2
        )
    elif goal == 'Appreciation':
        filtered_data['match_score'] = (
            filtered_data['cash_flow_score'] * 0.2 +
            filtered_data['appreciation_score'] * 0.6 +
            (1 - abs(filtered_data['risk_score'] - risk_score)) * 0.2
        )
    else:  # Balanced
        filtered_data['match_score'] = (
            filtered_data['cash_flow_score'] * 0.4 +
            filtered_data['appreciation_score'] * 0.4 +
            (1 - abs(filtered_data['risk_score'] - risk_score)) * 0.2
        )
    
    # Sort by match score and return top n recommendations
    recommendations = filtered_data.sort_values('match_score', ascending=False).head(n)
    
    return recommendations

def calculate_cash_flow_score(df):
    """
    Calculate cash flow score for properties.
    
    Args:
        df (DataFrame): Property data
    
    Returns:
        Series: Cash flow scores
    """
    # Calculate estimated monthly rent if not present
    if 'estimated_monthly_rent' not in df.columns:
        monthly_rent = df['price'] * 0.008  # Estimate as 0.8% of property value
    else:
        monthly_rent = df['estimated_monthly_rent']
    
    # Calculate estimated monthly expenses
    # Assume 20% down payment, 4.5% interest rate, 30-year loan
    loan_amount = df['price'] * 0.8
    monthly_rate = 0.045 / 12
    num_payments = 30 * 12
    monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    
    # Other monthly expenses (property tax, insurance, maintenance, vacancy)
    other_expenses = df['price'] * 0.02 / 12
    
    # Calculate monthly cash flow
    monthly_cash_flow = monthly_rent - monthly_mortgage - other_expenses
    
    # Calculate cash on cash return
    down_payment = df['price'] * 0.2
    closing_costs = df['price'] * 0.03
    total_investment = down_payment + closing_costs
    annual_cash_flow = monthly_cash_flow * 12
    cash_on_cash_return = annual_cash_flow / total_investment
    
    # Normalize cash on cash return to a 0-1 score
    # Assume 10% cash on cash return is excellent (score = 1)
    # and 0% or negative is poor (score = 0)
    cash_flow_score = cash_on_cash_return / 0.1
    cash_flow_score = cash_flow_score.clip(0, 1)
    
    return cash_flow_score

def calculate_appreciation_score(df, investment_years):
    """
    Calculate appreciation score for properties.
    
    Args:
        df (DataFrame): Property data
        investment_years (int): Investment horizon in years
    
    Returns:
        Series: Appreciation scores
    """
    # If we have historical appreciation data by location, use that
    # Otherwise, estimate based on location and property characteristics
    
    # Synthetic approach: estimate appreciation potential based on:
    # 1. Price relative to area average (lower = better potential)
    # 2. Location growth rate (estimated)
    # 3. Property age (newer = better)
    
    # Calculate price relative to area average
    df['area_avg_price'] = df.groupby('location')['price'].transform('mean')
    price_ratio = df['price'] / df['area_avg_price']
    price_potential = 1 - price_ratio.clip(0.5, 1.5) / 1.5
    
    # Estimate location growth rate
    # In a real application, this would come from historical data
    location_growth = df.groupby('location')['price'].transform(
        lambda x: 0.03 + np.random.uniform(-0.02, 0.04)  # Random growth between 1% and 7%
    )
    
    # Calculate property age factor if available
    if 'property_age' in df.columns:
        age_factor = 1 - df['property_age'] / 100
        age_factor = age_factor.clip(0.5, 1)
    else:
        age_factor = 0.8  # Default if age unknown
    
    # Combine factors to estimate annual appreciation rate
    annual_appreciation = 0.03 + (price_potential * 0.02) + (location_growth - 0.03) * 0.5
    annual_appreciation = annual_appreciation * age_factor
    
    # Calculate total appreciation over investment period
    total_appreciation = (1 + annual_appreciation) ** investment_years - 1
    
    # Normalize to 0-1 score
    # Assume doubling in value over investment period is excellent (score = 1)
    max_appreciation = (1 + 0.07) ** investment_years - 1  # Max at 7% annual growth
    appreciation_score = total_appreciation / max_appreciation
    appreciation_score = appreciation_score.clip(0, 1)
    
    return appreciation_score

def calculate_risk_score(df):
    """
    Calculate risk score for properties.
    
    Args:
        df (DataFrame): Property data
    
    Returns:
        Series: Risk scores (0 = low risk, 1 = high risk)
    """
    # Risk factors:
    # 1. Price volatility in the area
    # 2. Price to rent ratio (higher = riskier)
    # 3. Property age (older = riskier)
    # 4. Liquidity (days on market, if available)
    
    # Calculate price to rent ratio
    if 'estimated_monthly_rent' in df.columns:
        price_to_rent_ratio = df['price'] / (df['estimated_monthly_rent'] * 12)
    else:
        # Estimate rent as 0.8% of property value per month
        price_to_rent_ratio = 1 / (0.008 * 12)
    
    # Normalize price to rent ratio to a 0-1 score
    # Lower is better (less risky), higher is worse (more risky)
    # Assume ratio of 10 or less is low risk (0.2)
    # and ratio of 30 or more is high risk (0.8)
    p2r_risk = (price_to_rent_ratio - 10) / 20
    p2r_risk = p2r_risk.clip(0, 1) * 0.6 + 0.2  # Scale to 0.2-0.8 range
    
    # Property age risk (if available)
    if 'property_age' in df.columns:
        # Newer properties are less risky
        age_risk = df['property_age'] / 100
        age_risk = age_risk.clip(0, 1) * 0.6 + 0.2  # Scale to 0.2-0.8 range
    else:
        age_risk = 0.5  # Default if age unknown
    
    # Liquidity risk based on days on market (if available)
    if 'days_on_market' in df.columns:
        dom_risk = df['days_on_market'] / 180  # 6 months = high risk
        dom_risk = dom_risk.clip(0, 1) * 0.6 + 0.2  # Scale to 0.2-0.8 range
    else:
        dom_risk = 0.5  # Default if DOM unknown
    
    # Combine risk factors
    risk_score = (p2r_risk * 0.4 + age_risk * 0.3 + dom_risk * 0.3)
    
    return risk_score

def get_similar_properties(df, property_id, n=5):
    """
    Find properties similar to a given property.
    
    Args:
        df (DataFrame): Property data
        property_id: ID of the reference property
        n (int): Number of similar properties to return
    
    Returns:
        DataFrame: Similar properties
    """
    # Check if property exists
    if property_id not in df['property_id'].values:
        return pd.DataFrame()
    
    # Get reference property
    reference_property = df[df['property_id'] == property_id].iloc[0]
    
    # Features to use for similarity calculation
    numerical_features = ['price', 'bedrooms', 'bathrooms', 'sqft']
    if 'property_age' in df.columns:
        numerical_features.append('property_age')
    
    categorical_features = ['location', 'property_type']
    
    # Prepare numerical features
    X_numerical = df[numerical_features].copy()
    
    # Scale numerical features
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    
    # One-hot encode categorical features
    X_categorical = pd.get_dummies(df[categorical_features])
    
    # Combine features
    X_combined = np.hstack([X_numerical_scaled, X_categorical.values])
    
    # Get reference property features
    reference_idx = df[df['property_id'] == property_id].index[0]
    reference_features = X_combined[reference_idx].reshape(1, -1)
    
    # Calculate similarity
    similarities = cosine_similarity(reference_features, X_combined)[0]
    
    # Get indices of most similar properties (excluding the reference)
    similar_indices = similarities.argsort()[-n-1:-1][::-1]
    
    # Return similar properties
    return df.iloc[similar_indices]
