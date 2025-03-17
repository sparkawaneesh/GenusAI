import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from ml_models import MarketTrendModel

def analyze_market_trends(df, location=None, period="3 Years"):
    """
    Analyze real estate market trends.
    
    Args:
        df (DataFrame): Property data
        location (str): Location to analyze, None for all locations
        period (str): Time period for analysis ("1 Year", "3 Years", "5 Years", "10 Years")
    
    Returns:
        dict: Market trend analysis results
    """
    # Create a copy of the data
    data = df.copy()
    
    # Filter by location if specified
    if location:
        data = data[data['location'] == location]
    
    # Convert period to number of years
    years_mapping = {
        "1 Year": 1,
        "3 Years": 3,
        "5 Years": 5,
        "10 Years": 10
    }
    years = years_mapping.get(period, 3)
    
    # If we don't have temporal data, create synthetic data for demonstration
    # In a real application, you would use actual historical data
    
    # Calculate average price and growth rate
    avg_price = data['price'].mean()
    
    # Create synthetic historical price data
    current_date = datetime.now()
    dates = []
    prices = []
    
    # Generate synthetic historical data based on average price
    # and random fluctuations around a growth trend
    base_growth_rate = 0.03  # 3% annual growth
    monthly_growth = (1 + base_growth_rate) ** (1/12) - 1
    
    for month in range(years * 12, 0, -1):
        date = current_date - timedelta(days=30*month)
        dates.append(date)
        
        # Add some randomness to the growth
        monthly_factor = 1 + monthly_growth + random.uniform(-0.01, 0.01)
        
        # For the first month, start with the current average price and work backwards
        if month == years * 12:
            price = avg_price / ((1 + monthly_growth) ** (years * 12))
        else:
            price = prices[-1] * monthly_factor
        
        prices.append(price)
    
    # Calculate metrics from the synthetic data
    avg_price_growth = (prices[-1] / prices[0]) ** (1/years) - 1
    
    # Calculate median days on market
    if 'days_on_market' in data.columns:
        median_dom = data['days_on_market'].median()
    else:
        # Generate synthetic DOM data
        median_dom = 45
    
    # Calculate price to rent ratio
    if 'estimated_monthly_rent' in data.columns:
        price_to_rent_ratio = avg_price / (data['estimated_monthly_rent'].mean() * 12)
    else:
        # Assume rent is about 0.8% of property value per month
        price_to_rent_ratio = 1 / (0.008 * 12)
    
    # Calculate market health score (0-10 scale)
    # Higher score means healthier market for investors
    # Factors: price growth, days on market, price to rent ratio
    growth_score = min(10, max(0, avg_price_growth * 100))  # 10% growth or more = 10 points
    dom_score = min(10, max(0, 10 - median_dom / 15))  # 0 DOM = 10 points, 150 DOM = 0 points
    rent_ratio_score = min(10, max(0, 20 - price_to_rent_ratio))  # 10 P/R ratio or less = 10 points, 30 or more = 0 points
    
    market_health_score = (growth_score * 0.4 + dom_score * 0.3 + rent_ratio_score * 0.3)
    
    # Create forecast data
    # Assume future growth will be similar to historical growth with some uncertainty
    forecast_dates = []
    forecast_prices = []
    upper_bound = []
    lower_bound = []
    
    future_growth = avg_price_growth * 0.9  # Slightly conservative forecast
    current_price = prices[-1]
    
    for month in range(1, 13):  # 12-month forecast
        date = current_date + timedelta(days=30*month)
        forecast_dates.append(date)
        
        # Calculate forecasted price
        future_price = current_price * (1 + future_growth) ** (month/12)
        forecast_prices.append(future_price)
        
        # Add uncertainty bands (wider as we go further in the future)
        uncertainty = 0.02 * month  # Increasing uncertainty
        upper_bound.append(future_price * (1 + uncertainty))
        lower_bound.append(future_price * (1 - uncertainty))
    
    # Generate market insights based on the analysis
    insights = []
    
    if avg_price_growth > 0.05:
        insights.append(f"Strong price growth of {avg_price_growth:.1%} annually indicates a seller's market with potential for appreciation.")
    elif avg_price_growth > 0.02:
        insights.append(f"Moderate price growth of {avg_price_growth:.1%} annually suggests a balanced market with steady appreciation.")
    elif avg_price_growth > 0:
        insights.append(f"Slow price growth of {avg_price_growth:.1%} annually indicates a stable market with limited appreciation potential.")
    else:
        insights.append(f"Negative price growth of {avg_price_growth:.1%} annually suggests a buyer's market with potential deals available.")
    
    if price_to_rent_ratio < 15:
        insights.append(f"Low price-to-rent ratio of {price_to_rent_ratio:.1f} indicates strong rental yield potential and favorable cash flow opportunities.")
    elif price_to_rent_ratio < 20:
        insights.append(f"Moderate price-to-rent ratio of {price_to_rent_ratio:.1f} suggests balanced investment potential for both cash flow and appreciation.")
    else:
        insights.append(f"High price-to-rent ratio of {price_to_rent_ratio:.1f} indicates lower rental yields; investors should focus on appreciation potential.")
    
    if median_dom < 30:
        insights.append(f"Properties selling quickly (median {median_dom:.0f} days on market) indicates high demand and competitive market conditions.")
    elif median_dom < 60:
        insights.append(f"Average time on market ({median_dom:.0f} days) suggests balanced supply and demand.")
    else:
        insights.append(f"Longer selling times ({median_dom:.0f} days on market) indicates excess inventory and potential negotiating power for buyers.")
    
    # Generate supply and demand metrics (synthetic for demonstration)
    new_listings = int(len(data) * 0.1)  # Assume 10% of inventory is new listings
    total_inventory = len(data)
    absorption_rate = 0.2  # 20% of inventory sells each month
    months_of_supply = 1 / absorption_rate
    
    # Generate YoY changes (synthetic)
    new_listings_change = random.uniform(-0.1, 0.2)
    total_inventory_change = random.uniform(-0.15, 0.15)
    absorption_rate_change = random.uniform(-0.05, 0.1)
    months_of_supply_change = -absorption_rate_change  # Inverse relationship
    
    # Return market trend analysis
    return {
        'avg_price': avg_price,
        'avg_price_growth': avg_price_growth,
        'median_dom': median_dom,
        'price_to_rent_ratio': price_to_rent_ratio,
        'market_health_score': market_health_score,
        'historical_data': {
            'dates': dates,
            'prices': prices
        },
        'forecast_data': {
            'historical_dates': dates,
            'historical_prices': prices,
            'forecast_dates': forecast_dates,
            'forecast_prices': forecast_prices,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        },
        'market_insights': insights,
        'new_listings': new_listings,
        'total_inventory': total_inventory,
        'absorption_rate': absorption_rate,
        'months_of_supply': months_of_supply,
        'new_listings_change': new_listings_change,
        'total_inventory_change': total_inventory_change,
        'absorption_rate_change': absorption_rate_change,
        'months_of_supply_change': months_of_supply_change
    }

def compare_properties(df, property_ids):
    """
    Compare multiple properties.
    
    Args:
        df (DataFrame): Property data
        property_ids (list): List of property IDs to compare
    
    Returns:
        DataFrame: Comparison data
    """
    # Filter properties by ID
    comparison_data = df[df['property_id'].isin(property_ids)].copy()
    
    if comparison_data.empty:
        return pd.DataFrame()
    
    # Add additional comparison metrics
    
    # Price per square foot
    if 'sqft' in comparison_data.columns:
        comparison_data['price_per_sqft'] = comparison_data['price'] / comparison_data['sqft']
    
    # Price vs. area average
    comparison_data['area_avg_price'] = comparison_data.apply(
        lambda row: df[df['location'] == row['location']]['price'].mean(), axis=1
    )
    comparison_data['price_vs_area_avg'] = comparison_data['price'] / comparison_data['area_avg_price'] - 1
    
    # Calculate estimated ROI if not present
    if 'estimated_roi' not in comparison_data.columns:
        # Simple ROI calculation based on rent and price
        monthly_rent = comparison_data.apply(
            lambda row: row['estimated_monthly_rent'] if 'estimated_monthly_rent' in comparison_data.columns else row['price'] * 0.008,
            axis=1
        )
        annual_rent = monthly_rent * 12
        annual_expenses = comparison_data['price'] * 0.02  # Assume 2% annual expenses
        comparison_data['estimated_roi'] = (annual_rent - annual_expenses) / comparison_data['price']
    
    # Calculate cap rate
    if 'estimated_monthly_rent' in comparison_data.columns:
        annual_rent = comparison_data['estimated_monthly_rent'] * 12
    else:
        annual_rent = comparison_data['price'] * 0.008 * 12  # Estimate rent as 0.8% of price monthly
    
    annual_expenses = comparison_data['price'] * 0.02  # Estimate expenses as 2% of price annually
    comparison_data['cap_rate'] = (annual_rent - annual_expenses) / comparison_data['price']
    
    # Calculate neighborhood score if not present (synthetic for demonstration)
    if 'neighborhood_score' not in comparison_data.columns:
        comparison_data['neighborhood_score'] = comparison_data.apply(
            lambda row: min(10, max(1, 5 + (row['price_vs_area_avg'] * 10))),
            axis=1
        )
    
    return comparison_data

def get_market_insights(df, location=None, property_type=None, price_range=None):
    """
    Get market insights based on property data.
    
    Args:
        df (DataFrame): Property data
        location (str): Location filter
        property_type (str): Property type filter
        price_range (tuple): Price range filter (min, max)
    
    Returns:
        dict: Market insights
    """
    # Create a copy of the data
    data = df.copy()
    
    # Apply filters
    if location and location != 'All':
        data = data[data['location'] == location]
    
    if property_type and property_type != 'All':
        data = data[data['property_type'] == property_type]
    
    if price_range:
        data = data[(data['price'] >= price_range[0]) & (data['price'] <= price_range[1])]
    
    if data.empty:
        return {
            'count': 0,
            'avg_price': 0,
            'median_price': 0,
            'avg_price_per_sqft': 0,
            'avg_days_on_market': 0,
            'price_distribution': {},
            'bedrooms_distribution': {},
            'property_type_distribution': {},
            'insights': ["No properties match the selected criteria."]
        }
    
    # Calculate basic statistics
    count = len(data)
    avg_price = data['price'].mean()
    median_price = data['price'].median()
    
    # Calculate average price per square foot
    if 'sqft' in data.columns:
        avg_price_per_sqft = (data['price'] / data['sqft']).mean()
    else:
        avg_price_per_sqft = 0
    
    # Calculate average days on market
    if 'days_on_market' in data.columns:
        avg_days_on_market = data['days_on_market'].mean()
    else:
        avg_days_on_market = 45  # Default value
    
    # Calculate price distribution
    price_bins = [0, 200000, 400000, 600000, 800000, 1000000, float('inf')]
    price_labels = ['<$200k', '$200k-$400k', '$400k-$600k', '$600k-$800k', '$800k-$1M', '>$1M']
    
    data['price_range'] = pd.cut(data['price'], bins=price_bins, labels=price_labels, right=False)
    price_distribution = data['price_range'].value_counts().sort_index().to_dict()
    
    # Calculate bedrooms distribution
    if 'bedrooms' in data.columns:
        bedrooms_distribution = data['bedrooms'].value_counts().sort_index().to_dict()
    else:
        bedrooms_distribution = {}
    
    # Calculate property type distribution
    property_type_distribution = data['property_type'].value_counts().to_dict()
    
    # Generate insights
    insights = []
    
    # Price insights
    if avg_price > df['price'].mean() * 1.2:
        insights.append(f"Properties in this market are priced {((avg_price / df['price'].mean()) - 1) * 100:.1f}% higher than the overall average.")
    elif avg_price < df['price'].mean() * 0.8:
        insights.append(f"Properties in this market are priced {(1 - (avg_price / df['price'].mean())) * 100:.1f}% lower than the overall average.")
    else:
        insights.append(f"Property prices in this market are close to the overall average.")
    
    # Days on market insights
    if 'days_on_market' in data.columns:
        if avg_days_on_market < 30:
            insights.append(f"Properties in this market sell quickly (average {avg_days_on_market:.0f} days), indicating high demand.")
        elif avg_days_on_market > 60:
            insights.append(f"Properties in this market take longer to sell (average {avg_days_on_market:.0f} days), suggesting a buyer's market.")
        else:
            insights.append(f"Properties in this market sell at an average pace ({avg_days_on_market:.0f} days).")
    
    # Price per square foot insights
    if 'sqft' in data.columns:
        overall_avg_price_per_sqft = (df['price'] / df['sqft']).mean()
        
        if avg_price_per_sqft > overall_avg_price_per_sqft * 1.2:
            insights.append(f"Price per square foot (${avg_price_per_sqft:.0f}) is {((avg_price_per_sqft / overall_avg_price_per_sqft) - 1) * 100:.1f}% higher than the overall average.")
        elif avg_price_per_sqft < overall_avg_price_per_sqft * 0.8:
            insights.append(f"Price per square foot (${avg_price_per_sqft:.0f}) is {(1 - (avg_price_per_sqft / overall_avg_price_per_sqft)) * 100:.1f}% lower than the overall average.")
        else:
            insights.append(f"Price per square foot (${avg_price_per_sqft:.0f}) is close to the overall average.")
    
    # Return market insights
    return {
        'count': count,
        'avg_price': avg_price,
        'median_price': median_price,
        'avg_price_per_sqft': avg_price_per_sqft,
        'avg_days_on_market': avg_days_on_market,
        'price_distribution': price_distribution,
        'bedrooms_distribution': bedrooms_distribution,
        'property_type_distribution': property_type_distribution,
        'insights': insights
    }
