import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_property_data(num_properties=500):
    """
    Generate sample property data for the real estate analytics platform.
    
    Args:
        num_properties (int): Number of properties to generate
    
    Returns:
        DataFrame: Generated property data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define cities and states
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA']
    
    # Define locations with their relative price levels
    locations = {
        'Downtown': 1.5,          # Higher price multiplier
        'Suburban Heights': 1.2,  
        'Westside': 1.3,
        'Eastwood': 0.9,         # Lower price multiplier
        'Northend': 1.1,
        'Southbay': 0.8,
        'Central District': 1.0,
        'Riverside': 0.95,
        'Lakefront': 1.4,
        'Mountain View': 1.25
    }
    
    # Define property types with their relative price levels
    property_types = {
        'Single Family Home': 1.0,
        'Condominium': 0.8,
        'Townhouse': 0.9,
        'Multi-Family': 1.3,
        'Luxury Home': 2.0
    }
    
    # Generate property IDs
    property_ids = list(range(1, num_properties + 1))
    
    # Generate random property locations (neighborhoods)
    property_locations = np.random.choice(list(locations.keys()), size=num_properties, p=[0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05])
    
    # Generate random city/state combinations
    # Each property will be in one of our cities
    city_indices = np.random.randint(0, len(cities), size=num_properties)
    property_cities = [cities[i] for i in city_indices]
    property_states = [states[i] for i in city_indices]
    
    # Generate random zip codes
    zip_codes = np.random.randint(10000, 99999, size=num_properties)
    
    # Generate latitude and longitude based on city
    # These are approximate coordinates for the cities
    city_coordinates = {
        'New York': (40.7128, -74.0060),
        'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298),
        'Houston': (29.7604, -95.3698),
        'Phoenix': (33.4484, -112.0740),
        'Philadelphia': (39.9526, -75.1652),
        'San Antonio': (29.4241, -98.4936),
        'San Diego': (32.7157, -117.1611),
        'Dallas': (32.7767, -96.7970),
        'San Jose': (37.3382, -121.8863)
    }
    
    latitudes = []
    longitudes = []
    
    for city in property_cities:
        base_lat, base_lon = city_coordinates[city]
        # Add some random variation within the city (approximately within a few miles)
        lat = base_lat + np.random.uniform(-0.05, 0.05)
        lon = base_lon + np.random.uniform(-0.05, 0.05)
        latitudes.append(lat)
        longitudes.append(lon)
    
    # Generate random property types
    property_types_list = np.random.choice(list(property_types.keys()), size=num_properties, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    
    # Generate property features
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], size=num_properties, p=[0.05, 0.2, 0.4, 0.25, 0.07, 0.03])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], size=num_properties, p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.05, 0.05])
    
    # Generate square footage based on bedrooms and bathrooms
    base_sqft = 500 + bedrooms * 300 + bathrooms * 100
    sqft = base_sqft + np.random.normal(0, 200, num_properties).astype(int)
    sqft = np.maximum(500, sqft)  # Ensure minimum square footage
    
    # Generate year built
    current_year = datetime.now().year
    year_built = np.random.randint(1950, current_year + 1, size=num_properties)
    property_age = current_year - year_built
    
    # Generate base prices based on features
    base_price = 100000 + bedrooms * 50000 + bathrooms * 30000 + sqft * 100
    
    # Adjust prices based on location and property type
    prices = []
    for i in range(num_properties):
        location_multiplier = locations[property_locations[i]]
        property_type_multiplier = property_types[property_types_list[i]]
        age_factor = 1 - (property_age[i] / 200)  # Newer properties are worth more
        
        # Add some randomness
        random_factor = np.random.uniform(0.9, 1.1)
        
        # Calculate price
        price = base_price[i] * location_multiplier * property_type_multiplier * age_factor * random_factor
        prices.append(int(price))
    
    # Generate price per square foot
    price_per_sqft = [p / s for p, s in zip(prices, sqft)]
    
    # Generate estimated monthly rent
    # Typically, monthly rent is about 0.8% of property value for properties < $500K
    # and about 0.6% for more expensive properties
    estimated_monthly_rent = []
    for price in prices:
        if price < 500000:
            rent = price * 0.008 * np.random.uniform(0.9, 1.1)
        else:
            rent = price * 0.006 * np.random.uniform(0.9, 1.1)
        estimated_monthly_rent.append(int(rent))
    
    # Generate days on market
    days_on_market = np.random.exponential(45, size=num_properties).astype(int)
    days_on_market = np.minimum(days_on_market, 365)  # Cap at 365 days
    
    # Generate estimated ROI
    estimated_roi = []
    for i in range(num_properties):
        annual_rent = estimated_monthly_rent[i] * 12
        annual_expenses = prices[i] * 0.02  # Assume 2% annual expenses (taxes, maintenance, etc.)
        annual_cash_flow = annual_rent - annual_expenses
        roi = annual_cash_flow / prices[i]
        estimated_roi.append(roi)
    
    # Create DataFrame
    property_data = pd.DataFrame({
        'property_id': property_ids,
        'location': property_locations,  # This is the neighborhood
        'property_type': property_types_list,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'year_built': year_built,
        'property_age': property_age,
        'price': prices,
        'price_per_sqft': price_per_sqft,
        'estimated_monthly_rent': estimated_monthly_rent,
        'days_on_market': days_on_market,
        'estimated_roi': estimated_roi,
        'city': property_cities,
        'state': property_states,
        'zip_code': zip_codes,
        'neighborhood': property_locations,  # Add neighborhood as a separate column
        'latitude': latitudes,
        'longitude': longitudes
    })
    
    # Add some additional features for a subset of properties
    has_pool = np.random.choice([True, False], size=num_properties, p=[0.2, 0.8])
    has_garage = np.random.choice([True, False], size=num_properties, p=[0.7, 0.3])
    lot_size = np.random.uniform(0.1, 1.0, size=num_properties) * (1 + 0.5 * (bedrooms / 3))
    
    property_data['has_pool'] = has_pool
    property_data['has_garage'] = has_garage
    property_data['lot_size'] = lot_size.round(2)
    
    # Add neighborhood scores
    neighborhood_scores = []
    for location in property_locations:
        base_score = locations[location] * 5  # Scale location multiplier to a 1-10 score
        random_factor = np.random.uniform(-1, 1)  # Add some randomness
        score = min(10, max(1, base_score + random_factor))
        neighborhood_scores.append(score)
    
    property_data['neighborhood_score'] = neighborhood_scores
    
    # Add date listed (synthetic)
    current_date = datetime.now()
    date_listed = []
    for dom in days_on_market:
        # Convert numpy.int64 to regular Python int for timedelta
        list_date = current_date - timedelta(days=int(dom))
        date_listed.append(list_date)
    
    property_data['date_listed'] = date_listed
    
    # Ensure all numerical values are reasonable
    property_data['price'] = property_data['price'].clip(lower=50000)
    property_data['estimated_monthly_rent'] = property_data['estimated_monthly_rent'].clip(lower=500)
    property_data['estimated_roi'] = property_data['estimated_roi'].clip(lower=0, upper=0.2)
    
    return property_data
