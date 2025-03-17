import streamlit as st
import os
import pandas as pd

from pages.home import show as show_home
from pages.valuation import show as show_valuation
from pages.investment import show as show_investment
from pages.market_trends import show as show_market_trends
from utils.data_loader import load_real_estate_data
from models.property_valuation import PropertyValuationModel
from models.investment_analysis import InvestmentAnalysis

# Page configuration
st.set_page_config(
    page_title="Real Estate Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default menu and footer
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        transition: all 0.3s ease-in-out;
    }
    .stButton>button {
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Initialize session state
if 'selected_property' not in st.session_state:
    st.session_state.selected_property = None
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'budget_min': 100000,
        'budget_max': 1000000,
        'bedrooms': 2,
        'bathrooms': 2,
        'property_type': 'Single Family Home',
        'location': 'All'
    }

# Initialize models
if 'valuation_model' not in st.session_state:
    st.session_state.valuation_model = PropertyValuationModel()

if 'investment_analysis' not in st.session_state:
    st.session_state.investment_analysis = InvestmentAnalysis()

# Initialize data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def ensure_data_loaded():
    """Ensure data is loaded and cached"""
    # Check if sample data file exists
    data_path = os.path.join('data', 'sample_real_estate_data.csv')
    if os.path.exists(data_path):
        # If file exists but might be corrupted or empty, regenerate it
        try:
            df = pd.read_csv(data_path)
            if df.empty or 'city' not in df.columns or 'property_type' not in df.columns:
                # Force reload by removing file
                os.remove(data_path)
                df = load_real_estate_data()
            return df
        except:
            # If there's an error reading the file, regenerate it
            if os.path.exists(data_path):
                os.remove(data_path)
            df = load_real_estate_data()
            return df
    else:
        # File doesn't exist, generate it
        df = load_real_estate_data()
        return df

# Load data and ensure models are trained
df = ensure_data_loaded()

# Train the valuation model if not already trained
if not getattr(st.session_state.valuation_model, 'trained', False):
    if not df.empty and 'price' in df.columns:
        with st.spinner("Training valuation model..."):
            metrics = st.session_state.valuation_model.train(df, model_type='xgboost')
            st.session_state.model_metrics = metrics

# Store data in session state for easy access
st.session_state.df = df

# Main title and description
st.title("🏠 Real Estate Analytics Platform")
st.markdown("""
    An AI-powered platform for property valuation, investment analysis, and market insights.
    Use machine learning models to make informed real estate decisions.
""")


# Display the selected page
show_market_trends()