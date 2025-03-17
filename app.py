import streamlit as st

from pages.home import show as show_home
from pages.valuation import show as show_valuation
from pages.investment import show as show_investment
from pages.market_trends import show as show_market_trends

# Page configuration
st.set_page_config(
    page_title="Real Estate Analytics Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Main title and description
st.title("🏠 Real Estate Analytics Platform")
st.markdown("""
    An AI-powered platform for property valuation, investment analysis, and market insights.
    Use machine learning models to make informed real estate decisions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Home", "Property Valuation", "Investment Analysis", "Market Trends"]
)

# Display the selected page
if page == "Home":
    show_home()
elif page == "Property Valuation":
    show_valuation()
elif page == "Investment Analysis":
    show_investment()
elif page == "Market Trends":
    show_market_trends()