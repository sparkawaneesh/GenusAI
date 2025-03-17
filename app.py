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
# Custom fonts CSS
font_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Lexend:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Lexend', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-1629p8f h1 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .st-emotion-cache-10trblm {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .st-emotion-cache-1egp75f {
        font-family: 'Poppins', sans-serif !important;
    }
    </style>
"""
st.markdown(font_css + hide_menu_style, unsafe_allow_html=True)


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

# Custom CSS for responsive design and navigation
st.markdown("""
    <style>
    /* Responsive layout */
    .stApp {
        max-width: 100%;
        padding: 1rem;
        box-sizing: border-box;
    }
    
    /* Sidebar responsiveness */
    .css-1d391kg {
        width: 100%;
        max-width: 300px;
    }
    
    /* Navigation buttons */
    .stButton>button {
        width: 100%;
        min-height: 3rem;
        margin: 0.25rem 0;
        background: white;
        color: #1F1F1F;
        border: 1px solid #E0E0E0;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-family: 'Lexend', sans-serif;
    }
    
    .stButton>button:hover {
        background: rgba(255, 75, 75, 0.1);
        transform: translateX(5px);
        border-color: #FF4B4B;
    }
    
    .stButton>button.active {
        background: #FF4B4B;
        color: white;
        border-color: #FF4B4B;
    }
    
    /* Icons in buttons */
    .nav-icon {
        margin-right: 0.75rem;
        font-size: 1.2rem;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stApp {
            padding: 0.5rem;
        }
        
        .css-1d391kg {
            max-width: 100%;
        }
        
        .stButton>button {
            min-height: 2.5rem;
            font-size: 0.9rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize navigation if not in session state
if 'nav_page' not in st.session_state:
    st.session_state.nav_page = 'Home'

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    st.markdown("---")
    
    # Navigation links with icons
    pages = {
        'Home': '🏠',
        'Market Trends': '📈',
        'Property Valuation': '💰',
        'Investment Analysis': '📊'
    }
    
    # Create navigation buttons
    for page, icon in pages.items():
        # Create unique key for each button
        button_key = f"nav_{page.lower().replace(' ', '_')}"
        
        # Check if this is the active page
        is_active = st.session_state.nav_page == page
        
        # Create button with conditional styling
        if st.button(
            f"{icon} {page}",
            key=button_key,
            help=f"Navigate to {page}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.nav_page = page
            st.rerun()

# Main content area
if st.session_state.nav_page == 'Home':
    show_home()
elif st.session_state.nav_page == 'Market Trends':
    show_market_trends()
elif st.session_state.nav_page == 'Property Valuation':
    show_valuation()
elif st.session_state.nav_page == 'Investment Analysis':
    show_investment()