import streamlit as st
import os
from pages import home, valuation, investment, market_trends, comparative_analysis, recommendations

# Set page config
st.set_page_config(
    page_title="Real Estate Analytics Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a navigation sidebar
st.sidebar.title("Real Estate Analytics")
st.sidebar.image("https://www.svgrepo.com/show/529382/building-02.svg", width=80)

# Navigation options
pages = {
    "Home": home,
    "Property Valuation": valuation,
    "Investment Analysis": investment,
    "Market Trends": market_trends,
    "Comparative Analysis": comparative_analysis,
    "Recommendations": recommendations
}

# Create a radio button for navigation
selection = st.sidebar.radio("Navigate", list(pages.keys()))

# Display the selected page
pages[selection].show()

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("AI-Powered Real Estate Analytics Platform")

if __name__ == "__main__":
    # The app is meant to be run with: streamlit run app.py
    pass
