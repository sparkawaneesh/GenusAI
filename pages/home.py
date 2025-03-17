import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_real_estate_data, get_available_cities

def show():
    """Display the Home page content"""
    st.title("AI-Powered Real Estate Analytics")
    
    # Load data
    df = load_real_estate_data()
    
    # Header section with key metrics
    st.markdown("### Real Estate Market Overview")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Compute metrics
    if not df.empty and 'price' in df.columns:
        avg_price = df['price'].mean()
        median_price = df['price'].median()
        total_properties = len(df)
        
        with col1:
            st.metric("Average Price", f"${avg_price:,.0f}")
        
        with col2:
            st.metric("Median Price", f"${median_price:,.0f}")
            
        with col3:
            st.metric("Total Properties", f"{total_properties:,}")
            
        with col4:
            if 'days_on_market' in df.columns:
                avg_days = df['days_on_market'].mean()
                st.metric("Avg. Days on Market", f"{avg_days:.1f}")
    else:
        st.warning("No property data available. Please check the data source.")
    
    # Market trends section
    st.markdown("### Market Price Distribution")
    
    if not df.empty and 'price' in df.columns:
        # Price distribution histogram
        fig = px.histogram(
            df, 
            x="price", 
            nbins=30,
            title="Property Price Distribution",
            labels={"price": "Price ($)", "count": "Number of Properties"},
            color_discrete_sequence=['#FF4B4B']
        )
        fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Number of Properties",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Property type breakdown
    st.markdown("### Property Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not df.empty and 'property_type' in df.columns:
            property_counts = df['property_type'].value_counts().reset_index()
            property_counts.columns = ['Property Type', 'Count']
            
            fig = px.pie(
                property_counts, 
                values='Count', 
                names='Property Type',
                title="Property Types",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Property type data not available")
    
    with col2:
        if not df.empty and all(col in df.columns for col in ['bedrooms', 'price']):
            bed_price = df.groupby('bedrooms')['price'].median().reset_index()
            
            fig = px.bar(
                bed_price,
                x='bedrooms',
                y='price',
                title="Median Price by Bedroom Count",
                labels={'bedrooms': 'Bedrooms', 'price': 'Median Price ($)'},
                color='price',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                xaxis_title="Number of Bedrooms",
                yaxis_title="Median Price ($)",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bedroom and price data not available")
    
    # Map of properties
    st.markdown("### Geographic Distribution")
    
    if not df.empty and all(col in df.columns for col in ['latitude', 'longitude', 'price']):
        st.markdown("#### Property Map")
        
        # Create a price category for better visualization
        df['price_category'] = pd.cut(
            df['price'], 
            bins=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color="price_category",
            size="price",
            size_max=15,
            zoom=10,
            mapbox_style="open-street-map",
            title="Property Locations and Prices",
            hover_data=["price", "bedrooms", "bathrooms", "sqft", "property_type"]
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Featured Investment Opportunities
    st.markdown("### Featured Investment Opportunities")
    
    if not df.empty and 'monthly_rent_estimate' in df.columns:
        # Calculate rental yield
        df['annual_rent'] = df['monthly_rent_estimate'] * 12
        df['rental_yield'] = (df['annual_rent'] / df['price']) * 100
        
        # Get top 5 properties by rental yield
        top_investments = df.nlargest(5, 'rental_yield')
        
        for i, (_, prop) in enumerate(top_investments.iterrows()):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Use a placeholder image
                st.image("https://www.svgrepo.com/show/530661/villa.svg", width=100)
            
            with col2:
                property_type = prop.get('property_type', 'Property')
                bedrooms = prop.get('bedrooms', 'N/A')
                bathrooms = prop.get('bathrooms', 'N/A')
                location = f"{prop.get('city', '')}, {prop.get('state', '')}"
                
                st.markdown(f"**{property_type} - {bedrooms} bed, {bathrooms} bath**")
                st.markdown(f"Location: {location}")
                st.markdown(f"Price: ${prop.get('price', 0):,.0f}")
                st.markdown(f"Rental Yield: {prop.get('rental_yield', 0):.2f}%")
            
            st.markdown("---")
    
    # Quick access section
    st.markdown("### Quick Access")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("Property Valuation", on_click=lambda: st.session_state.update({"navigation": "Property Valuation"}))
    
    with col2:
        st.button("Investment Analysis", on_click=lambda: st.session_state.update({"navigation": "Investment Analysis"}))
    
    with col3:
        st.button("Market Trends", on_click=lambda: st.session_state.update({"navigation": "Market Trends"}))
