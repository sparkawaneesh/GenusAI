import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_processor import load_data, preprocess_data
from ml_models import PropertyValuationModel
from investment_analyzer import calculate_roi, calculate_rental_yield
from market_analyzer import analyze_market_trends, compare_properties
from property_recommender import recommend_properties
from visualization import (
    create_price_distribution_chart, 
    create_market_trends_chart,
    create_property_comparison_chart,
    create_investment_analysis_chart,
    create_heatmap
)
from utils import format_currency, calculate_monthly_mortgage, format_percentage

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

# Sidebar for user inputs and navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Home", "Property Valuation", "Investment Analysis", "Market Trends", "Property Comparison", "Recommendations"]
)

# Load and preprocess data
df = load_data()
df = preprocess_data(df)

# User preference filters in sidebar
st.sidebar.title("Filters")
locations = ['All'] + sorted(df['location'].unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations, index=0)

price_min = int(df['price'].min())
price_max = int(df['price'].max())
price_range = st.sidebar.slider(
    "Price Range ($)",
    price_min,
    price_max,
    (st.session_state.user_preferences['budget_min'], st.session_state.user_preferences['budget_max'])
)
st.session_state.user_preferences['budget_min'] = price_range[0]
st.session_state.user_preferences['budget_max'] = price_range[1]

property_types = ['All'] + sorted(df['property_type'].unique().tolist())
selected_property_type = st.sidebar.selectbox(
    "Property Type",
    property_types,
    index=property_types.index(st.session_state.user_preferences['property_type']) if st.session_state.user_preferences['property_type'] in property_types else 0
)
st.session_state.user_preferences['property_type'] = selected_property_type

bedrooms = sorted(df['bedrooms'].unique().tolist())
selected_bedrooms = st.sidebar.selectbox(
    "Bedrooms",
    bedrooms,
    index=bedrooms.index(st.session_state.user_preferences['bedrooms']) if st.session_state.user_preferences['bedrooms'] in bedrooms else 0
)
st.session_state.user_preferences['bedrooms'] = selected_bedrooms

bathrooms = sorted(df['bathrooms'].unique().tolist())
selected_bathrooms = st.sidebar.selectbox(
    "Bathrooms",
    bathrooms,
    index=bathrooms.index(st.session_state.user_preferences['bathrooms']) if st.session_state.user_preferences['bathrooms'] in bathrooms else 0
)
st.session_state.user_preferences['bathrooms'] = selected_bathrooms

# Filter the data based on user selections
filtered_df = df.copy()

if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
    
filtered_df = filtered_df[
    (filtered_df['price'] >= price_range[0]) &
    (filtered_df['price'] <= price_range[1])
]

if selected_property_type != 'All':
    filtered_df = filtered_df[filtered_df['property_type'] == selected_property_type]
    
filtered_df = filtered_df[
    (filtered_df['bedrooms'] == selected_bedrooms) &
    (filtered_df['bathrooms'] == selected_bathrooms)
]

# Home page
if page == "Home":
    st.header("Real Estate Market Overview")
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Price", format_currency(df['price'].mean()), f"{df['price'].pct_change().mean():.2%}")
    with col2:
        st.metric("Average Price/Sqft", format_currency(df['price_per_sqft'].mean()), f"{df['price_per_sqft'].pct_change().mean():.2%}")
    with col3:
        st.metric("Average ROI", f"{df['estimated_roi'].mean():.2%}", f"{df['estimated_roi'].pct_change().mean():.2%}")
    with col4:
        st.metric("Listings", f"{len(df)}", f"{len(df) - len(df[df['days_on_market'] > 30])}")
    
    # Price distribution by location
    st.subheader("Price Distribution by Location")
    price_dist_chart = create_price_distribution_chart(df)
    st.plotly_chart(price_dist_chart, use_container_width=True)
    
    # Property heatmap
    st.subheader("Property Price Heatmap")
    heatmap = create_heatmap(df)
    st.plotly_chart(heatmap, use_container_width=True)
    
    # Recent properties table
    st.subheader("Recent Properties")
    st.dataframe(
        df.sort_values('days_on_market')[['property_id', 'location', 'property_type', 'bedrooms', 'bathrooms', 'sqft', 'price', 'days_on_market']].head(10),
        use_container_width=True
    )

# Property Valuation page
elif page == "Property Valuation":
    st.header("AI-Powered Property Valuation")
    
    st.markdown("""
        Our machine learning model estimates property values based on location, size, features, and market trends.
        Enter property details below to get an estimated valuation.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox("Location", sorted(df['location'].unique().tolist()))
        property_type = st.selectbox("Property Type", sorted(df['property_type'].unique().tolist()))
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    
    with col2:
        sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=2000)
        lot_size = st.number_input("Lot Size (acres)", min_value=0.1, max_value=10.0, value=0.25, step=0.1)
        has_pool = st.checkbox("Has Pool")
    
    features = {
        'location': location,
        'property_type': property_type,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'year_built': year_built,
        'lot_size': lot_size,
        'has_pool': has_pool
    }
    
    valuation_model = PropertyValuationModel()
    valuation_model.train(df)
    
    if st.button("Estimate Property Value"):
        with st.spinner("Calculating property valuation..."):
            estimated_value = valuation_model.predict(features)
            confidence_interval = valuation_model.get_confidence_interval(features)
            
            st.success(f"Estimated Property Value: **{format_currency(estimated_value)}**")
            st.info(f"Confidence Interval: {format_currency(confidence_interval[0])} - {format_currency(confidence_interval[1])}")
            
            # Show similar properties for comparison
            st.subheader("Similar Properties")
            similar_properties = valuation_model.find_similar_properties(df, features, n=5)
            st.dataframe(
                similar_properties[['location', 'property_type', 'bedrooms', 'bathrooms', 'sqft', 'price']],
                use_container_width=True
            )
            
            # Show valuation factors
            st.subheader("Valuation Factors")
            valuation_factors = valuation_model.get_feature_importance()
            
            # Create a horizontal bar chart for feature importance
            fig = px.bar(
                x=valuation_factors.values(),
                y=valuation_factors.keys(),
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                title='Feature Importance in Valuation Model'
            )
            st.plotly_chart(fig, use_container_width=True)

# Investment Analysis page
elif page == "Investment Analysis":
    st.header("Investment Analysis")
    
    st.markdown("""
        Analyze potential investment opportunities based on ROI, rental yield, and appreciation potential.
        Select a property from our database or enter custom property details.
    """)
    
    # Option to select existing property or enter custom details
    analysis_type = st.radio(
        "Analysis Type",
        ["Select from Database", "Enter Custom Property"]
    )
    
    if analysis_type == "Select from Database":
        selected_property_id = st.selectbox(
            "Select Property",
            filtered_df['property_id'].tolist(),
            format_func=lambda x: f"ID: {x} - {filtered_df[filtered_df['property_id']==x]['location'].iloc[0]}, {int(filtered_df[filtered_df['property_id']==x]['bedrooms'].iloc[0])}bd {int(filtered_df[filtered_df['property_id']==x]['bathrooms'].iloc[0])}ba, ${int(filtered_df[filtered_df['property_id']==x]['price'].iloc[0]):,}"
        )
        property_data = filtered_df[filtered_df['property_id'] == selected_property_id].iloc[0]
        
        purchase_price = property_data['price']
        monthly_rent = property_data['estimated_monthly_rent'] if 'estimated_monthly_rent' in property_data else purchase_price * 0.008
        property_tax_rate = 0.01
        maintenance_cost = purchase_price * 0.01
        vacancy_rate = 0.05
        appreciation_rate = 0.03
        mortgage_rate = 0.045
        loan_term = 30
        down_payment_pct = 0.20
        
    else:  # Enter Custom Property
        col1, col2 = st.columns(2)
        
        with col1:
            purchase_price = st.number_input("Purchase Price ($)", min_value=50000, max_value=10000000, value=350000, step=10000)
            monthly_rent = st.number_input("Expected Monthly Rent ($)", min_value=500, max_value=50000, value=int(purchase_price * 0.008), step=100)
            property_tax_rate = st.slider("Annual Property Tax Rate (%)", 0.5, 3.0, 1.0) / 100
            maintenance_cost = st.number_input("Annual Maintenance Cost ($)", min_value=0, max_value=100000, value=int(purchase_price * 0.01), step=500)
        
        with col2:
            vacancy_rate = st.slider("Vacancy Rate (%)", 0.0, 20.0, 5.0) / 100
            appreciation_rate = st.slider("Annual Appreciation Rate (%)", 0.0, 10.0, 3.0) / 100
            mortgage_rate = st.slider("Mortgage Interest Rate (%)", 2.0, 8.0, 4.5) / 100
            loan_term = st.selectbox("Loan Term (years)", [15, 20, 30], index=2)
            down_payment_pct = st.slider("Down Payment (%)", 5.0, 50.0, 20.0) / 100
    
    # Calculate investment metrics
    if st.button("Analyze Investment"):
        with st.spinner("Analyzing investment potential..."):
            down_payment = purchase_price * down_payment_pct
            loan_amount = purchase_price - down_payment
            
            # Calculate mortgage payment
            monthly_rate = mortgage_rate / 12
            num_payments = loan_term * 12
            monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            
            # Annual expenses
            annual_property_tax = purchase_price * property_tax_rate
            annual_vacancy_cost = monthly_rent * 12 * vacancy_rate
            total_annual_expenses = annual_property_tax + maintenance_cost + annual_vacancy_cost
            
            # Cash flow
            annual_rental_income = monthly_rent * 12
            annual_mortgage_payments = monthly_mortgage * 12
            annual_cash_flow = annual_rental_income - annual_mortgage_payments - total_annual_expenses
            
            # ROI and yield calculations
            cash_on_cash_roi = calculate_roi(annual_cash_flow, down_payment)
            rental_yield = calculate_rental_yield(annual_rental_income, purchase_price)
            cap_rate = calculate_rental_yield(annual_rental_income - total_annual_expenses, purchase_price)
            
            # Five-year projection
            years = list(range(1, 6))
            property_values = [purchase_price * (1 + appreciation_rate)**year for year in years]
            equity_values = [property_values[i-1] - (loan_amount * (1 - (i / loan_term))) for i in years]
            cash_flows = [annual_cash_flow * (1.02)**year for year in years]  # Assuming 2% increase in rent annually
            cumulative_cash_flow = [sum(cash_flows[:i+1]) for i in range(len(cash_flows))]
            
            # Display investment metrics
            st.subheader("Investment Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cash-on-Cash ROI", f"{cash_on_cash_roi:.2%}")
            with col2:
                st.metric("Gross Rental Yield", f"{rental_yield:.2%}")
            with col3:
                st.metric("Cap Rate", f"{cap_rate:.2%}")
            with col4:
                st.metric("Monthly Cash Flow", format_currency(annual_cash_flow / 12))
            
            # Monthly breakdown
            st.subheader("Monthly Financial Breakdown")
            monthly_data = {
                "Income/Expense": ["Rental Income", "Mortgage Payment", "Property Tax", "Maintenance", "Vacancy Loss", "Net Cash Flow"],
                "Amount": [
                    monthly_rent,
                    -monthly_mortgage,
                    -annual_property_tax / 12,
                    -maintenance_cost / 12,
                    -annual_vacancy_cost / 12,
                    annual_cash_flow / 12
                ]
            }
            monthly_df = pd.DataFrame(monthly_data)
            monthly_df["Amount"] = monthly_df["Amount"].apply(lambda x: format_currency(x))
            
            st.table(monthly_df)
            
            # Five-year projection chart
            st.subheader("5-Year Investment Projection")
            investment_chart = create_investment_analysis_chart(years, property_values, equity_values, cumulative_cash_flow)
            st.plotly_chart(investment_chart, use_container_width=True)
            
            # Payback period calculation
            total_investment = down_payment + total_annual_expenses  # Initial investment plus first year expenses
            payback_years = 0
            cumulative_cf = 0
            
            for year, cf in enumerate(cash_flows, 1):
                cumulative_cf += cf
                if cumulative_cf >= total_investment:
                    payback_years = year
                    break
            
            if payback_years > 0:
                st.success(f"Payback Period: Approximately {payback_years} years")
            else:
                st.warning("Payback Period: More than 5 years")
                
            # Break-even analysis
            break_even_rent = (annual_mortgage_payments + total_annual_expenses) / 12
            st.info(f"Break-even Rent: {format_currency(break_even_rent)} per month")

# Market Trends page
elif page == "Market Trends":
    st.header("Market Trends Analysis")
    
    st.markdown("""
        Analyze real estate market trends to identify opportunities and make informed decisions.
        View historical data, price trends, and market forecasts.
    """)
    
    # Select location and time period
    col1, col2 = st.columns(2)
    
    with col1:
        trend_location = st.selectbox("Select Location", ['All'] + sorted(df['location'].unique().tolist()), key='trend_loc')
    
    with col2:
        trend_period = st.selectbox("Time Period", ["1 Year", "3 Years", "5 Years", "10 Years"], index=1)
    
    # Get market trends data
    market_data = analyze_market_trends(df, location=trend_location if trend_location != 'All' else None, period=trend_period)
    
    # Display market metrics
    st.subheader(f"Market Metrics for {trend_location if trend_location != 'All' else 'All Locations'}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg. Price Growth", f"{market_data['avg_price_growth']:.2%}")
    with col2:
        st.metric("Median Days on Market", f"{market_data['median_dom']:.0f} days")
    with col3:
        st.metric("Price to Rent Ratio", f"{market_data['price_to_rent_ratio']:.2f}x")
    with col4:
        st.metric("Market Health Score", f"{market_data['market_health_score']:.1f}/10")
    
    # Market trend charts
    st.subheader("Price Trends")
    market_trend_chart = create_market_trends_chart(market_data)
    st.plotly_chart(market_trend_chart, use_container_width=True)
    
    # Market forecast
    st.subheader("Market Forecast (Next 12 Months)")
    
    forecast_data = market_data['forecast_data']
    
    # Create forecast chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=forecast_data['historical_dates'],
        y=forecast_data['historical_prices'],
        mode='lines',
        name='Historical Prices',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data['forecast_dates'],
        y=forecast_data['forecast_prices'],
        mode='lines',
        name='Price Forecast',
        line=dict(color='red', width=2)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_data['forecast_dates'] + forecast_data['forecast_dates'][::-1],
        y=forecast_data['upper_bound'] + forecast_data['lower_bound'][::-1],
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title='Price Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Average Price',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Market insights
    st.subheader("Key Market Insights")
    
    insights = market_data['market_insights']
    for i, insight in enumerate(insights):
        st.info(insight)
    
    # Supply and demand metrics
    st.subheader("Supply and Demand Metrics")
    
    supply_demand_data = {
        "Metric": ["New Listings", "Total Inventory", "Absorption Rate", "Months of Supply"],
        "Value": [
            market_data['new_listings'],
            market_data['total_inventory'],
            f"{market_data['absorption_rate']:.2%}",
            f"{market_data['months_of_supply']:.1f}"
        ],
        "Change": [
            f"{market_data['new_listings_change']:.2%}",
            f"{market_data['total_inventory_change']:.2%}",
            f"{market_data['absorption_rate_change']:.2%}",
            f"{market_data['months_of_supply_change']:.2%}"
        ]
    }
    
    supply_demand_df = pd.DataFrame(supply_demand_data)
    st.table(supply_demand_df)

# Property Comparison page
elif page == "Property Comparison":
    st.header("Property Comparison")
    
    st.markdown("""
        Compare multiple properties side by side to make better investment decisions.
        Select up to 3 properties to compare their features, prices, and investment metrics.
    """)
    
    # Property selection
    properties_to_compare = st.multiselect(
        "Select Properties to Compare (Up to 3)",
        filtered_df['property_id'].tolist(),
        format_func=lambda x: f"ID: {x} - {filtered_df[filtered_df['property_id']==x]['location'].iloc[0]}, {int(filtered_df[filtered_df['property_id']==x]['bedrooms'].iloc[0])}bd {int(filtered_df[filtered_df['property_id']==x]['bathrooms'].iloc[0])}ba, ${int(filtered_df[filtered_df['property_id']==x]['price'].iloc[0]):,}",
        max_selections=3
    )
    
    if len(properties_to_compare) > 1:
        # Get data for selected properties
        comparison_data = filtered_df[filtered_df['property_id'].isin(properties_to_compare)]
        
        # Display comparison table
        st.subheader("Property Comparison Table")
        
        # Format comparison table
        display_columns = ['property_id', 'location', 'property_type', 'price', 'sqft', 'bedrooms', 'bathrooms', 'year_built', 'price_per_sqft', 'estimated_roi']
        comparison_display = comparison_data[display_columns].copy()
        comparison_display['price'] = comparison_display['price'].apply(lambda x: format_currency(x))
        comparison_display['price_per_sqft'] = comparison_display['price_per_sqft'].apply(lambda x: format_currency(x))
        comparison_display['estimated_roi'] = comparison_display['estimated_roi'].apply(lambda x: f"{x:.2%}")
        
        st.table(comparison_display.set_index('property_id'))
        
        # Create comparison charts
        st.subheader("Visual Comparison")
        comparison_chart = create_property_comparison_chart(comparison_data)
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Investment comparison
        st.subheader("Investment Metrics Comparison")
        
        # Calculate investment metrics for each property
        investment_metrics = []
        
        for _, prop in comparison_data.iterrows():
            # Basic property info
            property_info = {
                'property_id': prop['property_id'],
                'location': prop['location'],
                'price': prop['price']
            }
            
            # Calculate investment metrics
            monthly_rent = prop['estimated_monthly_rent'] if 'estimated_monthly_rent' in prop else prop['price'] * 0.008
            annual_rental_income = monthly_rent * 12
            
            # Assumptions
            property_tax = prop['price'] * 0.01  # 1% property tax
            maintenance = prop['price'] * 0.01  # 1% maintenance cost
            vacancy_cost = annual_rental_income * 0.05  # 5% vacancy rate
            total_expenses = property_tax + maintenance + vacancy_cost
            
            # Mortgage assumptions
            down_payment = prop['price'] * 0.2  # 20% down payment
            loan_amount = prop['price'] - down_payment
            monthly_rate = 0.045 / 12  # 4.5% interest rate
            num_payments = 30 * 12  # 30-year loan
            
            monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            annual_mortgage = monthly_mortgage * 12
            
            # Cash flow
            annual_cash_flow = annual_rental_income - annual_mortgage - total_expenses
            monthly_cash_flow = annual_cash_flow / 12
            
            # ROI metrics
            cash_on_cash_roi = calculate_roi(annual_cash_flow, down_payment)
            rental_yield = calculate_rental_yield(annual_rental_income, prop['price'])
            cap_rate = calculate_rental_yield(annual_rental_income - total_expenses, prop['price'])
            
            # Add to metrics list
            property_info.update({
                'monthly_rent': monthly_rent,
                'monthly_cash_flow': monthly_cash_flow,
                'cash_on_cash_roi': cash_on_cash_roi,
                'rental_yield': rental_yield,
                'cap_rate': cap_rate
            })
            
            investment_metrics.append(property_info)
        
        # Create investment metrics DataFrame
        metrics_df = pd.DataFrame(investment_metrics)
        
        # Display investment metrics in a readable format
        display_metrics = metrics_df.copy()
        display_metrics['price'] = display_metrics['price'].apply(lambda x: format_currency(x))
        display_metrics['monthly_rent'] = display_metrics['monthly_rent'].apply(lambda x: format_currency(x))
        display_metrics['monthly_cash_flow'] = display_metrics['monthly_cash_flow'].apply(lambda x: format_currency(x))
        display_metrics['cash_on_cash_roi'] = display_metrics['cash_on_cash_roi'].apply(lambda x: f"{x:.2%}")
        display_metrics['rental_yield'] = display_metrics['rental_yield'].apply(lambda x: f"{x:.2%}")
        display_metrics['cap_rate'] = display_metrics['cap_rate'].apply(lambda x: f"{x:.2%}")
        
        st.table(display_metrics.set_index('property_id')[['location', 'price', 'monthly_rent', 'monthly_cash_flow', 'cash_on_cash_roi', 'rental_yield', 'cap_rate']])
        
        # Property comparison radar chart
        st.subheader("Property Comparison Radar Chart")
        
        # Create radar chart for property comparison
        categories = ['Price', 'Size', 'ROI', 'Rental Yield', 'Neighborhood Score']
        
        fig = go.Figure()
        
        for _, prop in comparison_data.iterrows():
            prop_id = prop['property_id']
            location_name = prop['location']
            
            # Normalize values for radar chart (0-1 scale)
            price_norm = 1 - (prop['price'] - comparison_data['price'].min()) / (comparison_data['price'].max() - comparison_data['price'].min() + 1e-10)
            size_norm = (prop['sqft'] - comparison_data['sqft'].min()) / (comparison_data['sqft'].max() - comparison_data['sqft'].min() + 1e-10)
            roi_norm = (prop['estimated_roi'] - comparison_data['estimated_roi'].min()) / (comparison_data['estimated_roi'].max() - comparison_data['estimated_roi'].min() + 1e-10)
            
            # Calculate rental yield if not available
            if 'rental_yield' in prop:
                rental_yield = prop['rental_yield']
            else:
                monthly_rent = prop['estimated_monthly_rent'] if 'estimated_monthly_rent' in prop else prop['price'] * 0.008
                rental_yield = (monthly_rent * 12) / prop['price']
                
            rental_yield_norm = (rental_yield - 0.03) / 0.07  # Normalize between 3% and 10%
            rental_yield_norm = max(0, min(1, rental_yield_norm))  # Clamp to 0-1
            
            # Use neighborhood_score if available, otherwise estimate based on price
            if 'neighborhood_score' in prop:
                neighborhood_score_norm = prop['neighborhood_score'] / 10
            else:
                # Estimate based on price per sqft compared to area average
                location_avg_price_per_sqft = df[df['location'] == prop['location']]['price_per_sqft'].mean()
                prop_price_per_sqft = prop['price'] / prop['sqft']
                neighborhood_score_norm = min(1, max(0, 0.5 + (prop_price_per_sqft - location_avg_price_per_sqft) / (2 * location_avg_price_per_sqft)))
            
            # Add to radar chart
            fig.add_trace(go.Scatterpolar(
                r=[price_norm, size_norm, roi_norm, rental_yield_norm, neighborhood_score_norm],
                theta=categories,
                fill='toself',
                name=f"ID: {prop_id} ({location_name})"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation based on comparison
        st.subheader("Comparison Analysis")
        
        best_value_id = metrics_df.iloc[metrics_df['cash_on_cash_roi'].argmax()]['property_id']
        best_yield_id = metrics_df.iloc[metrics_df['rental_yield'].argmax()]['property_id']
        best_cash_flow_id = metrics_df.iloc[metrics_df['monthly_cash_flow'].argmax()]['property_id']
        
        st.markdown(f"""
            **Comparison Analysis:**
            
            - **Best Return on Investment:** Property ID {best_value_id} with {metrics_df.iloc[metrics_df['cash_on_cash_roi'].argmax()]['cash_on_cash_roi']:.2%} cash-on-cash ROI
            - **Best Rental Yield:** Property ID {best_yield_id} with {metrics_df.iloc[metrics_df['rental_yield'].argmax()]['rental_yield']:.2%} gross rental yield
            - **Best Cash Flow:** Property ID {best_cash_flow_id} with {format_currency(metrics_df.iloc[metrics_df['monthly_cash_flow'].argmax()]['monthly_cash_flow'])} monthly cash flow
        """)
        
        if best_value_id == best_yield_id and best_value_id == best_cash_flow_id:
            st.success(f"Property ID {best_value_id} is the best overall investment based on all metrics.")
        else:
            best_overall = metrics_df.iloc[((metrics_df['cash_on_cash_roi'] / metrics_df['cash_on_cash_roi'].max()) + 
                                             (metrics_df['rental_yield'] / metrics_df['rental_yield'].max()) + 
                                             (metrics_df['monthly_cash_flow'] / metrics_df['monthly_cash_flow'].max())).argmax()]['property_id']
            st.info(f"Property ID {best_overall} appears to be the best overall investment when considering all metrics.")
    
    else:
        st.info("Please select at least 2 properties to compare.")

# Recommendations page
elif page == "Recommendations":
    st.header("Personalized Property Recommendations")
    
    st.markdown("""
        Get personalized property recommendations based on your investment goals and preferences.
        Our AI analyzes property data to find the best matches for your criteria.
    """)
    
    # Investment preference inputs
    st.subheader("Your Investment Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        goal = st.selectbox(
            "Investment Goal",
            ["Cash Flow", "Appreciation", "Balanced"]
        )
        
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ["Short-term (1-3 years)", "Medium-term (4-7 years)", "Long-term (8+ years)"],
            index=1
        )
    
    with col2:
        budget_min = st.number_input(
            "Minimum Budget",
            min_value=50000,
            max_value=10000000,
            value=st.session_state.user_preferences['budget_min'],
            step=10000
        )
        
        budget_max = st.number_input(
            "Maximum Budget",
            min_value=budget_min,
            max_value=10000000,
            value=st.session_state.user_preferences['budget_max'],
            step=10000
        )
        
        preferred_locations = st.multiselect(
            "Preferred Locations",
            sorted(df['location'].unique().tolist()),
            default=st.session_state.user_preferences['location'] if st.session_state.user_preferences['location'] != 'All' else []
        )
    
    # Update user preferences in session state
    st.session_state.user_preferences['budget_min'] = budget_min
    st.session_state.user_preferences['budget_max'] = budget_max
    st.session_state.user_preferences['location'] = preferred_locations[0] if preferred_locations else 'All'
    
    # Get recommendations when button is clicked
    if st.button("Get Recommendations"):
        with st.spinner("Analyzing properties and generating recommendations..."):
            # Define user profile
            user_profile = {
                'goal': goal,
                'risk_tolerance': risk_tolerance,
                'investment_horizon': investment_horizon,
                'budget_min': budget_min,
                'budget_max': budget_max,
                'preferred_locations': preferred_locations
            }
            
            # Generate recommendations
            recommendations = recommend_properties(df, user_profile)
            
            if recommendations.empty:
                st.warning("No properties match your criteria. Try broadening your search parameters.")
            else:
                st.success(f"Found {len(recommendations)} properties matching your criteria!")
                
                # Display recommendations
                st.subheader("Recommended Properties")
                
                # Create tabs for different recommendation categories
                tabs = st.tabs(["All Recommendations", "Best Cash Flow", "Best Appreciation", "Best Value"])
                
                with tabs[0]:
                    # Display all recommendations
                    for i, (_, prop) in enumerate(recommendations.iterrows()):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Use a placeholder for property image
                            st.info(f"Property ID: {prop['property_id']}")
                        
                        with col2:
                            st.subheader(f"{prop['property_type']} in {prop['location']}")
                            st.write(f"{int(prop['bedrooms'])} bed, {int(prop['bathrooms'])} bath | {int(prop['sqft'])} sqft")
                            st.write(f"Price: {format_currency(prop['price'])}")
                            st.write(f"Estimated ROI: {prop['estimated_roi']:.2%}")
                            
                            # Calculate key metrics
                            monthly_rent = prop['estimated_monthly_rent'] if 'estimated_monthly_rent' in prop else prop['price'] * 0.008
                            cap_rate = (monthly_rent * 12 - (prop['price'] * 0.02)) / prop['price']  # Simple cap rate calculation
                            
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            metrics_col1.metric("Monthly Rent", format_currency(monthly_rent))
                            metrics_col2.metric("Cap Rate", f"{cap_rate:.2%}")
                            metrics_col3.metric("Price/Sqft", format_currency(prop['price'] / prop['sqft']))
                        
                        match_score = prop['match_score'] if 'match_score' in prop else 0.85
                        st.progress(match_score, text=f"Match Score: {match_score:.0%}")
                        st.markdown("---")
                
                with tabs[1]:
                    # Best cash flow properties
                    cash_flow_recs = recommendations.copy()
                    cash_flow_recs['monthly_cash_flow'] = cash_flow_recs.apply(
                        lambda x: (x['estimated_monthly_rent'] if 'estimated_monthly_rent' in x else x['price'] * 0.008) - 
                                (x['price'] * 0.8 * 0.045 / 12) - (x['price'] * 0.02 / 12),
                        axis=1
                    )
                    cash_flow_recs = cash_flow_recs.sort_values('monthly_cash_flow', ascending=False).head(3)
                    
                    for i, (_, prop) in enumerate(cash_flow_recs.iterrows()):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.info(f"Property ID: {prop['property_id']}")
                        
                        with col2:
                            st.subheader(f"{prop['property_type']} in {prop['location']}")
                            st.write(f"Price: {format_currency(prop['price'])}")
                            st.write(f"Monthly Cash Flow: {format_currency(prop['monthly_cash_flow'])}")
                            st.progress(min(1.0, prop['monthly_cash_flow'] / 1000), text="Cash Flow Rating")
                        
                        st.markdown("---")
                
                with tabs[2]:
                    # Best appreciation properties
                    appreciation_recs = recommendations.copy()
                    appreciation_recs['appreciation_potential'] = appreciation_recs.apply(
                        lambda x: (0.7 * (df[df['location'] == x['location']]['price'].pct_change().mean())) + 
                                 (0.3 * (x['estimated_roi'] if 'estimated_roi' in x else 0.05)),
                        axis=1
                    )
                    appreciation_recs = appreciation_recs.sort_values('appreciation_potential', ascending=False).head(3)
                    
                    for i, (_, prop) in enumerate(appreciation_recs.iterrows()):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.info(f"Property ID: {prop['property_id']}")
                        
                        with col2:
                            st.subheader(f"{prop['property_type']} in {prop['location']}")
                            st.write(f"Price: {format_currency(prop['price'])}")
                            st.write(f"Appreciation Potential: {prop['appreciation_potential']:.2%} annually")
                            st.progress(min(1.0, prop['appreciation_potential'] / 0.1), text="Appreciation Rating")
                        
                        st.markdown("---")
                
                with tabs[3]:
                    # Best value properties
                    value_recs = recommendations.copy()
                    value_recs['value_score'] = value_recs.apply(
                        lambda x: (x['estimated_roi'] if 'estimated_roi' in x else 0.05) * 
                                 (df['price_per_sqft'].mean() / (x['price'] / x['sqft'])),
                        axis=1
                    )
                    value_recs = value_recs.sort_values('value_score', ascending=False).head(3)
                    
                    for i, (_, prop) in enumerate(value_recs.iterrows()):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.info(f"Property ID: {prop['property_id']}")
                        
                        with col2:
                            st.subheader(f"{prop['property_type']} in {prop['location']}")
                            st.write(f"Price: {format_currency(prop['price'])}")
                            st.write(f"Price/Sqft: {format_currency(prop['price'] / prop['sqft'])}")
                            location_avg = df[df['location'] == prop['location']]['price_per_sqft'].mean()
                            discount = (1 - (prop['price'] / prop['sqft']) / location_avg) * 100
                            st.write(f"{discount:.1f}% below area average price/sqft")
                            st.progress(min(1.0, prop['value_score'] / 2), text="Value Rating")
                        
                        st.markdown("---")
                
                # Market insights for recommended properties
                st.subheader("Market Insights for Recommended Areas")
                
                # Get unique locations from recommendations
                rec_locations = recommendations['location'].unique()
                
                for location in rec_locations:
                    location_data = df[df['location'] == location]
                    avg_price = location_data['price'].mean()
                    avg_price_change = location_data['price'].pct_change().mean()
                    avg_dom = location_data['days_on_market'].mean() if 'days_on_market' in location_data else 45
                    
                    st.write(f"### {location}")
                    
                    # Location metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Price", format_currency(avg_price))
                    col2.metric("Price Change (YoY)", f"{avg_price_change:.2%}")
                    col3.metric("Avg. Days on Market", f"{avg_dom:.0f}")
                    
                    # Market description - generated based on the data
                    if avg_price_change > 0.05:
                        market_type = "Hot Seller's Market"
                        description = "Rapidly appreciating area with strong demand. Good for appreciation but cash flow may be limited."
                    elif avg_price_change > 0.02:
                        market_type = "Strong Market"
                        description = "Steady growth with good appreciation potential. Balance of cash flow and equity growth."
                    elif avg_price_change > -0.02:
                        market_type = "Balanced Market"
                        description = "Stable prices with moderate growth. Good for balanced investment strategies."
                    else:
                        market_type = "Buyer's Market"
                        description = "Declining prices but potential for cash flow. Look for undervalued properties."
                    
                    st.info(f"**Market Type:** {market_type}\n\n{description}")
    else:
        st.info("Fill in your preferences and click 'Get Recommendations' to see personalized property suggestions.")

# Footer with information
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>Real Estate Analytics Platform - Powered by AI</p>
        <p>Data is for demonstration purposes only. Always consult with a licensed real estate professional before making investment decisions.</p>
    </div>
""", unsafe_allow_html=True)
