import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_real_estate_data, get_available_cities, get_available_property_types
from datetime import datetime, timedelta

def show():
    """Display the Market Trends page"""
    st.title("Market Trends Analysis")

    # Load data
    df = load_real_estate_data()

    if df.empty:
        st.warning("No data available for analysis. Please check the data source.")
        return

    # Add tabs for different trend analyses
    tabs = st.tabs(["Price Trends", "Inventory Analysis", "Market Heatmap", "Appreciation Forecast", "Market Indicators"])

    with tabs[0]:
        show_price_trends(df)

    with tabs[1]:
        show_inventory_analysis(df)

    with tabs[2]:
        show_market_heatmap(df)

    with tabs[3]:
        show_appreciation_forecast(df)

    with tabs[4]:
        show_market_indicators(df)

def show_price_trends(df):
    """Display price trends analysis"""
    st.subheader("Price Trends Analysis")
    st.write("Analyze price trends across different locations and property types over time.")

    # Create filters for analysis
    col1, col2, col3 = st.columns(3)

    with col1:
        # Get available cities
        cities = get_available_cities(df)
        selected_cities = st.multiselect(
            "Select Cities",
            options=cities,
            default=cities[:3] if len(cities) > 3 else cities
        )

    with col2:
        # Get available property types
        property_types = get_available_property_types(df)
        selected_property_types = st.multiselect(
            "Select Property Types",
            options=property_types,
            default=property_types[0] if property_types else None
        )

    with col3:
        # Get time range if available
        if 'last_sold_date' in df.columns:
            df['last_sold_date'] = pd.to_datetime(df['last_sold_date'], errors='coerce')
            min_date = df['last_sold_date'].min()
            max_date = df['last_sold_date'].max()

            # Default to last 2 years if dates are available
            default_start = max_date - timedelta(days=730) if pd.notnull(max_date) else None
            default_end = max_date if pd.notnull(max_date) else None

            # Date selection
            date_range = st.date_input(
                "Date Range",
                value=[default_start, default_end] if default_start and default_end else None,
                min_value=min_date.date() if pd.notnull(min_date) else None,
                max_value=max_date.date() if pd.notnull(max_date) else None,
                key="price_trends_date"
            )

            # Convert to list if it's a tuple
            if isinstance(date_range, tuple):
                date_range = list(date_range)

            # Handle cases where date_range is a single date
            if isinstance(date_range, list) and len(date_range) == 1:
                date_range = [date_range[0], date_range[0]]

            # Apply date filter if valid range selected
            if isinstance(date_range, list) and len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df[
                    (df['last_sold_date'] >= pd.Timestamp(start_date)) & 
                    (df['last_sold_date'] <= pd.Timestamp(end_date))
                ]
            else:
                df_filtered = df.copy()
        else:
            df_filtered = df.copy()
            st.info("Date information not available in the dataset.")

    # Apply city and property type filters
    if selected_cities:
        df_filtered = df_filtered[df_filtered['city'].isin(selected_cities)]

    if selected_property_types:
        df_filtered = df_filtered[df_filtered['property_type'].isin(selected_property_types)]

    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return

    # Analyze price trends
    st.markdown("#### Price Trends Over Time")

    if 'last_sold_date' in df_filtered.columns and 'price' in df_filtered.columns:
        # Group by date and calculate average price
        df_filtered['year_month'] = df_filtered['last_sold_date'].dt.to_period('M')

        # Group by city and month
        if selected_cities and len(selected_cities) > 1:
            price_trends = df_filtered.groupby(['year_month', 'city'])['price'].mean().reset_index()
            price_trends['year_month'] = price_trends['year_month'].dt.to_timestamp()

            # Create line chart
            fig = px.line(
                price_trends,
                x='year_month',
                y='price',
                color='city',
                title="Average Property Price by City",
                labels={'year_month': 'Date', 'price': 'Average Price ($)', 'city': 'City'},
                markers=True
            )
        else:
            # Group by property type and month
            price_trends = df_filtered.groupby(['year_month', 'property_type'])['price'].mean().reset_index()
            price_trends['year_month'] = price_trends['year_month'].dt.to_timestamp()

            # Create line chart
            fig = px.line(
                price_trends,
                x='year_month',
                y='price',
                color='property_type',
                title="Average Property Price by Property Type",
                labels={'year_month': 'Date', 'price': 'Average Price ($)', 'property_type': 'Property Type'},
                markers=True
            )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Price ($)",
            legend_title="",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Price trend analysis requires date and price information, which is not available in the dataset.")

    # Price per square foot trends
    st.markdown("#### Price per Square Foot Analysis")

    if all(col in df_filtered.columns for col in ['price', 'sqft']):
        # Calculate price per square foot
        df_filtered['price_per_sqft'] = df_filtered['price'] / df_filtered['sqft']

        # Group by city
        if selected_cities and len(selected_cities) > 1:
            price_sqft_by_city = df_filtered.groupby('city')['price_per_sqft'].mean().reset_index()

            # Create bar chart
            fig = px.bar(
                price_sqft_by_city,
                x='city',
                y='price_per_sqft',
                color='price_per_sqft',
                title="Average Price per Square Foot by City",
                labels={'city': 'City', 'price_per_sqft': 'Price per sqft ($)'},
                color_continuous_scale='Viridis'
            )
        else:
            # Group by property type
            price_sqft_by_type = df_filtered.groupby('property_type')['price_per_sqft'].mean().reset_index()

            # Create bar chart
            fig = px.bar(
                price_sqft_by_type,
                x='property_type',
                y='price_per_sqft',
                color='price_per_sqft',
                title="Average Price per Square Foot by Property Type",
                labels={'property_type': 'Property Type', 'price_per_sqft': 'Price per sqft ($)'},
                color_continuous_scale='Viridis'
            )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Price per sqft ($)",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Price per square foot analysis requires price and square footage information, which is not available in the dataset.")

    # Price distribution by property type
    st.markdown("#### Price Distribution Analysis")

    if 'price' in df_filtered.columns:
        if 'property_type' in df_filtered.columns:
            # Create box plot
            fig = px.box(
                df_filtered,
                x='property_type',
                y='price',
                color='property_type',
                title="Price Distribution by Property Type",
                labels={'property_type': 'Property Type', 'price': 'Price ($)'}
            )

            fig.update_layout(
                xaxis_title="Property Type",
                yaxis_title="Price ($)",
                showlegend=False,
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create histogram
            fig = px.histogram(
                df_filtered,
                x='price',
                nbins=30,
                title="Price Distribution",
                labels={'price': 'Price ($)', 'count': 'Number of Properties'},
                color_discrete_sequence=['#FF4B4B']
            )

            fig.update_layout(
                xaxis_title="Price ($)",
                yaxis_title="Number of Properties",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Price distribution analysis requires price information, which is not available in the dataset.")

    # Year-over-year price change
    st.markdown("#### Year-over-Year Price Change")

    if 'last_sold_date' in df_filtered.columns and 'price' in df_filtered.columns:
        df_filtered['year'] = df_filtered['last_sold_date'].dt.year

        # Group by year and calculate average price
        yearly_prices = df_filtered.groupby('year')['price'].mean().reset_index()

        if len(yearly_prices) > 1:
            # Calculate year-over-year change
            yearly_prices['previous_price'] = yearly_prices['price'].shift(1)
            yearly_prices['yoy_change_pct'] = (yearly_prices['price'] - yearly_prices['previous_price']) / yearly_prices['previous_price'] * 100
            yearly_prices = yearly_prices.dropna()

            # Create bar chart
            fig = px.bar(
                yearly_prices,
                x='year',
                y='yoy_change_pct',
                title="Year-over-Year Price Change (%)",
                labels={'year': 'Year', 'yoy_change_pct': 'Price Change (%)'},
                color='yoy_change_pct',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )

            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Price Change (%)",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Year-over-year analysis requires data from multiple years, which is not available in the dataset.")
    else:
        st.info("Year-over-year analysis requires date and price information, which is not available in the dataset.")

def show_inventory_analysis(df):
    """Display inventory analysis"""
    st.subheader("Inventory Analysis")
    st.write("Analyze real estate inventory trends and market supply.")

    # Create filters
    col1, col2 = st.columns(2)

    with col1:
        # Get available cities
        cities = get_available_cities(df)
        selected_city = st.selectbox(
            "Select City",
            options=["All"] + cities,
            key="inventory_city"
        )

    with col2:
        # Get available property types
        property_types = get_available_property_types(df)
        selected_property_type = st.selectbox(
            "Select Property Type",
            options=["All"] + property_types,
            key="inventory_property_type"
        )

    # Apply filters
    df_filtered = df.copy()

    if selected_city != "All":
        df_filtered = df_filtered[df_filtered['city'] == selected_city]

    if selected_property_type != "All":
        df_filtered = df_filtered[df_filtered['property_type'] == selected_property_type]

    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return

    # Inventory by city
    st.markdown("#### Inventory by City")

    if 'city' in df_filtered.columns:
        city_inventory = df_filtered['city'].value_counts().reset_index()
        city_inventory.columns = ['city', 'property_count']

        # Create bar chart
        fig = px.bar(
            city_inventory.head(10),
            x='city',
            y='property_count',
            title="Property Inventory by City (Top 10)",
            labels={'city': 'City', 'property_count': 'Number of Properties'},
            color='property_count',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title="City",
            yaxis_title="Number of Properties",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Inventory by city analysis requires city information, which is not available in the dataset.")

    # Inventory by property type
    st.markdown("#### Inventory by Property Type")

    if 'property_type' in df_filtered.columns:
        type_inventory = df_filtered['property_type'].value_counts().reset_index()
        type_inventory.columns = ['property_type', 'property_count']

        # Create pie chart
        fig = px.pie(
            type_inventory,
            values='property_count',
            names='property_type',
            title="Property Inventory by Type",
            hole=0.4
        )

        fig.update_layout(
            legend_title="Property Type",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Inventory by property type analysis requires property type information, which is not available in the dataset.")

    # Days on market analysis
    st.markdown("#### Days on Market Analysis")

    if 'days_on_market' in df_filtered.columns:
        # Calculate average days on market by city
        if 'city' in df_filtered.columns:
            days_by_city = df_filtered.groupby('city')['days_on_market'].mean().reset_index()
            days_by_city = days_by_city.sort_values('days_on_market', ascending=False)

            # Create bar chart
            fig = px.bar(
                days_by_city.head(10),
                x='city',
                y='days_on_market',
                title="Average Days on Market by City (Top 10)",
                labels={'city': 'City', 'days_on_market': 'Average Days on Market'},
                color='days_on_market',
                color_continuous_scale='Reds'
            )

            fig.update_layout(
                xaxis_title="City",
                yaxis_title="Average Days on Market",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Days on market distribution
        fig = px.histogram(
            df_filtered,
            x='days_on_market',
            nbins=30,
            title="Days on Market Distribution",
            labels={'days_on_market': 'Days on Market', 'count': 'Number of Properties'},
            color_discrete_sequence=['#FF4B4B']
        )

        fig.update_layout(
            xaxis_title="Days on Market",
            yaxis_title="Number of Properties",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Market speed metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_days = df_filtered['days_on_market'].mean()
            st.metric("Average Days on Market", f"{avg_days:.1f} days")

        with col2:
            median_days = df_filtered['days_on_market'].median()
            st.metric("Median Days on Market", f"{median_days:.1f} days")

        with col3:
            fast_market = (df_filtered['days_on_market'] < 30).mean() * 100
            st.metric("Properties Sold < 30 Days", f"{fast_market:.1f}%")
    else:
        st.info("Days on market analysis requires days on market information, which is not available in the dataset.")

    # Bedrooms inventory analysis
    st.markdown("#### Bedrooms Inventory Analysis")

    if 'bedrooms' in df_filtered.columns:
        bedroom_inventory = df_filtered['bedrooms'].value_counts().reset_index()
        bedroom_inventory.columns = ['bedrooms', 'property_count']
        bedroom_inventory = bedroom_inventory.sort_values('bedrooms')

        # Create bar chart
        fig = px.bar(
            bedroom_inventory,
            x='bedrooms',
            y='property_count',
            title="Property Inventory by Bedroom Count",
            labels={'bedrooms': 'Number of Bedrooms', 'property_count': 'Number of Properties'},
            color='bedrooms',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title="Number of Bedrooms",
            yaxis_title="Number of Properties",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Bedrooms inventory analysis requires bedroom information, which is not available in the dataset.")

def show_market_heatmap(df):
    """Display market heatmap analysis"""
    st.subheader("Market Heatmap")
    st.write("Visualize geographic heat maps of real estate prices and market activity.")

    # Check for required coordinates
    if not all(col in df.columns for col in ['latitude', 'longitude']):
        st.warning("Geographic visualization requires latitude and longitude coordinates, which are not available in the dataset.")
        return

    # Create filters
    col1, col2 = st.columns(2)

    with col1:
        # Get available cities
        cities = get_available_cities(df)
        selected_city = st.selectbox(
            "Select City",
            options=["All"] + cities,
            key="heatmap_city"
        )

    with col2:
        # Metric selection
        available_metrics = []

        if 'price' in df.columns:
            available_metrics.append("Price")

        if 'price_per_sqft' in df.columns or all(col in df.columns for col in ['price', 'sqft']):
            available_metrics.append("Price per sqft")

        if 'days_on_market' in df.columns:
            available_metrics.append("Days on Market")

        if 'monthly_rent_estimate' in df.columns:
            available_metrics.append("Rental Estimate")

        if all(col in df.columns for col in ['monthly_rent_estimate', 'price']):
            available_metrics.append("Rental Yield")

        selected_metric = st.selectbox(
            "Select Metric",
            options=available_metrics,
            key="heatmap_metric",
            index=0 if available_metrics else None
        )

    # Apply city filter
    df_filtered = df.copy()

    if selected_city != "All":
        df_filtered = df_filtered[df_filtered['city'] == selected_city]

    # Prepare data for visualization
    if selected_metric == "Price":
        metric_col = 'price'
        title = "Property Price Heatmap"
        hover_data = ['price', 'bedrooms', 'bathrooms', 'sqft', 'property_type']
    elif selected_metric == "Price per sqft":
        if 'price_per_sqft' not in df_filtered.columns and all(col in df_filtered.columns for col in ['price', 'sqft']):
            df_filtered['price_per_sqft'] = df_filtered['price'] / df_filtered['sqft']

        metric_col = 'price_per_sqft'
        title = "Price per sqft Heatmap"
        hover_data = ['price_per_sqft', 'price', 'sqft', 'property_type']
    elif selected_metric == "Days on Market":
        metric_col = 'days_on_market'
        title = "Days on Market Heatmap"
        hover_data = ['days_on_market', 'price', 'property_type']
    elif selected_metric == "Rental Estimate":
        metric_col = 'monthly_rent_estimate'
        title = "Monthly Rental Estimate Heatmap"
        hover_data = ['monthly_rent_estimate', 'price', 'bedrooms', 'bathrooms']
    elif selected_metric == "Rental Yield":
        df_filtered['annual_rent'] = df_filtered['monthly_rent_estimate'] * 12
        df_filtered['rental_yield'] = (df_filtered['annual_rent'] / df_filtered['price']) * 100
        metric_col = 'rental_yield'
        title = "Rental Yield Heatmap"
        hover_data = ['rental_yield', 'monthly_rent_estimate', 'price']
    else:
        st.warning("Please select a valid metric for the heatmap.")
        return

    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return

    # Create density heatmap
    st.markdown(f"#### {title}")

    fig = px.density_mapbox(
        df_filtered,
        lat='latitude',
        lon='longitude',
        z=metric_col,
        radius=15,
        center=dict(lat=df_filtered['latitude'].mean(), lon=df_filtered['longitude'].mean()),
        zoom=10,
        mapbox_style="open-street-map",
        hover_data=hover_data
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Property scatter map
    st.markdown("#### Property Map")

    fig = px.scatter_mapbox(
        df_filtered,
        lat='latitude',
        lon='longitude',
        color=metric_col,
        size=metric_col,
        size_max=15,
        zoom=10,
        mapbox_style="open-street-map",
        hover_data=hover_data
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Neighborhood analysis
    st.markdown("#### Neighborhood Analysis")

    if 'neighborhood' in df_filtered.columns and metric_col in df_filtered.columns:
        neighborhood_metrics = df_filtered.groupby('neighborhood')[metric_col].mean().reset_index()
        neighborhood_metrics = neighborhood_metrics.sort_values(metric_col, ascending=False)

        # Create bar chart
        fig = px.bar(
            neighborhood_metrics.head(10),
            x='neighborhood',
            y=metric_col,
            title=f"Top 10 Neighborhoods by {selected_metric}",
            labels={'neighborhood': 'Neighborhood', metric_col: selected_metric},
            color=metric_col,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title="Neighborhood",
            yaxis_title=selected_metric,
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Neighborhood analysis requires neighborhood information, which is not available in the dataset.")

def show_appreciation_forecast(df):
    """Display appreciation forecast analysis"""
    st.subheader("Appreciation Forecast")
    st.write("Analyze historical appreciation rates and forecast future trends.")

    # Check for required data
    if not all(col in df.columns for col in ['last_sold_date', 'price']):
        st.warning("Appreciation forecast requires historical sales data with dates and prices, which is not available in the dataset.")
        return

    # Create filters
    col1, col2 = st.columns(2)

    with col1:
        # Get available cities
        cities = get_available_cities(df)
        selected_city = st.selectbox(
            "Select City",
            options=["All"] + cities,
            key="forecast_city"
        )

    with col2:
        # Get available property types
        property_types = get_available_property_types(df)
        selected_property_type = st.selectbox(
            "Select Property Type",
            options=["All"] + property_types,
            key="forecast_property_type"
        )

    # Apply filters
    df_filtered = df.copy()

    if selected_city != "All":
        df_filtered = df_filtered[df_filtered['city'] == selected_city]

    if selected_property_type != "All":
        df_filtered = df_filtered[df_filtered['property_type'] == selected_property_type]

    # Convert to datetime
    df_filtered['last_sold_date'] = pd.to_datetime(df_filtered['last_sold_date'], errors='coerce')

    # Extract year and calculate annual metrics
    df_filtered['year'] = df_filtered['last_sold_date'].dt.year

    # Group by year
    yearly_prices = df_filtered.groupby('year')['price'].mean().reset_index()

    if len(yearly_prices) < 2:
        st.warning("Appreciation forecast requires data from multiple years, which is not available in the dataset.")
        return

    # Calculate year-over-year appreciation
    yearly_prices['previous_price'] = yearly_prices['price'].shift(1)
    yearly_prices['yoy_appreciation'] = (yearly_prices['price'] - yearly_prices['previous_price']) / yearly_prices['previous_price'] * 100
    yearly_prices = yearly_prices.dropna()

    # Calculate average annual appreciation rate
    avg_appreciation = yearly_prices['yoy_appreciation'].mean()

    # Display historical appreciation
    st.markdown("#### Historical Appreciation Rates")

    # Create bar chart
    fig = px.bar(
        yearly_prices,
        x='year',
        y='yoy_appreciation',
        title="Historical Annual Appreciation Rates",
        labels={'year': 'Year', 'yoy_appreciation': 'Appreciation Rate (%)'},
        color='yoy_appreciation',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )

    # Add average line
    fig.add_hline(
        y=avg_appreciation,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Avg: {avg_appreciation:.2f}%",
        annotation_position="bottom right"
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Appreciation Rate (%)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast future appreciation
    st.markdown("#### Appreciation Forecast")

    # User inputs for forecast
    col1, col2, col3 = st.columns(3)

    with col1:
        forecast_years = st.slider(
            "Forecast Years",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="forecast_years"
        )

    with col2:
        baseline_rate = st.number_input(
            "Baseline Appreciation Rate (%)",
            min_value=-10.0,
            max_value=20.0,
            value=avg_appreciation,
            step=0.1,
            key="baseline_rate"
        )

    with col3:
        volatility = st.slider(
            "Market Volatility",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            key="market_volatility"
        )

    # Generate forecast
    last_year = yearly_prices['year'].max()
    last_price = yearly_prices[yearly_prices['year'] == last_year]['price'].values[0]

    forecast_years_list = list(range(last_year + 1, last_year + forecast_years + 1))
    forecast_prices = [last_price]

    # Generate multiple scenarios
    scenarios = ['Pessimistic', 'Baseline', 'Optimistic']
    scenario_rates = [baseline_rate - 2, baseline_rate, baseline_rate + 2]  # Adjust rates for scenarios

    forecast_data = []

    for scenario, rate in zip(scenarios, scenario_rates):
        scenario_prices = [last_price]

        for year in range(forecast_years):
            # Add some randomness based on volatility
            random_factor = np.random.normal(0, volatility)
            year_rate = rate + random_factor
            new_price = scenario_prices[-1] * (1 + year_rate / 100)
            scenario_prices.append(new_price)

        # Skip the first price (it's the last historical price)
        scenario_prices = scenario_prices[1:]

        for year, price in zip(forecast_years_list, scenario_prices):
            forecast_data.append({
                'year': year,
                'price': price,
                'scenario': scenario
            })

    # Create forecast dataframe
    forecast_df = pd.DataFrame(forecast_data)

    # Create line chart
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=yearly_prices['year'],
        y=yearly_prices['price'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='black', width=3)
    ))

    # Add forecast scenarios
    colors = {'Pessimistic': 'rgba(255, 0, 0, 0.7)', 'Baseline': 'rgba(0, 0, 255, 0.7)', 'Optimistic': 'rgba(0, 255, 0, 0.7)'}

    for scenario in scenarios:
        scenario_data = forecast_df[forecast_df['scenario'] == scenario]

        fig.add_trace(go.Scatter(
            x=scenario_data['year'],
            y=scenario_data['price'],
            mode='lines',
            name=f"{scenario} ({scenario_rates[scenarios.index(scenario)]:.1f}%)",
            line=dict(color=colors[scenario], width=2, dash='dash')
        ))

    fig.update_layout(
        title="Property Value Forecast",
        xaxis_title="Year",
        yaxis_title="Property Value ($)",
        legend_title="Scenario",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display forecast table
    st.markdown("#### Forecast Summary")

    # Create summary table
    summary_data = []

    for scenario in scenarios:
        scenario_data = forecast_df[forecast_df['scenario'] == scenario]
        final_price = scenario_data[scenario_data['year'] == forecast_years_list[-1]]['price'].values[0]
        total_appreciation = (final_price - last_price) / lastprice * 100
        annual_rate = scenario_rates[scenarios.index(scenario)]

        summary_data.append({
            'Scenario': scenario,
            'Annual Rate': f"{annual_rate:.2f}%",
            'Final Value': f"${final_price:,.2f}",
            'Total Appreciation': f"{total_appreciation:.2f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, hide_index=True)

    # Market factors affecting appreciation
    st.markdown("#### Factors Affecting Appreciation")

    # List of common factors
    factors = [
        {
            'Factor': 'Local Economy',
            'Impact': 'High',
            'Description': 'Job growth, wage increases, and economic development directly influence property values.'
        },
        {
            'Factor': 'Population Growth',
            'Impact': 'High',
            'Description': 'Areas with increasing population typically see stronger appreciation due to housing demand.'
        },
        {
            'Factor': 'Housing Supply',
            'Impact': 'Medium',
            'Description': 'Limited new construction or housing inventory can drive price increases in desirable areas.'
        },
        {
            'Factor': 'Interest Rates',
            'Impact': 'Medium',
            'Description': 'Lower rates increase buying power and can push property values higher.'
        },
        {
            'Factor': 'Infrastructure Development',
            'Impact': 'Medium',
            'Description': 'New transportation, schools, and amenities can boost neighborhood values.'
        }
    ]

    factors_df = pd.DataFrame(factors)
    st.dataframe(factors_df, hide_index=True)

    # Appreciation hotspots
    if 'city' in df.columns and 'last_sold_date' in df.columns and 'price' in df.columns:
        st.markdown("#### Appreciation Hotspots")

        # Calculate appreciation by city
        df_with_year = df.copy()
        df_with_year['last_sold_date'] = pd.to_datetime(df_with_year['last_sold_date'], errors='coerce')
        df_with_year['year'] = df_with_year['last_sold_date'].dt.year

        # Get recent years
        recent_years = sorted(df_with_year['year'].unique())[-2:]

        if len(recent_years) >= 2:
            previous_year, current_year = recent_years

            # Calculate average prices by city for each year and reset index
            previous_prices = df_with_year[df_with_year['year'] == previous_year].groupby('city')['price'].mean().reset_index()
            current_prices = df_with_year[df_with_year['year'] == current_year].groupby('city')['price'].mean().reset_index()

            # Merge the dataframes
            appreciation_rates = pd.merge(previous_prices, current_prices, 
                                        on='city', 
                                        suffixes=('_previous', '_current'))

            # Calculate appreciation
            appreciation_rates['appreciation'] = ((appreciation_rates['price_current'] - 
                                                appreciation_rates['price_previous']) / 
                                                appreciation_rates['price_previous'] * 100)

            # Display top appreciation markets
            fig = px.bar(
                appreciation_rates.head(10),
                x='city',
                y='appreciation',
                title=f"Top Appreciation Markets ({previous_year} to {current_year})",
                labels={'city': 'City', 'appreciation': 'Appreciation Rate (%)'},
                color='appreciation',
                color_continuous_scale='Blues'
            )

            fig.update_layout(
                xaxis_title="City",
                yaxis_title="Appreciation Rate (%)",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Appreciation hotspots require data from multiple years, which is not available.")
    else:
        st.info("Appreciation hotspots analysis requires city, date, and price information, which is not available.")

def show_market_indicators(df):
    """Display market indicators analysis"""
    st.subheader("Market Indicators")
    st.write("Analyze key market health indicators and metrics.")

    # Check for required data
    if df.empty:
        st.warning("No data available for market indicators analysis.")
        return

    # Create key market metrics dashboard
    st.markdown("#### Key Market Metrics")

    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Median price
        if 'price' in df.columns:
            median_price = df['price'].median()
            st.metric("Median Price", f"${median_price:,.0f}")
        else:
            st.metric("Median Price", "N/A")

    with col2:
        # Average days on market
        if 'days_on_market' in df.columns:
            avg_dom = df['days_on_market'].mean()
            st.metric("Avg. Days on Market", f"{avg_dom:.1f}")
        else:
            st.metric("Avg. Days on Market", "N/A")

    with col3:
        # Inventory count
        inventory_count = len(df)
        st.metric("Active Listings", f"{inventory_count:,}")

    with col4:
        # Price to rent ratio
        if all(col in df.columns for col in ['price', 'monthly_rent_estimate']):
            df['annual_rent'] = df['monthly_rent_estimate'] * 12
            df['price_to_rent_ratio'] = df['price'] / df['annual_rent']
            avg_p2r = df['price_to_rent_ratio'].median()
            st.metric("Price-to-Rent Ratio", f"{avg_p2r:.1f}")
        else:
            st.metric("Price-to-Rent Ratio", "N/A")

    # Market health indicators
    st.markdown("#### Market Health Indicators")

    # Calculate market health score
    market_health = {}

    # Inventory turnover
    if 'days_on_market' in df.columns:
        avg_dom = df['days_on_market'].mean()
        if avg_dom < 30:
            turnover_score = 5  # Hot market
        elif avg_dom < 60:
            turnover_score = 4  # Strong market
        elif avg_dom < 90:
            turnover_score = 3  # Balanced market
        elif avg_dom < 120:
            turnover_score = 2  # Slow market
        else:
            turnover_score = 1  # Cold market

        market_health['Inventory Turnover'] = turnover_score

    # Price trends
    if 'last_sold_date' in df.columns and 'price' in df.columns:
        df['last_sold_date'] = pd.to_datetime(df['last_sold_date'], errors='coerce')
        df['year'] = df['last_sold_date'].dt.year

        yearly_prices = df.groupby('year')['price'].mean().reset_index()

        if len(yearly_prices) > 1:
            # Calculate year-over-year price change
            yearly_prices['previous_price'] = yearly_prices['price'].shift(1)
            yearly_prices['yoy_change'] = (yearly_prices['price'] - yearly_prices['previous_price']) / yearly_prices['previous_price'] * 100
            yearly_prices = yearly_prices.dropna()

            recent_change = yearly_prices.iloc[-1]['yoy_change']

            if recent_change > 10:
                price_score = 5  # Strong appreciation
            elif recent_change > 5:
                price_score = 4  # Healthy appreciation
            elif recent_change > 0:
                price_score = 3  # Mild appreciation
            elif recent_change > -5:
                price_score = 2  # Slight depreciation
            else:
                price_score = 1  # Strong depreciation

            market_health['Price Trends'] = price_score

    # Supply-demand balance
    if 'days_on_market' in df.columns and len(df) > 0:
        # Use days on market as a proxy for supply-demand balance
        fast_selling = (df['days_on_market'] < 30).mean()

        if fast_selling > 0.8:
            balance_score = 5  # Strong seller's market
        elif fast_selling > 0.6:
            balance_score = 4  # Seller's market
        elif fast_selling > 0.4:
            balance_score = 3  # Balanced market
        elif fast_selling > 0.2:
            balance_score = 2  # Buyer's market
        else:
            balance_score = 1  # Strong buyer's market

        market_health['Supply-Demand Balance'] = balance_score

    # Affordability
    if all(col in df.columns for col in ['price', 'monthly_rent_estimate']):
        df['price_to_rent_ratio'] = df['price'] / (df['monthly_rent_estimate'] * 12)
        avg_p2r = df['price_to_rent_ratio'].median()

        if avg_p2r < 15:
            affordability_score = 5  # Highly affordable
        elif avg_p2r < 20:
            affordability_score = 4  # Affordable
        elif avg_p2r < 25:
            affordability_score = 3  # Moderately affordable
        elif avg_p2r < 30:
            affordability_score = 2  # Expensive
        else:
            affordability_score = 1  # Highly expensive

        market_health['Affordability'] = affordability_score

    # Display market health indicators
    if market_health:
        # Convert to dataframe
        health_df = pd.DataFrame({
            'Indicator': list(market_health.keys()),
            'Score': list(market_health.values())
        })

        # Create radar chart
        categories = health_df['Indicator'].tolist()
        values = health_df['Score'].tolist()

        # Close the polygon
        values.append(values[0])
        categories_closed = categories + [categories[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name='Market Health'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display score table with interpretations
        interpretations = {
            'Inventory Turnover': [
                'Cold market with very slow sales',
                'Slow market with high inventory',
                'Balanced market with moderate sales pace',
                'Strong market with quick sales',
                'Hot market with very quick sales'
            ],
            'Price Trends': [
                'Strong price depreciation',
                'Slight price depreciation',
                'Mild price appreciation',
                'Healthy price appreciation',
                'Strong price appreciation'
            ],
            'Supply-Demand Balance': [
                'Strong buyer\'s market',
                'Buyer\'s market',
                'Balanced market',
                'Seller\'s market',
                'Strong seller\'s market'
            ],
            'Affordability': [
                'Highly expensive market',
                'Expensive market',
                'Moderately affordable market',
                'Affordable market',
                'Highly affordable market'
            ]
        }

        # Create interpretation column
        health_df['Interpretation'] = health_df.apply(
            lambda row: interpretations.get(row['Indicator'], [''] * 5)[int(row['Score']) - 1],
            axis=1
        )

        st.dataframe(health_df, hide_index=True)
    else:
        st.info("Not enough data to calculate market health indicators.")

    # Price distribution analysis
    st.markdown("#### Price Distribution Analysis")

    if 'price' in df.columns:
        # Calculate price percentiles
        percentiles = [10, 25, 50, 75, 90]
        price_percentiles = np.percentile(df['price'], percentiles)

        # Create price distribution plot
        fig = px.histogram(
            df,
            x='price',
            nbins=30,
            title="Price Distribution",
            labels={'price': 'Price ($)', 'count': 'Number of Properties'},
            color_discrete_sequence=['#FF4B4B']
        )

        # Add percentile lines
        colors = ['green', 'blue', 'red', 'blue', 'green']

        for percentile, price, color in zip(percentiles, price_percentiles, colors):
            fig.add_vline(
                x=price,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{percentile}th Percentile: ${price:,.0f}",
                annotation_position="top right"
            )

        fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Number of Properties",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display price tiers
        st.markdown("#### Price Tiers")

        # Create price tiers
        price_ranges = [
            (0, price_percentiles[0]),
            (price_percentiles[0], price_percentiles[1]),
            (price_percentiles[1], price_percentiles[3]),
            (price_percentiles[3], price_percentiles[4]),
            (price_percentiles[4], df['price'].max())
        ]

        tier_labels = ["Entry Level", "Affordable", "Mid-Range", "Premium", "Luxury"]

        price_tiers = []

        for (min_price, max_price), label in zip(price_ranges, tier_labels):
            tier_count = df[(df['price'] >= min_price) & (df['price'] < max_price)].shape[0]
            tier_percentage = tier_count / len(df) * 100

            price_tiers.append({
                'Tier': label,
                'Price Range': f"${min_price:,.0f} - ${max_price:,.0f}",
                'Count': tier_count,
                'Percentage': f"{tier_percentage:.1f}%"
            })

        # Display price tiers
        price_tiers_df = pd.DataFrame(price_tiers)
        st.dataframe(price_tiers_df, hide_index=True)
    else:
        st.info("Price distribution analysis requires price information, which is not available in the dataset.")

    # Market comparison
    st.markdown("#### Market Comparison")

    if 'city' in df.columns and 'price' in df.columns:
        # Calculate metrics by city
        city_metrics = df.groupby('city').agg({
            'price': 'median',
            'days_on_market': 'mean' if 'days_on_market' in df.columns else 'count',
            'sqft': 'median' if 'sqft' in df.columns else 'count'
        }).reset_index()

        # Calculate price per sqft if available
        if 'sqft' in df.columns:
            city_metrics['price_per_sqft'] = city_metrics['price'] / city_metrics['sqft']

        # Rename columns
        # Rename columns based on available data
        new_columns = ['City', 'Median Price']
        if 'days_on_market' in df.columns:
            new_columns.append('Avg. Days on Market')
        else:
            new_columns.append('Count')
        if 'sqft' in df.columns:
            new_columns.append('Median Sqft')
        else:
            new_columns.append('Count')
        city_metrics.columns = new_columns

        if 'price_per_sqft' in city_metrics.columns:
            city_metrics = city_metrics.rename(columns={'price_per_sqft': 'Price per Sqft'})

        # Sort by median price
        city_metrics = city_metrics.sort_values('Median Price', ascending=False)

        # Format columns
        formatted_metrics = city_metrics.copy()
        formatted_metrics['Median Price'] = formatted_metrics['Median Price'].apply(lambda x: f"${x:,.0f}")

        if 'Price per Sqft' in formatted_metrics.columns:
            formatted_metrics['Price per Sqft'] = formatted_metrics['Price per Sqft'].apply(lambda x: f"${x:.2f}")

        if 'Avg. Days on Market' in formatted_metrics.columns:
            formatted_metrics['Avg. Days on Market'] = formatted_metrics['Avg. Days on Market'].apply(lambda x: f"{x:.1f}")

        # Display comparison table
        st.dataframe(formatted_metrics, hide_index=True)

        # Create comparison chart
        st.markdown("#### Visual Market Comparison")

        metric_options = ['Median Price']

        if 'Price per Sqft' in city_metrics.columns:
            metric_options.append('Price per Sqft')

        if 'Avg. Days on Market' in city_metrics.columns:
            metric_options.append('Avg. Days on Market')

        comparison_metric = st.selectbox(
            "Select Comparison Metric",
            options=metric_options,
            key="comparison_metric"
        )

        # Filter for top 10 cities
        top_cities = city_metrics.head(10)

        # Create bar chart
        fig = px.bar(
            top_cities,
            x='City',
            y=comparison_metric,
            title=f"Top 10 Markets by {comparison_metric}",
            labels={'City': 'City', comparison_metric: comparison_metric},
            color=comparison_metric,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title="City",
            yaxis_title=comparison_metric,
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market comparison requires city and price information, which is not available in the dataset.")