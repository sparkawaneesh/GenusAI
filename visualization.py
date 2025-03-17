import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_price_distribution_chart(df):
    """
    Create a bar chart showing price distribution by location.
    
    Args:
        df (DataFrame): Property data
    
    Returns:
        Figure: Plotly figure object
    """
    # Group data by location and calculate statistics
    location_stats = df.groupby('location').agg(
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        count=('price', 'count')
    ).reset_index()
    
    # Sort by average price
    location_stats = location_stats.sort_values('avg_price', ascending=False)
    
    # Create figure
    fig = px.bar(
        location_stats,
        x='location',
        y='avg_price',
        text='count',
        title='Average Property Price by Location',
        labels={'location': 'Location', 'avg_price': 'Average Price ($)', 'count': 'Number of Properties'},
        color='avg_price',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Location',
        yaxis_title='Average Price ($)',
        coloraxis_showscale=False,
        xaxis_tickangle=-45
    )
    
    # Format price labels
    fig.update_traces(
        texttemplate='%{text} props',
        textposition='outside'
    )
    
    # Add median price as a line
    fig.add_trace(
        go.Scatter(
            x=location_stats['location'],
            y=location_stats['median_price'],
            mode='markers',
            name='Median Price',
            marker=dict(
                color='red',
                size=8,
                symbol='diamond'
            )
        )
    )
    
    return fig

def create_market_trends_chart(market_data):
    """
    Create a line chart showing market price trends.
    
    Args:
        market_data (dict): Market trend data
    
    Returns:
        Figure: Plotly figure object
    """
    # Extract historical data
    dates = market_data['historical_data']['dates']
    prices = market_data['historical_data']['prices']
    
    # Create DataFrame from data
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Create figure
    fig = px.line(
        df,
        x='date',
        y='price',
        title='Price Trends Over Time',
        labels={'date': 'Date', 'price': 'Average Price ($)'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Average Price ($)',
        hovermode='x unified'
    )
    
    # Add trend line (linear regression)
    x_numeric = np.arange(len(dates))
    z = np.polyfit(x_numeric, prices, 1)
    p = np.poly1d(z)
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=p(x_numeric),
            mode='lines',
            name='Trend Line',
            line=dict(
                color='red',
                dash='dash'
            )
        )
    )
    
    return fig

def create_property_comparison_chart(comparison_data):
    """
    Create a radar chart comparing properties.
    
    Args:
        comparison_data (DataFrame): Property comparison data
    
    Returns:
        Figure: Plotly figure object
    """
    # Normalize data for comparison
    normalized_data = comparison_data.copy()
    
    # Columns to normalize
    columns_to_normalize = ['price', 'sqft', 'bedrooms', 'bathrooms', 'estimated_roi']
    available_columns = [col for col in columns_to_normalize if col in normalized_data.columns]
    
    # Normalize each column to 0-1 scale
    for col in available_columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        if max_val > min_val:  # Avoid division by zero
            normalized_data[col + '_normalized'] = (normalized_data[col] - min_val) / (max_val - min_val)
        else:
            normalized_data[col + '_normalized'] = 0.5
    
    # Create radar chart
    fig = go.Figure()
    
    # Categories for radar chart
    categories = [
        'Price (inv)',      # Inverse of price (lower is better)
        'Square Footage',
        'Bedrooms',
        'Bathrooms',
        'ROI'
    ]
    
    # Add trace for each property
    for _, row in normalized_data.iterrows():
        property_id = row['property_id']
        location = row['location']
        
        # Inverse of price normalization (lower price is better)
        price_inv = 1 - row['price_normalized'] if 'price_normalized' in row else 0.5
        
        # Values for radar chart
        values = [
            price_inv,
            row['sqft_normalized'] if 'sqft_normalized' in row else 0.5,
            row['bedrooms_normalized'] if 'bedrooms_normalized' in row else 0.5,
            row['bathrooms_normalized'] if 'bathrooms_normalized' in row else 0.5,
            row['estimated_roi_normalized'] if 'estimated_roi_normalized' in row else 0.5
        ]
        
        # Close the polygon
        values.append(values[0])
        categories_closed = categories + [categories[0]]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=f"ID: {property_id} - {location}"
            )
        )
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Property Comparison',
        showlegend=True
    )
    
    return fig

def create_investment_analysis_chart(years, property_values, equity_values, cumulative_cash_flow):
    """
    Create a chart showing investment analysis over time.
    
    Args:
        years (list): Years for x-axis
        property_values (list): Property values over time
        equity_values (list): Equity values over time
        cumulative_cash_flow (list): Cumulative cash flow over time
    
    Returns:
        Figure: Plotly figure object
    """
    # Create DataFrame from data
    df = pd.DataFrame({
        'Year': years,
        'Property Value': property_values,
        'Equity': equity_values,
        'Cumulative Cash Flow': cumulative_cash_flow
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Bar(
            x=df['Year'],
            y=df['Property Value'],
            name='Property Value',
            marker_color='blue'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=df['Year'],
            y=df['Equity'],
            name='Equity',
            marker_color='green'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Year'],
            y=df['Cumulative Cash Flow'],
            mode='lines+markers',
            name='Cumulative Cash Flow',
            line=dict(color='red', width=3)
        )
    )
    
    # Update layout
    fig.update_layout(
        title='5-Year Investment Projection',
        xaxis_title='Year',
        yaxis_title='Amount ($)',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_heatmap(df):
    """
    Create a heatmap showing property prices by location and property type.
    
    Args:
        df (DataFrame): Property data
    
    Returns:
        Figure: Plotly figure object
    """
    # Group data by location and property type
    heatmap_data = df.groupby(['location', 'property_type'])['price'].mean().reset_index()
    
    # Pivot table for heatmap
    pivot_table = heatmap_data.pivot(
        index='location',
        columns='property_type',
        values='price'
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_table,
        text_auto='.0f',
        aspect="auto",
        color_continuous_scale='Viridis',
        title='Average Property Price by Location and Type',
        labels=dict(x="Property Type", y="Location", color="Price ($)")
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Property Type',
        yaxis_title='Location',
        coloraxis_colorbar=dict(title='Price ($)'),
    )
    
    return fig

def create_roi_comparison_chart(df):
    """
    Create a scatter plot comparing ROI and property prices.
    
    Args:
        df (DataFrame): Property data
    
    Returns:
        Figure: Plotly figure object
    """
    # Create figure
    fig = px.scatter(
        df,
        x='price',
        y='estimated_roi',
        color='location',
        size='sqft',
        hover_name='property_id',
        hover_data=['bedrooms', 'bathrooms', 'property_type'],
        title='ROI vs Price by Location',
        labels={
            'price': 'Property Price ($)',
            'estimated_roi': 'Estimated ROI',
            'location': 'Location',
            'sqft': 'Square Footage'
        }
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Property Price ($)',
        yaxis_title='Estimated ROI (%)',
        xaxis=dict(
            type='log',
            title='Property Price ($) - Log Scale'
        ),
        yaxis=dict(
            tickformat='.1%',
            title='Estimated ROI (%)'
        )
    )
    
    # Add average ROI line
    avg_roi = df['estimated_roi'].mean()
    fig.add_hline(
        y=avg_roi,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg ROI: {avg_roi:.1%}",
        annotation_position="top right"
    )
    
    return fig

def create_price_history_chart(property_history):
    """
    Create a line chart showing property price history.
    
    Args:
        property_history (DataFrame): Property historical data
    
    Returns:
        Figure: Plotly figure object
    """
    # Create figure
    fig = px.line(
        property_history,
        x='date',
        y='price',
        title='Property Price History',
        labels={'date': 'Date', 'price': 'Price ($)'},
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x'
    )
    
    return fig

def create_price_forecast_chart(dates, historical_prices, forecast_dates, forecast_prices, lower_bound, upper_bound):
    """
    Create a chart showing price forecasts.
    
    Args:
        dates (list): Historical dates
        historical_prices (list): Historical prices
        forecast_dates (list): Forecast dates
        forecast_prices (list): Forecasted prices
        lower_bound (list): Lower confidence bound
        upper_bound (list): Upper confidence bound
    
    Returns:
        Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=historical_prices,
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_prices,
            mode='lines',
            name='Price Forecast',
            line=dict(color='red', width=2)
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(231,107,243,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Price Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig
