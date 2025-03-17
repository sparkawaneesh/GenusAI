import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.property_valuation import PropertyValuationModel, valuate_property_with_adjustments, estimate_renovation_impact
from utils.data_loader import load_real_estate_data, get_available_cities, get_available_property_types

def show():
    """Display the Property Valuation page"""
    st.title("AI-Powered Property Valuation")
    
    # Load data
    df = load_real_estate_data()
    
    # Initialize or load valuation model
    if 'valuation_model' not in st.session_state:
        st.session_state.valuation_model = PropertyValuationModel()
    
    # Make sure the model is trained
    if not getattr(st.session_state.valuation_model, 'trained', False):
        if not df.empty and 'price' in df.columns:
            with st.spinner("Training valuation model..."):
                metrics = st.session_state.valuation_model.train(df, model_type='xgboost')
                st.session_state.model_metrics = metrics
    
    # Create tabs for different valuation scenarios
    tabs = st.tabs(["Quick Valuation", "Detailed Valuation", "Renovation Impact", "Valuation Model Info"])
    
    with tabs[0]:
        show_quick_valuation(df)
    
    with tabs[1]:
        show_detailed_valuation(df)
    
    with tabs[2]:
        show_renovation_impact()
    
    with tabs[3]:
        show_model_info()

def show_quick_valuation(df):
    """Display the quick valuation form"""
    st.subheader("Quick Property Valuation")
    st.write("Get an instant estimate of your property's value based on key characteristics.")
    
    # Create a form for property information
    with st.form("quick_valuation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Get property details
            cities = get_available_cities(df)
            selected_city = st.selectbox("City", options=cities if cities else [""])
            
            property_types = get_available_property_types(df)
            property_type = st.selectbox("Property Type", options=property_types if property_types else [""])
            
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        
        with col2:
            sqft = st.number_input("Square Footage", min_value=100, max_value=10000, value=1500, step=100)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=2000, step=1)
            lot_size = st.number_input("Lot Size (sqft)", min_value=0, max_value=100000, value=5000, step=500)
            
            # Additional features
            has_garage = st.checkbox("Has Garage")
            has_pool = st.checkbox("Has Pool")
        
        # Submit button
        submit_button = st.form_submit_button("Calculate Valuation")
    
    # Process form submission
    if submit_button:
        with st.spinner("Calculating property value..."):
            try:
                # Create property dataframe
                property_data = pd.DataFrame({
                    'city': [selected_city],
                    'property_type': [property_type],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'sqft': [sqft],
                    'year_built': [year_built],
                    'lot_size': [lot_size],
                    'days_on_market': [0],  # placeholder
                    'latitude': [0],         # placeholder
                    'longitude': [0]         # placeholder
                })
                
                # Get base value prediction
                base_value = st.session_state.valuation_model.predict(property_data)
                
                # Apply adjustments for additional features
                adjustments = {}
                if has_garage:
                    adjustments['garage'] = 3.5  # 3.5% increase for garage
                if has_pool:
                    adjustments['pool'] = 5.0   # 5% increase for pool
                
                adjusted_value = valuate_property_with_adjustments(base_value, adjustments)
                
                # Display valuation
                st.success(f"**Estimated Property Value: ${adjusted_value:,.2f}**")
                
                # Show valuation breakdown
                st.markdown("#### Valuation Breakdown")
                
                # Create columns for base value and adjustments
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Base Property Value", f"${base_value:,.2f}")
                    
                    # Show comparable properties in the area
                    similar_props = df[
                        (df['city'] == selected_city) & 
                        (df['property_type'] == property_type) &
                        (df['bedrooms'] == bedrooms) &
                        (abs(df['sqft'] - sqft) < 300)
                    ]
                    
                    if not similar_props.empty:
                        st.markdown("##### Comparable Properties")
                        st.dataframe(
                            similar_props[['price', 'bedrooms', 'bathrooms', 'sqft', 'year_built']].head(3),
                            hide_index=True
                        )
                
                with col2:
                    # Show adjustments
                    st.markdown("##### Value Adjustments")
                    
                    adjustment_df = pd.DataFrame({
                        'Feature': list(adjustments.keys()) + ['Base Value'],
                        'Adjustment': [f"+{adj}%" for adj in adjustments.values()] + [""],
                        'Value': [base_value * (adj / 100) for adj in adjustments.values()] + [base_value]
                    })
                    
                    st.dataframe(adjustment_df, hide_index=True)
                    
                    # Show total
                    st.metric("Final Adjusted Value", f"${adjusted_value:,.2f}")
                
                # Show confidence interval
                st.markdown("##### Valuation Range")
                
                # Calculate confidence interval (±10% for simplicity)
                lower_bound = adjusted_value * 0.9
                upper_bound = adjusted_value * 1.1
                
                # Create a gauge chart for the value range
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=adjusted_value,
                    number={'prefix': "$", 'valueformat': ',.0f'},
                    gauge={
                        'axis': {'range': [lower_bound * 0.8, upper_bound * 1.2]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [lower_bound * 0.8, lower_bound], 'color': "lightgray"},
                            {'range': [lower_bound, upper_bound], 'color': "lightblue"},
                            {'range': [upper_bound, upper_bound * 1.2], 'color': "lightgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': adjusted_value
                        }
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Pricing recommendation
                st.markdown("##### Pricing Recommendation")
                
                if 'price' in df.columns:
                    avg_market_price = df[
                        (df['city'] == selected_city) & 
                        (df['property_type'] == property_type)
                    ]['price'].mean()
                    
                    if avg_market_price > 0:
                        if adjusted_value < avg_market_price * 0.9:
                            st.info("Your property is valued below the market average. Consider investigating if there are any issues affecting the value or if this represents a good buying opportunity.")
                        elif adjusted_value > avg_market_price * 1.1:
                            st.info("Your property is valued above the market average. This could indicate special features that add value or a potential overestimation.")
                        else:
                            st.info("Your property is valued within the typical market range for similar properties in this area.")
            
            except Exception as e:
                st.error(f"An error occurred during valuation: {e}")

def show_detailed_valuation(df):
    """Display the detailed valuation form with more property attributes"""
    st.subheader("Detailed Property Valuation")
    st.write("Get a comprehensive valuation with detailed property attributes and neighborhood factors.")
    
    # Create columns for property details and neighborhood factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Property Details")
        
        # Basic property information
        cities = get_available_cities(df)
        selected_city = st.selectbox("City", options=cities if cities else [""], key="detailed_city")
        
        property_types = get_available_property_types(df)
        property_type = st.selectbox("Property Type", options=property_types if property_types else [""], key="detailed_property_type")
        
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1, key="detailed_bedrooms")
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key="detailed_bathrooms")
        sqft = st.number_input("Square Footage", min_value=100, max_value=10000, value=1500, step=100, key="detailed_sqft")
        year_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=2000, step=1, key="detailed_year")
        lot_size = st.number_input("Lot Size (sqft)", min_value=0, max_value=100000, value=5000, step=500, key="detailed_lot")
        
        # Property condition
        condition = st.select_slider(
            "Property Condition",
            options=["Poor", "Fair", "Average", "Good", "Excellent"],
            value="Average"
        )
        
        # Interior features
        st.markdown("##### Interior Features")
        
        interior_features = {
            "Updated Kitchen": st.checkbox("Updated Kitchen"),
            "Updated Bathrooms": st.checkbox("Updated Bathrooms"),
            "Hardwood Floors": st.checkbox("Hardwood Floors"),
            "Fireplace": st.checkbox("Fireplace"),
            "Basement": st.checkbox("Basement"),
            "Finished Basement": st.checkbox("Finished Basement"),
            "Open Floor Plan": st.checkbox("Open Floor Plan")
        }
    
    with col2:
        st.markdown("##### Exterior Features")
        
        exterior_features = {
            "Garage": st.checkbox("Garage"),
            "Garage Spaces": st.number_input("Garage Spaces", min_value=0, max_value=4, value=0, step=1),
            "Pool": st.checkbox("Pool"),
            "Deck/Patio": st.checkbox("Deck/Patio"),
            "Large Yard": st.checkbox("Large Yard"),
            "Fenced Yard": st.checkbox("Fenced Yard"),
            "View": st.checkbox("Premium View"),
            "Waterfront": st.checkbox("Waterfront")
        }
        
        st.markdown("##### Neighborhood Factors")
        
        neighborhood_factors = {
            "School Quality": st.select_slider(
                "School Quality",
                options=["Poor", "Below Average", "Average", "Good", "Excellent"],
                value="Average"
            ),
            "Crime Rate": st.select_slider(
                "Crime Rate",
                options=["Very High", "High", "Average", "Low", "Very Low"],
                value="Average"
            ),
            "Proximity to Amenities": st.select_slider(
                "Proximity to Amenities",
                options=["Far", "Somewhat Far", "Average", "Close", "Very Close"],
                value="Average"
            ),
            "Public Transportation": st.select_slider(
                "Public Transportation",
                options=["None", "Limited", "Average", "Good", "Excellent"],
                value="Average"
            )
        }
    
    # Calculate detailed valuation button
    if st.button("Calculate Detailed Valuation"):
        with st.spinner("Calculating detailed property valuation..."):
            try:
                # Create property dataframe for base prediction
                property_data = pd.DataFrame({
                    'city': [selected_city],
                    'property_type': [property_type],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'sqft': [sqft],
                    'year_built': [year_built],
                    'lot_size': [lot_size],
                    'days_on_market': [0],  # placeholder
                    'latitude': [0],         # placeholder
                    'longitude': [0]         # placeholder
                })
                
                # Get base value prediction
                base_value = st.session_state.valuation_model.predict(property_data)
                
                # Calculate adjustments based on all features
                adjustments = {}
                
                # Condition adjustments
                condition_adjustments = {
                    "Poor": -15.0,
                    "Fair": -7.5,
                    "Average": 0.0,
                    "Good": 5.0,
                    "Excellent": 10.0
                }
                adjustments['condition'] = condition_adjustments.get(condition, 0.0)
                
                # Interior feature adjustments
                for feature, value in interior_features.items():
                    if value:
                        if feature == "Updated Kitchen":
                            adjustments[feature] = 4.0
                        elif feature == "Updated Bathrooms":
                            adjustments[feature] = 3.0
                        elif feature == "Hardwood Floors":
                            adjustments[feature] = 2.0
                        elif feature == "Fireplace":
                            adjustments[feature] = 1.0
                        elif feature == "Basement":
                            adjustments[feature] = 3.0
                        elif feature == "Finished Basement":
                            adjustments[feature] = 6.0
                        elif feature == "Open Floor Plan":
                            adjustments[feature] = 2.0
                
                # Exterior feature adjustments
                for feature, value in exterior_features.items():
                    if feature == "Garage":
                        if value:
                            adjustments[feature] = 3.0
                    elif feature == "Garage Spaces":
                        if value > 0:
                            adjustments[feature] = value * 1.5
                    elif feature == "Pool":
                        if value:
                            adjustments[feature] = 4.0
                    elif feature == "Deck/Patio":
                        if value:
                            adjustments[feature] = 2.0
                    elif feature == "Large Yard":
                        if value:
                            adjustments[feature] = 2.5
                    elif feature == "Fenced Yard":
                        if value:
                            adjustments[feature] = 1.0
                    elif feature == "View":
                        if value:
                            adjustments[feature] = 5.0
                    elif feature == "Waterfront":
                        if value:
                            adjustments[feature] = 15.0
                
                # Neighborhood factor adjustments
                neighborhood_adj = {
                    "School Quality": {
                        "Poor": -5.0,
                        "Below Average": -2.5,
                        "Average": 0.0,
                        "Good": 3.0,
                        "Excellent": 7.0
                    },
                    "Crime Rate": {
                        "Very High": -8.0,
                        "High": -4.0,
                        "Average": 0.0,
                        "Low": 3.0,
                        "Very Low": 6.0
                    },
                    "Proximity to Amenities": {
                        "Far": -3.0,
                        "Somewhat Far": -1.5,
                        "Average": 0.0,
                        "Close": 2.0,
                        "Very Close": 4.0
                    },
                    "Public Transportation": {
                        "None": -2.0,
                        "Limited": -1.0,
                        "Average": 0.0,
                        "Good": 1.5,
                        "Excellent": 3.0
                    }
                }
                
                for factor, value in neighborhood_factors.items():
                    if factor in neighborhood_adj:
                        adjustments[factor] = neighborhood_adj[factor].get(value, 0.0)
                
                # Calculate adjusted value
                adjusted_value = valuate_property_with_adjustments(base_value, adjustments)
                
                # Display valuation results
                st.success(f"**Detailed Property Valuation: ${adjusted_value:,.2f}**")
                
                # Show valuation breakdown
                st.markdown("#### Valuation Analysis")
                
                # Create a visual breakdown of the valuation
                breakdown_df = pd.DataFrame({
                    'Factor': list(adjustments.keys()) + ['Base Value'],
                    'Adjustment (%)': [adj for adj in adjustments.values()] + [0],
                    'Value Impact ($)': [base_value * (adj / 100) for adj in adjustments.values()] + [base_value]
                })
                
                # Sort by absolute value impact
                breakdown_df['Abs_Impact'] = breakdown_df['Value Impact ($)'].abs()
                breakdown_df = breakdown_df.sort_values('Abs_Impact', ascending=False).drop('Abs_Impact', axis=1)
                
                # Positive and negative factors
                positive_factors = breakdown_df[breakdown_df['Adjustment (%)'] > 0]
                negative_factors = breakdown_df[breakdown_df['Adjustment (%)'] < 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Value-Adding Factors")
                    if not positive_factors.empty:
                        fig = px.bar(
                            positive_factors,
                            y='Factor',
                            x='Value Impact ($)',
                            orientation='h',
                            color='Value Impact ($)',
                            color_continuous_scale='Blues',
                            labels={'Value Impact ($)': 'Added Value ($)', 'Factor': ''},
                            height=max(100, len(positive_factors) * 30)
                        )
                        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No positive adjustments found")
                
                with col2:
                    st.markdown("##### Value-Reducing Factors")
                    if not negative_factors.empty:
                        fig = px.bar(
                            negative_factors,
                            y='Factor',
                            x='Value Impact ($)',
                            orientation='h',
                            color='Value Impact ($)',
                            color_continuous_scale='Reds_r',
                            labels={'Value Impact ($)': 'Reduced Value ($)', 'Factor': ''},
                            height=max(100, len(negative_factors) * 30)
                        )
                        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No negative adjustments found")
                
                # Waterfall chart showing how we got from base to final value
                st.markdown("##### Valuation Waterfall")
                
                # Create data for waterfall chart
                waterfall_measures = ['absolute'] + ['relative'] * len(adjustments) + ['total']
                waterfall_values = [base_value] + [base_value * (adj / 100) for adj in adjustments.values()] + [adjusted_value]
                waterfall_labels = ['Base Value'] + list(adjustments.keys()) + ['Final Value']
                
                fig = go.Figure(go.Waterfall(
                    name="Valuation",
                    orientation="v",
                    measure=waterfall_measures,
                    x=waterfall_labels,
                    y=waterfall_values,
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "rgba(44, 160, 101, 0.7)"}},
                    decreasing={"marker": {"color": "rgba(255, 50, 50, 0.7)"}}
                ))
                
                fig.update_layout(
                    title="From Base Value to Final Valuation",
                    showlegend=False,
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence interval and valuation range
                st.markdown("##### Value Range and Confidence")
                
                # Calculate confidence interval (adjusted for quality of data)
                model_r2 = st.session_state.model_metrics.get('r2', 0.7) if hasattr(st.session_state, 'model_metrics') else 0.7
                confidence_margin = max(0.1, 0.3 * (1 - model_r2))
                
                lower_bound = adjusted_value * (1 - confidence_margin)
                upper_bound = adjusted_value * (1 + confidence_margin)
                
                st.markdown(f"**Value Range:** ${lower_bound:,.2f} - ${upper_bound:,.2f}")
                st.markdown(f"**Confidence Level:** {int((1 - confidence_margin) * 100)}%")
                
                # Market context
                st.markdown("##### Market Context")
                
                # Find relevant market data if available
                if 'price' in df.columns:
                    market_data = df[
                        (df['city'] == selected_city) & 
                        (df['property_type'] == property_type)
                    ]
                    
                    if not market_data.empty:
                        avg_price = market_data['price'].mean()
                        median_price = market_data['price'].median()
                        price_percentile = sum(market_data['price'] < adjusted_value) / len(market_data) * 100
                        
                        st.markdown(f"- Your property is valued higher than {price_percentile:.1f}% of similar properties")
                        st.markdown(f"- Average market price for similar properties: ${avg_price:,.2f}")
                        st.markdown(f"- Median market price for similar properties: ${median_price:,.2f}")
                        
                        # Price distribution with your property highlighted
                        fig = px.histogram(
                            market_data,
                            x='price',
                            nbins=20,
                            labels={'price': 'Property Price ($)', 'count': 'Number of Properties'},
                            title="Market Price Distribution (Your Property in Red)",
                            color_discrete_sequence=['lightblue']
                        )
                        
                        # Add a line for this property valuation
                        fig.add_vline(
                            x=adjusted_value,
                            line_width=3,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Your Property",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"An error occurred during detailed valuation: {e}")

def show_renovation_impact():
    """Display the renovation impact analysis"""
    st.subheader("Renovation Impact Analysis")
    st.write("Analyze how different renovations might affect your property's value.")
    
    # Current property value input
    current_value = st.number_input(
        "Current Property Value ($)",
        min_value=10000,
        max_value=10000000,
        value=300000,
        step=10000,
        key="renovation_current_value"
    )
    
    # Renovation types
    renovation_options = [
        "Kitchen Remodel",
        "Bathroom Remodel",
        "Exterior Improvements",
        "Basement Finishing",
        "Room Addition",
        "Energy Efficiency Upgrades",
        "Roof Replacement",
        "Landscaping"
    ]
    
    # Map UI options to backend renovation types
    renovation_mapping = {
        "Kitchen Remodel": "kitchen",
        "Bathroom Remodel": "bathroom",
        "Exterior Improvements": "exterior",
        "Basement Finishing": "basement",
        "Room Addition": "addition",
        "Energy Efficiency Upgrades": "energy_efficiency",
        "Roof Replacement": "roof",
        "Landscaping": "landscaping"
    }
    
    selected_renovations = st.multiselect(
        "Select Renovations to Analyze",
        options=renovation_options,
        default=["Kitchen Remodel"]
    )
    
    # Analyze button
    if st.button("Analyze Renovation Impact"):
        if not selected_renovations:
            st.warning("Please select at least one renovation type to analyze.")
        else:
            # Calculate impact for each selected renovation
            results = []
            
            for renovation in selected_renovations:
                backend_type = renovation_mapping.get(renovation, "kitchen")
                impact = estimate_renovation_impact(current_value, backend_type)
                
                results.append({
                    'Renovation': renovation,
                    'Value Increase': impact['value_increase'],
                    'Cost': impact['renovation_cost'],
                    'ROI (%)': impact['roi'],
                    'Value Increase (%)': impact['value_increase_percentage']
                })
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Display summary
            st.markdown("#### Renovation Impact Summary")
            
            # Format the dataframe for display
            formatted_df = results_df.copy()
            formatted_df['Value Increase'] = formatted_df['Value Increase'].apply(lambda x: f"${x:,.2f}")
            formatted_df['Cost'] = formatted_df['Cost'].apply(lambda x: f"${x:,.2f}")
            formatted_df['ROI (%)'] = formatted_df['ROI (%)'].apply(lambda x: f"{x:.1f}%")
            formatted_df['Value Increase (%)'] = formatted_df['Value Increase (%)'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(formatted_df, hide_index=True)
            
            # Visualize the ROI comparison
            st.markdown("#### ROI Comparison")
            
            fig = px.bar(
                results_df,
                x='Renovation',
                y='ROI (%)',
                color='ROI (%)',
                color_continuous_scale='Viridis',
                labels={'ROI (%)': 'Return on Investment (%)', 'Renovation': ''},
                title="ROI by Renovation Type"
            )
            
            fig.update_layout(
                xaxis={'categoryorder': 'total descending'},
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost vs. Value Increase visualization
            st.markdown("#### Cost vs. Value Added")
            
            fig = px.scatter(
                results_df,
                x='Cost',
                y='Value Increase',
                size='ROI (%)',
                color='Renovation',
                labels={
                    'Cost': 'Renovation Cost ($)',
                    'Value Increase': 'Value Added ($)',
                    'ROI (%)': 'ROI (%)'
                },
                title="Renovation Cost vs. Value Added"
            )
            
            # Add a breakeven line
            max_value = max(results_df['Cost'].max(), results_df['Value Increase'].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_value],
                    y=[0, max_value],
                    mode='lines',
                    name='Breakeven Line',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on results
            st.markdown("#### Recommendations")
            
            # Sort by ROI
            top_roi = results_df.sort_values('ROI (%)', ascending=False).iloc[0]
            
            st.markdown(f"**Best ROI:** {top_roi['Renovation']} with {top_roi['ROI (%)']:.1f}% return")
            
            # Filter for projects with ROI > 100%
            good_roi_projects = results_df[results_df['ROI (%)'] > 100]
            if not good_roi_projects.empty:
                st.markdown("**Recommended Projects (ROI > 100%):**")
                for _, project in good_roi_projects.iterrows():
                    st.markdown(f"- {project['Renovation']}: {project['ROI (%)']:.1f}% ROI, adds ${project['Value Increase']:,.2f} in value")
            
            # Low ROI projects
            low_roi_projects = results_df[results_df['ROI (%)'] < 50]
            if not low_roi_projects.empty:
                st.markdown("**Projects to Reconsider (ROI < 50%):**")
                for _, project in low_roi_projects.iterrows():
                    st.markdown(f"- {project['Renovation']}: Only {project['ROI (%)']:.1f}% ROI")
            
            # Total impact if all renovations are done
            total_cost = results_df['Cost'].sum()
            total_value_added = results_df['Value Increase'].sum()
            overall_roi = (total_value_added / total_cost) * 100 if total_cost > 0 else 0
            
            st.markdown("#### Overall Impact")
            st.markdown(f"**Total Renovation Cost:** ${total_cost:,.2f}")
            st.markdown(f"**Total Value Added:** ${total_value_added:,.2f}")
            st.markdown(f"**Combined ROI:** {overall_roi:.1f}%")
            st.markdown(f"**New Property Value:** ${current_value + total_value_added:,.2f}")
            
            # Show final value impact
            fig = go.Figure(go.Waterfall(
                name="Property Value",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(results_df) + ["total"],
                x=["Current Value"] + results_df['Renovation'].tolist() + ["New Value"],
                y=[current_value] + results_df['Value Increase'].tolist() + [current_value + total_value_added],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "rgba(44, 160, 101, 0.7)"}},
                decreasing={"marker": {"color": "rgba(255, 50, 50, 0.7)"}}
            ))
            
            fig.update_layout(
                title="Property Value Progression with Renovations",
                showlegend=False,
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_model_info():
    """Display information about the valuation model"""
    st.subheader("Valuation Model Information")
    
    if hasattr(st.session_state, 'model_metrics') and st.session_state.model_metrics:
        metrics = st.session_state.model_metrics
        
        # Display model performance metrics
        st.markdown("#### Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"${metrics.get('mae', 0):,.2f}")
            st.caption("Mean Absolute Error")
        
        with col2:
            st.metric("RMSE", f"${metrics.get('rmse', 0):,.2f}")
            st.caption("Root Mean Squared Error")
        
        with col3:
            st.metric("R²", f"{metrics.get('r2', 0):.3f}")
            st.caption("Coefficient of Determination")
        
        with col4:
            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
            st.caption("Mean Absolute Percentage Error")
        
        # Feature importance
        st.markdown("#### Feature Importance")
        st.write("The importance of different property characteristics in determining value:")
        
        try:
            # Get feature importance from the model
            importance_df = st.session_state.valuation_model.get_feature_importance()
            
            if not importance_df.empty:
                fig = px.bar(
                    importance_df,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    labels={'Importance': 'Importance Score', 'Feature': ''},
                    title="Feature Importance in Valuation Model"
                )
                
                fig.update_layout(
                    xaxis_title="Relative Importance",
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation of feature importance
                st.markdown("**Interpretation:**")
                
                if not importance_df.empty:
                    top_feature = importance_df.iloc[0]['Feature']
                    st.markdown(f"- **{top_feature}** has the strongest influence on property valuation")
                    
                    # Explain a few more features
                    if len(importance_df) > 2:
                        for i in range(1, min(3, len(importance_df))):
                            feature = importance_df.iloc[i]['Feature']
                            st.markdown(f"- **{feature}** is also a significant factor in determining property value")
            else:
                st.info("Feature importance data is not available for this model")
        
        except Exception as e:
            st.error(f"Error displaying feature importance: {e}")
        
        # Model information and methodology
        st.markdown("#### Valuation Methodology")
        st.markdown("""
        This valuation model uses machine learning to predict property values based on historical data and property characteristics. The model:
        
        1. **Analyzes comparable properties** in the same area with similar characteristics
        2. **Identifies patterns** in how various features impact property values
        3. **Applies adjustments** for specific property attributes and market conditions
        4. **Validates results** against recent sales data to ensure accuracy
        
        The model is periodically retrained with new market data to maintain accuracy.
        """)
        
        # Limitations
        st.markdown("#### Limitations and Considerations")
        st.markdown("""
        - Valuations are **estimates** and not guaranteed market values
        - Unique property features may not be fully captured by the model
        - Market conditions change rapidly, which can affect accuracy
        - Local knowledge and professional appraisals are still valuable complements
        - Data quality and availability impact model performance
        """)
    else:
        st.info("Model information is not available. Please train the model first.")
