import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models.investment_analysis import InvestmentAnalysis, calculate_mortgage_payment, calculate_amortization_schedule
from utils.data_loader import load_real_estate_data, get_available_cities, get_available_property_types

def show():
    """Display the Investment Analysis page"""
    st.title("Investment Analysis")
    
    # Load data
    df = load_real_estate_data()
    
    # Initialize or get investment analysis class
    if 'investment_analysis' not in st.session_state:
        st.session_state.investment_analysis = InvestmentAnalysis()
    
    # Create tabs for different investment analysis tools
    tabs = st.tabs(["ROI Calculator", "Investment Property Finder", "Mortgage Calculator", "Cash Flow Analysis"])
    
    with tabs[0]:
        show_roi_calculator(df)
    
    with tabs[1]:
        show_investment_property_finder(df)
    
    with tabs[2]:
        show_mortgage_calculator()
    
    with tabs[3]:
        show_cash_flow_analysis(df)

def show_roi_calculator(df):
    """Display the ROI calculator"""
    st.subheader("Return on Investment (ROI) Calculator")
    st.write("Calculate expected returns for your real estate investment.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Property Details")
        
        # Property information
        purchase_price = st.number_input(
            "Purchase Price ($)",
            min_value=10000,
            max_value=10000000,
            value=300000,
            step=10000,
            key="roi_purchase_price"
        )
        
        property_type_options = ["Single Family", "Multi Family", "Condo"]
        property_type_mapping = {
            "Single Family": "single_family",
            "Multi Family": "multi_family",
            "Condo": "condo"
        }
        
        property_type = st.selectbox(
            "Property Type",
            options=property_type_options,
            key="roi_property_type"
        )
        
        monthly_rent = st.number_input(
            "Monthly Rental Income ($)",
            min_value=0,
            max_value=100000,
            value=1800,
            step=100,
            key="roi_monthly_rent"
        )
        
        # Calculate annual rental income
        annual_rental_income = monthly_rent * 12
        
        # Use the investment analysis class to estimate typical expenses
        expenses = st.session_state.investment_analysis.estimate_expenses(
            purchase_price,
            property_type=property_type_mapping.get(property_type, "single_family")
        )
        
        # Allow user to adjust expenses
        st.markdown("##### Estimated Annual Expenses")
        
        property_tax = st.number_input(
            "Property Tax ($)",
            min_value=0,
            max_value=100000,
            value=int(expenses['property_tax']),
            step=100,
            key="roi_property_tax"
        )
        
        insurance = st.number_input(
            "Insurance ($)",
            min_value=0,
            max_value=50000,
            value=int(expenses['insurance']),
            step=100,
            key="roi_insurance"
        )
        
        maintenance = st.number_input(
            "Maintenance ($)",
            min_value=0,
            max_value=50000,
            value=int(expenses['maintenance']),
            step=100,
            key="roi_maintenance"
        )
        
        vacancy = st.number_input(
            "Vacancy ($)",
            min_value=0,
            max_value=50000,
            value=int(expenses['vacancy']),
            step=100,
            key="roi_vacancy"
        )
    
    with col2:
        st.markdown("##### Financing Details")
        
        # Financing options
        down_payment_percentage = st.slider(
            "Down Payment (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            key="roi_down_payment_pct"
        )
        
        interest_rate = st.slider(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.5,
            step=0.1,
            key="roi_interest_rate"
        )
        
        loan_term = st.slider(
            "Loan Term (years)",
            min_value=5,
            max_value=30,
            value=30,
            step=5,
            key="roi_loan_term"
        )
        
        st.markdown("##### Additional Expenses")
        
        property_management = st.number_input(
            "Property Management ($)",
            min_value=0,
            max_value=50000,
            value=int(expenses['property_management']),
            step=100,
            key="roi_property_management"
        )
        
        utilities = st.number_input(
            "Utilities ($)",
            min_value=0,
            max_value=20000,
            value=int(expenses['utilities']),
            step=100,
            key="roi_utilities"
        )
        
        hoa = st.number_input(
            "HOA Fees ($)",
            min_value=0,
            max_value=10000,
            value=int(expenses['hoa']),
            step=100,
            key="roi_hoa"
        )
        
        st.markdown("##### Investment Parameters")
        
        appreciation_rate = st.slider(
            "Annual Appreciation Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            key="roi_appreciation_rate"
        )
        
        holding_period = st.slider(
            "Holding Period (years)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            key="roi_holding_period"
        )
    
    # Create updated expenses dictionary
    updated_expenses = {
        'property_tax': property_tax,
        'insurance': insurance,
        'maintenance': maintenance,
        'vacancy': vacancy,
        'property_management': property_management,
        'utilities': utilities,
        'hoa': hoa
    }
    
    # Calculate ROI button
    if st.button("Calculate ROI"):
        with st.spinner("Calculating investment returns..."):
            # Call the investment analysis class to calculate ROI
            roi_results = st.session_state.investment_analysis.calculate_roi(
                purchase_price=purchase_price,
                annual_rental_income=annual_rental_income,
                expenses=updated_expenses,
                appreciation_rate=appreciation_rate / 100,
                holding_period=holding_period,
                down_payment_percentage=down_payment_percentage,
                interest_rate=interest_rate,
                loan_term=loan_term,
                closing_costs_percentage=3,  # Default
                selling_costs_percentage=6   # Default
            )
            
            # Display results
            st.markdown("### Investment Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Cash on Cash ROI",
                    f"{roi_results['cash_on_cash_roi']:.2f}%",
                    help="Annual cash flow divided by initial investment"
                )
            
            with col2:
                st.metric(
                    "Total ROI",
                    f"{roi_results['total_roi']:.2f}%",
                    help="Total profit divided by initial investment over the holding period"
                )
            
            with col3:
                st.metric(
                    "Cap Rate",
                    f"{roi_results['cap_rate']:.2f}%",
                    help="Net operating income divided by property value"
                )
            
            with col4:
                st.metric(
                    "Monthly Cash Flow",
                    f"${roi_results['monthly_cash_flow']:.2f}",
                    delta="per month",
                    help="Monthly rental income minus all expenses including mortgage"
                )
            
            # More detailed breakdown
            st.markdown("#### Financial Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Initial Investment")
                
                investment_data = {
                    'Category': ['Down Payment', 'Closing Costs', 'Total Initial Investment'],
                    'Amount': [
                        purchase_price * (down_payment_percentage / 100),
                        roi_results['initial_investment'] - (purchase_price * (down_payment_percentage / 100)),
                        roi_results['initial_investment']
                    ]
                }
                
                investment_df = pd.DataFrame(investment_data)
                investment_df['Amount'] = investment_df['Amount'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(investment_df, hide_index=True)
                
                st.markdown("##### Monthly Expenses")
                
                # Calculate mortgage payment
                monthly_mortgage = roi_results['monthly_mortgage_payment']
                
                expense_data = {
                    'Expense': [
                        'Mortgage Payment',
                        'Property Tax',
                        'Insurance',
                        'Maintenance',
                        'Vacancy',
                        'Property Management',
                        'Utilities',
                        'HOA Fees',
                        'Total Expenses'
                    ],
                    'Monthly Amount': [
                        monthly_mortgage,
                        property_tax / 12,
                        insurance / 12,
                        maintenance / 12,
                        vacancy / 12,
                        property_management / 12,
                        utilities / 12,
                        hoa / 12,
                        monthly_mortgage + sum(expense / 12 for expense in updated_expenses.values())
                    ]
                }
                
                expense_df = pd.DataFrame(expense_data)
                expense_df['Monthly Amount'] = expense_df['Monthly Amount'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(expense_df, hide_index=True)
            
            with col2:
                st.markdown("##### Cash Flow Analysis")
                
                # Monthly income, expenses, and cash flow
                monthly_income = annual_rental_income / 12
                monthly_expenses = sum(updated_expenses.values()) / 12
                monthly_total_expenses = monthly_expenses + monthly_mortgage
                monthly_cash_flow = monthly_income - monthly_total_expenses
                
                # Create a cash flow waterfall
                cash_flow_data = {
                    'Category': ['Rental Income', 'Operating Expenses', 'Mortgage Payment', 'Monthly Cash Flow'],
                    'Amount': [monthly_income, -monthly_expenses, -monthly_mortgage, monthly_cash_flow],
                    'Type': ['income', 'expense', 'expense', 'total']
                }
                
                fig = go.Figure(go.Waterfall(
                    name="Cash Flow",
                    orientation="v",
                    measure=cash_flow_data['Type'],
                    x=cash_flow_data['Category'],
                    y=cash_flow_data['Amount'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "rgba(44, 160, 101, 0.7)"}},
                    decreasing={"marker": {"color": "rgba(255, 50, 50, 0.7)"}},
                    text=[f"${abs(x):,.2f}" for x in cash_flow_data['Amount']],
                    textposition="outside"
                ))
                
                fig.update_layout(
                    title="Monthly Cash Flow",
                    showlegend=False,
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### Projected Equity Growth")
                
                # Calculate equity growth over the holding period
                years = list(range(holding_period + 1))
                property_values = [purchase_price * ((1 + appreciation_rate / 100) ** year) for year in years]
                
                # Calculate remaining loan balance for each year
                loan_amount = purchase_price - (purchase_price * (down_payment_percentage / 100))
                monthly_rate = (interest_rate / 100) / 12
                months = loan_term * 12
                
                remaining_balances = []
                for year in years:
                    remaining_months = months - (year * 12)
                    if remaining_months <= 0:
                        remaining_balances.append(0)
                    else:
                        balance = loan_amount * ((1 + monthly_rate) ** months - (1 + monthly_rate) ** (year * 12)) / ((1 + monthly_rate) ** months - 1)
                        remaining_balances.append(max(0, balance))
                
                # Calculate equity for each year
                equity_values = [property_values[i] - remaining_balances[i] for i in range(len(years))]
                
                # Create equity growth chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=property_values,
                    fill=None,
                    mode='lines',
                    line_color='rgba(44, 160, 101, 0.7)',
                    name='Property Value'
                ))
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=remaining_balances,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255, 50, 50, 0.7)',
                    name='Loan Balance'
                ))
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=equity_values,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(66, 114, 255, 0.7)',
                    name='Equity'
                ))
                
                fig.update_layout(
                    title="Equity Growth Over Time",
                    xaxis_title="Year",
                    yaxis_title="Amount ($)",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Overall profitability analysis
            st.markdown("#### Overall Profitability Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Cash Flow",
                    f"${roi_results['annual_cash_flow'] * holding_period:,.2f}",
                    help="Total cash flow over the holding period"
                )
                
                st.metric(
                    "Final Property Value",
                    f"${roi_results['final_property_value']:,.2f}",
                    delta=f"{appreciation_rate * holding_period:.1f}% growth",
                    help="Estimated property value after the holding period"
                )
            
            with col2:
                st.metric(
                    "Equity Upon Sale",
                    f"${roi_results['equity']:,.2f}",
                    help="Property value minus remaining loan balance and selling costs"
                )
                
                st.metric(
                    "Annualized ROI",
                    f"{roi_results['annualized_roi']:.2f}%",
                    help="Average annual return accounting for all factors"
                )
            
            with col3:
                st.metric(
                    "Total Profit",
                    f"${roi_results['total_profit']:,.2f}",
                    help="Sum of cash flow and equity growth minus initial investment"
                )
                
                st.metric(
                    "Debt Service Coverage Ratio",
                    f"{roi_results['dscr']:.2f}",
                    help="Net operating income divided by annual debt service (above 1.0 is positive)"
                )
            
            # Provide investment insights
            st.markdown("#### Investment Insights")
            
            # Cash flow assessment
            if roi_results['monthly_cash_flow'] < 0:
                st.warning(f"This property is cash flow negative by ${abs(roi_results['monthly_cash_flow']):.2f} per month.")
            elif roi_results['monthly_cash_flow'] < 100:
                st.warning(f"This property has minimal cash flow of ${roi_results['monthly_cash_flow']:.2f} per month.")
            else:
                st.success(f"This property generates positive cash flow of ${roi_results['monthly_cash_flow']:.2f} per month.")
            
            # ROI assessment
            if roi_results['total_roi'] < 20:
                st.info("The total ROI is below average for real estate investments.")
            elif roi_results['total_roi'] > 50:
                st.success("The total ROI is excellent compared to typical real estate investments.")
            else:
                st.info("The total ROI is within the average range for real estate investments.")
            
            # Cap rate assessment
            if roi_results['cap_rate'] < 4:
                st.info("The cap rate is low, indicating lower risk but potentially lower returns.")
            elif roi_results['cap_rate'] > 8:
                st.success("The cap rate is high, suggesting strong income relative to property value.")
            else:
                st.info("The cap rate is within the average range for investment properties.")
            
            # DSCR assessment
            if roi_results['dscr'] < 1:
                st.warning("The DSCR is below 1.0, indicating the property income doesn't fully cover the debt service.")
            elif roi_results['dscr'] > 1.5:
                st.success("The DSCR is strong, showing the property generates significantly more income than needed for debt service.")
            else:
                st.info("The DSCR is adequate, showing the property can cover its debt service.")

def show_investment_property_finder(df):
    """Display the investment property finder"""
    st.subheader("Investment Property Finder")
    st.write("Find properties with the best investment potential based on your criteria.")
    
    # Create investment filters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Investment Criteria")
        
        budget = st.number_input(
            "Investment Budget ($)",
            min_value=0,
            max_value=5000000,
            value=100000,
            step=10000,
            key="finder_budget"
        )
        
        investment_goal = st.selectbox(
            "Investment Goal",
            options=["Rental Income", "Appreciation", "Balanced"],
            index=2,
            key="finder_goal"
        )
        
        # Map UI options to backend values
        goal_mapping = {
            "Rental Income": "rental_income",
            "Appreciation": "appreciation",
            "Balanced": "balanced"
        }
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=["Low", "Medium", "High"],
            index=1,
            key="finder_risk"
        )
        
        # Map UI options to backend values
        risk_mapping = {
            "Low": "low",
            "Medium": "medium",
            "High": "high"
        }
    
    with col2:
        st.markdown("##### Property Filters (Optional)")
        
        cities = ["Any"] + get_available_cities(df)
        selected_city = st.selectbox(
            "City",
            options=cities,
            index=0,
            key="finder_city"
        )
        
        property_types = ["Any"] + get_available_property_types(df)
        property_type = st.selectbox(
            "Property Type",
            options=property_types,
            index=0,
            key="finder_property_type"
        )
        
        min_bedrooms = st.number_input(
            "Minimum Bedrooms",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="finder_min_bedrooms"
        )
        
        min_bathrooms = st.number_input(
            "Minimum Bathrooms",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="finder_min_bathrooms"
        )
    
    # Find properties button
    if st.button("Find Investment Properties"):
        with st.spinner("Analyzing investment opportunities..."):
            # Check if we have the necessary data
            if df.empty:
                st.error("No property data available for analysis.")
                return
            
            # Set property data for the investment analysis class
            st.session_state.investment_analysis.set_property_data(df)
            
            # Prepare user preferences
            preferences = {
                'city': None if selected_city == "Any" else selected_city,
                'property_type': None if property_type == "Any" else property_type,
                'min_bedrooms': min_bedrooms if min_bedrooms > 0 else None,
                'min_bathrooms': min_bathrooms if min_bathrooms > 0 else None,
                'investment_goal': goal_mapping.get(investment_goal, "balanced")
            }
            
            # Calculate investment metrics for filtering
            if df is not None and not df.empty:
                df_with_metrics = st.session_state.investment_analysis.calculate_financial_metrics(df)
                
                # Filter properties based on user preferences
                filtered_properties = df_with_metrics.copy()
                
                # Apply filters
                if preferences['city'] is not None:
                    filtered_properties = filtered_properties[filtered_properties['city'] == preferences['city']]
                
                if preferences['property_type'] is not None:
                    filtered_properties = filtered_properties[filtered_properties['property_type'] == preferences['property_type']]
                
                if preferences['min_bedrooms'] is not None:
                    filtered_properties = filtered_properties[filtered_properties['bedrooms'] >= preferences['min_bedrooms']]
                
                if preferences['min_bathrooms'] is not None:
                    filtered_properties = filtered_properties[filtered_properties['bathrooms'] >= preferences['min_bathrooms']]
                
                # Calculate maximum purchase price based on budget (assuming 20% down payment)
                max_price = budget * 5  # 20% down payment
                filtered_properties = filtered_properties[filtered_properties['price'] <= max_price]
                
                # Sort properties based on investment goal
                if preferences['investment_goal'] == 'rental_income':
                    if 'cap_rate' in filtered_properties.columns:
                        filtered_properties = filtered_properties.sort_values('cap_rate', ascending=False)
                elif preferences['investment_goal'] == 'appreciation':
                    if 'price_to_rent_ratio' in filtered_properties.columns:
                        # Lower price-to-rent ratio often indicates better appreciation potential
                        filtered_properties = filtered_properties.sort_values('price_to_rent_ratio', ascending=True)
                else:  # balanced
                    if 'cash_on_cash_return' in filtered_properties.columns:
                        filtered_properties = filtered_properties.sort_values('cash_on_cash_return', ascending=False)
                
                # Display top investment properties
                if filtered_properties.empty:
                    st.warning("No properties match your investment criteria. Try adjusting your filters.")
                else:
                    st.success(f"Found {len(filtered_properties)} properties matching your criteria.")
                    
                    st.markdown("#### Top Investment Opportunities")
                    
                    # Create metrics for top 5 properties
                    top_properties = filtered_properties.head(5)
                    
                    for i, (_, prop) in enumerate(top_properties.iterrows()):
                        col1, col2, col3 = st.columns([1, 2, 2])
                        
                        with col1:
                            # Use a placeholder image
                            st.image("https://www.svgrepo.com/show/530661/villa.svg", width=100)
                        
                        with col2:
                            address = f"{prop.get('city', '')}, {prop.get('state', '')}"
                            property_type = prop.get('property_type', 'Property')
                            bedrooms = prop.get('bedrooms', 'N/A')
                            bathrooms = prop.get('bathrooms', 'N/A')
                            sqft = prop.get('sqft', 'N/A')
                            
                            st.markdown(f"**{property_type} - {bedrooms} bed, {bathrooms} bath**")
                            st.markdown(f"**Location:** {address}")
                            st.markdown(f"**Size:** {sqft} sqft")
                            st.markdown(f"**Price:** ${prop.get('price', 0):,.0f}")
                        
                        with col3:
                            # Display key investment metrics
                            if 'cap_rate' in prop:
                                st.markdown(f"**Cap Rate:** {prop.get('cap_rate', 0):.2f}%")
                            
                            if 'cash_on_cash_return' in prop:
                                st.markdown(f"**Cash on Cash Return:** {prop.get('cash_on_cash_return', 0):.2f}%")
                            
                            if 'monthly_rent_estimate' in prop:
                                st.markdown(f"**Est. Monthly Rent:** ${prop.get('monthly_rent_estimate', 0):,.0f}")
                            
                            if 'price_per_sqft' in prop:
                                st.markdown(f"**Price/sqft:** ${prop.get('price_per_sqft', 0):,.0f}")
                            
                            # Display link to detailed analysis
                            st.button(f"Analyze Property {i+1}", key=f"analyze_property_{i}")
                        
                        st.markdown("---")
                    
                    # Show map of investment properties
                    if all(col in filtered_properties.columns for col in ['latitude', 'longitude', 'price']):
                        st.markdown("#### Investment Properties Map")
                        
                        # Color by investment metric
                        color_by = 'cap_rate' if 'cap_rate' in filtered_properties.columns else 'price'
                        
                        fig = px.scatter_mapbox(
                            filtered_properties.head(20),  # Limit to top 20 properties
                            lat='latitude',
                            lon='longitude',
                            color=color_by,
                            size='price',
                            hover_name='property_id' if 'property_id' in filtered_properties.columns else None,
                            hover_data=['price', 'bedrooms', 'bathrooms', 'cap_rate' if 'cap_rate' in filtered_properties.columns else None],
                            zoom=10,
                            height=400,
                            mapbox_style="open-street-map"
                        )
                        
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show comparative metrics
                    st.markdown("#### Investment Metrics Comparison")
                    
                    # Prepare data for radar chart
                    if not top_properties.empty and len(top_properties) > 0:
                        # Check which columns are available
                        available_metrics = []
                        metric_names = []
                        
                        # Check for cap rate
                        if 'cap_rate' in top_properties.columns:
                            available_metrics.append('cap_rate')
                            metric_names.append('Cap Rate')
                            
                        # Check for cash on cash return
                        if 'cash_on_cash_return' in top_properties.columns:
                            available_metrics.append('cash_on_cash_return')
                            metric_names.append('Cash on Cash')
                            
                        # Check for GRM
                        grm_available = 'grm' in top_properties.columns
                        if grm_available:
                            available_metrics.append('grm_inverted')
                            metric_names.append('Rent Multiplier')
                        
                        # Only proceed if we have metrics to show
                        if available_metrics and len(available_metrics) > 0:
                            # Create metrics dataframe with available columns
                            metrics_cols = ['price'] + [col for col in available_metrics if col != 'grm_inverted']
                            if grm_available:
                                metrics_cols.append('grm')
                            
                            metrics_df = top_properties[metrics_cols].copy()
                            
                            # Add inverted GRM if available
                            if grm_available:
                                # Flip GRM so lower is better (like price_to_rent_ratio)
                                metrics_df['grm_inverted'] = 1 / metrics_df['grm']
                            
                            # Normalize to 0-1 scale
                            for col in available_metrics:
                                if col in metrics_df.columns and metrics_df[col].max() > metrics_df[col].min():
                                    metrics_df[col] = (metrics_df[col] - metrics_df[col].min()) / (metrics_df[col].max() - metrics_df[col].min())
                        
                        # Check if we have metrics to show
                        if available_metrics and metric_names and len(available_metrics) > 0:
                            # Prepare for radar chart
                            categories = metric_names
                            fig = go.Figure()
                            
                            # Make sure metrics_df is defined
                            if 'metrics_df' in locals() and not metrics_df.empty:
                                for i, (_, prop) in enumerate(top_properties.iterrows()):
                                    # Get values for available metrics
                                    values = []
                                    for metric in available_metrics:
                                        if metric in metrics_df.columns:
                                            values.append(metrics_df.iloc[i][metric])
                                    
                                    if values:
                                        # Close the polygon
                                        values.append(values[0])
                                        categories_closed = categories + [categories[0]]
                                        
                                        # Add trace for this property
                                        fig.add_trace(go.Scatterpolar(
                                            r=values,
                                            theta=categories_closed,
                                            fill='toself',
                                            name=f"Property {i+1}"
                                        ))
                                
                                # Only update layout and display chart if we added traces
                                if hasattr(fig, 'data') and len(fig.data) > 0:
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1]
                                            )
                                        ),
                                        showlegend=True,
                                        height=400,
                                        margin=dict(l=40, r=40, t=20, b=20)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Unable to generate radar chart with available metrics.")
                    
                    # Display table with all properties
                    st.markdown("#### All Matching Properties")
                    
                    # Select columns to display
                    display_columns = [
                        'price', 'bedrooms', 'bathrooms', 'sqft', 'property_type', 'city',
                        'cap_rate', 'cash_on_cash_return', 'monthly_rent_estimate'
                    ]
                    
                    # Filter for columns that actually exist
                    display_columns = [col for col in display_columns if col in filtered_properties.columns]
                    
                    # Format columns for display
                    formatted_properties = filtered_properties[display_columns].copy()
                    
                    if 'price' in formatted_properties.columns:
                        formatted_properties['price'] = formatted_properties['price'].apply(lambda x: f"${x:,.0f}")
                    
                    if 'monthly_rent_estimate' in formatted_properties.columns:
                        formatted_properties['monthly_rent_estimate'] = formatted_properties['monthly_rent_estimate'].apply(lambda x: f"${x:,.0f}")
                    
                    if 'cap_rate' in formatted_properties.columns:
                        formatted_properties['cap_rate'] = formatted_properties['cap_rate'].apply(lambda x: f"{x:.2f}%")
                    
                    if 'cash_on_cash_return' in formatted_properties.columns:
                        formatted_properties['cash_on_cash_return'] = formatted_properties['cash_on_cash_return'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(formatted_properties.head(10), hide_index=True)
            else:
                st.error("Unable to analyze investment properties. Please try again.")

def show_mortgage_calculator():
    """Display the mortgage calculator"""
    st.subheader("Mortgage Calculator")
    st.write("Calculate mortgage payments and view amortization schedule.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Loan details
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=10000,
            max_value=10000000,
            value=240000,
            step=10000,
            key="mortgage_loan_amount"
        )
        
        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.1,
            max_value=20.0,
            value=4.5,
            step=0.1,
            key="mortgage_interest_rate"
        )
        
        loan_term = st.number_input(
            "Loan Term (years)",
            min_value=5,
            max_value=30,
            value=30,
            step=5,
            key="mortgage_loan_term"
        )
    
    with col2:
        # Additional options
        show_amortization = st.checkbox("Show Amortization Schedule", value=True)
        
        include_taxes_insurance = st.checkbox("Include Property Taxes and Insurance", value=False)
        
        if include_taxes_insurance:
            annual_property_tax = st.number_input(
                "Annual Property Tax ($)",
                min_value=0,
                max_value=50000,
                value=3000,
                step=500,
                key="mortgage_property_tax"
            )
            
            annual_insurance = st.number_input(
                "Annual Insurance ($)",
                min_value=0,
                max_value=10000,
                value=1200,
                step=100,
                key="mortgage_insurance"
            )
    
    # Calculate button
    if st.button("Calculate Mortgage"):
        # Calculate monthly payment
        monthly_payment = calculate_mortgage_payment(
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            loan_term=loan_term
        )
        
        # Add taxes and insurance if selected
        monthly_tax = 0
        monthly_insurance = 0
        
        if include_taxes_insurance and 'annual_property_tax' in locals() and 'annual_insurance' in locals():
            monthly_tax = annual_property_tax / 12
            monthly_insurance = annual_insurance / 12
            
        total_monthly_payment = monthly_payment + monthly_tax + monthly_insurance
        
        # Display payment summary
        st.markdown("### Mortgage Payment Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Monthly Principal & Interest",
                f"${monthly_payment:.2f}",
                help="Monthly payment for principal and interest only"
            )
        
        with col2:
            if include_taxes_insurance:
                st.metric(
                    "Taxes & Insurance",
                    f"${monthly_tax + monthly_insurance:.2f}",
                    help="Monthly property tax and insurance payment"
                )
            else:
                st.metric(
                    "Loan Term",
                    f"{loan_term} years",
                    help="Duration of the mortgage"
                )
        
        with col3:
            if include_taxes_insurance:
                st.metric(
                    "Total Monthly Payment",
                    f"${total_monthly_payment:.2f}",
                    help="Total monthly payment including principal, interest, taxes, and insurance"
                )
            else:
                st.metric(
                    "Total Interest Paid",
                    f"${(monthly_payment * loan_term * 12) - loan_amount:,.2f}",
                    help="Total interest paid over the life of the loan"
                )
        
        # Loan summary
        st.markdown("#### Loan Summary")
        
        # Calculate total payments
        total_payments = monthly_payment * loan_term * 12
        total_interest = total_payments - loan_amount
        
        # Create a pie chart for principal vs. interest
        fig = px.pie(
            values=[loan_amount, total_interest],
            names=['Principal', 'Interest'],
            title="Principal vs. Interest",
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Show amortization schedule if selected
        if show_amortization:
            st.markdown("#### Amortization Schedule")
            
            # Calculate amortization schedule
            amortization_df = calculate_amortization_schedule(
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                loan_term=loan_term
            )
            
            # Display amortization table
            formatted_amortization = amortization_df.copy()
            for col in ['principal_paid', 'interest_paid', 'total_payment', 'remaining_balance']:
                formatted_amortization[col] = formatted_amortization[col].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(formatted_amortization, hide_index=True)
            
            # Create visualization of principal and interest over time
            fig = go.Figure()
            
            # Add traces for principal and interest
            fig.add_trace(go.Bar(
                x=amortization_df['year'],
                y=amortization_df['principal_paid'],
                name='Principal',
                marker_color='#636EFA'
            ))
            
            fig.add_trace(go.Bar(
                x=amortization_df['year'],
                y=amortization_df['interest_paid'],
                name='Interest',
                marker_color='#EF553B'
            ))
            
            # Add trace for remaining balance
            fig.add_trace(go.Scatter(
                x=amortization_df['year'],
                y=amortization_df['remaining_balance'],
                name='Remaining Balance',
                yaxis='y2',
                line=dict(color='#00CC96', width=3)
            ))
            
            # Update layout
            fig.update_layout(
                title="Amortization Schedule",
                xaxis_title="Year",
                yaxis_title="Annual Payment ($)",
                yaxis2=dict(
                    title="Remaining Balance ($)",
                    overlaying='y',
                    side='right'
                ),
                barmode='stack',
                legend=dict(x=0.01, y=0.99),
                height=500,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_cash_flow_analysis(df):
    """Display the cash flow analysis"""
    st.subheader("Cash Flow Analysis")
    st.write("Analyze projected cash flow for your investment property.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Property Details")
        
        # Property information
        purchase_price = st.number_input(
            "Purchase Price ($)",
            min_value=10000,
            max_value=10000000,
            value=300000,
            step=10000,
            key="cf_purchase_price"
        )
        
        property_type_options = ["Single Family", "Multi Family", "Condo"]
        property_type_mapping = {
            "Single Family": "single_family",
            "Multi Family": "multi_family",
            "Condo": "condo"
        }
        
        property_type = st.selectbox(
            "Property Type",
            options=property_type_options,
            key="cf_property_type"
        )
        
        monthly_rent = st.number_input(
            "Monthly Rental Income ($)",
            min_value=0,
            max_value=100000,
            value=1800,
            step=100,
            key="cf_monthly_rent"
        )
        
        # Financing section
        st.markdown("##### Financing")
        
        down_payment_percentage = st.slider(
            "Down Payment (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            key="cf_down_payment_pct"
        )
        
        interest_rate = st.slider(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.5,
            step=0.1,
            key="cf_interest_rate"
        )
        
        loan_term = st.slider(
            "Loan Term (years)",
            min_value=5,
            max_value=30,
            value=30,
            step=5,
            key="cf_loan_term"
        )
    
    with col2:
        st.markdown("##### Annual Operating Expenses")
        
        # Use the investment analysis class to estimate typical expenses
        expenses = st.session_state.investment_analysis.estimate_expenses(
            purchase_price,
            property_type=property_type_mapping.get(property_type, "single_family")
        )
        
        # Allow user to adjust expenses
        property_tax = st.number_input(
            "Property Tax ($)",
            min_value=0,
            max_value=100000,
            value=int(expenses['property_tax']),
            step=100,
            key="cf_property_tax"
        )
        
        insurance = st.number_input(
            "Insurance ($)",
            min_value=0,
            max_value=50000,
            value=int(expenses['insurance']),
            step=100,
            key="cf_insurance"
        )
        
        maintenance = st.number_input(
            "Maintenance ($)",
            min_value=0,
            max_value=50000,
            value=int(expenses['maintenance']),
            step=100,
            key="cf_maintenance"
        )
        
        # Additional expenses
        vacancy_rate = st.slider(
            "Vacancy Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="cf_vacancy_rate"
        )
        
        management_fee_rate = st.slider(
            "Property Management Fee (%)",
            min_value=0.0,
            max_value=15.0,
            value=8.0,
            step=0.5,
            key="cf_management_fee_rate"
        )
        
        # Other expenses
        other_expenses = st.number_input(
            "Other Monthly Expenses ($)",
            min_value=0,
            max_value=10000,
            value=100,
            step=50,
            key="cf_other_expenses"
        )
        
        # Projection period
        projection_years = st.slider(
            "Projection Period (years)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="cf_projection_years"
        )
        
        # Growth rates
        rent_growth_rate = st.slider(
            "Annual Rent Growth Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            key="cf_rent_growth_rate"
        )
        
        expense_growth_rate = st.slider(
            "Annual Expense Growth Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key="cf_expense_growth_rate"
        )
    
    # Calculate cash flow button
    if st.button("Analyze Cash Flow"):
        with st.spinner("Calculating projected cash flow..."):
            # Calculate mortgage payment
            loan_amount = purchase_price * (1 - down_payment_percentage / 100)
            monthly_mortgage = calculate_mortgage_payment(
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                loan_term=loan_term
            )
            annual_mortgage = monthly_mortgage * 12
            
            # Calculate annual rental income
            annual_rental_income = monthly_rent * 12
            
            # Calculate expenses based on rates
            vacancy_expense = annual_rental_income * (vacancy_rate / 100)
            management_fee = annual_rental_income * (management_fee_rate / 100)
            annual_other_expenses = other_expenses * 12
            
            # Total operating expenses (excluding mortgage)
            operating_expenses = {
                'property_tax': property_tax,
                'insurance': insurance,
                'maintenance': maintenance,
                'vacancy': vacancy_expense,
                'property_management': management_fee,
                'other': annual_other_expenses
            }
            
            total_operating_expenses = sum(operating_expenses.values())
            
            # Calculate NOI (Net Operating Income)
            noi = annual_rental_income - total_operating_expenses
            
            # Calculate cash flow
            cash_flow = noi - annual_mortgage
            
            # Calculate cash on cash return
            initial_investment = purchase_price * (down_payment_percentage / 100)
            cash_on_cash_return = (cash_flow / initial_investment) * 100 if initial_investment > 0 else 0
            
            # Project cash flow for future years
            years = list(range(1, projection_years + 1))
            projected_data = []
            
            cumulative_cash_flow = 0
            
            for year in years:
                # Apply growth rates
                year_rental_income = annual_rental_income * ((1 + rent_growth_rate / 100) ** (year - 1))
                year_operating_expenses = total_operating_expenses * ((1 + expense_growth_rate / 100) ** (year - 1))
                
                # NOI remains the same calculation
                year_noi = year_rental_income - year_operating_expenses
                
                # Mortgage payment stays constant (assuming fixed-rate)
                year_cash_flow = year_noi - annual_mortgage
                
                # Calculate cash on cash return for each year
                year_cash_on_cash = (year_cash_flow / initial_investment) * 100 if initial_investment > 0 else 0
                
                # Track cumulative cash flow
                cumulative_cash_flow += year_cash_flow
                
                # Add to projected data
                projected_data.append({
                    'Year': year,
                    'Rental Income': year_rental_income,
                    'Operating Expenses': year_operating_expenses,
                    'NOI': year_noi,
                    'Mortgage Payment': annual_mortgage,
                    'Cash Flow': year_cash_flow,
                    'Cash on Cash Return': year_cash_on_cash,
                    'Cumulative Cash Flow': cumulative_cash_flow
                })
            
            # Create dataframe with projections
            projection_df = pd.DataFrame(projected_data)
            
            # Display cash flow summary
            st.markdown("### Cash Flow Analysis Results")
            
            # Year 1 metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Monthly Cash Flow",
                    f"${cash_flow / 12:.2f}",
                    help="Monthly cash flow after all expenses and mortgage"
                )
            
            with col2:
                st.metric(
                    "Annual Cash Flow",
                    f"${cash_flow:.2f}",
                    help="Annual cash flow after all expenses and mortgage"
                )
            
            with col3:
                st.metric(
                    "Cash on Cash Return",
                    f"{cash_on_cash_return:.2f}%",
                    help="Annual cash flow divided by initial investment"
                )
            
            with col4:
                st.metric(
                    "Cap Rate",
                    f"{(noi / purchase_price) * 100:.2f}%",
                    help="Net operating income divided by property value"
                )
            
            # Cash flow breakdown visualization
            st.markdown("#### Cash Flow Breakdown (Year 1)")
            
            # Create data for waterfall chart
            waterfall_values = [annual_rental_income, -total_operating_expenses, -annual_mortgage, cash_flow]
            waterfall_labels = ['Rental Income', 'Operating Expenses', 'Mortgage Payment', 'Cash Flow']
            waterfall_measures = ['absolute', 'relative', 'relative', 'total']
            
            fig = go.Figure(go.Waterfall(
                name="Cash Flow",
                orientation="v",
                measure=waterfall_measures,
                x=waterfall_labels,
                y=waterfall_values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "rgba(44, 160, 101, 0.7)"}},
                decreasing={"marker": {"color": "rgba(255, 50, 50, 0.7)"}},
                text=[f"${abs(x):,.2f}" for x in waterfall_values],
                textposition="outside"
            ))
            
            fig.update_layout(
                title="Annual Cash Flow Breakdown",
                showlegend=False,
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Operating expense breakdown
            st.markdown("#### Operating Expense Breakdown")
            
            # Create pie chart for operating expenses
            expense_labels = list(operating_expenses.keys())
            expense_values = list(operating_expenses.values())
            
            fig = px.pie(
                values=expense_values,
                names=expense_labels,
                title="Operating Expenses",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Projected cash flow
            st.markdown("#### Projected Cash Flow")
            
            # Create line chart for projected cash flow
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=projection_df['Year'],
                y=projection_df['Cash Flow'],
                mode='lines+markers',
                name='Annual Cash Flow',
                line=dict(color='rgba(44, 160, 101, 0.7)', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=projection_df['Year'],
                y=projection_df['Cumulative Cash Flow'],
                mode='lines+markers',
                name='Cumulative Cash Flow',
                line=dict(color='rgba(66, 114, 255, 0.7)', width=3),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Projected Cash Flow Over Time",
                xaxis_title="Year",
                yaxis_title="Annual Cash Flow ($)",
                yaxis2=dict(
                    title="Cumulative Cash Flow ($)",
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.01, y=0.99),
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cash flow projection table
            st.markdown("#### Cash Flow Projection Table")
            
            # Format the projection dataframe for display
            formatted_projection = projection_df.copy()
            
            # Format currency columns
            currency_cols = ['Rental Income', 'Operating Expenses', 'NOI', 'Mortgage Payment', 'Cash Flow', 'Cumulative Cash Flow']
            for col in currency_cols:
                formatted_projection[col] = formatted_projection[col].apply(lambda x: f"${x:,.2f}")
            
            # Format percentage columns
            formatted_projection['Cash on Cash Return'] = formatted_projection['Cash on Cash Return'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(formatted_projection, hide_index=True)
            
            # Cash flow highlights and insights
            st.markdown("#### Cash Flow Insights")
            
            # Calculate metrics for insights
            total_cash_flow = projection_df['Cash Flow'].sum()
            avg_cash_on_cash = projection_df['Cash on Cash Return'].mean()
            
            # Payback period calculation
            payback_year = None
            for i, row in projection_df.iterrows():
                if row['Cumulative Cash Flow'] >= initial_investment:
                    payback_year = row['Year']
                    break
            
            # Display insights
            st.markdown(f"**Total Cash Flow (over {projection_years} years):** ${total_cash_flow:,.2f}")
            st.markdown(f"**Average Cash on Cash Return:** {avg_cash_on_cash:.2f}%")
            
            if payback_year:
                st.markdown(f"**Investment Payback Period:** {payback_year} years")
            else:
                st.markdown("**Investment Payback Period:** Beyond projection period")
            
            # Cash flow assessment
            if cash_flow < 0:
                st.warning(f"This property is cash flow negative by ${abs(cash_flow / 12):.2f} per month. Consider raising rent or reducing expenses.")
            elif cash_flow < 100 * 12:  # Less than $100/month
                st.warning(f"This property has minimal cash flow of ${cash_flow / 12:.2f} per month, which may not provide adequate buffer for unexpected expenses.")
            else:
                st.success(f"This property generates positive cash flow of ${cash_flow / 12:.2f} per month.")
            
            # Break-even rent calculation
            break_even_annual = total_operating_expenses + annual_mortgage
            break_even_monthly = break_even_annual / 12
            
            st.markdown(f"**Break-even Monthly Rent:** ${break_even_monthly:.2f}")
            
            if monthly_rent < break_even_monthly:
                st.warning(f"Current rent (${monthly_rent:.2f}) is below break-even rent (${break_even_monthly:.2f}). Consider increasing rent or reducing expenses.")
            else:
                rent_buffer = monthly_rent - break_even_monthly
                buffer_percentage = (rent_buffer / break_even_monthly) * 100
                st.markdown(f"**Rent Buffer:** ${rent_buffer:.2f} per month ({buffer_percentage:.1f}% above break-even)")
            
            # Long-term profitability
            if avg_cash_on_cash < 5:
                st.info("The average cash on cash return is below 5%, which is generally considered low for rental property investments.")
            elif avg_cash_on_cash > 8:
                st.success("The average cash on cash return is above 8%, which is generally considered excellent for rental property investments.")
