import numpy as np
import pandas as pd
from ml_models import PropertyValuationModel, RentalPriceModel

def calculate_roi(annual_cash_flow, initial_investment):
    """
    Calculate Return on Investment (ROI).
    
    Args:
        annual_cash_flow (float): Annual cash flow from the property
        initial_investment (float): Initial investment amount
    
    Returns:
        float: ROI as a decimal (e.g., 0.08 for 8%)
    """
    if initial_investment == 0:
        return 0
    return annual_cash_flow / initial_investment

def calculate_rental_yield(annual_rental_income, property_value):
    """
    Calculate Rental Yield.
    
    Args:
        annual_rental_income (float): Annual rental income
        property_value (float): Current property value
    
    Returns:
        float: Rental yield as a decimal (e.g., 0.05 for 5%)
    """
    if property_value == 0:
        return 0
    return annual_rental_income / property_value

def calculate_cap_rate(annual_noi, property_value):
    """
    Calculate Capitalization Rate (Cap Rate).
    
    Args:
        annual_noi (float): Annual Net Operating Income
        property_value (float): Current property value
    
    Returns:
        float: Cap rate as a decimal (e.g., 0.06 for 6%)
    """
    if property_value == 0:
        return 0
    return annual_noi / property_value

def calculate_cash_on_cash_return(annual_cash_flow, down_payment):
    """
    Calculate Cash on Cash Return.
    
    Args:
        annual_cash_flow (float): Annual cash flow
        down_payment (float): Down payment amount
    
    Returns:
        float: Cash on cash return as a decimal
    """
    if down_payment == 0:
        return 0
    return annual_cash_flow / down_payment

def calculate_mortgage_payment(loan_amount, interest_rate, loan_term_years):
    """
    Calculate monthly mortgage payment.
    
    Args:
        loan_amount (float): Loan amount
        interest_rate (float): Annual interest rate as a decimal (e.g., 0.04 for 4%)
        loan_term_years (int): Loan term in years
    
    Returns:
        float: Monthly mortgage payment
    """
    monthly_rate = interest_rate / 12
    num_payments = loan_term_years * 12
    
    if monthly_rate == 0:
        return loan_amount / num_payments
    
    return loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)

def analyze_investment(property_data, investment_params):
    """
    Perform a comprehensive investment analysis.
    
    Args:
        property_data (dict): Property data including price, location, etc.
        investment_params (dict): Investment parameters like down payment percentage, interest rate, etc.
    
    Returns:
        dict: Investment analysis results
    """
    # Extract property data
    property_price = property_data.get('price', 0)
    monthly_rent = property_data.get('estimated_monthly_rent', 0)
    
    # If monthly rent is not provided, estimate it
    if monthly_rent == 0:
        monthly_rent = property_price * 0.008  # Simple estimation: 0.8% of property value
    
    # Extract investment parameters
    down_payment_pct = investment_params.get('down_payment_pct', 0.2)
    interest_rate = investment_params.get('interest_rate', 0.045)
    loan_term_years = investment_params.get('loan_term_years', 30)
    closing_costs_pct = investment_params.get('closing_costs_pct', 0.03)
    property_tax_rate = investment_params.get('property_tax_rate', 0.01)
    insurance_rate = investment_params.get('insurance_rate', 0.005)
    maintenance_pct = investment_params.get('maintenance_pct', 0.01)
    vacancy_rate = investment_params.get('vacancy_rate', 0.05)
    property_mgmt_pct = investment_params.get('property_mgmt_pct', 0.1)
    appreciation_rate = investment_params.get('appreciation_rate', 0.03)
    income_tax_rate = investment_params.get('income_tax_rate', 0.25)
    
    # Calculate loan details
    down_payment = property_price * down_payment_pct
    closing_costs = property_price * closing_costs_pct
    total_investment = down_payment + closing_costs
    loan_amount = property_price - down_payment
    
    # Calculate monthly mortgage payment
    monthly_mortgage = calculate_mortgage_payment(loan_amount, interest_rate, loan_term_years)
    
    # Calculate annual expenses
    annual_property_tax = property_price * property_tax_rate
    annual_insurance = property_price * insurance_rate
    annual_maintenance = property_price * maintenance_pct
    annual_vacancy_cost = monthly_rent * 12 * vacancy_rate
    annual_property_mgmt = monthly_rent * 12 * property_mgmt_pct
    
    # Calculate annual income and cash flow
    annual_rental_income = monthly_rent * 12
    annual_operating_expenses = annual_property_tax + annual_insurance + annual_maintenance + annual_vacancy_cost + annual_property_mgmt
    annual_noi = annual_rental_income - annual_operating_expenses
    annual_mortgage_payment = monthly_mortgage * 12
    annual_cash_flow = annual_noi - annual_mortgage_payment
    
    # Calculate investment metrics
    cap_rate = calculate_cap_rate(annual_noi, property_price)
    cash_on_cash_return = calculate_cash_on_cash_return(annual_cash_flow, total_investment)
    rental_yield = calculate_rental_yield(annual_rental_income, property_price)
    
    # Calculate tax implications
    annual_depreciation = property_price * 0.8 / 27.5  # Building value (80% of property) depreciated over 27.5 years
    taxable_income = annual_rental_income - annual_operating_expenses - annual_depreciation - (loan_amount * interest_rate)
    tax_liability = max(0, taxable_income * income_tax_rate)
    after_tax_cash_flow = annual_cash_flow - tax_liability
    
    # Calculate five-year projection
    five_year_projection = []
    current_property_value = property_price
    current_loan_balance = loan_amount
    
    for year in range(1, 6):
        # Update property value
        current_property_value *= (1 + appreciation_rate)
        
        # Update loan balance (simplified calculation)
        interest_portion = current_loan_balance * interest_rate
        principal_portion = annual_mortgage_payment - interest_portion
        current_loan_balance -= principal_portion
        
        # Calculate equity
        equity = current_property_value - current_loan_balance
        
        # Increase rent by 2% per year (typical rent growth)
        current_annual_rent = annual_rental_income * (1.02 ** (year - 1))
        
        # Update expenses (simplified, assuming expenses grow with inflation)
        current_operating_expenses = annual_operating_expenses * (1.02 ** (year - 1))
        current_noi = current_annual_rent - current_operating_expenses
        current_cash_flow = current_noi - annual_mortgage_payment
        
        # Calculate current returns
        current_cap_rate = current_noi / current_property_value
        current_cash_on_cash = current_cash_flow / total_investment
        
        # Add to projection
        five_year_projection.append({
            'year': year,
            'property_value': current_property_value,
            'loan_balance': current_loan_balance,
            'equity': equity,
            'annual_rent': current_annual_rent,
            'noi': current_noi,
            'cash_flow': current_cash_flow,
            'cap_rate': current_cap_rate,
            'cash_on_cash': current_cash_on_cash
        })
    
    # Calculate payback period (simplified)
    cumulative_cash_flow = 0
    payback_years = 0
    
    for year in range(1, 31):
        # Simplifying assumption: Cash flow grows by 2% annually
        current_cash_flow = annual_cash_flow * (1.02 ** (year - 1))
        cumulative_cash_flow += current_cash_flow
        
        if cumulative_cash_flow >= total_investment and payback_years == 0:
            payback_years = year
    
    # Return investment analysis results
    return {
        'property_price': property_price,
        'monthly_rent': monthly_rent,
        'down_payment': down_payment,
        'closing_costs': closing_costs,
        'total_investment': total_investment,
        'loan_amount': loan_amount,
        'monthly_mortgage': monthly_mortgage,
        'annual_rental_income': annual_rental_income,
        'annual_operating_expenses': annual_operating_expenses,
        'annual_noi': annual_noi,
        'annual_cash_flow': annual_cash_flow,
        'cap_rate': cap_rate,
        'cash_on_cash_return': cash_on_cash_return,
        'rental_yield': rental_yield,
        'after_tax_cash_flow': after_tax_cash_flow,
        'five_year_projection': five_year_projection,
        'payback_period': payback_years,
        'first_year_roi': cash_on_cash_return,
        'five_year_roi': (five_year_projection[4]['equity'] + sum(proj['cash_flow'] for proj in five_year_projection)) / total_investment - 1,
        'monthly_expenses': {
            'mortgage': monthly_mortgage,
            'property_tax': annual_property_tax / 12,
            'insurance': annual_insurance / 12,
            'maintenance': annual_maintenance / 12,
            'vacancy': annual_vacancy_cost / 12,
            'property_management': annual_property_mgmt / 12
        },
        'break_even_occupancy': (annual_mortgage_payment + annual_operating_expenses - annual_vacancy_cost) / (annual_rental_income - annual_vacancy_cost)
    }

def calculate_investment_score(investment_analysis, investor_profile):
    """
    Calculate an investment score based on the analysis and investor profile.
    
    Args:
        investment_analysis (dict): Results from analyze_investment
        investor_profile (dict): Investor profile with preferences
    
    Returns:
        float: Investment score from 0-100
    """
    # Extract investor preferences
    goal = investor_profile.get('goal', 'balanced')  # 'cash_flow', 'appreciation', or 'balanced'
    risk_tolerance = investor_profile.get('risk_tolerance', 'medium')  # 'low', 'medium', or 'high'
    investment_horizon = investor_profile.get('investment_horizon', 'medium')  # 'short', 'medium', or 'long'
    
    # Convert risk tolerance to numeric value
    risk_factor = {'low': 0.3, 'medium': 0.5, 'high': 0.7}.get(risk_tolerance, 0.5)
    
    # Convert investment horizon to numeric value
    horizon_factor = {'short': 0.3, 'medium': 0.5, 'long': 0.7}.get(investment_horizon, 0.5)
    
    # Base score components
    cash_flow_score = min(100, max(0, investment_analysis['cash_on_cash_return'] * 1000))  # 7% COC return = 70 points
    appreciation_score = min(100, max(0, investment_analysis['five_year_projection'][4]['property_value'] / investment_analysis['property_price'] * 20))  # 5x in 5 years = 100 points
    safety_score = min(100, max(0, 100 - investment_analysis['break_even_occupancy'] * 100))  # Lower break-even occupancy is better
    
    # Adjust weights based on investor goal
    if goal == 'cash_flow':
        weights = {'cash_flow': 0.6, 'appreciation': 0.2, 'safety': 0.2}
    elif goal == 'appreciation':
        weights = {'cash_flow': 0.2, 'appreciation': 0.6, 'safety': 0.2}
    else:  # balanced
        weights = {'cash_flow': 0.4, 'appreciation': 0.4, 'safety': 0.2}
    
    # Calculate weighted score
    weighted_score = (
        weights['cash_flow'] * cash_flow_score +
        weights['appreciation'] * appreciation_score +
        weights['safety'] * safety_score
    )
    
    # Adjust score based on risk tolerance and investment horizon
    adjusted_score = weighted_score * (1 + (risk_factor - 0.5) * 0.2) * (1 + (horizon_factor - 0.5) * 0.2)
    
    # Ensure score is in 0-100 range
    final_score = min(100, max(0, adjusted_score))
    
    return final_score
