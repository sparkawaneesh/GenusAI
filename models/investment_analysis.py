import pandas as pd
import numpy as np
import streamlit as st

class InvestmentAnalysis:
    """
    Analyze real estate investments and calculate ROI metrics
    """
    
    def __init__(self, property_data=None):
        self.property_data = property_data
        
    def set_property_data(self, property_data):
        """
        Set the property data for analysis
        
        Args:
            property_data (DataFrame): Property information
        """
        self.property_data = property_data
    
    def calculate_roi(self, purchase_price, annual_rental_income, expenses,
                     appreciation_rate=0.03, holding_period=5, 
                     down_payment_percentage=20, interest_rate=4.5,
                     loan_term=30, closing_costs_percentage=3,
                     selling_costs_percentage=6):
        """
        Calculate Return on Investment for a property
        
        Args:
            purchase_price (float): Purchase price of the property
            annual_rental_income (float): Annual rental income
            expenses (dict): Dictionary with annual expenses
            appreciation_rate (float): Annual appreciation rate
            holding_period (int): Number of years to hold property
            down_payment_percentage (float): Down payment percentage
            interest_rate (float): Annual interest rate
            loan_term (int): Loan term in years
            closing_costs_percentage (float): Closing costs as percentage of purchase
            selling_costs_percentage (float): Selling costs as percentage of final value
            
        Returns:
            dict: ROI metrics
        """
        # Calculate down payment and loan amount
        down_payment = purchase_price * (down_payment_percentage / 100)
        loan_amount = purchase_price - down_payment
        
        # Calculate closing costs
        closing_costs = purchase_price * (closing_costs_percentage / 100)
        
        # Calculate initial investment
        initial_investment = down_payment + closing_costs
        
        # Calculate monthly mortgage payment
        monthly_interest_rate = (interest_rate / 100) / 12
        months = loan_term * 12
        monthly_mortgage_payment = 0
        
        if monthly_interest_rate > 0:
            monthly_mortgage_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** months) / ((1 + monthly_interest_rate) ** months - 1)
        
        annual_mortgage_payment = monthly_mortgage_payment * 12
        
        # Sum up annual expenses
        total_annual_expenses = sum(expenses.values()) + annual_mortgage_payment
        
        # Calculate annual cash flow
        annual_cash_flow = annual_rental_income - total_annual_expenses
        
        # Calculate property value after holding period
        final_property_value = purchase_price * ((1 + appreciation_rate) ** holding_period)
        
        # Calculate remaining loan balance after holding period
        remaining_balance = 0
        if loan_amount > 0:
            remaining_payments = loan_term * 12 - holding_period * 12
            if remaining_payments > 0:
                remaining_balance = loan_amount * ((1 + monthly_interest_rate) ** (loan_term * 12) - (1 + monthly_interest_rate) ** (holding_period * 12)) / ((1 + monthly_interest_rate) ** (loan_term * 12) - 1)
        
        # Calculate selling costs
        selling_costs = final_property_value * (selling_costs_percentage / 100)
        
        # Calculate equity upon sale
        equity = final_property_value - remaining_balance - selling_costs
        
        # Calculate total profit
        total_cash_flow = annual_cash_flow * holding_period
        total_profit = equity - initial_investment + total_cash_flow
        
        # Calculate ROI metrics
        cash_on_cash_roi = (annual_cash_flow / initial_investment) * 100
        total_roi = (total_profit / initial_investment) * 100
        annualized_roi = ((1 + (total_roi / 100)) ** (1 / holding_period) - 1) * 100
        
        # Calculate cap rate
        net_operating_income = annual_rental_income - sum(expenses.values())
        cap_rate = (net_operating_income / purchase_price) * 100
        
        # Calculate cash flow metrics
        monthly_cash_flow = annual_cash_flow / 12
        cash_flow_per_door = monthly_cash_flow  # For single property
        
        # Calculate debt service coverage ratio
        dscr = net_operating_income / annual_mortgage_payment if annual_mortgage_payment > 0 else float('inf')
        
        # Calculate gross rent multiplier
        grm = purchase_price / annual_rental_income if annual_rental_income > 0 else float('inf')
        
        return {
            'initial_investment': initial_investment,
            'monthly_mortgage_payment': monthly_mortgage_payment,
            'annual_cash_flow': annual_cash_flow,
            'monthly_cash_flow': monthly_cash_flow,
            'cash_flow_per_door': cash_flow_per_door,
            'final_property_value': final_property_value,
            'equity': equity,
            'total_profit': total_profit,
            'cash_on_cash_roi': cash_on_cash_roi,
            'total_roi': total_roi,
            'annualized_roi': annualized_roi,
            'cap_rate': cap_rate,
            'dscr': dscr,
            'grm': grm
        }
    
    def estimate_expenses(self, purchase_price, property_type='single_family'):
        """
        Estimate typical expenses for a property
        
        Args:
            purchase_price (float): Purchase price of the property
            property_type (str): Type of property
            
        Returns:
            dict: Estimated annual expenses
        """
        # Default expense percentages based on property type
        expense_rates = {
            'single_family': {
                'property_tax': 1.0,  # % of purchase price annually
                'insurance': 0.5,     # % of purchase price annually
                'maintenance': 1.0,   # % of purchase price annually
                'vacancy': 5.0,       # % of rental income
                'property_management': 8.0,  # % of rental income
                'utilities': 0.0,     # Owner-paid utilities
                'hoa': 0.0           # HOA fees
            },
            'multi_family': {
                'property_tax': 1.2,
                'insurance': 0.6,
                'maintenance': 1.2,
                'vacancy': 6.0,
                'property_management': 10.0,
                'utilities': 0.3,
                'hoa': 0.0
            },
            'condo': {
                'property_tax': 1.0,
                'insurance': 0.4,
                'maintenance': 0.5,
                'vacancy': 5.0,
                'property_management': 8.0,
                'utilities': 0.0,
                'hoa': 0.5
            }
        }
        
        # Use single_family as default if property_type not found
        if property_type not in expense_rates:
            property_type = 'single_family'
        
        rates = expense_rates[property_type]
        
        # Estimate monthly rental income (used for vacancy and management calculation)
        # Assumption: Monthly rent is roughly 0.8% of purchase price for a typical property
        estimated_monthly_rent = purchase_price * 0.008
        estimated_annual_rent = estimated_monthly_rent * 12
        
        # Calculate expenses
        expenses = {
            'property_tax': purchase_price * (rates['property_tax'] / 100),
            'insurance': purchase_price * (rates['insurance'] / 100),
            'maintenance': purchase_price * (rates['maintenance'] / 100),
            'vacancy': estimated_annual_rent * (rates['vacancy'] / 100),
            'property_management': estimated_annual_rent * (rates['property_management'] / 100),
            'utilities': purchase_price * (rates['utilities'] / 100),
            'hoa': purchase_price * (rates['hoa'] / 100)
        }
        
        return expenses
    
    def calculate_financial_metrics(self, df):
        """
        Calculate financial metrics for a set of properties
        
        Args:
            df (DataFrame): Properties dataframe
            
        Returns:
            DataFrame: Original dataframe with added financial metrics
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_cols = ['price', 'monthly_rent_estimate']
        if not all(col in result_df.columns for col in required_cols):
            return result_df
        
        # Calculate cap rate
        result_df['annual_rent'] = result_df['monthly_rent_estimate'] * 12
        
        # Estimate expenses (simplified approach)
        result_df['estimated_expenses'] = result_df.apply(
            lambda x: sum(self.estimate_expenses(x['price']).values()), axis=1
        )
        
        # Calculate NOI (Net Operating Income)
        result_df['noi'] = result_df['annual_rent'] - result_df['estimated_expenses']
        
        # Calculate cap rate
        result_df['cap_rate'] = (result_df['noi'] / result_df['price']) * 100
        
        # Calculate gross rent multiplier
        result_df['grm'] = result_df['price'] / result_df['annual_rent']
        
        # Calculate price to rent ratio
        result_df['price_to_rent_ratio'] = result_df['price'] / (result_df['monthly_rent_estimate'] * 12)
        
        # Calculate cash on cash return (simplified, assuming 20% down payment and 4.5% interest rate)
        down_payment_pct = 20
        interest_rate = 4.5
        loan_term = 30
        
        result_df['down_payment'] = result_df['price'] * (down_payment_pct / 100)
        result_df['loan_amount'] = result_df['price'] - result_df['down_payment']
        
        # Calculate monthly mortgage payment
        monthly_interest_rate = (interest_rate / 100) / 12
        months = loan_term * 12
        
        result_df['monthly_mortgage'] = result_df.apply(
            lambda x: x['loan_amount'] * (monthly_interest_rate * (1 + monthly_interest_rate) ** months) / 
                     ((1 + monthly_interest_rate) ** months - 1) if monthly_interest_rate > 0 else 0,
            axis=1
        )
        
        result_df['annual_mortgage'] = result_df['monthly_mortgage'] * 12
        result_df['annual_cash_flow'] = result_df['noi'] - result_df['annual_mortgage']
        result_df['cash_on_cash_return'] = (result_df['annual_cash_flow'] / result_df['down_payment']) * 100
        
        return result_df

def calculate_mortgage_payment(loan_amount, interest_rate, loan_term):
    """
    Calculate monthly mortgage payment
    
    Args:
        loan_amount (float): Loan principal amount
        interest_rate (float): Annual interest rate (percentage)
        loan_term (int): Loan term in years
        
    Returns:
        float: Monthly payment amount
    """
    # Convert annual rate to monthly rate and years to months
    monthly_rate = (interest_rate / 100) / 12
    months = loan_term * 12
    
    # Calculate payment using the mortgage formula
    if monthly_rate == 0:
        return loan_amount / months
    
    payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    
    return payment

def calculate_amortization_schedule(loan_amount, interest_rate, loan_term):
    """
    Calculate the amortization schedule for a mortgage
    
    Args:
        loan_amount (float): Loan principal amount
        interest_rate (float): Annual interest rate (percentage)
        loan_term (int): Loan term in years
        
    Returns:
        DataFrame: Amortization schedule by year
    """
    # Calculate monthly payment
    monthly_payment = calculate_mortgage_payment(loan_amount, interest_rate, loan_term)
    
    # Convert annual rate to monthly rate
    monthly_rate = (interest_rate / 100) / 12
    
    # Create amortization schedule
    schedule = []
    remaining_balance = loan_amount
    
    for year in range(1, loan_term + 1):
        yearly_principal = 0
        yearly_interest = 0
        
        for month in range(1, 13):
            # Calculate interest for this month
            interest_payment = remaining_balance * monthly_rate
            
            # Calculate principal for this month
            principal_payment = monthly_payment - interest_payment
            
            # Update remaining balance
            remaining_balance -= principal_payment
            
            # Track yearly totals
            yearly_principal += principal_payment
            yearly_interest += interest_payment
        
        schedule.append({
            'year': year,
            'principal_paid': yearly_principal,
            'interest_paid': yearly_interest,
            'total_payment': yearly_principal + yearly_interest,
            'remaining_balance': max(0, remaining_balance)
        })
    
    return pd.DataFrame(schedule)
