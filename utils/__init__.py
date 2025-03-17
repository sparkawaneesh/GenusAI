# Make utils directory a package

def format_currency(value):
    """
    Format a number as currency.
    
    Args:
        value (float): Number to format
    
    Returns:
        str: Formatted currency string
    """
    return f"${value:,.2f}"

def calculate_monthly_mortgage(loan_amount, interest_rate, loan_term_years):
    """
    Calculate monthly mortgage payment.
    
    Args:
        loan_amount (float): Loan amount
        interest_rate (float): Annual interest rate as a decimal (e.g., 0.04)
        loan_term_years (int): Loan term in years
    
    Returns:
        float: Monthly mortgage payment
    """
    monthly_rate = interest_rate / 12
    num_payments = loan_term_years * 12
    if monthly_rate == 0:
        return loan_amount / num_payments
    return loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)

def calculate_price_to_rent_ratio(price, monthly_rent):
    """
    Calculate price to rent ratio.
    
    Args:
        price (float): Property price
        monthly_rent (float): Monthly rent
    
    Returns:
        float: Price to rent ratio
    """
    annual_rent = monthly_rent * 12
    return price / annual_rent

def calculate_cap_rate(annual_noi, property_value):
    """
    Calculate capitalization rate.
    
    Args:
        annual_noi (float): Annual net operating income
        property_value (float): Property value
    
    Returns:
        float: Cap rate as a decimal
    """
    return annual_noi / property_value

def calculate_cash_flow(monthly_rent, monthly_expenses):
    """
    Calculate monthly cash flow.
    
    Args:
        monthly_rent (float): Monthly rental income
        monthly_expenses (float): Monthly expenses
    
    Returns:
        float: Monthly cash flow
    """
    return monthly_rent - monthly_expenses

def format_percentage(value):
    """
    Format a decimal as a percentage.
    
    Args:
        value (float): Decimal value to format (e.g., 0.05 for 5%)
    
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.2%}"

def abbreviate_number(number):
    """
    Abbreviate a large number (e.g., 1,000,000 -> 1M).
    
    Args:
        number (float): Number to abbreviate
    
    Returns:
        str: Abbreviated number
    """
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return f"{number:.0f}"

def safe_divide(numerator, denominator):
    """
    Safely divide two numbers, returning 0 if denominator is 0.
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
    
    Returns:
        float: Division result, or 0 if denominator is 0
    """
    if denominator == 0:
        return 0
    return numerator / denominator

def color_scale(value, min_val, max_val, reverse=False):
    """
    Generate a color scale value between 0 and 1.
    
    Args:
        value (float): The value to convert
        min_val (float): Minimum value in the range
        max_val (float): Maximum value in the range
        reverse (bool): If True, reverse the scale (1 becomes min, 0 becomes max)
    
    Returns:
        float: Color scale value between 0 and 1
    """
    if min_val == max_val:
        return 0.5
    
    scale = (value - min_val) / (max_val - min_val)
    scale = max(0, min(1, scale))
    
    if reverse:
        scale = 1 - scale
        
    return scale