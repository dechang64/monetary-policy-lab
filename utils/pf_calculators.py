"""
Personal Finance Lab — Financial Calculators
=============================================
Core calculators for teaching personal finance concepts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compound_interest(principal: float, annual_rate: float, years: int,
                      contributions: float = 0, contrib_freq: str = "monthly") -> pd.DataFrame:
    """
    Project compound interest over time with optional contributions.
    
    Returns DataFrame: Year, Balance, Contributions, Interest Earned
    """
    if contrib_freq == "monthly":
        periods_per_year = 12
    elif contrib_freq == "annually":
        periods_per_year = 1
    elif contrib_freq == "biweekly":
        periods_per_year = 26
    else:
        periods_per_year = 12
    
    rate_per_period = annual_rate / periods_per_year
    total_periods = years * periods_per_year
    contrib_per_period = contributions / periods_per_year if contributions else 0
    
    balance = principal
    total_contrib = principal
    rows = [{"Year": 0, "Balance": round(balance, 2),
             "Contributions": round(total_contrib, 2),
             "Interest Earned": 0}]
    
    for year in range(1, years + 1):
        year_interest = 0
        for _ in range(periods_per_year):
            interest = balance * rate_per_period
            balance = balance + interest + contrib_per_period
            year_interest += interest
            total_contrib += contrib_per_period
        rows.append({
            "Year": year, "Balance": round(balance, 2),
            "Contributions": round(total_contrib, 2),
            "Interest Earned": round(year_interest, 2),
        })
    
    return pd.DataFrame(rows)


def monte_carlo_retirement(
    initial_savings: float, annual_contribution: float,
    years_to_retire: int, years_in_retirement: int,
    expected_return: float, volatility: float,
    withdrawal_rate: float = 0.04, n_simulations: int = 1000, seed: int = 42
) -> Dict:
    """
    Monte Carlo simulation for retirement planning.
    
    Returns dict with success_rate, median_outcome, percentile_10, percentile_90, simulations
    """
    rng = np.random.default_rng(seed)
    n_periods = years_to_retire + years_in_retirement
    all_final = []
    
    for _ in range(n_simulations):
        balance = initial_savings
        # Accumulation phase
        for y in range(years_to_retire):
            ret = rng.normal(expected_return, volatility)
            balance = balance * (1 + ret) + annual_contribution
        # Withdrawal phase
        for y in range(years_in_retirement):
            ret = rng.normal(expected_return, volatility)
            balance = balance * (1 + ret) - balance * withdrawal_rate
        
        all_final.append(balance)
    
    all_final = np.array(all_final)
    return {
        "success_rate": (all_final > 0).mean(),
        "median_outcome": np.median(all_final),
        "percentile_10": np.percentile(all_final, 10),
        "percentile_90": np.percentile(all_final, 90),
        "simulations": all_final,
    }


def buy_vs_rent(
    home_price: float, down_payment_pct: float, mortgage_rate: float,
    mortgage_years: int, property_tax_rate: float, maintenance_rate: float,
    rent_monthly: float, appreciation_rate: float, hold_years: int,
    opportunity_return: float = 0.07
) -> Dict:
    """
    Compare buying vs renting over hold period.
    """
    down_payment = home_price * down_payment_pct
    loan_amount = home_price - down_payment
    monthly_rate = mortgage_rate / 12
    n_payments = mortgage_years * 12
    
    # Monthly mortgage payment (fixed rate)
    if monthly_rate > 0:
        mortgage_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / \
                            ((1 + monthly_rate)**n_payments - 1)
    else:
        mortgage_payment = loan_amount / n_payments
    
    # Annual costs
    annual_property_tax = home_price * property_tax_rate
    annual_maintenance = home_price * maintenance_rate
    annual_mortgage = mortgage_payment * 12
    
    # Total buy cost over hold period
    total_buy_cost = (annual_mortgage + annual_property_tax + annual_maintenance) * hold_years
    home_value_at_sale = home_price * (1 + appreciation_rate) ** hold_years
    loan_balance_at_sale = loan_amount * (1 + monthly_rate)**(hold_years * 12) - \
                            mortgage_payment * (((1 + monthly_rate)**(hold_years * 12) - 1) / monthly_rate)
    loan_balance_at_sale = max(0, loan_balance_at_sale)
    
    buy_equity = home_value_at_sale - loan_balance_at_sale
    buy_net = buy_equity - down_payment  # net gain from buying
    
    # Rent scenario: invest down payment + difference in monthly costs
    rent_annual = rent_monthly * 12
    monthly_diff = mortgage_payment + (annual_property_tax + annual_maintenance) / 12 - rent_monthly
    if monthly_diff < 0:
        monthly_diff = 0  # can't invest negative
    
    # Future value of invested down payment + monthly savings
    invested_value = down_payment * (1 + opportunity_return) ** hold_years
    for y in range(hold_years):
        invested_value += monthly_diff * 12 * (1 + opportunity_return) ** (hold_years - y - 1)
    
    rent_net = invested_value - down_payment  # net gain from renting
    
    return {
        "down_payment": round(down_payment, 2),
        "loan_amount": round(loan_amount, 2),
        "monthly_mortgage": round(mortgage_payment, 2),
        "annual_mortgage": round(annual_mortgage, 2),
        "annual_property_tax": round(annual_property_tax, 2),
        "annual_maintenance": round(annual_maintenance, 2),
        "home_value_at_sale": round(home_value_at_sale, 2),
        "loan_balance_at_sale": round(loan_balance_at_sale, 2),
        "buy_equity": round(buy_equity, 2),
        "buy_net_gain": round(buy_net, 2),
        "rent_total_cost": round(rent_annual * hold_years, 2),
        "invested_value": round(invested_value, 2),
        "rent_net_gain": round(rent_net, 2),
        "advantage": "Buy" if buy_net > rent_net else "Rent",
        "advantage_amount": round(abs(buy_net - rent_net), 2),
    }


def credit_card_payoff(
    balance: float, apr: float, monthly_payment: float,
    new_charges: float = 0
) -> Dict:
    """
    Calculate credit card payoff timeline and total interest.
    """
    monthly_rate = apr / 12
    months = 0
    total_interest = 0
    current = balance
    
    while current > 0 and months < 600:  # cap at 50 years
        interest = current * monthly_rate
        total_interest += interest
        current = current + interest + new_charges - monthly_payment
        if current < 0:
            current = 0
        months += 1
        if monthly_payment <= current * monthly_rate + new_charges:
            return {"error": "Monthly payment too low — balance will grow forever. Increase payment.",
                    "min_payment_needed": round((current * monthly_rate + new_charges + 1), 2)}
    
    return {
        "months_to_payoff": months,
        "years_to_payoff": round(months / 12, 1),
        "total_interest_paid": round(total_interest, 2),
        "total_paid": round(balance + total_interest, 2),
        "interest_pct_of_balance": round((total_interest / balance) * 100, 1) if balance > 0 else 0,
    }


def portfolio_optimization(
    assets: List[str], expected_returns: np.ndarray, cov_matrix: np.ndarray,
    n_portfolios: int = 10000, risk_free_rate: float = 0.02, seed: int = 42
) -> Dict:
    """
    Generate efficient frontier using Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(assets)
    
    # Generate random weights
    weights_all = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    
    returns = weights_all @ expected_returns
    vols = np.sqrt(np.diag(weights_all @ cov_matrix @ weights_all.T))
    sharpe = (returns - risk_free_rate) / vols
    
    # Max Sharpe portfolio
    max_sharpe_idx = np.argmax(sharpe)
    max_sharpe_weights = weights_all[max_sharpe_idx]
    
    # Min volatility portfolio
    min_vol_idx = np.argmin(vols)
    min_vol_weights = weights_all[min_vol_idx]
    
    return {
        "all_returns": returns,
        "all_vols": vols,
        "all_sharpe": sharpe,
        "max_sharpe_weights": max_sharpe_weights,
        "max_sharpe_return": returns[max_sharpe_idx],
        "max_sharpe_vol": vols[max_sharpe_idx],
        "max_sharpe_ratio": sharpe[max_sharpe_idx],
        "min_vol_weights": min_vol_weights,
        "min_vol_return": returns[min_vol_idx],
        "min_vol_vol": vols[min_vol_idx],
    }


def calculate_federal_tax(income: float, brackets: List[Tuple[float, float]]) -> Dict:
    """
    Progressive tax calculation.
    """
    tax = 0
    for i, (threshold, rate) in enumerate(brackets):
        if i == len(brackets) - 1:
            if income > threshold:
                tax += (income - threshold) * rate
        else:
            next_threshold = brackets[i+1][0]
            if income > threshold:
                taxable = min(income, next_threshold) - threshold
                tax += taxable * rate
    
    return {
        "gross_income": round(income, 2),
        "federal_tax": round(tax, 2),
        "effective_rate": round((tax / income) * 100, 2) if income > 0 else 0,
        "net_income": round(income - tax, 2),
    }


def estimate_fico(
    payment_history_score: int, utilization_pct: float,
    credit_history_years: int, credit_mix_score: int,
    new_inquiries: int
) -> Dict:
    """
    Simplified FICO score estimator.
    Each factor scored 300-850.
    """
    # Payment history (35%)
    ph_component = payment_history_score
    
    # Utilization (30%) — lower is better, <10% = max, >50% = min
    if utilization_pct < 10:
        u_component = 850
    elif utilization_pct < 30:
        u_component = 750
    elif utilization_pct < 50:
        u_component = 650
    else:
        u_component = 550
    
    # Credit history length (15%) — 7+ years = max
    if credit_history_years >= 10:
        h_component = 800
    elif credit_history_years >= 7:
        h_component = 750
    elif credit_history_years >= 3:
        h_component = 650
    else:
        h_component = 550
    
    # Credit mix (10%)
    cm_component = credit_mix_score
    
    # New inquiries (10%) — fewer is better
    if new_inquiries == 0:
        nq_component = 800
    elif new_inquiries <= 2:
        nq_component = 720
    elif new_inquiries <= 4:
        nq_component = 650
    else:
        nq_component = 580
    
    score = (ph_component * 0.35 + u_component * 0.30 +
             h_component * 0.15 + cm_component * 0.10 + nq_component * 0.10)
    
    if score >= 800:
        rating = "Exceptional"
    elif score >= 740:
        rating = "Very Good"
    elif score >= 670:
        rating = "Good"
    elif score >= 580:
        rating = "Fair"
    else:
        rating = "Poor"
    
    return {
        "estimated_score": round(score),
        "rating": rating,
        "components": {
            "Payment History (35%)": round(ph_component),
            "Credit Utilization (30%)": round(u_component),
            "Credit History Length (15%)": round(h_component),
            "Credit Mix (10%)": round(cm_component),
            "New Inquiries (10%)": round(nq_component),
        },
    }


def student_loan_payoff(
    principal: float, apr: float, years: int,
    income: float = 0, disposable_income_pct: float = 0.10
) -> Dict:
    """
    Standard student loan payoff with optional income-driven payment.
    """
    monthly_rate = apr / 12
    n_payments = years * 12
    
    # Standard payment (amortized)
    if monthly_rate > 0:
        standard_payment = principal * (monthly_rate * (1 + monthly_rate)**n_payments) / \
                           ((1 + monthly_rate)**n_payments - 1)
    else:
        standard_payment = principal / n_payments
    
    total_paid_standard = standard_payment * n_payments
    total_interest_standard = total_paid_standard - principal
    
    # Income-driven (10% of disposable income — simplified)
    income_driven_payment = income * disposable_income_pct / 12 if income > 0 else 0
    
    return {
        "standard_monthly": round(standard_payment, 2),
        "standard_total_paid": round(total_paid_standard, 2),
        "standard_total_interest": round(total_interest_standard, 2),
        "standard_years": years,
        "income_driven_monthly": round(income_driven_payment, 2),
        "income_driven_eligible": income > 0,
    }
