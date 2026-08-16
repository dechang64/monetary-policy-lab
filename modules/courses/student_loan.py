"""Student Loan Planner — standard vs income-driven repayment."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.pf_calculators import student_loan_payoff
from utils.pf_constants import STUDENT_LOAN_TYPES

def render():
    st.markdown('<div class="main-header"><h1>🎓 Student Loan Planner</h1>'
                '<p>Standard vs income-driven repayment strategies</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    **Student loans** are often the second-largest debt after a mortgage.

    **Repayment options:**
    - **Standard**: Fixed monthly payment over 10 years. Lowest total interest.
    - **Income-Driven (IDR)**: Payment = 10-15% of discretionary income.
      Longer term (20-25 years), higher total interest, but lower monthly burden.
    - **Refinance**: Combine federal + private loans at lower rate (loses federal protections).

    **Key trade-off**: Lower monthly payment = more total interest paid.
    """)

    st.markdown("### 🎛️ Your Student Loans")
    col1, col2, col3 = st.columns(3)
    with col1:
        principal = st.number_input("Total Loan Principal ($)", 1000, 500000, 35000, step=1000)
    with col2:
        loan_type = st.selectbox("Loan Type", list(STUDENT_LOAN_TYPES.keys()))
    with col3:
        custom_apr = st.checkbox("Use custom APR?")
        if custom_apr:
            apr = st.slider("Custom APR (%)", 1.0, 15.0, 5.0, 0.1) / 100
        else:
            apr = STUDENT_LOAN_TYPES[loan_type]
            st.metric("Current APR", f"{apr*100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        years = st.selectbox("Repayment Term", [10, 15, 20, 25], index=0)
    with col2:
        income = st.number_input("Annual Income ($)", 0, 1000000, 60000, step=5000)

    result = student_loan_payoff(principal, apr, years, income)

    st.markdown("---")
    st.markdown("### 📊 Standard Repayment (10-year amortized)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Payment", f"${result['standard_monthly']}")
    col2.metric("Total Paid", f"${result['standard_total_paid']:,.0f}")
    col3.metric("Total Interest", f"${result['standard_total_interest']:,.0f}")

    if result['income_driven_eligible']:
        st.markdown("### 📊 Income-Driven Repayment (10% of income)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly Payment", f"${result['income_driven_monthly']}")
        col2.metric("Annual Payment", f"${result['income_driven_monthly']*12:,.0f}")
        col3.metric("Payment vs Standard",
                    f"${result['income_driven_monthly'] - result['standard_monthly']:+,.0f}/mo",
                    delta_color="inverse" if result['income_driven_monthly'] > result['standard_monthly'] else "normal")

    # Comparison chart
    st.markdown("### 📈 Payment & Interest Comparison")
    scenarios = []
    for y in [10, 15, 20, 25]:
        r = student_loan_payoff(principal, apr, y, income)
        scenarios.append({
            "Term": f"{y} years",
            "Monthly": r["standard_monthly"],
            "Total Interest": r["standard_total_interest"],
            "Total Paid": r["standard_total_paid"],
        })

    df = pd.DataFrame(scenarios)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Term"], y=df["Monthly"], name="Monthly Payment",
                          marker_color="#3b82f6", yaxis="y"))
    fig.add_trace(go.Bar(x=df["Term"], y=df["Total Interest"], name="Total Interest",
                          marker_color="#dc2626", yaxis="y2"))
    fig.update_layout(
        title="Longer Term = Lower Payment but More Interest",
        yaxis=dict(title="Monthly Payment ($)"),
        yaxis2=dict(title="Total Interest ($)", overlaying="y", side="right"),
        barmode="group", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Amortization preview
    st.markdown("### 📋 Year-by-Year Breakdown (Standard 10-year)")
    import numpy as np
    monthly_rate = apr / 12
    balance = principal
    rows = []
    year_interest = 0
    for month in range(1, 121):
        interest = balance * monthly_rate
        principal_payment = result['standard_monthly'] - interest
        balance -= principal_payment
        year_interest += interest
        if month % 12 == 0:
            rows.append({"Year": month // 12,
                         "Principal Paid": f"${principal - balance:,.0f}",
                         "Interest Paid": f"${year_interest:,.0f}",
                         "Remaining Balance": f"${max(0, balance):,.0f}"})
            year_interest = 0
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
    **Discussion Questions:**
    1. If you pay $100 extra per month, how many years earlier can you pay off the loan?
    2. When does it make sense to choose IDR over standard repayment?
    3. Should you invest extra money or pay off student loans early? (Compare APR vs. investment return)
    """)
