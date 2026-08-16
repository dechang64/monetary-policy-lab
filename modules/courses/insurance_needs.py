"""Insurance Needs — life insurance coverage estimation."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render():
    st.markdown('<div class="main-header"><h1>🛡️ Insurance Needs</h1>'
                '<p>How much life insurance do you actually need?</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    **Life insurance** replaces your income for dependents if you die.

    **Three common methods to estimate need:**
    1. **DIME** (Debt + Income + Mortgage + Education) — most comprehensive
    2. **10x Income** — simple rule of thumb
    3. **Human Life Value** — present value of future earnings

    **Types of insurance:**
    - **Term Life**: Pure protection for a set period (10/20/30 years). Cheapest.
    - **Whole Life**: Lifetime coverage + cash value. 5-10x more expensive.
    - Most financial advisors recommend **term life** and invest the difference.
    """)

    st.markdown("### 🎛️ Your Situation")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Your Age", 18, 80, 35)
        annual_income = st.number_input("Annual Income ($)", 0, 1000000, 80000, step=5000)
        retirement_age = st.slider("Planned Retirement Age", 50, 75, 65)
        dependents = st.slider("Number of Dependents", 0, 10, 2)

    with col2:
        mortgage = st.number_input("Outstanding Mortgage ($)", 0, 2000000, 300000, step=10000)
        other_debt = st.number_input("Other Debts ($)", 0, 500000, 20000, step=5000)
        college_fund = st.number_input("Education Need per Child ($)", 0, 200000, 100000, step=10000)
        existing_insurance = st.number_input("Existing Life Insurance ($)", 0, 1000000, 0, step=50000)
        savings = st.number_input("Current Savings/Investments ($)", 0, 5000000, 50000, step=10000)

    # DIME Method
    income_need = annual_income * (retirement_age - age) * 0.6  # 60% income replacement
    total_debt = mortgage + other_debt
    education_need = college_fund * dependents
    dime_total = total_debt + income_need + education_need - savings - existing_insurance
    dime_total = max(0, dime_total)

    # 10x Income
    ten_x = annual_income * 10

    # Human Life Value (PV of future earnings)
    years_to_retire = retirement_age - age
    hlv = 0
    discount_rate = 0.05
    for y in range(years_to_retire):
        hlv += annual_income * (1.03 ** y) / (1 + discount_rate) ** y
    hlv -= savings

    st.markdown("---")
    st.markdown("### 📊 Coverage Estimates (3 Methods)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("DIME Method", f"${dime_total:,.0f}")
        st.caption("Debt + Income + Mortgage + Education")
    with col2:
        st.metric("10× Income Rule", f"${ten_x:,.0f}")
        st.caption("Simple rule of thumb")
    with col3:
        st.metric("Human Life Value", f"${hlv:,.0f}")
        st.caption("PV of future earnings")

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["DIME", "10× Income", "Human Life Value"],
        y=[dime_total, ten_x, hlv],
        marker_color=["#3b82f6", "#16a34a", "#f59e0b"],
    ))
    fig.update_layout(title="Insurance Coverage Need — Three Methods",
                       yaxis_title="$", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # DIME breakdown
    st.markdown("### 📋 DIME Breakdown")
    dime_df = pd.DataFrame({
        "Component": ["Debt (Mortgage + Other)", "Income Replacement",
                       "Education Fund", "Less: Savings", "Less: Existing Insurance"],
        "Amount ($)": [total_debt, income_need, education_need, -savings, -existing_insurance],
    })
    st.dataframe(dime_df, use_container_width=True, hide_index=True)

    # Recommended coverage
    recommended = max(dime_total, ten_x * 0.8)
    st.success(f"💡 **Recommended coverage**: ${recommended:,.0f}")

    # Term vs Whole Life cost comparison
    st.markdown("### 💰 Term vs Whole Life Cost Comparison")
    col1, col2 = st.columns(2)
    with col1:
        term_years = st.selectbox("Term Length", [10, 15, 20, 30], index=2)
        # Approximate monthly cost per $100k of coverage
        age_multiplier = 1 + (age - 30) * 0.08 if age > 30 else 1
        term_monthly_per_100k = 8 * age_multiplier  # rough estimate
        term_total = (recommended / 100000) * term_monthly_per_100k
        st.metric("Term Life Monthly Cost", f"${term_total:.0f}/mo")
        st.metric("Term Life Annual Cost", f"${term_total*12:.0f}/yr")

    with col2:
        whole_monthly_per_100k = 50 * age_multiplier
        whole_total = (recommended / 100000) * whole_monthly_per_100k
        st.metric("Whole Life Monthly Cost", f"${whole_total:.0f}/mo")
        st.metric("Whole Life Annual Cost", f"${whole_total*12:.0f}/yr")

    st.info(f"""
    **💡 Recommendation**: Buy **term life** for ${recommended:,.0f} coverage
    over {term_years} years, and invest the difference
    (${whole_total*12 - term_total*12:,.0f}/yr saved vs whole life).

    Over {term_years} years, that's
    ${(whole_total*12 - term_total*12) * term_years:,.0f} saved —
    invested at 7% → ${((whole_total*12 - term_total*12) * (((1.07**term_years) - 1) / 0.07)):,.0f}
    """)

    st.markdown("""
    **Discussion Questions:**
    1. Which method gives you the highest coverage need? Why?
    2. Do you need life insurance if you have no dependents?
    3. Why do most financial advisors recommend term over whole life?
    """)
