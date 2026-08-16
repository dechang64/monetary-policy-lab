"""Tax Calculator — federal + state income tax."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.pf_calculators import calculate_federal_tax
from utils.pf_constants import TAX_BRACKETS_SINGLE, STATE_TAX

def render():
    st.markdown('<div class="main-header"><h1>🧾 Tax Calculator</h1>'
                '<p>Understand progressive taxation and take-home pay</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    The US uses **progressive taxation**: different portions of your income are taxed
    at different rates (brackets).

    **Important**: Being in the "22% bracket" does NOT mean you pay 22% on everything.
    You pay 10% on the first portion, 12% on the next, 22% on the portion above that.

    **Marginal rate** = rate on your next dollar earned
    **Effective rate** = actual % of income paid in tax (always lower than marginal)
    """)

    st.markdown("### 🎛️ Your Income")
    col1, col2, col3 = st.columns(3)
    with col1:
        gross_income = st.number_input("Gross Annual Income ($)", 0, 10000000, 75000, step=5000)
    with col2:
        filing_status = st.selectbox("Filing Status", ["Single", "Married Jointly"])
    with col3:
        state = st.selectbox("State Tax", list(STATE_TAX.keys()))

    # Pre-tax deductions
    st.markdown("### 📝 Pre-Tax Deductions (reduce taxable income)")
    col1, col2, col3 = st.columns(3)
    with col1:
        retirement_contrib = st.number_input("401(k) Contribution ($)", 0, 23000, 6000, step=500)
    with col2:
        hsa_contrib = st.number_input("HSA Contribution ($)", 0, 4150, 0, step=500)
    with col3:
        other_deductions = st.number_input("Other Deductions ($)", 0, 50000, 0, step=500)

    # Standard deduction
    std_deduction = 14600 if filing_status == "Single" else 29200
    total_deductions = std_deduction + retirement_contrib + hsa_contrib + other_deductions
    taxable_income = max(0, gross_income - total_deductions)

    st.info(f"**Standard Deduction**: ${std_deduction:,}  |  **Total Deductions**: ${total_deductions:,}  |  **Taxable Income**: ${taxable_income:,}")

    # Calculate federal tax
    fed_result = calculate_federal_tax(taxable_income, TAX_BRACKETS_SINGLE)
    state_rate = STATE_TAX[state]
    state_tax = taxable_income * state_rate
    total_tax = fed_result["federal_tax"] + state_tax
    net_income = gross_income - total_tax

    st.markdown("---")
    st.markdown("### 📊 Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gross Income", f"${gross_income:,.0f}")
    col2.metric("Federal Tax", f"${fed_result['federal_tax']:,.0f}",
                delta=f"{fed_result['effective_rate']:.1f}% effective")
    col3.metric(f"State Tax ({state_rate*100:.1f}%)", f"${state_tax:,.0f}")
    col4.metric("Net Take-Home", f"${net_income:,.0f}",
                delta=f"{(net_income/gross_income)*100:.1f}% of gross" if gross_income > 0 else "")

    # Visualization
    col_left, col_right = st.columns(2)
    with col_left:
        fig = go.Figure(data=[go.Pie(
            labels=["Take-Home", "Federal Tax", "State Tax", "Pre-Tax Deductions"],
            values=[net_income, fed_result["federal_tax"], state_tax,
                    total_deductions - std_deduction],
            hole=.4,
        )])
        fig.update_layout(title="Where Your Income Goes", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Marginal vs effective
        marginal_rate = 0
        for threshold, rate in TAX_BRACKETS_SINGLE:
            if taxable_income > threshold:
                marginal_rate = rate

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Effective Rate", "Marginal Rate"],
            y=[fed_result["effective_rate"], marginal_rate*100],
            marker_color=["#16a34a", "#dc2626"],
        ))
        fig2.update_layout(title="Effective vs Marginal Tax Rate",
                           yaxis_title="%", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Bracket breakdown
    st.markdown("### 📋 Tax Bracket Breakdown")
    bracket_rows = []
    income_remaining = taxable_income
    for i, (threshold, rate) in enumerate(TAX_BRACKETS_SINGLE):
        if i == len(TAX_BRACKETS_SINGLE) - 1:
            if income_remaining > 0:
                taxable = income_remaining
                tax = taxable * rate
                bracket_rows.append({"Bracket": f"${threshold:,}+",
                                      "Rate": f"{rate*100:.0f}%",
                                      "Taxable Amount": f"${taxable:,.0f}",
                                      "Tax": f"${tax:,.0f}"})
                income_remaining = 0
        else:
            next_threshold = TAX_BRACKETS_SINGLE[i+1][0]
            bracket_size = next_threshold - threshold
            if income_remaining > 0:
                taxable = min(income_remaining, bracket_size)
                tax = taxable * rate
                bracket_rows.append({"Bracket": f"${threshold:,}-${next_threshold:,}",
                                      "Rate": f"{rate*100:.0f}%",
                                      "Taxable Amount": f"${taxable:,.0f}",
                                      "Tax": f"${tax:,.0f}"})
                income_remaining -= taxable

    st.dataframe(pd.DataFrame(bracket_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    **Discussion Questions:**
    1. What's the difference between your marginal and effective rate? Why does it matter?
    2. How much does contributing $6,000 to a 401(k) actually save you in taxes?
    3. If you get a $5,000 raise, how much of it do you actually keep?
    """)
