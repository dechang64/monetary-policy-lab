"""Compound Interest — the magic of compounding over time."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.pf_calculators import compound_interest

def render():
    st.markdown('<div class="main-header"><h1>✨ Compound Interest</h1>'
                '<p>See the most powerful force in finance — Albert Einstein</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    **Compound interest** = earning interest on your interest.

    The formula: $A = P(1 + r)^t$

    Where:
    - $A$ = Final amount
    - $P$ = Principal (initial investment)
    - $r$ = Annual interest rate
    - $t$ = Time in years

    **Key insight**: Time matters more than the amount invested.
    Starting early with small amounts beats starting late with large amounts.
    """)

    st.markdown("### 🎛️ Calculator")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        principal = st.number_input("Initial Investment ($)", 0, 1000000, 10000, step=1000)
    with col2:
        annual_rate = st.slider("Annual Return (%)", 0.0, 20.0, 7.0, 0.1) / 100
    with col3:
        years = st.slider("Time Horizon (years)", 1, 50, 30)
    with col4:
        monthly_contrib = st.number_input("Monthly Contribution ($)", 0, 10000, 500, step=100)

    df = compound_interest(principal, annual_rate, years, monthly_contrib, "monthly")

    # Key metrics
    final = df.iloc[-1]
    total_contrib = final["Contributions"]
    total_interest = final["Balance"] - total_contrib

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Balance", f"${final['Balance']:,.0f}")
    col2.metric("Total Contributed", f"${total_contrib:,.0f}")
    col3.metric("Interest Earned", f"${total_interest:,.0f}")
    col4.metric("Interest % of Final",
                f"{total_interest/final['Balance']*100:.1f}%" if final['Balance'] > 0 else "0%")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Contributions"], name="Contributions",
                              fill='tozeroy', line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Balance"], name="Total Balance",
                              fill='tonexty', line=dict(color="#16a34a")))
    fig.update_layout(title="Growth Over Time",
                       xaxis_title="Years", yaxis_title="$",
                       height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Comparison: Start Early vs Start Late
    st.markdown("### ⚡ The Power of Starting Early")
    st.markdown("""
    **Person A** invests $500/month from age 25 to 35 (10 years, $60k total),
    then stops. **Person B** waits until age 35, then invests $500/month
    from 35 to 65 (30 years, $180k total). Who has more at age 65?
    """)

    early_rate = st.slider("Assumed Annual Return for Comparison (%)", 3.0, 12.0, 7.0, 0.5) / 100

    df_a = compound_interest(0, early_rate, 40, 500, "monthly")
    # Person A: contributes only years 0-10, then just grows
    balance_a = 0
    for y in range(40):
        if y < 10:
            balance_a = balance_a * (1 + early_rate) + 500 * 12
        else:
            balance_a = balance_a * (1 + early_rate)

    # Person B: starts at year 10, contributes 30 years
    balance_b = 0
    for y in range(40):
        if y >= 10:
            balance_b = balance_b * (1 + early_rate) + 500 * 12
        else:
            balance_b = 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Person A (10 yrs × $500/mo, age 25-35)",
                  f"${balance_a:,.0f}",
                  delta=f"Contributed: $60,000")
    with col2:
        st.metric("Person B (30 yrs × $500/mo, age 35-65)",
                  f"${balance_b:,.0f}",
                  delta=f"Contributed: $180,000")

    if balance_a > balance_b:
        st.success(f"🎉 Person A wins by ${balance_a - balance_b:,.0f} — despite contributing only 1/3 as much!")
    else:
        st.warning(f"Person B wins by ${balance_b - balance_a:,.0f}. Try a higher return rate to see the magic of compounding.")

    st.markdown("""
    **Discussion Questions:**
    1. At what return rate does Person A overtake Person B? Why?
    2. How does the gap change if contributions are $1,000/month instead of $500?
    3. What does this tell you about the value of starting to invest early?
    """)
