"""Retirement Planner — Monte Carlo simulation."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.pf_calculators import monte_carlo_retirement

def render():
    st.markdown('<div class="main-header"><h1>👴 Retirement Planner</h1>'
                '<p>Monte Carlo simulation: will you outlive your money?</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    The **4% rule** (Bengen, 1994): withdraw 4% of your portfolio annually in retirement,
    adjusted for inflation, and your savings should last 30+ years.

    But this assumes constant returns. In reality, markets are volatile.
    **Monte Carlo simulation** runs thousands of scenarios with random returns
    to estimate your probability of success.
    """)

    st.markdown("### 🎛️ Your Scenario")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_savings = st.number_input("Current Savings ($)", 0, 10000000, 50000, step=5000)
    with col2:
        annual_contrib = st.number_input("Annual Contribution ($)", 0, 100000, 12000, step=1000)
    with col3:
        years_to_retire = st.slider("Years to Retirement", 1, 40, 25)
    with col4:
        years_in_retirement = st.slider("Years in Retirement", 5, 50, 30)

    col1, col2, col3 = st.columns(3)
    with col1:
        expected_return = st.slider("Expected Annual Return (%)", 0.0, 15.0, 7.0, 0.1) / 100
    with col2:
        volatility = st.slider("Volatility / Std Dev (%)", 0.0, 40.0, 16.0, 0.5) / 100
    with col3:
        withdrawal_rate = st.slider("Withdrawal Rate (%)", 1.0, 10.0, 4.0, 0.1) / 100

    n_sims = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=0)

    if st.button("🎲 Run Simulation", type="primary"):
        with st.spinner(f"Running {n_sims} simulations..."):
            result = monte_carlo_retirement(
                current_savings, annual_contrib, years_to_retire,
                years_in_retirement, expected_return, volatility,
                withdrawal_rate, n_simulations=n_sims
            )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Success Rate", f"{result['success_rate']*100:.1f}%",
                     delta="Safe" if result['success_rate'] > 0.9 else "Risky" if result['success_rate'] > 0.7 else "Danger",
                     delta_color="normal" if result['success_rate'] > 0.9 else "inverse")
        col2.metric("Median Outcome", f"${result['median_outcome']:,.0f}")
        col3.metric("Worst 10%", f"${result['percentile_10']:,.0f}")
        col4.metric("Best 10%", f"${result['percentile_90']:,.0f}")

        # Distribution chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=result['simulations'], nbinsx=50,
                                    marker_color='#667eea', name='Final Balance'))
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                       annotation_text="Bankrupt")
        fig.add_vline(x=result['median_outcome'], line_dash="dash", line_color="green",
                       annotation_text=f"Median: ${result['median_outcome']:,.0f}")
        fig.update_layout(title="Distribution of Retirement Outcomes",
                           xaxis_title="Final Balance ($)",
                           yaxis_title="Count",
                           height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Assessment
        sr = result['success_rate']
        if sr >= 0.95:
            st.success(f"✅ Excellent! {sr*100:.1f}% success rate. Your plan looks robust.")
        elif sr >= 0.85:
            st.info(f"👍 Good. {sr*100:.1f}% success rate. Consider increasing savings or delaying retirement.")
        elif sr >= 0.70:
            st.warning(f"⚠️ Risky. {sr*100:.1f}% success rate. You may run out of money. Reduce withdrawal rate or save more.")
        else:
            st.error(f"🚨 Danger. Only {sr*100:.1f}% success rate. Significant risk of outliving savings.")

    st.markdown("""
    **Discussion Questions:**
    1. What withdrawal rate gives you 95% confidence? Is it livable?
    2. How does increasing savings rate vs. delaying retirement affect success?
    3. What if returns are lower than expected? How sensitive is your plan?
    """)
