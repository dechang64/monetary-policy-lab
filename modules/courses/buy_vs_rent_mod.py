"""Buy vs Rent — NPV comparison."""
import streamlit as st
import plotly.graph_objects as go
from utils.pf_calculators import buy_vs_rent

def render():
    st.markdown('<div class="main-header"><h1>🏠 Buy vs Rent</h1>'
                '<p>The biggest financial decision — should you buy or rent?</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    **Buying a home** isn't always better than renting. The decision depends on:
    - Home price appreciation vs. investment returns
    - Transaction costs (closing, maintenance, property tax)
    - How long you'll stay (longer = better for buying)

    The **opportunity cost** of a down payment is what you could earn investing it.
    """)

    st.markdown("### 🏡 Property Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_price = st.number_input("Home Price ($)", 50000, 5000000, 500000, step=10000)
    with col2:
        down_pct = st.slider("Down Payment (%)", 0, 50, 20) / 100
    with col3:
        mortgage_rate = st.slider("Mortgage Rate (%)", 1.0, 10.0, 6.5, 0.1) / 100

    col1, col2, col3 = st.columns(3)
    with col1:
        mortgage_years = st.selectbox("Loan Term", [15, 20, 30], index=2)
    with col2:
        property_tax_rate = st.slider("Property Tax (%/yr)", 0.0, 3.0, 1.2, 0.1) / 100
    with col3:
        maintenance_rate = st.slider("Maintenance (%/yr)", 0.0, 3.0, 1.0, 0.1) / 100

    st.markdown("### 💰 Rent & Market Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        rent_monthly = st.number_input("Monthly Rent ($)", 500, 10000, 2500, step=100)
    with col2:
        appreciation = st.slider("Home Appreciation (%/yr)", -2.0, 10.0, 3.0, 0.5) / 100
    with col3:
        opportunity_return = st.slider("Alt Investment Return (%/yr)", 0.0, 15.0, 7.0, 0.5) / 100

    hold_years = st.slider("Hold Period (years)", 1, 30, 7)

    result = buy_vs_rent(home_price, down_pct, mortgage_rate, mortgage_years,
                          property_tax_rate, maintenance_rate, rent_monthly,
                          appreciation, hold_years, opportunity_return)

    st.markdown("---")
    st.markdown("### 📊 Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Down Payment", f"${result['down_payment']:,.0f}")
        st.metric("Monthly Mortgage", f"${result['monthly_mortgage']:,.0f}")
    with col2:
        st.metric("Home Value at Sale", f"${result['home_value_at_sale']:,.0f}")
        st.metric("Loan Balance at Sale", f"${result['loan_balance_at_sale']:,.0f}")
    with col3:
        st.metric("Buy Equity at Sale", f"${result['buy_equity']:,.0f}")
        st.metric("Buy Net Gain", f"${result['buy_net_gain']:,.0f}")

    st.markdown("### 🏠 Renting + Investing the Difference")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Invested Value (down payment + savings)",
                  f"${result['invested_value']:,.0f}")
    with col2:
        st.metric("Rent Net Gain", f"${result['rent_net_gain']:,.0f}")

    # Verdict
    st.markdown("---")
    if result['advantage'] == 'Buy':
        st.success(f"✅ **Buying wins** by ${result['advantage_amount']:,.0f} over {hold_years} years.")
    else:
        st.success(f"✅ **Renting wins** by ${result['advantage_amount']:,.0f} over {hold_years} years.")

    # Break-even analysis
    st.markdown("### 📈 Break-Even Analysis")
    years = list(range(1, 31))
    buy_gains = []
    rent_gains = []
    for y in years:
        r = buy_vs_rent(home_price, down_pct, mortgage_rate, mortgage_years,
                         property_tax_rate, maintenance_rate, rent_monthly,
                         appreciation, y, opportunity_return)
        buy_gains.append(r['buy_net_gain'])
        rent_gains.append(r['rent_net_gain'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=buy_gains, name="Buy Net Gain",
                              line=dict(color="#3b82f6", width=3)))
    fig.add_trace(go.Scatter(x=years, y=rent_gains, name="Rent Net Gain",
                              line=dict(color="#16a34a", width=3)))
    fig.add_trace(go.Scatter(x=years, y=[b-r for b,r in zip(buy_gains, rent_gains)],
                              name="Buy Advantage", line=dict(color="#f59e0b", dash="dash")))
    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    fig.update_layout(title="Net Gain Over Time — When Does Buying Overtake Renting?",
                       xaxis_title="Years Held", yaxis_title="$",
                       height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Discussion Questions:**
    1. At what year does buying become better than renting? What if the home doesn't appreciate?
    2. How does the mortgage rate change the break-even point?
    3. What non-financial factors should influence this decision?
    """)
