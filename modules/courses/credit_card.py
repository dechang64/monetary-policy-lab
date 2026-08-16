"""Credit Card Payoff — true cost of minimum payments."""
import streamlit as st
import plotly.graph_objects as go
from utils.pf_calculators import credit_card_payoff
from utils.pf_constants import CC_APR_RANGE

def render():
    st.markdown('<div class="main-header"><h1>💳 Credit Card Payoff</h1>'
                '<p>See the real cost of carrying a balance</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    Credit cards typically charge **15-29% APR**. If you only pay the minimum,
    a $5,000 balance can take **15+ years** to pay off and cost **2x** the original amount.

    **Key terms:**
    - **APR** = Annual Percentage Rate (interest rate)
    - **Minimum payment** = usually 1-3% of balance or $25, whichever is higher
    - **Principal** = original amount borrowed
    """)

    st.markdown("### 🎛️ Your Credit Card")
    col1, col2, col3 = st.columns(3)
    with col1:
        balance = st.number_input("Current Balance ($)", 100, 100000, 5000, step=500)
    with col2:
        apr = st.slider("APR (%)", 5.0, 35.0, 22.0, 0.5) / 100
    with col3:
        monthly_payment = st.number_input("Monthly Payment ($)", 10, 10000, 200, step=50)

    # Minimum payment reference
    min_payment = max(25, balance * 0.02)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Typical Minimum Payment", f"${min_payment:.0f}/mo")
    with col2:
        st.metric("Your Payment vs Minimum",
                  f"${monthly_payment:.0f} vs ${min_payment:.0f}",
                  delta="Above minimum ✅" if monthly_payment > min_payment else "Below minimum ⚠️")

    result = credit_card_payoff(balance, apr, monthly_payment)

    if "error" in result:
        st.error(result["error"])
        if "min_payment_needed" in result:
            st.info(f"Minimum payment needed: **${result['min_payment_needed']}/mo**")
    else:
        st.markdown("---")
        st.markdown("### 📊 Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Time to Pay Off", f"{result['years_to_payoff']} years")
        col2.metric("Total Interest Paid", f"${result['total_interest_paid']:,.0f}")
        col3.metric("Total Paid", f"${result['total_paid']:,.0f}")
        col4.metric("Interest as % of Balance", f"{result['interest_pct_of_balance']:.1f}%")

        # Comparison chart
        st.markdown("### 📈 Payoff Comparison")
        scenarios = {
            f"${monthly_payment}/mo": monthly_payment,
            f"${min_payment:.0f}/mo (minimum)": min_payment,
            f"${monthly_payment + 100}/mo (+$100)": monthly_payment + 100,
            f"${monthly_payment * 2}/mo (double)": monthly_payment * 2,
        }

        names = []
        times = []
        interests = []
        for name, pmt in scenarios.items():
            r = credit_card_payoff(balance, apr, pmt)
            if "error" not in r:
                names.append(name)
                times.append(r["years_to_payoff"])
                interests.append(r["total_interest_paid"])

        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=times, name="Years to Pay Off",
                              marker_color="#3b82f6", yaxis="y"))
        fig.add_trace(go.Bar(x=names, y=interests, name="Total Interest",
                              marker_color="#dc2626", yaxis="y2"))
        fig.update_layout(
            title="Impact of Payment Amount",
            yaxis=dict(title="Years to Pay Off"),
            yaxis2=dict(title="Total Interest ($)", overlaying="y", side="right"),
            barmode="group", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Warning
        if result['interest_pct_of_balance'] > 50:
            st.warning(f"⚠️ You'll pay {result['interest_pct_of_balance']:.0f}% of your balance in interest alone!")
        if result['years_to_payoff'] > 5:
            st.error(f"🚨 At this rate, it'll take {result['years_to_payoff']:.1f} years to pay off.")

    st.markdown("""
    **Discussion Questions:**
    1. What happens if you double your monthly payment? How much interest do you save?
    2. Why is paying only the minimum so dangerous?
    3. If you have $10,000 in credit card debt at 22% APR and $10,000 in savings earning 4%,
       should you use savings to pay off the card?
    """)
