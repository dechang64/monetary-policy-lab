"""FICO Score Simulator — interactive credit score factors."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.pf_calculators import estimate_fico
from utils.pf_constants import FICO_FACTORS

def render():
    st.markdown('<div class="main-header"><h1>🏆 FICO Score Simulator</h1>'
                '<p>Understand what drives your credit score</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    Your **FICO score** (300-850) determines whether you get approved for loans,
    credit cards, and even apartments. It's calculated from 5 factors:

    | Factor | Weight |
    |--------|--------|
    | Payment History | 35% |
    | Credit Utilization | 30% |
    | Length of Credit History | 15% |
    | Credit Mix | 10% |
    | New Credit Inquiries | 10% |
    """)

    st.markdown("### 🎛️ Simulate Your Credit Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Payment History (35%)")
        payment_score = st.slider(
            "How good is your payment history? (0-100)",
            0, 100, 90,
            help="100 = never missed a payment. 50 = several late payments. 0 = collections/defaults."
        )
        ph_score = 300 + payment_score * 5.5  # map 0-100 → 300-850

        st.markdown("#### Credit Utilization (30%)")
        utilization = st.slider(
            "Current Credit Utilization (%)",
            0, 100, 20,
            help="Balance / Total Credit Limit. Below 10% is best, above 50% hurts."
        )

    with col2:
        st.markdown("#### Length of Credit History (15%)")
        history_years = st.slider(
            "Years of Credit History",
            0, 30, 5,
            help="Longer is better. 7+ years is good, 10+ is excellent."
        )

        st.markdown("#### Credit Mix (10%)")
        mix_score = st.slider(
            "Credit Mix Diversity (0-100)",
            0, 100, 60,
            help="Having both revolving (credit cards) and installment (loans) credit helps."
        )

        st.markdown("#### New Inquiries (10%)")
        new_inquiries = st.slider(
            "Recent Hard Inquiries (last 12 months)",
            0, 15, 2,
            help="Each hard inquiry can drop your score 5-10 points. 6+ in a year signals risk."
        )

    # Calculate
    result = estimate_fico(ph_score, utilization, history_years, mix_score, new_inquiries)

    st.markdown("---")
    st.markdown("### 📊 Your Estimated FICO Score")

    # Big score display
    score = result["estimated_score"]
    rating = result["rating"]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        color = "#16a34a" if score >= 740 else "#3b82f6" if score >= 670 else "#f59e0b" if score >= 580 else "#dc2626"
        st.markdown(f"""
        <div style="text-align:center; padding:2rem; background:{color}; border-radius:12px;">
            <h1 style="color:white; font-size:4rem; margin:0;">{score}</h1>
            <h3 style="color:white; margin:0;">{rating}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "FICO Score"},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': color},
            'steps': [
                {'range': [300, 580], 'color': "#fef2f2"},
                {'range': [580, 670], 'color': "#fffbeb"},
                {'range': [670, 740], 'color': "#eff6ff"},
                {'range': [740, 850], 'color': "#f0fdf4"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score,
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown
    st.markdown("### 📋 Score Factor Breakdown")
    for factor, value in result["components"].items():
        with st.expander(f"{factor}: {value}"):
            weight = float(factor.split("(")[1].split("%)")[0]) / 100
            fig = go.Figure(go.Bar(
                x=[value], orientation='h',
                marker_color="#667eea",
            ))
            fig.update_layout(xaxis_range=[300, 850], height=150,
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # What-if analysis
    st.markdown("### 🔍 What-If Analysis")
    st.markdown("See how changes affect your score:")

    scenarios = [
        ("Current", ph_score, utilization, history_years, mix_score, new_inquiries),
        ("Miss 1 payment", max(0, ph_score - 15), utilization, history_years, mix_score, new_inquiries),
        ("Max out cards", ph_score, 90, history_years, mix_score, new_inquiries),
        ("Pay down cards to 5%", ph_score, 5, history_years, mix_score, new_inquiries),
        ("Apply for 3 new cards", ph_score, utilization, history_years, mix_score, new_inquiries + 3),
        ("History +5 years", ph_score, utilization, history_years + 5, mix_score, new_inquiries),
    ]
    rows = []
    for name, ph, u, h, m, ni in scenarios:
        r = estimate_fico(ph, u, h, m, ni)
        rows.append({"Scenario": name, "Estimated Score": r["estimated_score"],
                      "Rating": r["rating"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
    **Discussion Questions:**
    1. Which factor has the biggest impact on your score? Why?
    2. If you could only improve one thing, what would it be?
    3. Why does closing an old credit card hurt your score?
    """)
