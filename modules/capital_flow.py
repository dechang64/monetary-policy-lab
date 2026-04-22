"""
Capital Flow Analysis Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.constants import FOMC_DATES
from utils.helpers import generate_synthetic_returns, generate_portfolio_flows
from analysis.capital_flow import CapitalFlowAnalyzer
from visualization.charts import capital_flow_sankey, regime_timeline


def render():
    st.markdown('<div class="main-header"><h1>🔄 Capital Flow Analysis</h1><p>Portfolio rebalancing patterns around FOMC announcements</p></div>', unsafe_allow_html=True)
    
    # ── Theory ──
    with st.expander("📖 Theoretical Background"):
        st.markdown("""
        **Portfolio Balance Channel** (Tobin, 1969; Brunner & Meltzer, 1993)
        
        When monetary policy changes, investors rebalance across asset classes:
        
        - **Tightening** → Bonds less attractive → Flow to equities? Or risk-off to cash?
        - **Easing** → Search for yield → Flow to riskier assets
        
        **Risk-Taking Channel** (Borio & Zhu, 2012)
        
        Low rates → Lower perceived risk → Higher risk appetite → 
        Rebalance from safe to risky assets
        
        **Key Question**: How does the *composition* of portfolios change,
        not just total flows?
        """)
    
    # ── Load Data ──
    @st.cache_data
    def load_data():
        return generate_synthetic_returns()
    
    returns = load_data()
    
    # ── Parameters ──
    col1, col2 = st.columns(2)
    with col1:
        pre_window = st.slider("Pre-FOMC Window (days)", 5, 60, 30)
    with col2:
        post_window = st.slider("Post-FOMC Window (days)", 5, 60, 30)
    
    # ── Run Analysis ──
    if st.button("🌊 Analyze Capital Flows", type="primary", use_container_width=True):
        with st.spinner("Analyzing portfolio rebalancing patterns..."):
            analyzer = CapitalFlowAnalyzer(returns, FOMC_DATES)
            
            # 1. Risk regime analysis
            st.markdown("### 🏷️ Risk Regime Analysis")
            regime = analyzer.risk_regime_analysis()
            if not regime.empty:
                st.dataframe(regime, use_container_width=True, hide_index=True)
                
                fig = regime_timeline(regime)
                st.plotly_chart(fig, use_container_width=True)
            
            # 2. Portfolio flow data
            st.markdown("### 💰 Fund Flow Changes by Category")
            flows = generate_portfolio_flows(FOMC_DATES)
            
            # Average flow change by category
            avg_flows = flows.groupby("category").agg(
                avg_pre=("flow_pre_b", "mean"),
                avg_post=("flow_post_b", "mean"),
                avg_change=("flow_change_b", "mean"),
                pct_positive=("flow_change_b", lambda x: (x > 0).mean() * 100),
            ).sort_values("avg_change")
            
            st.dataframe(
                avg_flows.style.format("{:.2f}").background_gradient(
                    subset=["avg_change"],
                    cmap="RdYlGn",
                ),
                use_container_width=True,
            )
            
            # 3. Correlation changes
            st.markdown("### 🔗 Cross-Asset Correlation Changes")
            corr_changes = analyzer.correlation_change(
                pre_window=pre_window,
                post_window=post_window,
            )
            if not corr_changes.empty:
                st.dataframe(corr_changes, use_container_width=True, hide_index=True)
                
                avg_corr_change = corr_changes["corr_change"].mean()
                if avg_corr_change > 0:
                    st.info(f"📊 Average correlation **increases** by {avg_corr_change:.4f} post-FOMC → suggests **herding behavior**")
                else:
                    st.info(f"📊 Average correlation **decreases** by {abs(avg_corr_change):.4f} post-FOMC → suggests **diversification**")
    
    # ── Sankey Diagram ──
    st.markdown("---")
    st.markdown("### 🌊 Capital Flow Sankey Diagram")
    st.caption("Visualize how capital flows between asset categories around FOMC events")
    
    if st.button("Generate Sankey", use_container_width=True):
        flows = generate_portfolio_flows(FOMC_DATES)
        avg_flows = flows.groupby("category")["flow_change_b"].mean().sort_values()
        
        fig = capital_flow_sankey(avg_flows)
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Methodology ──
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Data Sources (Production)**:
        - Thomson Reuters Mutual Fund Holdings (via WRDS)
        - CRSP Survivor-Bias-Free US Mutual Fund Database
        - EPFR Global Fund Flows
        
        **Flow Calculation**:
        $$Flow_{i,t} = TNA_{i,t} - TNA_{i,t-1} \\times (1 + R_{i,t})$$
        
        where TNA = Total Net Assets, R = return
        
        **Risk-Safe Spread**:
        $$Spread_t = R_{risk-on,t} - R_{risk-off,t}$$
        
        **Correlation Change**:
        $$\\Delta\\rho = \\bar{\\rho}_{post} - \\bar{\\rho}_{pre}$$
        
        Positive Δρ suggests herding; negative suggests diversification.
        """)
