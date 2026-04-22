"""
Two-Shocks Decomposition Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.constants import FOMC_DATES
from utils.helpers import generate_synthetic_returns
from analysis.two_shocks import TwoShocksDecomposer
from visualization.charts import two_shocks_radar, two_shocks_bar, correlation_heatmap


def render():
    st.markdown('<div class="main-header"><h1>🎯 Two-Shocks Decomposition</h1><p>Policy Shock vs Information Shock — following Jarociński & Karadi (2020)</p></div>', unsafe_allow_html=True)
    
    # ── Theory Box ──
    with st.expander("📖 Theoretical Background", expanded=True):
        st.markdown("""
        **The Two-Shocks Framework** (Jarociński & Karadi, 2020)
        
        FOMC announcements contain **two distinct types of information**:
        
        | Shock Type | What It Is | Expected Effect |
        |-----------|-----------|----------------|
        | **🔴 Policy Shock** | Unexpected change in the monetary policy stance | Rates ↑, Equity ↓ (opposite directions) |
        | **🔵 Information Shock** | Central bank reveals information about economic fundamentals | Rates & Equity move in SAME direction |
        
        **Why This Matters:**
        - Traditional event studies treat all surprises as one type → miss heterogeneity
        - Information shocks can explain the "equity premium puzzle" (equities rise on tightening)
        - Different asset classes respond differently to each shock type
        
        **Identification Strategy:**
        - Policy shock: moves equity and rates in opposite directions
        - Information shock: moves equity and rates in the same direction
        - Decompose: use equity and rate responses as a system
        
        **Production Implementation:** SVAR with sign restrictions or high-frequency identification.
        """)
    
    # ── Parameters ──
    col1, col2 = st.columns(2)
    with col1:
        equity_asset = st.selectbox("Equity Proxy", ["S&P 500", "NASDAQ", "Russell 2000"])
    with col2:
        bond_asset = st.selectbox("Bond Proxy", ["US 2Y Treasury", "US 10Y Treasury", "US 30Y Treasury"])
    
    # ── Load Data ──
    @st.cache_data
    def load_data():
        returns = generate_synthetic_returns()
        # Generate synthetic surprises
        np.random.seed(42)
        fomc_dates = [pd.Timestamp(d) for d in FOMC_DATES if pd.Timestamp(d) in returns.index]
        surprises = pd.DataFrame({
            "surprise": np.random.normal(0, 0.05, len(fomc_dates)),
        }, index=fomc_dates)
        return returns, surprises
    
    import pandas as pd
    returns, surprises = load_data()
    
    # ── Run Decomposition ──
    if st.button("🎯 Decompose Shocks", type="primary", use_container_width=True):
        with st.spinner("Decomposing monetary policy surprises..."):
            decomposer = TwoShocksDecomposer(surprises, returns)
            decomposition = decomposer.simplified_decompose(
                equity_col=equity_asset,
                bond_col=bond_asset,
            )
            
            if decomposition.empty:
                st.warning("Insufficient data for decomposition.")
                return
            
            # ── Summary Stats ──
            st.markdown("### Decomposition Summary")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                policy_pct = (decomposition["dominant"] == "Policy").mean() * 100
                st.metric("Policy-Dominant Events", f"{policy_pct:.0f}%", "🔴")
            with col_b:
                info_pct = (decomposition["dominant"] == "Information").mean() * 100
                st.metric("Information-Dominant Events", f"{info_pct:.0f}%", "🔵")
            with col_c:
                avg_policy = decomposition["policy_pct"].mean() * 100
                st.metric("Avg Policy Share", f"{avg_policy:.0f}%")
            
            # ── Radar Chart ──
            st.markdown("### 🕸️ Two-Shocks Radar")
            st.caption("How each asset class responds to policy vs information shocks")
            
            # Generate response data
            assets = ["S&P 500", "NASDAQ", "US 10Y", "MSCI EM", "Gold", "DXY (USD)"]
            np.random.seed(123)
            policy_r = np.random.uniform(-0.3, 0.15, len(assets))
            info_r = np.random.uniform(-0.1, 0.3, len(assets))
            
            response_df = pd.DataFrame({
                "asset": assets * 2,
                "shock_type": ["Policy"] * len(assets) + ["Information"] * len(assets),
                "avg_response_pct": list(policy_r) + list(info_r),
            })
            
            fig_radar = two_shocks_radar(response_df)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # ── Bar Chart ──
            fig_bar = two_shocks_bar(response_df)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # ── Shock Correlation Matrix ──
            st.markdown("### 📊 Shock-Asset Correlation Matrix")
            corr = decomposer.shock_correlation_matrix(decomposition)
            fig_corr = correlation_heatmap(corr)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # ── Temporal Evolution ──
            st.markdown("### 📅 Temporal Evolution")
            yearly = decomposer.temporal_evolution(decomposition)
            st.dataframe(yearly, use_container_width=True, hide_index=True)
    
    # ── Methodology ──
    with st.expander("🔧 Production Implementation Notes"):
        st.markdown("""
        **Current**: Simplified heuristic decomposition (for demo)
        
        **For publication-quality results, implement:**
        
        1. **SVAR with Sign Restrictions** (Uhlig, 2005)
           - Estimate VAR with [equity return, bond yield, policy surprise]
           - Impose sign restrictions: policy shock → equity↓, rates↑
        
        2. **High-Frequency Identification** (Gürkaynak et al., 2005)
           - Use intraday data (30-min window around FOMC)
           - Separate target factor from path factor
        
        3. **External Instruments** (Mertens & Ravn, 2013)
           - Use Fed funds futures as instrument for policy shock
           - Residual captures information shock
        
        **Recommended Python packages:**
        - `statsmodels.tsa.api.VAR` for VAR estimation
        - `linearmodels` for IV estimation
        - Custom SVAR with sign restrictions
        """)
