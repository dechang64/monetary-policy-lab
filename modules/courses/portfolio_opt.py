"""Portfolio Optimization — Markowitz efficient frontier."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.pf_calculators import portfolio_optimization
from utils.pf_constants import ASSET_EXPECTED_RETURNS, ASSET_VOLS

def render():
    st.markdown('<div class="main-header"><h1>📈 Portfolio Optimization</h1>'
                '<p>Markowitz Efficient Frontier — maximize return, minimize risk</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💡 Concept")
    st.info("""
    **Modern Portfolio Theory** (Markowitz, 1952, Nobel Prize 1990):

    - Every asset has a **risk-return trade-off**
    - A **portfolio** can achieve higher return for the same risk by diversifying
    - The **efficient frontier** = set of optimal portfolios for each risk level

    Key insight: **diversification reduces risk** without sacrificing return.
    """)

    st.markdown("### 🎯 Choose Your Asset Universe")
    all_assets = list(ASSET_EXPECTED_RETURNS.keys())
    selected = st.multiselect("Select Assets", all_assets,
                               default=["US Large Cap (S&P 500)", "US Small Cap",
                                        "US Bonds", "Real Estate (REITs)", "Gold"])

    if len(selected) < 2:
        st.warning("Please select at least 2 assets.")
        return

    # Build inputs
    expected_returns = np.array([ASSET_EXPECTED_RETURNS[a] for a in selected])
    vols = np.array([ASSET_VOLS[a] for a in selected])

    # Simplified correlation matrix
    st.markdown("### 🔗 Asset Correlations (Simplified)")
    st.markdown("In production: use historical return correlations. Here we use a simplified correlation of 0.3 between different asset classes.")
    corr = 0.3
    n = len(selected)
    corr_matrix = np.eye(n) * 1.0 + (1 - np.eye(n)) * corr
    cov_matrix = np.outer(vols, vols) * corr_matrix

    # Display selected assets
    asset_df = st.dataframe(
        pd.DataFrame({
            "Asset": selected,
            "Expected Return (%)": [r*100 for r in expected_returns],
            "Volatility (%)": [v*100 for v in vols],
        }).set_index("Asset"),
        use_container_width=True,
    )

    n_portfolios = st.slider("Number of Portfolios to Simulate", 1000, 20000, 10000, step=1000)
    risk_free = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.1) / 100

    if st.button("🚀 Optimize", type="primary"):
        with st.spinner(f"Generating {n_portfolios} portfolios..."):
            result = portfolio_optimization(selected, expected_returns, cov_matrix,
                                              n_portfolios, risk_free)

        # Efficient frontier
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result['all_vols']*100, y=result['all_returns']*100,
            mode='markers', name='Random Portfolios',
            marker=dict(color=result['all_sharpe'], colorscale='Viridis', size=4,
                         showscale=True, colorbar=dict(title="Sharpe")),
        ))
        # Max Sharpe
        fig.add_trace(go.Scatter(
            x=[result['max_sharpe_vol']*100], y=[result['max_sharpe_return']*100],
            mode='markers+text', name='Max Sharpe (Tangency)',
            marker=dict(size=15, color='red', symbol='star'),
            text=["Max Sharpe"], textposition="top center",
        ))
        # Min Vol
        fig.add_trace(go.Scatter(
            x=[result['min_vol_vol']*100], y=[result['min_vol_return']*100],
            mode='markers+text', name='Min Volatility',
            marker=dict(size=15, color='blue', symbol='diamond'),
            text=["Min Vol"], textposition="top center",
        ))
        # Capital Market Line
        max_ret = max(result['all_returns'])
        cml_x = [0, max(result['all_vols'])]
        cml_y = [risk_free*100, risk_free*100 + (result['max_sharpe_ratio'])*(cml_x[1])*100]
        fig.add_trace(go.Scatter(x=cml_x, y=cml_y, name='Capital Market Line',
                                  line=dict(dash="dash", color="orange")))
        fig.update_layout(title="Efficient Frontier",
                           xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
                           height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Optimal portfolios
        st.markdown("### 📋 Optimal Portfolio Weights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⭐ Maximum Sharpe Ratio (Tangency Portfolio)")
            sharpe_df = pd.DataFrame({
                "Asset": selected,
                "Weight (%)": [w*100 for w in result['max_sharpe_weights']],
            }).set_index("Asset")
            st.dataframe(sharpe_df.style.format("{:.1f}%"), use_container_width=True)
            st.metric("Expected Return", f"{result['max_sharpe_return']*100:.2f}%")
            st.metric("Volatility", f"{result['max_sharpe_vol']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{result['max_sharpe_ratio']:.3f}")

        with col2:
            st.markdown("#### 💎 Minimum Volatility Portfolio")
            minvol_df = pd.DataFrame({
                "Asset": selected,
                "Weight (%)": [w*100 for w in result['min_vol_weights']],
            }).set_index("Asset")
            st.dataframe(minvol_df.style.format("{:.1f}%"), use_container_width=True)
            st.metric("Expected Return", f"{result['min_vol_return']*100:.2f}%")
            st.metric("Volatility", f"{result['min_vol_vol']*100:.2f}%")

    st.markdown("""
    **Discussion Questions:**
    1. Why does the Max Sharpe portfolio include bonds even though stocks have higher returns?
    2. What happens to the frontier if you add a low-correlation asset like Gold?
    3. How does the Capital Market Line relate to the risk-free rate?
    """)
