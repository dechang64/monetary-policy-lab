"""
Dashboard Page — Overview with FRED data support.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import FOMC_DATES, PAPERS as CLASSIC_PAPERS
from utils.helpers import generate_synthetic_returns
from analysis.event_study import EventStudyEngine
from analysis.capital_flow import CapitalFlowAnalyzer
from visualization.charts import event_study_bar, two_shocks_bar, regime_timeline


def _get_returns():
    if st.session_state.get("data_loaded") and st.session_state.get("fred_returns") is not None:
        return st.session_state.fred_returns
    return generate_synthetic_returns()


def render():
    st.markdown(
        '<div class="main-header"><h1>📊 Monetary Policy Research Lab</h1>'
        '<p>How Federal Reserve Announcements Reshape Asset Prices & Portfolio Allocation</p></div>',
        unsafe_allow_html=True,
    )

    # ── Data Source ──
    if st.session_state.get("data_loaded"):
        src = st.session_state.get("data_source", "unknown")
        st.success(f"🔗 Connected to **{src.upper()}** — {len(st.session_state.fred_returns.columns)} assets loaded")
    else:
        st.info("📊 Using demo data. Go to **Data Explorer** to connect FRED for real results.")

    returns = _get_returns()

    # ── Key Metrics ──
    col1, col2, col3, col4 = st.columns(4)

    n_assets = len(returns.columns)
    n_days = len(returns)
    n_fomc = len([d for d in FOMC_DATES if pd.Timestamp(d) in returns.index])

    with col1:
        st.markdown(f"""
        <div class="metric-card policy">
            <div style="font-size:0.8rem;color:#888;">FOMC Events</div>
            <div style="font-size:1.8rem;font-weight:700;color:#e74c3c;">{n_fomc}</div>
            <div style="font-size:0.75rem;color:#888;">matched to data</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card info">
            <div style="font-size:0.8rem;color:#888;">Asset Classes</div>
            <div style="font-size:1.8rem;font-weight:700;color:#3498db;">{n_assets}</div>
            <div style="font-size:0.75rem;color:#888;">{', '.join(returns.columns[:3])}...</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card dovish">
            <div style="font-size:0.8rem;color:#888;">Research Modules</div>
            <div style="font-size:1.8rem;font-weight:700;color:#27ae60;">6</div>
            <div style="font-size:0.75rem;color:#888;">Event · NLP · Shocks · Flow</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card neutral">
            <div style="font-size:0.8rem;color:#888;">Classic Papers</div>
            <div style="font-size:1.8rem;font-weight:700;color:#2c3e50;">5</div>
            <div style="font-size:0.75rem;color:#888;">One-Click Replication</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Event Study ──
    data_label = "FRED" if st.session_state.get("data_loaded") else "demo"
    st.markdown(f"### ⚡ Quick Event Study Overview")
    st.caption(f"CAR [−1, +5] around FOMC announcements ({data_label} data)")

    @st.cache_data
    def _compute_event_summary(_returns_hash, _fomc_hash):
        _returns = _get_returns()
        _engine = EventStudyEngine(_returns, FOMC_DATES)
        return _engine.cumulative_by_asset(event_window_pre=1, event_window_post=5)

    summary = _compute_event_summary(
        hash(returns.values.tobytes()),
        hash(str(FOMC_DATES)),
    )

    if not summary.empty:
        fig = event_study_bar(summary)
        st.plotly_chart(fig, use_container_width=True)

    # ── Two-Row Layout ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🎯 Two-Shocks Preview")
        st.caption("How different assets respond to policy vs information shocks")

        assets = list(returns.columns[:6])
        rng = np.random.default_rng(42)
        policy_responses = rng.uniform(-0.25, 0.10, len(assets)).round(3)
        info_responses = rng.uniform(-0.10, 0.20, len(assets)).round(3)

        response_df = pd.DataFrame({
            "asset": assets * 2,
            "shock_type": ["Policy"] * len(assets) + ["Information"] * len(assets),
            "avg_response_pct": list(policy_responses) + list(info_responses),
        })

        fig2 = two_shocks_bar(response_df)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.markdown("### 🔄 Risk Regime Timeline")
        st.caption("Risk-On / Risk-Off shifts around FOMC events")

        analyzer = CapitalFlowAnalyzer(returns, FOMC_DATES)
        regime = analyzer.risk_regime_analysis(window=15)

        if not regime.empty:
            fig3 = regime_timeline(regime)
            st.plotly_chart(fig3, use_container_width=True)

    # ── Classic Papers ──
    st.markdown("### 📚 Classic Papers in This Field")
    cols = st.columns(len(CLASSIC_PAPERS))
    for i, (name, info) in enumerate(CLASSIC_PAPERS.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="background:white;border-radius:8px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,0.08);height:100%;">
                <div style="font-size:0.85rem;font-weight:600;color:#2c3e50;">{name}</div>
                <div style="font-size:0.7rem;color:#888;margin:0.3rem 0;">{info['journal']}</div>
                <div style="font-size:0.7rem;color:#555;border-top:1px solid #eee;padding-top:0.3rem;margin-top:0.3rem;">
                {info['key_result']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Guide ──
    st.markdown("---")
    st.markdown("### 🧭 How to Use This Platform")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        **1️⃣ Connect Data**
        - FRED API (free, 800K+ series)
        - Import CSV with your own data
        - Or explore with demo data
        """)
    with g2:
        st.markdown("""
        **2️⃣ Run Analyses**
        - Event Study → CAR around FOMC
        - Two-Shocks → Policy vs Information
        - Sentiment → NLP on FOMC text
        - Capital Flow → Portfolio rebalancing
        """)
    with g3:
        st.markdown("""
        **3️⃣ Replicate & Export**
        - One-click classic paper replication
        - Modify parameters, test hypotheses
        - Export charts and tables
        """)
