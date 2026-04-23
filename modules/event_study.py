"""
Event Study Engine Page — supports both FRED real data and demo data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import FOMC_DATES
from utils.helpers import generate_synthetic_returns
from analysis.event_study import EventStudyEngine
from visualization.charts import event_study_bar, event_study_timeline


def _get_returns():
    """Get returns data — FRED if loaded, otherwise demo."""
    if st.session_state.get("data_loaded") and st.session_state.get("fred_returns") is not None:
        return st.session_state.fred_returns
    return generate_synthetic_returns()


def render():
    st.markdown(
        '<div class="main-header"><h1>⚡ Event Study Engine</h1>'
        '<p>Compute abnormal returns around FOMC announcements</p></div>',
        unsafe_allow_html=True,
    )

    # ── Data Source Badge ──
    if st.session_state.get("data_loaded"):
        src = st.session_state.get("data_source", "unknown")
        st.info(f"🔗 Using **{src.upper()}** data. {len(st.session_state.fred_returns.columns)} assets, {len(st.session_state.fred_returns)} days.")
    else:
        st.caption("📊 Using demo data. Connect FRED in Data Explorer for real results.")

    # ── Parameters ──
    st.markdown("### Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        est_window = st.slider("Estimation Window (trading days)", 50, 500, 250)
    with col2:
        pre_window = st.slider("Pre-Event Window (days)", 0, 10, 1)
    with col3:
        post_window = st.slider("Post-Event Window (days)", 1, 20, 5)

    returns = _get_returns()

    # ── Run Event Study ──
    if st.button("🚀 Run Event Study", type="primary", width='stretch'):
        with st.spinner("Computing abnormal returns..."):
            engine = EventStudyEngine(returns, FOMC_DATES)
            summary = engine.cumulative_by_asset(
                event_window_pre=pre_window,
                event_window_post=post_window,
            )

            if summary.empty:
                st.warning("Insufficient data. Try a shorter estimation window.")
                return

            st.markdown("### Results: Average CAR by Asset Class")
            st.caption(f"Event window: [{-pre_window}, +{post_window}] | Estimation: {est_window} days")

            fig = event_study_bar(summary)
            st.plotly_chart(fig, width='stretch')

            st.markdown("### Detailed Statistics")
            display_cols = ["asset", "avg_CAR_pct", "median_CAR", "std_CAR", "avg_t_stat", "n_events", "pct_positive"]
            display_df = summary[display_cols].copy()
            display_df.columns = ["Asset", "Avg CAR (%)", "Median CAR", "Std CAR", "Avg t-stat", "N Events", "% Positive"]
            display_df = display_df.round(4)
            st.dataframe(display_df, width='stretch', hide_index=True)

            st.markdown("### 🔍 Key Findings")
            most_positive = summary.loc[summary["avg_CAR_pct"].idxmax()]
            most_negative = summary.loc[summary["avg_CAR_pct"].idxmin()]
            most_significant = summary.loc[summary["avg_t_stat"].abs().idxmax()]

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Largest Positive CAR", f"{most_positive['avg_CAR_pct']:+.3f}%", most_positive["asset"])
            with col_b:
                st.metric("Largest Negative CAR", f"{most_negative['avg_CAR_pct']:+.3f}%", most_negative["asset"])
            with col_c:
                st.metric("Most Significant", f"|t| = {abs(most_significant['avg_t_stat']):.2f}", most_significant["asset"])

    # ── Individual Asset Timeline ──
    st.markdown("---")
    st.markdown("### 📈 Individual Asset Timeline")
    selected_asset = st.selectbox("Select Asset", returns.columns.tolist())

    if st.button("Show Timeline", width='stretch'):
        with st.spinner("Computing..."):
            engine = EventStudyEngine(returns, FOMC_DATES)
            market_candidates = [c for c in returns.columns if c != selected_asset]
            market = market_candidates[0] if market_candidates else selected_asset
            results = engine.market_model(
                selected_asset, market,
                estimation_window=est_window,
                event_window_pre=pre_window,
                event_window_post=post_window,
            )
            if not results.empty:
                fig = event_study_timeline(results, selected_asset)
                st.plotly_chart(fig, width='stretch')

    with st.expander("📖 Methodology"):
        st.markdown("""
        **Market Model Event Study** (Brown & Warner, 1985)

        1. **Estimation**: $R_{it} = \\alpha_i + \\beta_i R_{mt} + \\epsilon_{it}$
        2. **Abnormal Return**: $AR_{it} = R_{it} - (\\hat{\\alpha}_i + \\hat{\\beta}_i R_{mt})$
        3. **CAR**: $CAR_i = \\sum_{t=-T_1}^{T_2} AR_{it}$
        4. **t-stat**: $t = CAR_i / (\\sigma_{AR} \\sqrt{N})$

        **Reference**: Kuttner (2001), Bernanke & Kuttner (2005)
        """)
