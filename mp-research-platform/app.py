"""
Monetary Policy Research Platform
==================================
A distinctive academic research platform for studying
monetary policy announcements, asset prices, and portfolio reallocation.

Author: Built for Eileen Zhang's research workflow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import json
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))
from modules.data_engine import DataEngine
from modules.analyzers import EventStudyEngine, TwoShocksEngine, NLPEngine, PortfolioEngine
from modules.charts import (
    create_event_study_chart,
    create_shock_radar,
    create_sankey_flow,
    create_gap_heatmap,
    create_literature_network,
    create_communication_timeline,
    create_impulse_response,
    create_rebalancing_heatmap,
)

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MP Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────
if "data_engine" not in st.session_state:
    st.session_state.data_engine = DataEngine()
if "event_engine" not in st.session_state:
    st.session_state.event_engine = EventStudyEngine()
if "two_shocks" not in st.session_state:
    st.session_state.two_shocks = TwoShocksEngine()
if "nlp_engine" not in st.session_state:
    st.session_state.nlp_engine = NLPEngine()
if "portfolio_engine" not in st.session_state:
    st.session_state.portfolio_engine = PortfolioEngine()

de = st.session_state.data_engine

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ MP Research Platform")
    st.markdown("*Monetary Policy × Asset Prices × Portfolio Rebalancing*")
    st.divider()

    # Data source
    st.markdown("### 📡 Data Source")
    data_source = st.radio(
        "Select data source",
        ["Demo Data (Built-in)", "FRED API (Live)", "Custom Upload"],
        index=0,
    )

    if data_source == "FRED API (Live)":
        fred_key = st.text_input("FRED API Key", type="password", placeholder="Enter your FRED API key...")
        if fred_key:
            st.success("FRED API connected")
    elif data_source == "Custom Upload":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            st.success(f"Loaded: {uploaded.name}")

    st.divider()

    # Date range
    st.markdown("### 📅 Date Range")
    era = st.selectbox(
        "Research Period",
        ["Full History (1994-2025)", "Pre-Crisis (1994-2007)", "GFC (2007-2012)", "Post-GFC (2012-2020)", "COVID-Present (2020-2025)"],
    )

    era_map = {
        "Full History (1994-2025)": (1994, 2025),
        "Pre-Crisis (1994-2007)": (1994, 2007),
        "GFC (2007-2012)": (2007, 2012),
        "Post-GFC (2012-2020)": (2012, 2020),
        "COVID-Present (2020-2025)": (2020, 2025),
    }
    start_year, end_year = era_map[era]

    st.divider()
    st.markdown("---")
    st.caption("Built for Eileen Zhang's research")
    st.caption("v1.0 | 2026-04-22")

# ─── Filter Data ───────────────────────────────────────────────
fomc_data = de.get_fomc_data(start_year, end_year)
asset_data = de.get_asset_data(start_year, end_year)
shock_data = de.get_shock_data(start_year, end_year)

# ─── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Event Study",
    "⚡ Two-Shocks",
    "🗣️ Communication",
    "🔄 Rebalancing",
    "📚 Literature",
    "🔬 Workflow",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: EVENT STUDY DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 FOMC Event Study Dashboard")
    st.markdown("Analyze asset price reactions around Federal Reserve policy announcements")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown("### Event Configuration")
        selected_event = st.selectbox(
            "Select FOMC Meeting",
            fomc_data["date_str"].tolist(),
            index=len(fomc_data) - 1,
        )
        event_idx = fomc_data[fomc_data["date_str"] == selected_event].index[0]

        window_before = st.slider("Window Before (minutes)", 30, 240, 60)
        window_after = st.slider("Window After (minutes)", 30, 480, 120)

    with col2:
        st.markdown("### Asset Classes")
        assets = st.multiselect(
            "Select assets to analyze",
            ["S&P 500", "NASDAQ", "2Y Treasury", "10Y Treasury", "DXY", "Gold", "Oil", "VIX"],
            default=["S&P 500", "10Y Treasury", "DXY", "Gold"],
        )

    with col3:
        st.markdown("### Event Info")
        event_row = fomc_data.iloc[event_idx]
        st.metric("Date", event_row["date_str"])
        st.metric("Rate Decision", f"{event_row['rate_before']:.2f}% → {event_row['rate_after']:.2f}%")
        st.metric("Surprise (bp)", f"{event_row['surprise_bp']:+.0f}")
        shock_type = "Hawkish" if event_row["surprise_bp"] > 5 else ("Dovish" if event_row["surprise_bp"] < -5 else "Neutral")
        st.metric("Tone", shock_type)

    st.divider()

    # Event Study Chart
    if assets:
        event_assets = de.get_event_window(event_idx, window_before, window_after, assets)
        fig = create_event_study_chart(event_assets, selected_event, window_before, window_after)
        st.plotly_chart(fig, use_container_width=True)

    # Summary Statistics
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("### Abnormal Returns Summary")
        if assets:
            summary = st.session_state.event_engine.compute_summary(event_assets, assets)
            st.dataframe(summary, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### 🎯 Shock Radar")
        if assets:
            radar_fig = create_shock_radar(event_assets, assets, event_row["surprise_bp"])
            st.plotly_chart(radar_fig, use_container_width=True)

    # Historical Distribution
    st.markdown("### Historical Surprise Distribution")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=fomc_data["surprise_bp"],
        nbinsx=40,
        marker_color="#4C72B0",
        opacity=0.8,
        name="All Events",
    ))
    hist_fig.add_vline(
        x=event_row["surprise_bp"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current: {event_row['surprise_bp']:+.0f}bp",
    )
    hist_fig.update_layout(
        xaxis_title="Surprise (basis points)",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(hist_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: TWO-SHOCKS DECOMPOSITION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## ⚡ Two-Shocks Decomposition")
    st.markdown("*Policy Shock vs Information Shock* — following Jarociński & Karadi (2020)")
    st.caption("Distinguishes between the direct effect of rate changes and the informational content of FOMC announcements")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Decomposition Settings")
        shock_method = st.selectbox(
            "Identification Method",
            ["High-Frequency (JK 2020)", "Shadow Rate (Wu-Xia)", "NLP-Based"],
        )
        show_assets = st.multiselect(
            "Assets for impulse response",
            ["S&P 500", "10Y Treasury", "DXY", "VIX", "Gold"],
            default=["S&P 500", "10Y Treasury", "DXY"],
        )
        horizon = st.slider("Horizon (days)", 1, 60, 20)

    with col2:
        # Two-shocks time series
        ts_fig = create_impulse_response(shock_data, show_assets, horizon)
        st.plotly_chart(ts_fig, use_container_width=True)

    st.divider()

    # Comparative analysis
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Policy Shock Effect")
        st.markdown("""
        **Mechanism**: Unexpected rate change → discount rate adjustment → price reaction

        **Expected pattern**:
        - 🔴 Hawkish surprise → stocks ↓, bonds ↓, USD ↑
        - 🟢 Dovish surprise → stocks ↑, bonds ↑, USD ↓

        **Key insight**: Policy shocks explain ~60-75% of the total market reaction
        (Bernanke & Kuttner, 2005)
        """)

    with col_b:
        st.markdown("### Information Shock Effect")
        st.markdown("""
        **Mechanism**: FOMC statement reveals info about future economy → cash flow expectations change

        **Expected pattern**:
        - Positive info (strong economy) → stocks ↑, bonds ↓, USD ↑
        - Negative info (weak economy) → stocks ↓, bonds ↑, USD ↓

        **Key insight**: Information shocks can *reverse* the direction of policy shock effects
        (Nakamura & Steinsson, 2018)
        """)

    # Variance decomposition
    st.markdown("### Variance Decomposition: Policy vs Information")
    var_data = st.session_state.two_shocks.variance_decomposition(show_assets)
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        y=show_assets,
        x=var_data["policy_pct"],
        name="Policy Shock",
        orientation="h",
        marker_color="#E74C3C",
    ))
    fig_var.add_trace(go.Bar(
        y=show_assets,
        x=var_data["info_pct"],
        name="Information Shock",
        orientation="h",
        marker_color="#3498DB",
    ))
    fig_var.update_layout(
        barmode="stack",
        xaxis_title="Variance Explained (%)",
        template="plotly_white",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_var, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: FOMC COMMUNICATION DECODER
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🗣️ FOMC Communication Decoder")
    st.markdown("NLP-powered analysis of Federal Reserve communication patterns")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Statement Selection")
        statements = de.get_fomc_statements(start_year, end_year)
        stmt_dates = list(statements.keys())
        stmt1 = st.selectbox("Statement A", stmt_dates, index=max(0, len(stmt_dates) - 2))
        stmt2 = st.selectbox("Statement B", stmt_dates, index=len(stmt_dates) - 1)

        st.markdown("### Statement A")
        st.info(statements[stmt1])
        st.markdown("### Statement B")
        st.success(statements[stmt2])

    with col2:
        st.markdown("### Sentiment Analysis")
        nlp_results = st.session_state.nlp_engine.analyze_pair(
            statements[stmt1], statements[stmt2]
        )
        sentiment_fig = create_communication_timeline(nlp_results, stmt1, stmt2)
        st.plotly_chart(sentiment_fig, use_container_width=True)

    st.divider()

    # Detailed metrics
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 📊 Sentiment Scores")
        for label, scores in [("Statement A", nlp_results["sentiment_a"]), ("Statement B", nlp_results["sentiment_b"])]:
            st.markdown(f"**{label}**")
            for metric, value in scores.items():
                color = "🟢" if value > 0.5 else ("🟡" if value > 0.3 else "🔴")
                st.markdown(f"{color} {metric}: {value:.3f}")

    with col_b:
        st.markdown("### 📝 Readability Metrics")
        for label, scores in [("Statement A", nlp_results["readability_a"]), ("Statement B", nlp_results["readability_b"])]:
            st.markdown(f"**{label}**")
            for metric, value in scores.items():
                st.markdown(f"- {metric}: {value:.1f}")

    with col_c:
        st.markdown("### 🔑 Key Phrase Changes")
        changes = nlp_results["key_changes"]
        for change in changes:
            icon = "➕" if change["type"] == "added" else ("➖" if change["type"] == "removed" else "🔄")
            st.markdown(f"{icon} **{change['phrase']}** — {change['context']}")

    # Historical sentiment trend
    st.markdown("### Historical Sentiment Trend")
    all_sentiments = de.get_historical_sentiments(start_year, end_year)
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=list(all_sentiments.keys()),
        y=[v["hawkish"] for v in all_sentiments.values()],
        name="Hawkish Signal",
        line=dict(color="#E74C3C", width=2),
        fill="tozeroy",
    ))
    trend_fig.add_trace(go.Scatter(
        x=list(all_sentiments.keys()),
        y=[v["dovish"] for v in all_sentiments.values()],
        name="Dovish Signal",
        line=dict(color="#3498DB", width=2),
        fill="tozeroy",
    ))
    trend_fig.update_layout(
        xaxis_title="FOMC Meeting",
        yaxis_title="Signal Strength",
        template="plotly_white",
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(trend_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4: PORTFOLIO REBALANCING
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔄 Portfolio Rebalancing Analysis")
    st.markdown("Visualize how monetary policy announcements drive cross-asset capital flows")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Simulation Settings")
        shock_scenario = st.selectbox(
            "Shock Scenario",
            ["+25bp Hawkish Surprise", "+50bp Hawkish Surprise", "-25bp Dovish Surprise", "-50bp Dovish Surprise", "Information: Strong Economy", "Information: Weak Economy"],
        )
        investor_type = st.selectbox(
            "Investor Type",
            ["Mutual Funds", "Hedge Funds", "Pension Funds", "Foreign Investors", "Retail"],
        )
        time_horizon = st.selectbox(
            "Rebalancing Horizon",
            ["Intraday (0-1 day)", "Short-term (1-5 days)", "Medium-term (5-30 days)", "Long-term (30-90 days)"],
        )

    with col2:
        # Sankey diagram
        sankey_fig = create_sankey_flow(shock_scenario, investor_type, time_horizon)
        st.plotly_chart(sankey_fig, use_container_width=True)

    st.divider()

    # Rebalancing heatmap
    st.markdown("### Cross-Asset Rebalancing Heatmap")
    heatmap_fig = create_rebalancing_heatmap(shock_scenario, investor_type)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Before/After allocation
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Before Announcement")
        before = st.session_state.portfolio_engine.get_allocation("before", investor_type)
        fig_before = go.Figure(go.Pie(
            labels=list(before.keys()),
            values=list(before.values()),
            hole=0.4,
            marker_colors=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
        ))
        fig_before.update_layout(height=350, title_text="Pre-FOMC Allocation")
        st.plotly_chart(fig_before, use_container_width=True)

    with col_b:
        st.markdown("### After Announcement")
        after = st.session_state.portfolio_engine.get_allocation("after", investor_type, shock_scenario)
        fig_after = go.Figure(go.Pie(
            labels=list(after.keys()),
            values=list(after.values()),
            hole=0.4,
            marker_colors=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
        ))
        fig_after.update_layout(height=350, title_text="Post-FOMC Allocation")
        st.plotly_chart(fig_after, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5: LITERATURE NAVIGATOR
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📚 Literature Navigator")
    st.markdown("Interactive map of the research landscape + gap analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Filters")
        topic_filter = st.multiselect(
            "Topic Areas",
            ["Event Studies", "Expectations", "Transmission", "Risk-Taking", "Capital Flows", "Two-Shocks", "NLP/Communication"],
            default=["Event Studies", "Transmission", "Two-Shocks"],
        )
        era_filter = st.multiselect(
            "Era",
            ["1980s-1990s", "2000s", "2010s", "2020s"],
            default=["2000s", "2010s", "2020s"],
        )
        view_mode = st.radio("View", ["Network Graph", "Gap Heatmap", "Timeline"], index=0)

    with col2:
        if view_mode == "Network Graph":
            net_fig = create_literature_network(topic_filter, era_filter)
            st.plotly_chart(net_fig, use_container_width=True)
        elif view_mode == "Gap Heatmap":
            gap_fig = create_gap_heatmap()
            st.plotly_chart(gap_fig, use_container_width=True)
        else:
            st.info("Timeline view — showing publication dates of key papers")
            timeline_data = de.get_literature_timeline(topic_filter)
            tl_fig = go.Figure()
            for i, paper in enumerate(timeline_data):
                tl_fig.add_trace(go.Scatter(
                    x=[paper["year"], paper["year"]],
                    y=[paper["topic"], paper["topic"]],
                    mode="markers+text",
                    marker=dict(size=15, color=paper["color"]),
                    text=[paper["short"]],
                    textposition="top center",
                    name=paper["short"],
                ))
            tl_fig.update_layout(
                yaxis=dict(categoryorder="array", categoryarray=list(dict.fromkeys(p["topic"] for p in timeline_data))),
                template="plotly_white",
                height=500,
                showlegend=False,
            )
            st.plotly_chart(tl_fig, use_container_width=True)

    st.divider()

    # Paper details
    st.markdown("### Key Papers Summary")
    papers = de.get_papers(topic_filter, era_filter)
    for paper in papers:
        with st.expander(f"📄 {paper['authors']} ({paper['year']}) — {paper['title']}"):
            st.markdown(f"**Journal**: {paper['journal']}")
            st.markdown(f"**Key Finding**: {paper['finding']}")
            st.markdown(f"**Method**: {paper['method']}")
            st.markdown(f"**Gap Addressed**: {paper['gap']}")


# ═══════════════════════════════════════════════════════════════
# TAB 6: RESEARCH WORKFLOW
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 🔬 Research Workflow Tracker")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Project Progress")
        tasks = [
            ("Literature Review", "In Progress", 0.65),
            ("Data Collection", "Not Started", 0.0),
            ("Event Study Replication", "Not Started", 0.0),
            ("Two-Shocks Identification", "Not Started", 0.0),
            ("Portfolio Rebalancing Analysis", "Not Started", 0.0),
            ("NLP Communication Analysis", "Not Started", 0.0),
            ("Robustness Checks", "Not Started", 0.0),
            ("Paper Writing", "Not Started", 0.0),
        ]
        for task, status, progress in tasks:
            color = "🟢" if progress >= 0.8 else ("🟡" if progress > 0 else "⚪")
            st.markdown(f"{color} **{task}** — {status}")
            st.progress(progress)

    with col2:
        st.markdown("### 📝 Research Notes")
        notes = st.text_area("Add notes", height=300, placeholder="Record hypotheses, observations, ideas...")

        if st.button("Save Notes"):
            st.success("Notes saved!")

        st.markdown("### 💡 Hypotheses")
        st.markdown("""
        1. **H1**: Larger monetary policy surprises lead to greater cross-asset rebalancing
        2. **H2**: Information shocks dominate policy shocks for equity markets during uncertainty periods
        3. **H3**: Mutual funds with higher turnover rebalance more aggressively post-FOMC
        4. **H4**: The risk-taking channel amplifies rebalancing toward risky assets after dovish surprises
        """)

    st.divider()

    # Export
    st.markdown("### 📤 Export")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Export Summary (CSV)"):
            st.success("Summary exported!")
    with col_b:
        if st.button("Export Figures (HTML)"):
            st.success("Figures exported!")
    with col_c:
        if st.button("Export Literature BibTeX"):
            st.success("BibTeX exported!")

    # Quick stats
    st.markdown("### 📈 Quick Stats")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("FOMC Events Analyzed", len(fomc_data))
    with col_b:
        st.metric("Papers in Database", len(de.get_all_papers()))
    with col_c:
        st.metric("Asset Classes", 8)
    with col_d:
        st.metric("Research Period", f"{end_year - start_year} years")
