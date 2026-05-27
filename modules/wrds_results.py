"""
WRDS-Enhanced Results Module (v6)
===================================
Displays real regression results from the v6 analysis pipeline
using CRSP + GSS shocks + expanded sentiment dictionary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")


def render():
    st.markdown('<div class="main-header"><h1>📊 WRDS-Enhanced Results (v6)</h1>'
                '<p>Real regression results: CRSP + GSS Shocks + Expanded Sentiment</p></div>',
                unsafe_allow_html=True)

    # ── Version Comparison ──
    st.markdown("### Data Upgrade Impact")
    st.markdown("""
    | Version | Data | H1 R² | H1 p-value | Key Upgrade |
    |---------|------|--------|------------|-------------|
    | v4 | yfinance + rate_change | 0.17% | 0.712 | Baseline |
    | v5 | CRSP + GSS shocks | 1.57% | 0.032** | Correct surprise measure |
    | **v6** | **CRSP + GSS + expanded sentiment** | **4.12%** | **0.010*** | Better sentiment capture |
    
    *R² improved 24× from v4 to v6 by using proper high-frequency surprise identification 
    and an expanded central bank dictionary.*
    """)

    # ── Load Results ──
    results_file = os.path.join(RESULTS_DIR, "regression_results_v6.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        _render_results(results)
    else:
        st.warning("v6 results not found. Run `analysis/run_v6_comprehensive.py` first.")

    # ── Charts ──
    st.markdown("### 📈 Publication-Quality Charts")
    _render_charts()

    # ── Dataset Preview ──
    st.markdown("### 📋 Dataset Preview")
    dataset_file = os.path.join(RESULTS_DIR, "analysis_dataset_v6.csv")
    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file, index_col=0, parse_dates=True)
        st.markdown(f"**{len(df)} FOMC meetings** (2006-2022) with {len(df.columns)} variables")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Key stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Shock β", f"{results['H1']['beta_target']:.6f}")
        with col2:
            st.metric("Path Shock β", f"{results['H1']['beta_path']:.6f}")
        with col3:
            st.metric("Path Shock p-value", f"{results['H1']['p_path']:.4f}")
    else:
        st.info("Dataset file not found.")


def _render_results(results):
    """Render regression results tables."""
    
    # H1
    st.markdown("#### H1: Sentiment ~ Target Shock + Path Shock")
    h1 = results.get("H1", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{h1.get('r_squared', 0)*100:.2f}%")
    col2.metric("Target β", f"{h1.get('beta_target', 0):.6f}")
    col3.metric("Path β", f"{h1.get('beta_path', 0):.6f}")
    col4.metric("N", h1.get('n', ''))
    
    t_p = h1.get('p_target', 1)
    p_p = h1.get('p_path', 1)
    st.markdown(f"**Target shock**: p = {t_p:.4f} {'**' if t_p < 0.05 else '*' if t_p < 0.1 else ''}")
    st.markdown(f"**Path shock**: p = {p_p:.4f} {'***' if p_p < 0.01 else '**' if p_p < 0.05 else '*' if p_p < 0.1 else ''}")
    st.info("🔑 **Key finding**: Path shock dominates (p=0.010***) — FOMC language is primarily forward-looking")
    
    # H2
    st.markdown("#### H2: Asset Returns ~ Target Shock + Path Shock")
    h2 = results.get("H2", {})
    if h2:
        h2_data = []
        for asset, r in h2.items():
            sig = "***" if r.get('p_target', 1) < 0.01 else "**" if r.get('p_target', 1) < 0.05 else "*" if r.get('p_target', 1) < 0.1 else ""
            h2_data.append({
                "Asset": asset,
                "β(Target)": f"{r.get('beta_target', 0):.4f}",
                "p(Target)": f"{r.get('p_target', 1):.4f}{sig}",
                "β(Path)": f"{r.get('beta_path', 0):.4f}",
                "p(Path)": f"{r.get('p_path', 1):.4f}",
                "R²": f"{r.get('r_squared', 0)*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(h2_data), use_container_width=True)
        st.markdown("*Equal-weighted market responds most strongly to target shocks (t=-2.03**)*")

    # H3
    st.markdown("#### H3: Information Channel")
    h3 = results.get("H3", {})
    if h3:
        if h3.get("info_dominates"):
            st.success("✅ **Path shock dominates** — Information channel confirmed!")
        else:
            st.warning("❌ Target shock dominates — No evidence for information channel")
        st.markdown(f"Target |t| = {h3.get('target_t', 0):.3f} vs Path |t| = {h3.get('path_t', 0):.3f}")

    # Robustness
    st.markdown("#### Robustness Checks")
    rob = results.get("robustness", {})
    rob_data = []
    for check, r in rob.items():
        if isinstance(r, dict) and 'r_squared' in r:
            rob_data.append({
                "Check": check,
                "R²": f"{r['r_squared']*100:.2f}%",
                "N": r.get('n', ''),
            })
    if rob_data:
        st.dataframe(pd.DataFrame(rob_data), use_container_width=True)


def _render_charts():
    """Render charts from the v6 analysis."""
    charts = [
        ("fig10_version_comparison.png", "Model Improvement Across Versions"),
        ("fig2_h1_scatter.png", "H1: Sentiment vs Path Shock"),
        ("fig_target_vs_returns.png", "Target Shock vs Market Returns"),
        ("fig7_sentiment_by_regime.png", "Sentiment by Policy Regime"),
        ("fig6_shocks_timeseries.png", "Target & Path Shocks Over Time"),
        ("fig9_correlation_heatmap.png", "Correlation Matrix"),
        ("fig4_financial_event_study.png", "Financial Sector Event Study"),
        ("fig_rolling_r2.png", "Rolling R² Over Time"),
        ("fig_shock_distributions.png", "Distribution of Shocks"),
        ("fig_chair_comparison.png", "Comparison Across Fed Chairs"),
    ]
    
    for fname, label in charts:
        fpath = os.path.join(CHARTS_DIR, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=label, use_container_width=True)
        # Skip missing charts silently
