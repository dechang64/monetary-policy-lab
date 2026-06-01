"""
WRDS-Enhanced Results Module (v10.2)
=====================================
Displays verified regression results from the v10.2 analysis pipeline
using CRSP + GSS shocks + combined sentiment (LM + CB).
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
    st.markdown('<div class="main-header"><h1>📊 WRDS-Enhanced Results (v10.2)</h1>'
                '<p>Verified regression results: CRSP + GSS Shocks + Combined Sentiment (LM + CB)</p></div>',
                unsafe_allow_html=True)

    # ── Version Comparison ──
    st.markdown("### Data Upgrade Impact")
    st.markdown("""
    | Version | Data | H1 R² | Target p | Path p | Key Finding |
    |---------|------|--------|----------|--------|-------------|
    | v4 | yfinance + rate_change | 0.17% | 0.712 | — | Baseline |
    | v5 | CRSP + GSS shocks | 1.57% | 0.032** | 0.152 | Correct surprise measure |
    | **v10.2** | **CRSP + GSS + combined sentiment** | **1.57%** | **0.017**** | **0.152** | **Target shock significant; path not** |

    *Target shock has a significant positive effect on FOMC statement sentiment (p=0.017).
    Path shock is not statistically significant (p=0.152). Evidence favors policy implementation
    channel over information revelation channel.*
    """)

    # ── Load Results ──
    results_file = os.path.join(RESULTS_DIR, "verified_results.json")
    results = None
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        _render_results(results)
    else:
        st.warning("Verified results not found. Run audit pipeline first.")

    # ── Charts ──
    st.markdown("### 📈 Publication-Quality Charts")
    _render_charts()

    # ── Dataset Preview ──
    st.markdown("### 📋 Dataset Preview")
    dataset_file = os.path.join(RESULTS_DIR, "analysis_dataset_extended_v7.csv")
    if os.path.exists(dataset_file) and results:
        df = pd.read_csv(dataset_file, index_col=0, parse_dates=True)
        st.markdown(f"**{len(df)} FOMC meetings** (2006-2022) with {len(df.columns)} variables")
        st.dataframe(df.head(20), use_container_width=True)

        # Key stats from verified results
        h1 = results.get('H1', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Shock β", f"{h1.get('target_shock', {}).get('beta', 0):.6f}")
        with col2:
            st.metric("Path Shock β", f"{h1.get('path_shock', {}).get('beta', 0):.6f}")
        with col3:
            st.metric("Target Shock p-value", f"{h1.get('target_shock', {}).get('p', 1):.4f}")
    else:
        st.info("Dataset file not found.")


def _render_results(results):
    """Render regression results tables from verified_results.json format."""

    # H1: Sentiment ~ Target Shock + Path Shock
    st.markdown("#### H1: Sentiment ~ Target Shock + Path Shock")
    h1 = results.get("H1", {})
    target = h1.get("target_shock", {})
    path = h1.get("path_shock", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{h1.get('r2', 0)*100:.2f}%")
    col2.metric("Target β", f"{target.get('beta', 0):.6f}")
    col3.metric("Path β", f"{path.get('beta', 0):.6f}")
    col4.metric("N", h1.get('n', ''))

    t_p = target.get('p', 1)
    p_p = path.get('p', 1)
    st.markdown(f"**Target shock**: β = {target.get('beta', 0):.6f}, t = {target.get('t', 0):.2f}, p = {t_p:.4f} {'**' if t_p < 0.05 else '*' if t_p < 0.1 else ''}")
    st.markdown(f"**Path shock**: β = {path.get('beta', 0):.6f}, t = {path.get('t', 0):.2f}, p = {p_p:.4f} {'**' if p_p < 0.05 else '*' if p_p < 0.1 else ''}")
    st.info("🔑 **Key finding**: Target shock is significant (p=0.017), path shock is not (p=0.152) — Evidence favors policy implementation channel")

    # H2: Asset Returns
    st.markdown("#### H2: Asset Returns ~ Target Shock + Path Shock")
    h2 = results.get("H2_crsp", {})
    if h2:
        h2_data = []
        asset_labels = {
            'crsp_vw': 'CRSP VW', 'crsp_ew': 'CRSP EW',
            'sp500_sprtrn': 'S&P 500', 'nasdaq': 'NASDAQ',
            'gold': 'Gold', 'ty10': '10Y Treasury', 'tb13w': '13W T-bill'
        }
        for asset, r in h2.items():
            if not isinstance(r, dict) or 'beta_target' not in r:
                continue
            sig = "***" if r.get('p_target', 1) < 0.01 else "**" if r.get('p_target', 1) < 0.05 else "*" if r.get('p_target', 1) < 0.1 else ""
            h2_data.append({
                "Asset": asset_labels.get(asset, asset),
                "β(Target)": f"{r.get('beta_target', 0):.3f}",
                "t(Target)": f"{r.get('t_target', 0):.2f}",
                "p(Target)": f"{r.get('p_target', 1):.3f}{sig}",
                "β(Path)": f"{r.get('beta_path', 0):.3f}",
                "p(Path)": f"{r.get('p_path', 1):.3f}",
                "R²": f"{r.get('r2', 0)*100:.1f}%",
            })
        if h2_data:
            st.dataframe(pd.DataFrame(h2_data), use_container_width=True)
            st.markdown("*Target shock significantly affects equities and gold; path shock not significant for any asset class*")

    # H3: Wald Test (Information Channel)
    st.markdown("#### H3: Information Channel (Wald Test)")
    h3 = results.get("H3_wald", {})
    if h3:
        col1, col2 = st.columns(2)
        col1.metric("Wald χ²", f"{h3.get('chi2', 0):.4f}")
        col2.metric("p-value", f"{h3.get('p', 1):.4f}")
        st.warning(f"⚠️ {h3.get('conclusion', 'Cannot confirm information channel')}")

    # H4: Forward Guidance Interaction
    st.markdown("#### H4: Forward Guidance Period Interaction")
    h4_vw = results.get("H4_crsp_vw", {})
    h4_nq = results.get("H4_nasdaq", {})
    if h4_vw or h4_nq:
        h4_data = []
        for label, h4 in [("CRSP VW", h4_vw), ("NASDAQ", h4_nq)]:
            if not h4:
                continue
            row = {"Asset": label}
            for key in ['target_shock', 'path_shock', 'sentiment']:
                if key in h4:
                    row[f"β({key})"] = f"{h4[key].get('beta', 0):.2f}"
                    row[f"p({key})"] = f"{h4[key].get('p', 1):.3f}"
            if 'sent_x_fg' in h4:
                row["β(Sent×FG)"] = f"{h4['sent_x_fg'].get('beta', 0):.2f}"
                row["p(Sent×FG)"] = f"{h4['sent_x_fg'].get('p', 1):.3f}"
            h4_data.append(row)
        if h4_data:
            st.dataframe(pd.DataFrame(h4_data), use_container_width=True)

    # Robustness
    st.markdown("#### Robustness Checks")
    rob = results.get("H1_robustness", {})
    rob_data = []
    rob_labels = {
        'rate_change_only': 'Rate Change Only',
        'kuttner_only': 'Kuttner Surprise',
        'cb_score_depvar': 'CB Score (Dep Var)',
    }
    for check, r in rob.items():
        if not isinstance(r, dict):
            continue
        row = {"Check": rob_labels.get(check, check)}
        if 'r2' in r:
            row["R²"] = f"{r['r2']*100:.2f}%"
        if 'p' in r:
            row["p-value"] = f"{r['p']:.4f}"
        if 'target_p' in r:
            row["Target p"] = f"{r['target_p']:.4f}"
        if 'path_p' in r:
            row["Path p"] = f"{r['path_p']:.4f}"
        rob_data.append(row)
    if rob_data:
        st.dataframe(pd.DataFrame(rob_data), use_container_width=True)


def _render_charts():
    """Render charts from the v10.2 analysis."""
    charts = [
        ("fig1_framework.png", "Research Framework"),
        ("fig2_h1_scatter.png", "H1: Sentiment vs Monetary Policy Shocks"),
        ("fig3_h2_returns.png", "H2: Asset Returns and Monetary Policy Shocks"),
        ("fig7_sentiment_by_regime.png", "Sentiment by Policy Regime"),
        ("fig9_correlation_heatmap.png", "Correlation Matrix"),
        ("fig6_shocks_timeseries.png", "Target & Path Shocks Over Time"),
        ("fig_target_vs_returns.png", "Target Shock vs Market Returns"),
        ("fig4_financial_event_study.png", "Financial Sector Event Study"),
        ("fig_rolling_r2.png", "Rolling R² Over Time"),
        ("fig_shock_distributions.png", "Distribution of Shocks"),
        ("fig_chair_comparison.png", "Comparison Across Fed Chairs"),
    ]

    for fname, label in charts:
        fpath = os.path.join(CHARTS_DIR, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=label, use_container_width=True)
