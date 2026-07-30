"""
Direction 2: Portfolio Rebalancing Module
==========================================
Two-Shocks Framework: MP vs CBI Effects on Fund Flows

Key Findings:
- H3: Government bonds show differential MP vs CBI effects (9/9 windows significant with B-S baselines)
- Diff window: H1 supported (MP drives immediate risk-off)
- Post window: H2 supported (CBI drives lagged risk-on)
- LLM sentiment correctly distinguishes hawkish/dovish; LM dictionary cannot (r=0.000)
- H5: Corporate bonds show ZLB regime effects

Paper: D2_AFA2027 — Submitted to AFA 2027 GenAI Session (deadline: Aug 31, 2026)
"""

import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2_DIR = os.path.join(BASE, "direction2")
RESULTS_DIR = os.path.join(D2_DIR, "results")
FIGURES_DIR = os.path.join(D2_DIR, "figures")


def render():
    st.markdown("## 🎯 Direction 2: Portfolio Rebalancing via Two-Shocks Framework")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;'>
        <h3 style='color: white; margin: 0;'>MP vs CBI: How Fed Announcements Reshape Fund Flows</h3>
        <p style='opacity: 0.85; margin: 0.5rem 0 0 0;'>
            Jarociński-Korstvedt decomposition → Monetary Policy (MP) shock vs Central Bank Information (CBI) shock<br>
            117 FOMC meetings · 7 asset classes · 3 event windows · 3 sentiment baselines
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FOMC Meetings", "117", "2006–2022")
    with col2:
        st.metric("Asset Classes", "7", "Gov→Small Cap")
    with col3:
        st.metric("Event Windows", "3", "Same/Post/Diff")
    with col4:
        st.metric("Sentiment Baselines", "3", "Raw JK / LM / LLM")

    st.markdown("---")

    # ── Section 1: LLM vs LM Sentiment ──
    st.markdown("### 1. LLM vs LM Dictionary Sentiment Comparison")
    st.markdown("""
    **Core finding**: LM dictionary sentiment and LLM hawkish score measure fundamentally different things.
    Pearson correlation: **r = 0.000**. LM cannot distinguish rate hikes from cuts; LLM can.
    """)

    fig1_path = os.path.join(FIGURES_DIR, "fig1_lm_vs_llm_sentiment.png")
    if os.path.exists(fig1_path):
        st.image(fig1_path, use_container_width=True)

    # Show comparison data
    llm_csv = os.path.join(RESULTS_DIR, "llm_sentiment_results.csv")
    if os.path.exists(llm_csv):
        df = pd.read_csv(llm_csv)
        df['date'] = pd.to_datetime(df['date'])
        df['llm_hawkish'] = pd.to_numeric(df['llm_hawkish'], errors='coerce')

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**LM Dictionary: Fails to discriminate**")
            hike_lm = df[df['decision'] == 'rate_hike']['lm_sentiment'].mean()
            cut_lm = df[df['decision'] == 'rate_cut']['lm_sentiment'].mean()
            st.metric("Avg LM (Hikes)", f"{hike_lm:.4f}")
            st.metric("Avg LM (Cuts)", f"{cut_lm:.4f}")
            st.markdown(f"→ Difference: **{abs(hike_lm - cut_lm):.4f}** (essentially zero)")

        with col_b:
            st.markdown("**LLM Hawkish: Correctly discriminates**")
            hike_llm = df[df['decision'] == 'rate_hike']['llm_hawkish'].mean()
            cut_llm = df[df['decision'] == 'rate_cut']['llm_hawkish'].mean()
            st.metric("Avg LLM (Hikes)", f"{hike_llm:.2f}")
            st.metric("Avg LLM (Cuts)", f"{cut_llm:.2f}")
            st.markdown(f"→ Difference: **{abs(hike_llm - cut_llm):.2f}** (clearly separated)")

    st.markdown("---")

    # ── Section 2: H1 — Fund Flow Response ──
    st.markdown("### 2. H1: Do Fund Flows Respond to FOMC Shocks?")
    st.markdown("""
    **H1**: Fund flows respond significantly to FOMC shocks.
    - **Diff window** (MP effect): H1 supported — MP drives immediate risk-off
    - **Post window** (CBI effect): H1 supported for corporate bonds — CBI drives lagged rebalancing
    """)

    fig2_path = os.path.join(FIGURES_DIR, "fig2_h1_coefficients_3windows.png")
    if os.path.exists(fig2_path):
        st.image(fig2_path, use_container_width=True)

    # H1 results table
    st.markdown("**H1 Regression Results (β coefficients)**")
    h1_data = []
    assets_display = ['government_bonds', 'corporate_bonds', 'real_assets',
                      'large_cap_equity', 'developed_market_equity',
                      'emerging_market_equity', 'small_cap_equity']
    for window in ['same', 'post', 'diff']:
        fpath = os.path.join(RESULTS_DIR, f"h1_h4_results_{window}_window.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            for asset in assets_display:
                h1 = d['h1_raw'].get(asset, {})
                h1_data.append({
                    'Window': window.capitalize(),
                    'Asset Class': asset.replace('_', ' ').title(),
                    'β_MP': f"{h1.get('beta_mp', np.nan):.3f}",
                    'p_MP': f"{h1.get('p_mp', np.nan):.3f}",
                    'β_CBI': f"{h1.get('beta_cbi', np.nan):.3f}",
                    'p_CBI': f"{h1.get('p_cbi', np.nan):.3f}",
                    'R²': f"{h1.get('r_squared', np.nan):.3f}",
                })
    if h1_data:
        st.dataframe(pd.DataFrame(h1_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section 3: H3 — MP vs CBI Differential Effects ──
    st.markdown("### 3. H3: Do MP and CBI Shocks Have Different Effects?")
    st.markdown("""
    **H3**: The effect of MP shocks on fund flows differs from the effect of CBI shocks.
    - **Government bonds**: Most robust finding — differential effects across all windows and baselines
    - **Corporate bonds**: Significant in post window (CBI-driven rebalancing)
    """)

    fig3_path = os.path.join(FIGURES_DIR, "fig3_h3_wald_test_3windows.png")
    if os.path.exists(fig3_path):
        st.image(fig3_path, use_container_width=True)

    # Significance heatmap
    st.markdown("#### CBI Shock Significance Heatmap")
    fig6_path = os.path.join(FIGURES_DIR, "fig6_significance_heatmap.png")
    if os.path.exists(fig6_path):
        st.image(fig6_path, use_container_width=True)

    st.markdown("---")

    # ── Section 4: Phase 1 LLM Robustness ──
    st.markdown("### 4. Phase 1 Robustness: LM vs LLM Incremental R²")
    st.markdown("""
    **Finding**: LM and LLM are complementary, not substitutes.
    - **LM dictionary** better for equities in Forward Guidance period
    - **LLM hawkish** better for gold and non-FG periods
    - Both add explanatory power beyond target+path factors, but capture different signals
    """)

    fig4_path = os.path.join(FIGURES_DIR, "fig4_phase1_llm_robustness.png")
    if os.path.exists(fig4_path):
        st.image(fig4_path, use_container_width=True)

    # Show robustness summary
    summary_path = os.path.join(RESULTS_DIR, "phase1_llm_robustness_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = f.read()
        with st.expander("View Full Robustness Summary"):
            st.text(summary)

    st.markdown("---")

    # ── Section 5: H5 — ZLB Regime Analysis ──
    st.markdown("### 5. H5: Zero Lower Bound Regime Effects")
    st.markdown("""
    **H5**: CBI shock effects are amplified during the Zero Lower Bound period.
    - Corporate bonds show *** significance in ZLB regime
    - Suggests forward guidance becomes more powerful when conventional policy is exhausted
    """)

    fig5_path = os.path.join(FIGURES_DIR, "fig5_h5_regime_zlb.png")
    if os.path.exists(fig5_path):
        st.image(fig5_path, use_container_width=True)

    st.markdown("---")

    # ── Section 6: Three-Window Robustness Matrix ──
    st.markdown("### 6. Three-Window × Three-Baseline Robustness Matrix")
    st.markdown("""
    The triple baseline design (Raw JK / B-S LM / B-S LLM) × three event windows (same / post / diff)
    provides 9 tests per hypothesis per asset class.

    **Most robust finding**: H3 for government bonds — differential MP vs CBI effects
    are significant across all 9 combinations.
    """)

    # Build robustness summary table
    robust_data = []
    for window in ['same', 'post', 'diff']:
        fpath = os.path.join(RESULTS_DIR, f"h1_h4_results_{window}_window.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            for asset in ['government_bonds', 'corporate_bonds']:
                h3 = d['h3_raw'].get(asset, {})
                h1 = d['h1_raw'].get(asset, {})
                robust_data.append({
                    'Window': window.capitalize(),
                    'Asset': asset.replace('_', ' ').title(),
                    'β_MP (H1)': f"{h1.get('beta_mp', np.nan):.3f}",
                    'β_CBI (H1)': f"{h1.get('beta_cbi', np.nan):.3f}",
                    'H3 Wald p': f"{h3.get('wald_p', np.nan):.3f}",
                    'H3 Rejected': str(h3.get('h3_rejected', 'N/A')),
                })
    if robust_data:
        st.dataframe(pd.DataFrame(robust_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Paper & Code ──
    st.markdown("### 7. Paper & Replication Package")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        **📄 AFA 2027 GenAI Session Paper**
        - Title: *Two Shocks, Two Flows: MP vs CBI Effects on Fund Rebalancing*
        - Authors: Eileen Zhang & Yang Dongsheng
        - Deadline: August 31, 2026
        - Status: Draft complete, revisions ongoing
        """)
    with col_p2:
        st.markdown("""
        **💻 Replication Code**
        - Pipeline: `direction2/code/run_pipeline.py`
        - Regression: `direction2/code/h1_h4_regression.py`
        - LLM Sentiment: `direction2/code/llm_sentiment_analysis.py`
        - Audit Chain: `direction2/code/audit_chain.py`
        """)

    st.markdown("""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #0f3460; margin-top: 1rem;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            <strong>Core Narrative:</strong> Central Bank Information (CBI) shocks drive portfolio rebalancing,
            not Monetary Policy (MP) shocks. The distinction matters — using LM dictionary sentiment
            masks this effect entirely; LLM-based sentiment measurement reveals it.
        </p>
    </div>
    """, unsafe_allow_html=True)
