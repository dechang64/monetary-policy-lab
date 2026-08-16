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
- Cross-paper comparison: Target shock drives daily prices, CBI shock drives monthly flows
- Frequency comparison: Both MP and CBI significant at daily (opposite signs), CBI fades at weekly

Paper: D2_AFA2027 — Submitted to AFA 2027 GenAI Session (deadline: Aug 31, 2026)
Authors: Eileen Zhang (Rutgers) & Dechang Xu (XJTLU)
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
            117 FOMC meetings · 7 asset classes · 3 event windows · 3 sentiment baselines · 3 frequencies
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
        st.metric("Frequencies", "3", "Daily/Weekly/Monthly")

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

    # LLM timeline
    fig9_path = os.path.join(FIGURES_DIR, "fig9_llm_sentiment_timeline.png")
    if os.path.exists(fig9_path):
        st.image(fig9_path, use_container_width=True)

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

    # Risk ladder coefficients
    fig7_path = os.path.join(FIGURES_DIR, "fig7_risk_ladder_coefficients.png")
    if os.path.exists(fig7_path):
        st.image(fig7_path, use_container_width=True)

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

    fig6_path = os.path.join(FIGURES_DIR, "fig6_significance_heatmap.png")
    if os.path.exists(fig6_path):
        st.image(fig6_path, use_container_width=True)

    st.markdown("---")

    # ── Section 4: H4 — Substitution Matrix ──
    st.markdown("### 4. H4: Asset Substitution Matrix")
    st.markdown("""
    CBI shocks show a monotonic decline in mean |β| with distance from the diagonal
    (Spearman ρ = -0.77, p = 0.072), suggesting risk-ladder substitution.
    """)

    fig8_path = os.path.join(FIGURES_DIR, "fig8_h4_substitution_matrix.png")
    if os.path.exists(fig8_path):
        st.image(fig8_path, use_container_width=True)

    st.markdown("---")

    # ── Section 5: H5 — ZLB Regime ──
    st.markdown("### 5. H5: Zero Lower Bound Regime Effects")
    st.markdown("""
    ZLB regime amplifies MP effects on corporate bonds (β = -0.462, p = 0.005).
    """)

    fig5_path = os.path.join(FIGURES_DIR, "fig5_h5_regime_zlb.png")
    if os.path.exists(fig5_path):
        st.image(fig5_path, use_container_width=True)

    st.markdown("---")

    # ── Section 6: Phase 1 LLM Robustness ──
    st.markdown("### 6. Phase 1 Robustness: LM vs LLM Incremental R²")
    st.markdown("""
    LM and LLM are complementary: LM better for equities in FG period, LLM better for gold.
    """)

    fig4_path = os.path.join(FIGURES_DIR, "fig4_phase1_llm_robustness.png")
    if os.path.exists(fig4_path):
        st.image(fig4_path, use_container_width=True)

    summary_path = os.path.join(RESULTS_DIR, "phase1_llm_robustness_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = f.read()
        with st.expander("View Full Robustness Summary"):
            st.text(summary)

    st.markdown("---")

    # ── Section 7: Triple Baseline Robustness ──
    st.markdown("### 7. Triple Baseline Robustness Matrix")
    st.markdown("""
    4 hypotheses × 9 specifications = 36 tests. H3 (government bonds) significant in all 9.
    """)

    fig10_path = os.path.join(FIGURES_DIR, "fig10_triple_baseline_robustness.png")
    if os.path.exists(fig10_path):
        st.image(fig10_path, use_container_width=True)

    st.markdown("---")

    # ── Section 8: Cross-Paper Comparison ──
    st.markdown("### 8. Cross-Paper Comparison: Asset Returns vs. Fund Flows")
    st.markdown("""
    Comparing Phase 1 (daily asset returns, GSS target/path shocks) with Direction 2 (monthly fund flows, JK MP/CBI shocks).

    **Three key findings:**
    1. **Target shock drives prices, CBI shock drives flows** — two-stage transmission
    2. **LM dictionary works for returns, LLM works for flows** — complementary measures
    3. **Government bonds are special** — no price reaction but significant flow response
    """)

    fig11_path = os.path.join(FIGURES_DIR, "fig11_phase1_vs_direction2_comparison.png")
    if os.path.exists(fig11_path):
        st.image(fig11_path, use_container_width=True)

    fig12_path = os.path.join(FIGURES_DIR, "fig12_cross_paper_scatter.png")
    if os.path.exists(fig12_path):
        st.image(fig12_path, use_container_width=True)

    st.markdown("---")

    # ── Section 9: Frequency Comparison ──
    st.markdown("### 9. Frequency Comparison: Daily vs. Weekly vs. Monthly")
    st.markdown("""
    Re-estimating H1/H3 with daily returns (FOMC day) and weekly returns (3-day, 2-day windows).

    **Five patterns:**
    1. **Daily CRSP VW**: Both MP and CBI significant with **opposite signs** (H3 χ²=40.75***)
    2. **S&P 500 & NASDAQ**: Only **CBI** significant — information drives stock prices
    3. **Gold**: Only **MP** significant — reacts to pure rate changes
    4. **Weekly**: CBI fades, MP persists — information effect is short-lived
    5. **Monthly flows**: Individual effects not significant, but H3 difference matters
    """)

    fig13_path = os.path.join(FIGURES_DIR, "fig13_frequency_comparison.png")
    if os.path.exists(fig13_path):
        st.image(fig13_path, use_container_width=True)

    # Show daily results table
    daily_path = os.path.join(RESULTS_DIR, "daily_return_jk_results.json")
    weekly_path = os.path.join(RESULTS_DIR, "weekly_return_jk_results.json")
    if os.path.exists(daily_path):
        with open(daily_path) as f:
            daily = json.load(f)
        with open(weekly_path) as f:
            weekly = json.load(f)

        freq_data = []
        for asset, r in daily.items():
            freq_data.append({
                'Frequency': 'Daily (FOMC day)',
                'Asset': asset,
                'β_MP': f"{r['beta_mp']:+.4f}",
                'p_MP': f"{r['p_mp']:.3f}",
                'β_CBI': f"{r['beta_cbi']:+.4f}",
                'p_CBI': f"{r['p_cbi']:.3f}",
                'H3 χ²': f"{r['wald_chi2']:.2f}",
                'p_H3': f"{r['wald_p']:.3f}",
            })
        for asset, r in weekly.items():
            freq_data.append({
                'Frequency': 'Weekly',
                'Asset': asset,
                'β_MP': f"{r['beta_mp']:+.4f}",
                'p_MP': f"{r['p_mp']:.3f}",
                'β_CBI': f"{r['beta_cbi']:+.4f}",
                'p_CBI': f"{r['p_cbi']:.3f}",
                'H3 χ²': f"{r['wald_chi2']:.2f}",
                'p_H3': f"{r['wald_p']:.3f}",
            })
        st.dataframe(pd.DataFrame(freq_data), use_container_width=True, hide_index=True)

    st.markdown("""
    **Transmission timeline:**
    - **Day 0**: MP + CBI both affect prices (opposite directions)
    - **Days 1–2**: CBI fades, MP persists
    - **Month t+1**: Fund flows adjust based on MP vs CBI difference
    """)

    st.markdown("---")

    # ── Section 10: Paper & Code ──
    st.markdown("### 10. Paper & Replication Package")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        **📄 AFA 2027 GenAI Session Paper**
        - Title: *Information vs. Action: How Central Bank Communication Drives Mutual Fund Rebalancing*
        - Authors: Eileen Zhang (Rutgers) & Dechang Xu (XJTLU)
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
            not Monetary Policy (MP) shocks. At daily frequency, both matter (opposite signs).
            At monthly frequency, only the <em>difference</em> matters for fund flows.
            LLM sentiment reveals what LM dictionary masks (r=0.000).
        </p>
    </div>
    """, unsafe_allow_html=True)
