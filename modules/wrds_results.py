"""
WRDS-Enhanced Results Module (v10.3)
=====================================
Displays verified regression results from the v10.3 analysis pipeline
including JK decomposition, Bauer-Swanson orthogonalization,
and Fernández-Fuertes (2025) positioning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
FIG_DIR = os.path.join(BASE_DIR, "presentation_v10.3", "figures")


def render():
    st.markdown('<div class="main-header"><h1>📊 Research Results (v10.3)</h1>'
                '<p>Implementation vs. Revelation · JK Decomposition · B-S Orthogonalization · FF Positioning</p></div>',
                unsafe_allow_html=True)

    # ── Version Timeline ──
    st.markdown("### 🔄 Version Evolution")
    st.markdown("""
    | Version | Data | H1 R² | Target p | Key Upgrade |
    |---------|------|--------|----------|-------------|
    | v4 | yfinance + rate_change | 0.17% | 0.712 | Baseline |
    | v5 | CRSP + GSS shocks | 1.57% | 0.032** | Correct surprise measure |
    | v10.2 | CRSP + GSS + combined sentiment | 1.57% | 0.017** | CB dictionary + NW HAC |
    | **v10.3** | **+ JK decomposition + B-S orthogonalization** | **1.57%** | **0.017**** | **Implementation vs Revelation + FF positioning** |
    """)

    # ── Tabs for major sections ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Core Results (H1-H4)",
        "🔬 JK Decomposition",
        "📐 B-S Orthogonalization",
        "📰 FF (2025) Positioning",
        "📈 Charts & Figures"
    ])

    with tab1:
        _render_core_results()

    with tab2:
        _render_jk_decomposition()

    with tab3:
        _render_bs_orthogonalization()

    with tab4:
        _render_ff_positioning()

    with tab5:
        _render_charts()


def _render_core_results():
    """Render H1-H4 core regression results."""
    results_file = os.path.join(RESULTS_DIR, "verified_results.json")
    if not os.path.exists(results_file):
        st.warning("Verified results not found.")
        return

    with open(results_file) as f:
        results = json.load(f)

    # H1
    st.markdown("#### H1: Sentiment ~ Target + Path Shock")
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
    st.markdown(f"**Target shock**: β = {target.get('beta', 0):.6f}, p = {t_p:.4f} {'✅ **' if t_p < 0.05 else ''}")
    st.markdown(f"**Path shock**: β = {path.get('beta', 0):.6f}, p = {p_p:.4f} {'❌' if p_p > 0.1 else ''}")
    st.success("🔑 Target shock significantly predicts sentiment (p=0.017); path shock does not (p=0.152) → Evidence favors **policy implementation** over informational revelation")

    # H2
    st.markdown("#### H2: Asset Returns ~ Target + Path Shock")
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
                "p(Target)": f"{r.get('p_target', 1):.3f}{sig}",
                "β(Path)": f"{r.get('beta_path', 0):.3f}",
                "p(Path)": f"{r.get('p_path', 1):.3f}",
                "R²": f"{r.get('r2', 0)*100:.1f}%",
            })
        if h2_data:
            st.dataframe(pd.DataFrame(h2_data), use_container_width=True)
            st.info("EW > VW: smaller firms more sensitive → financing-channel evidence")

    # H3
    st.markdown("#### H3: Wald Test (β_T = β_P?)")
    h3 = results.get("H3_wald", {})
    if h3:
        col1, col2 = st.columns(2)
        col1.metric("Wald χ²", f"{h3.get('chi2', 0):.4f}")
        col2.metric("p-value", f"{h3.get('p', 1):.4f}")
        st.warning(f"⚠️ Cannot reject β_T = β_P (p={h3.get('p', 1):.2f}) — cannot fully separate channels")

    # H4
    st.markdown("#### H4: Forward Guidance Interaction")
    h4_vw = results.get("H4_crsp_vw", {})
    h4_nq = results.get("H4_nasdaq", {})
    if h4_vw or h4_nq:
        h4_data = []
        for label, h4 in [("CRSP VW", h4_vw), ("NASDAQ", h4_nq)]:
            if not h4:
                continue
            row = {"Asset": label}
            if 'sent_x_fg' in h4:
                row["β(Sent×FG)"] = f"{h4['sent_x_fg'].get('beta', 0):.2f}"
                row["p(Sent×FG)"] = f"{h4['sent_x_fg'].get('p', 1):.3f}"
            h4_data.append(row)
        if h4_data:
            st.dataframe(pd.DataFrame(h4_data), use_container_width=True)
            st.error("❌ FG interaction **not significant** — sentiment does not become more important during ZLB")


def _render_jk_decomposition():
    """Render JK sign-restriction decomposition results."""
    jk_file = os.path.join(RESULTS_DIR, "jk_bs_results.json")
    if not os.path.exists(jk_file):
        st.warning("JK decomposition results not found. Run `jk_bs_decomposition.py` first.")
        return

    with open(jk_file) as f:
        data = json.load(f)

    jk = data.get("jk_decomposition", {})

    st.markdown("#### Jarociński-Karadi (2020) Sign-Restriction Decomposition")
    st.markdown("""
    Classify target shocks by co-movement with stock returns:
    - **MP shock**: rate ↑ + stocks ↓ (pure monetary policy) — 59% of meetings
    - **CBI shock**: rate ↑ + stocks ↑ (central bank information) — 41% of meetings
    """)

    # Classification
    col1, col2 = st.columns(2)
    col1.metric("MP Shocks", f"{jk.get('n_mp_shocks', 0)}", "59% of meetings")
    col2.metric("CBI Shocks", f"{jk.get('n_cbi_shocks', 0)}", "41% of meetings")

    # H1 Sentiment
    st.markdown("##### Sentiment Regression (H1)")
    decomp = jk.get("decomposed", {})
    sent_data = {
        "Coefficient": ["β_MP", "β_CBI", "β_Path"],
        "Estimate": [decomp.get("beta_mp", 0), decomp.get("beta_cbi", 0), decomp.get("beta_path", 0)],
        "p-value": [decomp.get("p_mp", 1), decomp.get("p_cbi", 1), decomp.get("p_path", 1)],
        "Significant": [
            "✅" if decomp.get("p_mp", 1) < 0.05 else "❌",
            "✅" if decomp.get("p_cbi", 1) < 0.05 else "❌",
            "✅" if decomp.get("p_path", 1) < 0.05 else "❌",
        ]
    }
    st.dataframe(pd.DataFrame(sent_data), use_container_width=True)

    ftest = jk.get("ftest_equal", {})
    st.info(f"F-test (β_MP = β_CBI): F = {ftest.get('f_stat', 0):.3f}, p = {ftest.get('p_value', 1):.3f} → Cannot reject equality")

    # H2 Asset Returns
    st.markdown("##### CRSP VW Returns (H2)")
    h2 = data.get("jk_h2_vw", {})
    ret_data = {
        "Coefficient": ["β_MP", "β_CBI", "β_Path"],
        "Estimate": [h2.get("beta_mp", 0), h2.get("beta_cbi", 0), h2.get("beta_path", 0)],
        "p-value": [h2.get("p_mp", 1), h2.get("p_cbi", 1), h2.get("p_path", 1)],
        "Significant": [
            "✅***" if h2.get("p_mp", 1) < 0.01 else "❌",
            "✅***" if h2.get("p_cbi", 1) < 0.01 else "❌",
            "❌" if h2.get("p_path", 1) > 0.1 else "✅",
        ]
    }
    st.dataframe(pd.DataFrame(ret_data), use_container_width=True)

    st.success(f"🔑 **Key insight**: R² jumps from 9.1% → **{h2.get('r_squared', 0):.1f}%** for asset returns. "
               "MP shocks push stocks down, CBI shocks push stocks up — **information effect exists in markets, not in language**")

    # JK figure
    jk_fig = os.path.join(FIG_DIR, "fig_jk_decomposition.png")
    if os.path.exists(jk_fig):
        st.image(jk_fig, caption="JK Decomposition: Classification, Sentiment, and Asset Returns", use_container_width=True)


def _render_bs_orthogonalization():
    """Render Bauer-Swanson orthogonalization results."""
    bs_file = os.path.join(RESULTS_DIR, "jk_bs_results.json")
    if not os.path.exists(bs_file):
        st.warning("B-S results not found.")
        return

    with open(bs_file) as f:
        data = json.load(f)

    bs = data.get("bauer_swanson", {})

    st.markdown("#### Bauer-Swanson (2023) Orthogonalization")
    st.markdown("""
    Address the critique that HF surprises are partially predictable from pre-FOMC macro information.
    Regress shocks on pre-FOMC controls → use residuals as orthogonalized shocks.
    """)

    # Predictability
    col1, col2 = st.columns(2)
    col1.metric("Target Predictability (R²)", f"{bs.get('predictability_target_r2', 0):.1f}%")
    col2.metric("Path Predictability (R²)", f"{bs.get('predictability_path_r2', 0):.1f}%")

    # Comparison table
    st.markdown("##### Original vs. Orthogonalized Shocks")
    orth_h1 = bs.get("orthogonalized_h1", {})
    orth_h2 = bs.get("orthogonalized_h2_vw", {})

    comp_data = {
        "": ["β_Target", "p(Target)", "β_Path", "p(Path)"],
        "H1 Sentiment (Original)": [
            "0.000592", "0.012**", "0.000666", "0.131"
        ],
        "H1 Sentiment (Orthogonalized)": [
            f"{orth_h1.get('beta_target', 0):.6f}",
            f"{orth_h1.get('p_target', 1):.3f}" + ("**" if orth_h1.get('p_target', 1) < 0.05 else ""),
            f"{orth_h1.get('beta_path', 0):.6f}",
            f"{orth_h1.get('p_path', 1):.3f}",
        ],
        "H2 CRSP VW (Original)": [
            "-0.435", "0.043**", "-0.186", "0.443"
        ],
        "H2 CRSP VW (Orthogonalized)": [
            f"{orth_h2.get('beta_target', 0):.3f}",
            f"{orth_h2.get('p_target', 1):.3f}" + ("***" if orth_h2.get('p_target', 1) < 0.01 else ""),
            f"{orth_h2.get('beta_path', 0):.3f}",
            f"{orth_h2.get('p_path', 1):.3f}",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    st.warning("⚠️ Target loses significance for sentiment (p: 0.012→0.108) but **strengthens** for asset returns (p: 0.043→0.005)")
    st.info("🔑 **Asymmetry**: Predictable component attenuates sentiment but not returns → sentiment captures broader communication channel")

    # B-S figure
    bs_fig = os.path.join(FIG_DIR, "fig_bs_orthogonalization.png")
    if os.path.exists(bs_fig):
        st.image(bs_fig, caption="Bauer-Swanson Orthogonalization: Original vs. Orthogonalized", use_container_width=True)


def _render_ff_positioning():
    """Render Fernández-Fuertes (2025) comparison and positioning."""
    st.markdown("#### Fernández-Fuertes (2025): Complementary, Not Competing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ##### 🔵 FF: Better Shock Measure
        - Multi-agent LLM framework
        - Processes Statements, Minutes, Beige Books, Press Conferences
        - Constructs narrative monetary policy surprises
        - **His Table 32**: target β_T = 0.047***, path β_P ≈ 0
        - R² = 12.4% (vs. our 1.57%)
        - 88% of variance outside GSS span
        """)

    with col2:
        st.markdown("""
        ##### 🔴 Us: What Does Language Convey?
        - Different question, not different method
        - 4 structured hypotheses + Wald + FG interaction
        - Statement vs. Minutes channel comparison
        - Fully transparent & reproducible
        - No API dependencies
        - **Path significant in Minutes** (p=0.015), not Statements
        """)

    st.success("🔑 **FF's Table 32 is independent confirmation** of our core finding — "
               "both methods converge on target-dominant pattern. Complementary, not competing.")

    st.markdown("##### Our Three Distinguishing Contributions")
    st.markdown("""
    1. **Systematic hypothesis testing**: Wald test (β_T = β_P), FG interaction — FF does not conduct these
    2. **Communication channel differentiation**: Path shock significant in Minutes but not Statements — FF processes all docs through one pipeline
    3. **Transparency & reproducibility**: Dictionary + OLS + NW HAC — any researcher can replicate without API access
    """)


def _render_charts():
    """Render publication-quality charts."""
    # v10.3 figures
    v103_figs = [
        ("fig1_sentiment_ts.png", "Sentiment Time Series"),
        ("fig2_scatter.png", "Sentiment vs. Shocks Scatter"),
        ("fig3_h2.png", "Asset Returns Response"),
        ("fig_jk_decomposition.png", "JK Decomposition (NEW)"),
        ("fig_bs_orthogonalization.png", "B-S Orthogonalization (NEW)"),
        ("figure2_sentiment_vs_shocks_reproduced.png", "Sentiment vs. Shocks (Reproduced)"),
        ("figure3_asset_returns_reproduced.png", "Asset Returns (Reproduced)"),
    ]

    st.markdown("### v10.3 Figures")
    for fname, label in v103_figs:
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=label, use_container_width=True)

    # Legacy charts
    st.markdown("### Legacy Charts (v10.2)")
    legacy_charts = [
        ("fig1_framework.png", "Research Framework"),
        ("fig2_h1_scatter.png", "H1: Sentiment vs Shocks"),
        ("fig3_h2_returns.png", "H2: Asset Returns"),
        ("fig7_sentiment_by_regime.png", "Sentiment by Regime"),
        ("fig9_correlation_heatmap.png", "Correlation Matrix"),
    ]
    for fname, label in legacy_charts:
        fpath = os.path.join(CHARTS_DIR, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=label, use_container_width=True)
