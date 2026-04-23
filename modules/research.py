"""
Phase 1 Research: Information Content of FOMC Language
======================================================
Research module for testing whether FOMC statement language
contains incremental information beyond interest rate decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import FOMC_DATES
from utils.helpers import generate_synthetic_returns


def _get_fed_chair(date_str: str) -> str:
    """Assign Fed Chair based on date."""
    from datetime import datetime
    d = datetime.strptime(date_str, "%Y-%m-%d")
    if d < pd.Timestamp("2006-02-01"):
        return "Greenspan"
    elif d < pd.Timestamp("2014-02-03"):
        return "Bernanke"
    elif d < pd.Timestamp("2018-02-05"):
        return "Yellen"
    else:
        return "Powell"


def render():
    # Lazy imports — only load when this page is actually visited
    from analysis.nlp_engine import FOMCSentimentEngine
    from analysis.surprise_calculator import SurpriseCalculator
    from analysis.regression_engine import RegressionEngine
    from analysis.two_shocks import TwoShocksDecomposer
    from data.fomc_scraper import FOMCScraper
    from data.fred_connector import FREDConnector
    from visualization.charts import (
        sentiment_vs_surprise_scatter,
        sentiment_trajectory_by_chair,
        incremental_r2_bar,
        regression_coefficient_plot,
        sentiment_trajectory,
    )

    st.markdown('<div class="main-header"><h1>🔬 Phase 1 Research</h1>'
                '<p>Information Content of FOMC Language</p></div>', unsafe_allow_html=True)

    st.markdown("""
    **Research Question:** Does FOMC statement language contain incremental information
    beyond interest rate decisions that explains asset price movements?

    **Hypotheses:**
    - H1: Language sentiment correlates with Kuttner surprise but is not collinear
    - H2: Sentiment has incremental explanatory power for asset prices (ΔR²)
    - H3: Sentiment maps to information shocks (Two-Shocks framework)
    - H4: Language matters more during Forward Guidance period (2008-2015)
    """)

    st.markdown("---")

    # ── Step 1: Data Collection ──
    st.markdown("### Step 1: Data Collection")

    tab1, tab2, tab3 = st.tabs(["FOMC Statements", "FRED Data", "Synthetic Demo"])

    with tab1:
        st.markdown("**FOMC Statement Scraper** — Fetch historical statements from the Fed website.")
        scraper = FOMCScraper()
        available = scraper.get_available_dates()
        st.success(f"✅ {len(available)} FOMC statements available (1994-2024)")

        if st.button("🔍 Preview Latest Statement", key="preview_stmt"):
            with st.spinner("Fetching..."):
                text = scraper.fetch_statement(available[-1])
                if text:
                    st.text_area(f"Statement: {available[-1]}", text[:2000], height=200)
                else:
                    st.warning("Could not fetch statement.")

    with tab2:
        st.markdown("**FRED Data** — Fed Funds Futures + Asset Prices")
        api_key = st.text_input("FRED API Key", value="", type="password", key="research_fred")
        if api_key:
            fred = FREDConnector(api_key=api_key)
            if st.button("📊 Fetch Data", key="fetch_research"):
                with st.spinner("Fetching from FRED..."):
                    df = fred.fetch_all(start="2020-01-01", end="2024-12-31")
                    if not df.empty:
                        st.success(f"✅ Fetched {len(df)} observations, {len(df.columns)} series")
                        st.dataframe(df.tail(5))
                    else:
                        st.warning("No data returned. Check API key.")

    with tab3:
        st.markdown("**Synthetic Demo** — Use generated data to test the pipeline.")
        if st.button("▶️ Run Full Pipeline (Synthetic)", key="run_pipeline"):
            _run_synthetic_pipeline()

    st.markdown("---")

    # ── Step 2: NLP Sentiment Analysis ──
    st.markdown("### Step 2: NLP Sentiment Analysis")

    engine = FOMCSentimentEngine()
    sample_texts = {
        "Hawkish": "The Committee remains attentive to inflationary pressures and "
                   "is prepared to take additional firming action if needed.",
        "Dovish": "The Committee anticipates that gradual adjustments in the stance "
                  "of monetary policy will be appropriate to support economic activity.",
        "Neutral": "The Committee decided to maintain the target range for the "
                   "federal funds rate at 5.25 to 5.50 percent.",
    }

    cols = st.columns(3)
    for i, (label, text) in enumerate(sample_texts.items()):
        with cols[i]:
            r = engine.analyze(text)
            color = "🔴" if r["label"] == "Hawkish" else ("🟢" if r["label"] == "Dovish" else "⚪")
            st.metric(f"{color} {label}", f"{r['sentiment_score']:.3f}")
            st.caption(f"Hawkish: {r['hawkish_found']} | Dovish: {r['dovish_found']}")

    st.markdown("---")

    # ── Step 3: Regression Analysis ──
    st.markdown("### Step 3: Regression Analysis Framework")

    st.markdown("""
    **Model 1 (H1):** `Sentiment = α + β₁ · Surprise + ε`
    **Model 2 (H2):** `Asset_Return = α + β₁ · Surprise + β₂ · Sentiment + ε`
    **Model 3 (H3):** `Sentiment = α + β₁ · Policy_Shock + β₂ · Info_Shock + ε`
    **Model 4 (H4):** `Asset_Return = α + β₁ · Surprise + β₂ · Sentiment + β₃ · (Sentiment × FG) + ε`
    """)

    st.info("💡 Connect real FRED data (Step 1) to run actual regressions. "
            "The Synthetic Demo tab shows the full pipeline with generated data.")


def _run_synthetic_pipeline():
    """Full pipeline with synthetic data for demonstration."""
    from visualization.charts import two_shocks_radar, two_shocks_bar

    st.markdown("#### Pipeline Output (Synthetic Data)")

    # 1. Generate synthetic data
    np.random.seed(42)
    n = 40  # ~40 FOMC meetings
    fomc_dates = sorted(FOMC_DATES[-n:])

    # Sentiment scores (correlated with but not equal to surprises)
    surprises = np.random.normal(0, 8, n)  # basis points
    sentiment = 0.03 * surprises + np.random.normal(0, 0.15, n)
    sentiment = np.clip(sentiment, -1, 1)

    # Asset returns on FOMC days
    equity_ret = -0.04 * surprises + 0.5 * sentiment + np.random.normal(0, 0.5, n)
    bond_ret = 0.02 * surprises + np.random.normal(0, 0.1, n)

    # Build dataset
    df = pd.DataFrame({
        "surprise_bp": surprises,
        "sentiment_score": sentiment,
        "S&P 500": equity_ret,
        "2Y Treasury": bond_ret,
        "fed_chair": [_get_fed_chair(d) for d in fomc_dates],
        "fg_period": [1 if pd.Timestamp(d).year <= 2015 else 0 for d in fomc_dates],
    }, index=[pd.Timestamp(d) for d in fomc_dates])

    st.dataframe(df.head(10).style.format("{:.3f}"), use_container_width=True)

    # 2. Scatter plot
    st.markdown("**H1 Test: Sentiment vs Surprise**")
    fig1 = sentiment_vs_surprise_scatter(df)
    st.plotly_chart(fig1, use_container_width=True)

    # 3. Trajectory by chair
    st.markdown("**Sentiment Trajectory by Fed Chair**")
    fig2 = sentiment_trajectory_by_chair(df)
    st.plotly_chart(fig2, use_container_width=True)

    # 4. Regression analysis
    st.markdown("**H2 Test: Incremental R²**")
    reg = RegressionEngine(df)

    inc_results = {}
    for asset in ["S&P 500", "2Y Treasury"]:
        inc = reg.incremental_r2(asset, ["surprise_bp"], ["surprise_bp", "sentiment_score"])
        if "error" not in inc:
            inc_results[asset] = inc

    if inc_results:
        fig3 = incremental_r2_bar(inc_results)
        st.plotly_chart(fig3, use_container_width=True)

        for asset, r in inc_results.items():
            st.markdown(f"**{asset}:** ΔR² = {r['incremental_r2']*100:.2f}%, "
                        f"F-stat = {r['f_stat']:.2f}, p = {r['p_value']:.4f}")

    # 5. Full regression table
    st.markdown("**Model 2: Full Regression Table**")
    for asset in ["S&P 500", "2Y Treasury"]:
        result = reg.ols(asset, ["surprise_bp", "sentiment_score"])
        if "error" not in result:
            table = reg.format_table(result, f"Dependent Variable: {asset}")
            st.dataframe(table, use_container_width=True)
            fig4 = regression_coefficient_plot(result, f"Coefficients: {asset}")
            st.plotly_chart(fig4, use_container_width=True)

    # 6. Two-Shocks linkage
    st.markdown("**H3 Test: Two-Shocks Linkage**")
    returns = generate_synthetic_returns()
    surprises_df = pd.DataFrame({"surprise": np.random.normal(0, 0.05, len(fomc_dates))},
                                 index=[pd.Timestamp(d) for d in fomc_dates])
    dec = TwoShocksDecomposer(surprises_df, returns)
    decomp = dec.simplified_decompose(equity_col="S&P 500", bond_col="US 10Y Treasury")
    if not decomp.empty:
        st.dataframe(decomp.head(10).style.format("{:.4f}"), use_container_width=True)

    st.success("✅ Full pipeline completed! Replace synthetic data with real FRED data for actual results.")
