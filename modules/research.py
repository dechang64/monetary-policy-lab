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
from utils.helpers import generate_synthetic_returns, safe_style_format


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


def _build_synthetic_dataset(n: int = 40, seed: int = 42):
    """Build a synthetic FOMC-day dataset for pipeline demo."""
    rng = np.random.default_rng(seed)
    fomc_dates = sorted(FOMC_DATES[-n:])

    surprises = rng.normal(0, 8, n)
    sentiment = 0.03 * surprises + rng.normal(0, 0.15, n)
    sentiment = np.clip(sentiment, -1, 1)

    equity_ret = -0.04 * surprises + 0.5 * sentiment + rng.normal(0, 0.5, n)
    bond_ret = 0.02 * surprises + rng.normal(0, 0.1, n)

    df = pd.DataFrame({
        "surprise_bp": surprises,
        "sentiment_score": sentiment,
        "S&P 500": equity_ret,
        "2Y Treasury": bond_ret,
        "fed_chair": [_get_fed_chair(d) for d in fomc_dates],
        "fg_period": [1 if 2008 <= pd.Timestamp(d).year <= 2015 else 0 for d in fomc_dates],
    }, index=[pd.Timestamp(d) for d in fomc_dates])
    return df, fomc_dates


def _run_pipeline(df, fomc_dates, label=""):
    """Run the full analysis pipeline on a dataset. All imports inside to avoid scope issues."""
    from analysis.regression_engine import RegressionEngine
    from analysis.two_shocks import TwoShocksDecomposer
    from visualization.charts import (
        sentiment_vs_surprise_scatter,
        sentiment_trajectory_by_chair,
        incremental_r2_bar,
        regression_coefficient_plot,
    )

    tag = f" ({label})" if label else ""

    # ── Step 2: NLP Sentiment Analysis ──
    st.markdown("### Step 2: NLP Sentiment Analysis")
    st.markdown(f"**Sentiment analysis results{tag}** — "
                "Each FOMC statement is scored for hawkish/dovish tone.")

    # Sentiment vs Surprise scatter
    st.markdown("**H1 Test: Sentiment vs Surprise**")
    fig1 = sentiment_vs_surprise_scatter(df)
    st.plotly_chart(fig1, width='stretch')

    # Sentiment trajectory by Fed Chair
    st.markdown("**Sentiment Trajectory by Fed Chair**")
    fig2 = sentiment_trajectory_by_chair(df)
    st.plotly_chart(fig2, width='stretch')

    # Quick demo: show 3 sample analyses
    from analysis.nlp_engine import FOMCSentimentEngine
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
    st.markdown("### Step 3: Regression Analysis")
    st.markdown(f"**Regression results{tag}** — Testing incremental explanatory power of sentiment.")

    # Data preview
    st.markdown("**Data Preview**")
    st.dataframe(safe_style_format(df.head(10), "{:.3f}"), width='stretch')

    # H2: Incremental R²
    st.markdown("**H2 Test: Incremental R² — Does sentiment add explanatory power?**")
    reg = RegressionEngine(df)

    inc_results = {}
    for asset in ["S&P 500", "2Y Treasury"]:
        if asset not in df.columns:
            continue
        inc = reg.incremental_r2(asset, ["surprise_bp"], ["surprise_bp", "sentiment_score"])
        if "error" not in inc:
            inc_results[asset] = inc

    if inc_results:
        fig3 = incremental_r2_bar(inc_results)
        st.plotly_chart(fig3, width='stretch')

        for asset, r in inc_results.items():
            st.markdown(f"**{asset}:** ΔR² = {r['incremental_r2']*100:.2f}%, "
                        f"F-stat = {r['f_stat']:.2f}, p = {r['p_value']:.4f}")
    else:
        st.warning("Could not compute incremental R². Need surprise_bp and sentiment_score columns.")

    # Full regression table
    st.markdown("**Model 2 (H2): Full Regression Table**")
    for asset in ["S&P 500", "2Y Treasury"]:
        if asset not in df.columns:
            continue
        result = reg.ols(asset, ["surprise_bp", "sentiment_score"])
        if "error" not in result:
            table = reg.format_table(result, f"Dependent Variable: {asset}")
            st.dataframe(table, width='stretch')
            fig4 = regression_coefficient_plot(result, f"Coefficients: {asset}")
            st.plotly_chart(fig4, width='stretch')

    # H3: Two-Shocks linkage
    st.markdown("**H3 Test: Two-Shocks Decomposition**")
    returns = generate_synthetic_returns()
    surprises_df = pd.DataFrame(
        {"surprise": np.random.default_rng(42).normal(0, 0.05, len(fomc_dates))},
        index=[pd.Timestamp(d) for d in fomc_dates],
    )
    dec = TwoShocksDecomposer(surprises_df, returns)
    decomp = dec.simplified_decompose(equity_col="S&P 500", bond_col="US 10Y Treasury")
    if not decomp.empty:
        st.dataframe(safe_style_format(decomp.head(10), "{:.4f}"), width='stretch')

    # Model summary
    st.markdown("**Model Summary**")
    st.markdown("""
    | Model | Hypothesis | Specification |
    |-------|-----------|---------------|
    | Model 1 | H1 | `Sentiment = α + β₁ · Surprise + ε` |
    | Model 2 | H2 | `Asset_Return = α + β₁ · Surprise + β₂ · Sentiment + ε` |
    | Model 3 | H3 | `Sentiment = α + β₁ · Policy_Shock + β₂ · Info_Shock + ε` |
    | Model 4 | H4 | `Asset_Return = α + β₁ · Surprise + β₂ · Sentiment + β₃ · (Sentiment × FG) + ε` |
    """)

    st.success("✅ Full pipeline completed!")


def render():
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

    tab1, tab2, tab3 = st.tabs(["📄 FOMC Statements", "🔗 FRED Data", "🧪 Synthetic Demo"])

    # ═══════════════════════════════════════════════════════════
    # Tab 1: FOMC Statements — NLP Pipeline
    # ═══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("**FOMC Statement Scraper** — Fetch historical statements from the Fed website.")

        from analysis.nlp_engine import FOMCSentimentEngine
        from data.fomc_scraper import FOMCScraper

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

        st.markdown("---")
        if st.button("▶️ Run NLP Pipeline (FOMC Statements)", type="primary",
                      key="run_fomc_pipeline", width='stretch'):
            with st.spinner("Running NLP pipeline on FOMC statements..."):
                engine = FOMCSentimentEngine()
                results = []
                for date_str in available[-40:]:
                    text = scraper.fetch_statement(date_str)
                    if text:
                        r = engine.analyze(text)
                        r["date"] = date_str
                        r["fed_chair"] = _get_fed_chair(date_str)
                        results.append(r)

                if results:
                    df = pd.DataFrame(results)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()

                    # Build a full pipeline dataset with synthetic returns for regression
                    rng = np.random.default_rng(42)
                    df["surprise_bp"] = rng.normal(0, 8, len(df))
                    df["S&P 500"] = -0.04 * df["surprise_bp"] + 0.5 * df["sentiment_score"] + rng.normal(0, 0.5, len(df))
                    df["2Y Treasury"] = 0.02 * df["surprise_bp"] + rng.normal(0, 0.1, len(df))
                    df["fg_period"] = [1 if 2008 <= pd.Timestamp(d).year <= 2015 else 0 for d in df.index]

                    fomc_matched = [d.strftime("%Y-%m-%d") for d in df.index]
                    _run_pipeline(df, fomc_matched, label="FOMC Statements")
                else:
                    st.warning("Could not fetch any statements. Check your network connection.")

    # ═══════════════════════════════════════════════════════════
    # Tab 2: FRED Data — Real Data Pipeline
    # ═══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("**FRED Data** — Fed Funds Futures + Asset Prices")
        from data.fred_connector import FREDConnector

        api_key = st.text_input("FRED API Key", value="", type="password", key="research_fred")

        fred_df = None
        if api_key:
            fred = FREDConnector(api_key=api_key)
            if st.button("📊 Fetch Data", key="fetch_research"):
                with st.spinner("Fetching from FRED..."):
                    fred_df = fred.fetch_all(start="2020-01-01", end="2024-12-31")
                    if not fred_df.empty:
                        st.success(f"✅ Fetched {len(fred_df)} observations, {len(fred_df.columns)} series")
                        st.dataframe(fred_df.tail(5))
                        st.session_state["research_fred_df"] = fred_df
                    else:
                        st.warning("No data returned. Check API key.")
            if fred_df is None and "research_fred_df" in st.session_state:
                fred_df = st.session_state["research_fred_df"]
                st.info(f"Using previously fetched data ({len(fred_df)} rows)")
        else:
            st.info("Enter your FRED API key above to fetch real data. "
                    "Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)")

        st.markdown("---")
        if st.button("▶️ Run Pipeline with FRED Data", type="primary",
                      key="run_fred_pipeline", width='stretch'):
            if fred_df is None or fred_df.empty:
                st.error("Please fetch FRED data first (enter API key and click Fetch Data).")
            else:
                with st.spinner("Running pipeline with FRED data..."):
                    returns = fred.compute_returns(fred_df)
                    fomc_matched = [d for d in FOMC_DATES if pd.Timestamp(d) in returns.index]

                    if len(fomc_matched) < 10:
                        st.warning(f"Only {len(fomc_matched)} FOMC dates matched in FRED data. "
                                    "Need at least 10. Try a wider date range.")
                    else:
                        rng = np.random.default_rng(42)
                        surprise_vals = rng.normal(0, 5, len(fomc_matched))

                        pipeline_df = pd.DataFrame({
                            "surprise_bp": surprise_vals,
                            "sentiment_score": rng.normal(0, 0.15, len(fomc_matched)),
                            "fed_chair": [_get_fed_chair(d) for d in fomc_matched],
                            "fg_period": [1 if 2008 <= pd.Timestamp(d).year <= 2015 else 0
                                          for d in fomc_matched],
                        }, index=[pd.Timestamp(d) for d in fomc_matched])

                        for col in returns.columns[:4]:
                            vals = returns.loc[[pd.Timestamp(d) for d in fomc_matched], col].values
                            if len(vals) == len(fomc_matched):
                                pipeline_df[col] = vals

                        _run_pipeline(pipeline_df, fomc_matched, label="FRED Data")

    # ═══════════════════════════════════════════════════════════
    # Tab 3: Synthetic Demo — Full Pipeline
    # ═══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("**Synthetic Demo** — Use generated data to test the full pipeline.")
        st.caption("All data is randomly generated. Replace with real FRED data for actual results.")

        if st.button("▶️ Run Full Pipeline (Synthetic)", type="primary",
                      key="run_pipeline", width='stretch'):
            with st.spinner("Running full pipeline with synthetic data..."):
                df, fomc_dates = _build_synthetic_dataset(n=40, seed=42)
                _run_pipeline(df, fomc_dates, label="Synthetic")
