"""
Data Explorer Page — with real FRED API integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.constants import FOMC_DATES
from utils.helpers import generate_synthetic_returns
from data.fred_connector import FREDConnector


def render():
    st.markdown(
        '<div class="main-header"><h1>⚙️ Data Explorer</h1>'
        '<p>Connect FRED API · Import data · Explore FOMC datasets</p></div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔗 FRED API (Live Data)",
        "📊 Demo Data",
        "📄 FOMC Statements",
        "📤 Import CSV",
    ])

    # ═══════════════════════════════════════════════════════════
    # Tab 1: FRED API — Real Data
    # ═══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Connect to FRED API")
        st.caption(
            "FRED provides 800,000+ economic data series for free. "
            "Get your API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)"
        )

        # ── API Key Input ──
        api_key = st.text_input(
            "FRED API Key",
            type="password",
            placeholder="Enter your FRED API key (stored in session only, never saved)",
            help="Free API key from https://fred.stlouisfed.org/docs/api/api_key.html",
        )

        if api_key:
            st.session_state["fred_api_key"] = api_key

        # ── Date Range ──
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.Timestamp("2015-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.Timestamp("2024-12-31"))

        # ── Series Selection ──
        st.markdown("#### Select Data Series")

        # Grouped selection
        groups = {
            "📈 Interest Rates": [
                "Fed Funds Rate", "2Y Treasury", "5Y Treasury",
                "10Y Treasury", "30Y Treasury", "2Y-10Y Spread",
            ],
            "📊 Equity Markets": [
                "S&P 500", "NASDAQ", "VIX",
            ],
            "💱 FX": [
                "DXY", "EUR/USD", "JPY/USD",
            ],
            "🛢️ Commodities": [
                "Gold", "Oil (WTI)",
            ],
            "💳 Credit": [
                "AAA Corporate", "BAA Corporate", "Credit Spread",
            ],
            "💰 Inflation": [
                "CPI YoY", "Core CPI YoY", "Breakeven 10Y",
            ],
            "🏦 Fed Balance Sheet": [
                "Fed Assets Total", "Reserve Balances",
            ],
            "📋 Labor Market": [
                "Unemployment Rate", "Nonfarm Payrolls",
            ],
        }

        selected_series = []
        for group_name, series_list in groups.items():
            with st.expander(group_name, expanded=(group_name == "📈 Interest Rates")):
                cols = st.columns(min(len(series_list), 3))
                for i, series in enumerate(series_list):
                    with cols[i % len(cols)]:
                        if st.checkbox(series, value=(series in [
                            "Fed Funds Rate", "10Y Treasury", "S&P 500",
                            "DXY", "Gold", "VIX",
                        ])):
                            selected_series.append(series)

        st.markdown(f"**{len(selected_series)} series selected**")

        # ── Fetch Button ──
        if st.button("🚀 Fetch Data from FRED", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your FRED API key first.")
                return

            with st.spinner("Connecting to FRED API..."):
                fred = FREDConnector(api_key=api_key)

                # Test connection
                if not fred.test_connection():
                    st.error(
                        "❌ Could not connect to FRED API. "
                        "Please check your API key."
                    )
                    return

                st.success("✅ Connected to FRED API!")

                # Fetch data
                with st.spinner(f"Fetching {len(selected_series)} series..."):
                    df = fred.fetch_all(
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        series_names=selected_series,
                    )

                if df.empty:
                    st.warning("No data returned. Try a different date range or series.")
                    return

                # Store in session
                st.session_state["fred_data"] = df
                st.session_state["fred_levels"] = df.copy()
                st.session_state["fred_returns"] = fred.compute_returns(df)
                st.session_state["data_loaded"] = True

                st.success(
                    f"✅ Loaded {len(df)} observations × {len(df.columns)} series"
                )

        # ── Display FRED Data ──
        if "fred_data" in st.session_state:
            df = st.session_state["fred_data"]

            view_mode = st.radio("View", ["Price Levels", "Daily Returns"], horizontal=True)
            display_df = st.session_state["fred_returns"] if view_mode == "Daily Returns" else df

            st.markdown("#### Data Preview")
            st.dataframe(
                display_df.tail(30).style.format("{:.4f}").background_gradient(
                    cmap="RdBu_r", vmin=-0.03, vmax=0.03, axis=0
                ),
                use_container_width=True,
                height=400,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Summary Statistics")
                st.dataframe(
                    display_df.describe().T.style.format("{:.4f}"),
                    use_container_width=True,
                )
            with col2:
                st.markdown("#### Correlation Matrix")
                st.dataframe(
                    display_df.corr().style.format("{:.3f}").background_gradient(
                        cmap="RdBu_r", vmin=-1, vmax=1
                    ),
                    use_container_width=True,
                )

            # Quick plot
            st.markdown("#### Time Series")
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = go.Figure()
            for col in display_df.columns[:6]:  # Plot up to 6 series
                fig.add_trace(go.Scatter(
                    x=display_df.index,
                    y=display_df[col],
                    name=col,
                    mode="lines",
                ))
            fig.update_layout(
                template="plotly_white",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # FOMC event windows
            st.markdown("#### FOMC Event Windows")
            if st.button("Extract FOMC Event Windows"):
                fred = FREDConnector(api_key=st.session_state.get("fred_api_key", ""))
                windows = fred.get_fomc_event_windows(
                    FOMC_DATES, display_df,
                    window_pre=1, window_post=5,
                )
                st.session_state["fomc_windows"] = windows
                st.success(f"Extracted {len(windows)} event windows")

    # ═══════════════════════════════════════════════════════════
    # Tab 2: Demo Data
    # ═══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Built-in Demo Dataset")
        st.caption("Synthetic daily returns for 12 asset classes (2015-2024)")

        returns = generate_synthetic_returns()

        st.dataframe(
            returns.head(20).style.format("{:.4f}"),
            use_container_width=True,
            height=400,
        )

        st.markdown(f"**{len(returns)} trading days** | **{len(returns.columns)} assets**")

        if st.button("Use Demo Data for Analysis", use_container_width=True):
            st.session_state["fred_returns"] = returns
            st.session_state["fred_levels"] = returns.cumsum().apply(np.exp)
            st.session_state["data_loaded"] = True
            st.success("✅ Demo data loaded! Navigate to other modules to analyze.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Summary Statistics")
            st.dataframe(
                returns.describe().T.style.format("{:.4f}"),
                use_container_width=True,
            )
        with col2:
            st.markdown("#### Correlation Matrix")
            st.dataframe(
                returns.corr().style.format("{:.3f}").background_gradient(
                    cmap="RdBu_r", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )

    # ═══════════════════════════════════════════════════════════
    # Tab 3: FOMC Statements
    # ═══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### FOMC Statements")
        st.caption("Fetch historical FOMC statements from federalreserve.gov")

        from data.fomc_scraper import FOMCScraper

        scraper = FOMCScraper()
        available = scraper.get_available_dates()

        st.markdown(f"**{len(available)} statements available** (2020-2024)")

        selected_date = st.selectbox("Select FOMC Date", available)

        if st.button("Fetch Statement", use_container_width=True):
            with st.spinner("Fetching from federalreserve.gov..."):
                text = scraper.fetch_statement(selected_date)

            if text:
                st.session_state["fomc_statement"] = text
                st.session_state["fomc_statement_date"] = selected_date
                st.success(f"✅ Fetched statement ({len(text)} characters)")
            else:
                st.error("Could not fetch statement. Try a different date.")

        if "fomc_statement" in st.session_state:
            st.markdown("#### Statement Text")
            st.text_area(
                "FOMC Statement",
                value=st.session_state["fomc_statement"],
                height=400,
                disabled=True,
            )

            # Quick sentiment preview
            from analysis.nlp_engine import FOMCSentimentEngine
            engine = FOMCSentimentEngine()
            result = engine.analyze(st.session_state["fomc_statement"])

            col1, col2, col3 = st.columns(3)
            with col1:
                label = result["sentiment_label"]
                color = "#e74c3c" if label == "Hawkish" else ("#27ae60" if label == "Dovish" else "#95a5a6")
                st.markdown(
                    f'<div style="text-align:center;padding:1rem;background:white;'
                    f'border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                    f'<div style="font-size:0.8rem;color:#888;">Sentiment</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:{color};">{label}</div>'
                    f'<div style="font-size:0.7rem;color:#888;">Score: {result["sentiment_score"]:.3f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div style="text-align:center;padding:1rem;background:white;'
                    f'border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                    f'<div style="font-size:0.8rem;color:#888;">Hawkish Keywords</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:#e74c3c;">'
                    f'{len(result["hawkish_found"])}</div></div>',
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f'<div style="text-align:center;padding:1rem;background:white;'
                    f'border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                    f'<div style="font-size:0.8rem;color:#888;">Dovish Keywords</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:#27ae60;">'
                    f'{len(result["dovish_found"])}</div></div>',
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════════════════════════════════════
    # Tab 4: Import CSV
    # ═══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### Import Your Own Data")

        uploaded = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="First column = dates, remaining columns = asset returns/prices",
        )

        if uploaded:
            try:
                df = pd.read_csv(uploaded, parse_dates=[0], index_col=0)
                st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
                st.dataframe(df.head(10), use_container_width=True)

                st.markdown("#### Data Summary")
                st.dataframe(df.describe().T.style.format("{:.4f}"), use_container_width=True)

                st.session_state["fred_returns"] = df.pct_change().iloc[1:]
                st.session_state["fred_levels"] = df
                st.session_state["data_loaded"] = True
                st.info("✅ Data loaded! Navigate to other modules to analyze.")

            except Exception as e:
                st.error(f"Error reading file: {e}")

        st.markdown("---")
        st.markdown("#### Expected CSV Format")
        st.code(
            """Date,S&P 500,NASDAQ,US 10Y,Gold,DXY
2015-01-02,2058.20,4726.81,2.12,1184.25,90.35
2015-01-05,2020.58,4663.63,2.04,1192.75,91.20
...""",
            language="text",
        )
