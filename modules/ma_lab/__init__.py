"""
M&A Research Lab — Main Streamlit Module
==========================================
Renders 4 sub-modules: M&A Event Study, LBO Valuation, SEC EDGAR Search, M&A Literature.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.ma_lab.constants import MA_DEALS, LBO_CASES, MA_PAPERS, INDUSTRIES, LBO_DEFAULTS
from modules.ma_lab.lbo_model import LBOModel
from modules.ma_lab.event_study import MAEventStudyEngine
from utils.helpers import generate_synthetic_returns
from utils.constants import COLORS


def render():
    st.markdown(
        '<div class="main-header"><h1>🤝 M&A / LBO Research Lab</h1>'
        '<p>Merger announcements · Event study · LBO valuation · SEC EDGAR · Literature</p></div>',
        unsafe_allow_html=True,
    )
    
    sub = st.selectbox("Module", [
        "📊 M&A Deal Database",
        "⚡ M&A Event Study",
        "💰 LBO Valuation Engine",
        "🔍 SEC EDGAR Search",
        "📚 M&A Literature",
    ])
    
    if sub == "📊 M&A Deal Database":
        _render_deal_database()
    elif sub == "⚡ M&A Event Study":
        _render_event_study()
    elif sub == "💰 LBO Valuation Engine":
        _render_lbo()
    elif sub == "🔍 SEC EDGAR Search":
        _render_edgar()
    elif sub == "📚 M&A Literature":
        _render_literature()


def _render_deal_database():
    st.markdown("### 📊 M&A Deal Database")
    st.caption("Sample of notable M&A deals 2013-2024. In production: WRDS SDC Platinum.")
    
    df = pd.DataFrame(MA_DEALS)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        industry_filter = st.multiselect("Industry", INDUSTRIES, default=[])
    with col2:
        payment_filter = st.multiselect("Payment", ["cash", "stock", "mixed"], default=[])
    with col3:
        completed_only = st.checkbox("Completed deals only", value=False)
    
    filtered = df.copy()
    if industry_filter:
        filtered = filtered[filtered["industry"].isin(industry_filter)]
    if payment_filter:
        filtered = filtered[filtered["payment"].isin(payment_filter)]
    if completed_only:
        filtered = filtered[filtered["completed"] == True]
    
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    
    # Summary stats
    st.markdown("### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Deals", len(filtered))
    with col2:
        st.metric("Total Value ($B)", f"{filtered['deal_value_b'].sum():.1f}")
    with col3:
        st.metric("Avg Premium (%)", f"{filtered['premium_pct'].mean():.1f}")
    with col4:
        avg_lev = filtered[filtered["leverage_ratio"] > 0]["leverage_ratio"].mean()
        st.metric("Avg LBO Leverage (%)", f"{avg_lev*100:.1f}" if avg_lev > 0 else "N/A")
    
    # Chart: deal value over time
    try:
        import plotly.express as px
        df_chart = filtered.sort_values("date")
        fig = px.bar(df_chart, x="date", y="deal_value_b", color="industry",
                     title="Deal Value by Date", labels={"deal_value_b": "Deal Value ($B)"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")


def _render_event_study():
    st.markdown("### ⚡ M&A Event Study")
    st.caption("Compute abnormal returns around M&A announcement dates using market model.")
    
    st.info("""
    **Method**: Brown & Warner (1985) market model, same as FOMC event study.
    
    **Key difference**: Replace FOMC dates with M&A announcement dates.
    
    **Hypotheses**:
    - H1: Target CAR [-1, +1] > 0 (significant positive abnormal return)
    - H2: Acquirer CAR [-1, +1] < Target CAR (winner's curse)
    """)
    
    # Use M&A dates as events
    ma_dates = [d["date"] for d in MA_DEALS]
    ma_labels = [f"{d['acquirer']} → {d['target']}" for d in MA_DEALS]
    
    # Generate synthetic returns (in production: CRSP)
    returns = generate_synthetic_returns(n_days=2500)
    
    # Add synthetic M&A event effects: target +5%, acquirer -1%
    rng = np.random.default_rng(42)
    for i, d in enumerate(MA_DEALS):
        target_date = pd.Timestamp(d["date"])
        if target_date in returns.index:
            # Inject M&A effect on target proxies (we use existing columns as demo)
            for col in ["S&P 500", "NASDAQ"]:
                if col in returns.columns:
                    returns.loc[target_date, col] += rng.normal(0.02, 0.01)
    
    engine = MAEventStudyEngine(returns, ma_dates, labels=ma_labels)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pre_window = st.slider("Pre-Event Window (days)", 0, 10, 1)
    with col2:
        post_window = st.slider("Post-Event Window (days)", 1, 20, 1)
    with col3:
        assets = st.multiselect("Assets", returns.columns.tolist()[:6],
                                default=["S&P 500", "NASDAQ", "Russell 2000"])
    
    if st.button("🚀 Run M&A Event Study", type="primary", use_container_width=True):
        with st.spinner("Computing CAR around M&A announcements..."):
            summary = engine.cross_sectional(
                assets if assets else ["S&P 500", "NASDAQ"],
                market="S&P 500",
                event_window_pre=pre_window,
                event_window_post=post_window,
            )
            if summary.empty:
                st.warning("Insufficient data. Try a shorter window.")
            else:
                st.markdown("### Average CAR by Asset")
                st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
                
                # Visualization
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=summary["asset"], y=summary["avg_CAR_pct"],
                        marker_color=["#16a34a" if v > 0 else "#dc2626"
                                      for v in summary["avg_CAR_pct"]],
                        text=summary["avg_CAR_pct"].round(2).astype(str) + "%",
                        textposition="outside",
                    ))
                    fig.update_layout(title="Average CAR around M&A Announcements",
                                      xaxis_title="Asset", yaxis_title="CAR (%)")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"Chart: {e}")
                
                st.markdown("### 🔍 Key Findings")
                col1, col2 = st.columns(2)
                with col1:
                    best = summary.loc[summary["avg_CAR_pct"].idxmax()]
                    st.metric("Largest Positive CAR", f"{best['avg_CAR_pct']:+.3f}%", best["asset"])
                with col2:
                    worst = summary.loc[summary["avg_CAR_pct"].idxmin()]
                    st.metric("Largest Negative CAR", f"{worst['avg_CAR_pct']:+.3f}%", worst["asset"])
    
    with st.expander("📖 Methodology"):
        st.markdown("""
        **Market Model** (Brown & Warner, 1985)
        
        1. **Estimation**: $R_{it} = \\alpha_i + \\beta_i R_{mt} + \\epsilon_{it}$ (250 trading days)
        2. **Abnormal Return**: $AR_{it} = R_{it} - (\\hat{\\alpha}_i + \\hat{\\beta}_i R_{mt})$
        3. **CAR**: $CAR_i = \\sum_{t=-T_1}^{T_2} AR_{it}$
        4. **t-stat**: $t = CAR_i / (\\sigma_{AR} \\sqrt{N})$
        
        **Note**: Sample data is synthetic. In production use CRSP daily returns + WRDS SDC deal dates.
        
        **Reference**: Andrade, Mitchell & Stafford (2001), Moeller, Schlingemann & Stulz (2005)
        """)


def _render_lbo():
    st.markdown("### 💰 LBO Valuation Engine")
    st.caption("Interactive LBO model. Adjust assumptions to see IRR/MOIC impact.")
    
    mode = st.radio("Mode", ["Custom Assumptions", "Historical Cases"], horizontal=True)
    
    if mode == "Historical Cases":
        case_name = st.selectbox(
            "Select LBO case",
            [f"{c['name']} — {c['sponsor']} ({c.get('hold_years',5)}y hold)" for c in LBO_CASES]
        )
        case_idx = [f"{c['name']} — {c['sponsor']} ({c.get('hold_years',5)}y hold)"
                    for c in LBO_CASES].index(case_name)
        case = LBO_CASES[case_idx]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entry EV ($B)", case["entry_ev_b"])
            st.metric("Entry EBITDA ($B)", case["entry_ebitda_b"])
        with col2:
            st.metric("Debt %", f"{case['debt_pct']*100:.0f}%")
            st.metric("Hold (years)", case["hold_years"])
        with col3:
            st.metric("Exit EV ($B)", case["exit_ev_b"])
            moic = case["exit_ev_b"] / (case["entry_ev_b"] * (1 - case["debt_pct"])) if case["exit_ev_b"] > 0 else 0
            irr = (moic ** (1/case["hold_years"]) - 1) if moic > 0 else -1
            st.metric("MOIC", f"{moic:.2f}x")
            st.metric("IRR (%)", f"{irr*100:.1f}")
        
        if case["exit_ev_b"] == 0:
            st.error("⚠️ This LBO ended in bankruptcy/restructuring — equity wiped out.")
        
        st.markdown("**Background**: " + _get_case_background(case["name"]))
        return
    
    # Custom assumptions
    st.markdown("#### Entry Assumptions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        revenue = st.number_input("Entry Revenue ($B)", 0.1, 100.0, LBO_DEFAULTS["entry_revenue_b"], 0.1)
    with col2:
        ebitda_margin = st.slider("EBITDA Margin (%)", 5, 50, int(LBO_DEFAULTS["entry_ebitda_margin"]*100))
    with col3:
        entry_multiple = st.number_input("Entry EV/EBITDA", 3.0, 20.0, LBO_DEFAULTS["entry_ev_ebitda"], 0.5)
    with col4:
        debt_pct = st.slider("Debt % of EV", 0, 90, int(LBO_DEFAULTS["debt_pct"]*100))
    
    st.markdown("#### Operating Assumptions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        revenue_growth = st.slider("Revenue Growth (%/yr)", 0, 20, int(LBO_DEFAULTS["revenue_growth"]*100))
    with col2:
        margin_exp = st.slider("Margin Expansion (bp/yr)", 0, 200, int(LBO_DEFAULTS["margin_expansion"]*10000))
    with col3:
        interest_rate = st.slider("Interest Rate (%)", 1, 15, int(LBO_DEFAULTS["interest_rate"]*100))
    with col4:
        tax_rate = st.slider("Tax Rate (%)", 0, 40, int(LBO_DEFAULTS["tax_rate"]*100))
    
    st.markdown("#### Exit Assumptions")
    col1, col2 = st.columns(2)
    with col1:
        exit_multiple = st.number_input("Exit EV/EBITDA", 3.0, 20.0, LBO_DEFAULTS["exit_ev_ebitda"], 0.5)
    with col2:
        hold_years = st.slider("Hold Period (years)", 2, 10, LBO_DEFAULTS["hold_years"])
    
    assumptions = {
        "entry_revenue_b": revenue,
        "entry_ebitda_margin": ebitda_margin / 100,
        "entry_ev_ebitda": entry_multiple,
        "debt_pct": debt_pct / 100,
        "interest_rate": interest_rate / 100,
        "tax_rate": tax_rate / 100,
        "revenue_growth": revenue_growth / 100,
        "margin_expansion": margin_exp / 10000,
        "exit_ev_ebitda": exit_multiple,
        "hold_years": hold_years,
    }
    
    model = LBOModel(assumptions)
    df = model.project()
    
    st.markdown("### 📊 LBO Projection")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary
    s = model.summary
    st.markdown("### Returns Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Entry Equity ($B)", s["entry_equity_b"])
    with col2:
        st.metric("Exit Equity ($B)", s["exit_equity_b"])
    with col3:
        color = "off" if s["moic"] >= 2 else "normal"
        st.metric("MOIC", f"{s['moic']:.2f}x")
    with col4:
        st.metric("IRR (%)", f"{s['irr_pct']:.1f}")
    
    # Capital structure visualization
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Entry"], y=[s["entry_debt_b"]], name="Debt",
            marker_color="#dc2626", text=f"{s['entry_debt_b']:.1f}", textposition="inside"
        ))
        fig.add_trace(go.Bar(
            x=["Entry"], y=[s["entry_equity_b"]], name="Equity",
            marker_color="#16a34a", text=f"{s['entry_equity_b']:.1f}", textposition="inside"
        ))
        fig.add_trace(go.Bar(
            x=["Exit"], y=[s["exit_debt_b"]], name="Debt (Exit)",
            marker_color="#dc2626", text=f"{s['exit_debt_b']:.1f}", textposition="inside",
            showlegend=False
        ))
        fig.add_trace(go.Bar(
            x=["Exit"], y=[s["exit_equity_b"]], name="Equity (Exit)",
            marker_color="#16a34a", text=f"{s['exit_equity_b']:.1f}", textposition="inside",
            showlegend=False
        ))
        fig.update_layout(barmode="stack", title="Capital Structure: Entry vs Exit",
                          yaxis_title="$B")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Chart: {e}")
    
    # Sensitivity
    st.markdown("### IRR Sensitivity to Exit Multiple")
    try:
        import plotly.graph_objects as go
        mult_range = np.linspace(exit_multiple * 0.6, exit_multiple * 1.4, 9)
        irr_list = []
        for m in mult_range:
            a2 = assumptions.copy()
            a2["exit_ev_ebitda"] = m
            m2 = LBOModel(a2)
            m2.project()
            irr_list.append(m2.summary["irr_pct"])
        
        fig = go.Figure(go.Bar(x=mult_range, y=irr_list,
                                marker_color=["#dc2626" if v < 8 else "#f59e0b" if v < 15 else "#16a34a"
                                              for v in irr_list]))
        fig.update_layout(title="IRR vs Exit EV/EBITDA",
                          xaxis_title="Exit EV/EBITDA", yaxis_title="IRR (%)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _get_case_background(name: str) -> str:
    backgrounds = {
        "RJR Nabisco (1989)": "Legendary LBO documented in 'Barbarians at the Gate'. KKR won bidding war at $31.4B, 85% leverage. Breakup and debt burden led to underperformance; exited via IPO at loss on equity relative to entry.",
        "Dollar General (2009)": "KKR took Dollar General private at $7.3B in financial crisis. Operational improvements + deleveraging led to successful IPO in 2009. Considered one of KKR's best-performing LBOs of that vintage.",
        "First Data (2007)": "KKR acquired at peak market (29B). Crisis hit, leverage covenants strained; KKR injected additional equity in 2010. Restructured and eventually re-IPO'd at lower valuation.",
        "Hertz (2005)": "CD&R/Carlyle/MLIM consortium. Successful LBO with operational improvements; re-IPO'd in 2006. Later bankrupt in 2020 due to COVID travel collapse.",
        "Heinz (2013)": "Berkshire + 3G's zero-based budgeting approach. Operational efficiency gains drove margin expansion. Later merged with Kraft to form Kraft Heinz.",
        "Toys 'R' Us (2005)": "KKR/Bain/Vornado LBO at $6.6B. Debt service burden + Amazon competition led to bankruptcy in 2017, liquidation in 2018. Cautionary tale of LBO leverage risk.",
    }
    return backgrounds.get(name, "No background available.")


def _render_edgar():
    st.markdown("### 🔍 SEC EDGAR Full-Text Search")
    st.caption("Search SEC filings (8-K, DEFM14A, SC 13D) for M&A-related disclosures. Free, no API key.")
    
    from modules.ma_lab.sec_edgar import search_ma_announcements, search_defm14a
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", pd.Timestamp("2024-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.Timestamp("2024-12-31"))
    with col3:
        form_type = st.selectbox("Form Type", ["8-K (merger agreement)",
                                                "DEFM14A (merger proxy)",
                                                "SC 13D (5% stake)"])
    
    if st.button("🔍 Search EDGAR", type="primary", use_container_width=True):
        with st.spinner("Querying SEC EDGAR..."):
            try:
                if "8-K" in form_type:
                    results = search_ma_announcements(
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                elif "DEFM14A" in form_type:
                    from modules.ma_lab.sec_edgar import search_defm14a
                    results = search_defm14a(
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                else:
                    from modules.ma_lab.sec_edgar import search_filings
                    results = search_filings(
                        query="5% beneficial ownership",
                        form_type="SC 13D",
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                    )
                
                if results and "error" not in results[0]:
                    st.success(f"Found {len(results)} filings")
                    for r in results[:15]:
                        with st.expander(f"📄 {r.get('company','—')} — {r.get('filing_date','—')} [{r.get('form','')}]"):
                            st.write(f"**Title**: {r.get('title','—')}")
                            st.write(f"**Form**: {r.get('form','—')}")
                            st.write(f"**Filed**: {r.get('filing_date','—')}")
                            st.write(f"**CIK**: {r.get('cik','—')}")
                            st.markdown(f"[🔗 View on EDGAR]({r.get('url','')})")
                else:
                    st.warning("No results or error connecting to EDGAR. Try a different date range.")
                    if results:
                        st.caption(f"Debug: {results[0].get('error','')}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.markdown("#### About SEC EDGAR")
    st.info("""
    **EDGAR Full-Text Search** (https://efts.sec.gov/LATEST/search-index) is a free public API
    that searches the full text of SEC filings since 2001.
    
    **M&A-relevant forms**:
    - **8-K**: Current report — filed within 4 business days of material events like merger agreements
    - **DEFM14A**: Definitive merger proxy statement — sent to shareholders for vote
    - **SC 13D**: Schedule 13D — filed when acquiring 5%+ of a public company
    
    **Rate limit**: 10 requests/second. **Cost**: Free.
    """)


def _render_literature():
    st.markdown("### 📚 M&A / LBO Literature")
    st.caption("Classic papers for replication and theoretical grounding.")
    
    for name, info in MA_PAPERS.items():
        with st.expander(f"📄 {name} — {info['title']}"):
            st.write(f"**Journal**: {info['journal']}")
            st.write(f"**Method**: {info['method']}")
            st.write(f"**Key Result**: {info['key_result']}")
    
    st.markdown("---")
    st.markdown("#### Research Questions for This Lab")
    st.markdown("""
    **H1 (M&A Announcement Effect)**:
    Target-firm shareholders earn significantly positive CAR in [-1, +1] window
    around announcement. Acquirer CAR ≈ 0 or slightly negative.
    
    **H2 (Winner's Curse)**:
    Competitive auctions / multiple bidders → higher target premium but lower
    acquirer return. Roll (1986) hubris hypothesis.
    
    **H3 (LBO Leverage-Returns Relationship)**:
    Non-monotonic: moderate leverage (~60-70%) → highest IRR via tax shield +
    discipline; excessive leverage (>80%) → distress costs dominate.
    Axelson et al. (2013) "Borrow Cheap, Buy High".
    
    **H4 (Merger Waves & Macro)**:
    M&A activity clusters in waves (Harford, 2005), correlated with capital
    liquidity, regulatory shocks, and industry-specific shocks.
    Connects to monetary policy transmission (links to Eileen's existing proposal).
    """)
    
    st.markdown("#### Connection to Monetary Policy Lab")
    st.info("""
    The Kuttner/Bauer-Swanson/JK framework from Eileen's existing proposal
    applies directly to H4:
    
    - **Identification**: Monetary policy surprises (path shocks) → cost of capital →
      M&A activity cluster
    - **Variables**: Replace "asset returns" with "industry M&A volume"
    - **Standard errors**: NW HAC(4) + Thompson double-clustered (same as proposal)
    
    This makes the M&A Lab a natural extension of the existing platform.
    """)
