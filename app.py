"""
Abigail-Eileen Zhang’s Research Lab
=============================
A distinctive research platform for studying how monetary policy announcements
affect asset prices and portfolio reallocation.

Unique Features:
- Two-Shocks Radar: Decompose FOMC into policy vs information shocks
- FOMC Sentiment Trajectory: FinBERT-FOMC scoring over time
- Capital Flow Sankey: Portfolio rebalancing visualization
- Classic Paper Replication Lab
- Real-time FRED API integration

Deployment:
  Docker:     docker compose up --build
  Streamlit:  streamlit run app.py
  Cloud:      Push to GitHub → deploy on Streamlit Community Cloud
"""

import streamlit as st
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from utils.constants import FOMC_DATES, PAPERS

# ── Page Config ──
st.set_page_config(
    page_title="Abigail-Eileen Zhang's Research Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
        padding: 2rem 2.5rem !important;
        border-radius: 12px !important;
        margin-bottom: 2rem !important;
        color: white !important;
    }
    .main-header h1 { font-size: 2rem !important; font-weight: 700 !important; margin: 0 0 0.5rem 0 !important; color: white !important; }
    .main-header p { font-size: 0.95rem !important; opacity: 0.8 !important; margin: 0 !important; color: white !important; }
    .metric-card {
        background: white !important; border-radius: 10px !important; padding: 1.2rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; border-left: 4px solid !important;
    }
    .metric-card.policy { border-color: #e74c3c !important; }
    .metric-card.info { border-color: #3498db !important; }
    .metric-card.success { border-color: #27ae60 !important; }
    .metric-card.warning { border-color: #f39c12 !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: #E8E8E8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Initialize Session State ──
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "data_source" not in st.session_state:
    st.session_state.data_source = "demo"
if "fred_returns" not in st.session_state:
    st.session_state.fred_returns = None

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>📊 Abigail-Eileen Zhang's Research Lab</h1>
    <p>Federal Reserve Announcements · Asset Prices · Portfolio Reallocation · NLP Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Data Source Indicator ──
if st.session_state.data_loaded:
    source = st.session_state.data_source
    icon = "🔗" if source == "fred" else ("📄" if source == "csv" else "📊")
    st.sidebar.success(f"{icon} Data loaded: **{source.upper()}**")
else:
    st.sidebar.info("📊 Using demo data. Go to **Data Explorer** to connect FRED or import CSV.")

# ── Sidebar Navigation ──
st.sidebar.title("🔬 Research Modules")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "⚡ Event Study Engine",
        "🎯 Two-Shocks Decomposition",
        "💬 FOMC Sentiment Analysis",
        "🔄 Capital Flow Analysis",
        "📚 Paper Replication Lab",
        "⚙️ Data Explorer",
        "🔬 Phase 1 Research",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This Platform**
Built for studying how monetary policy announcements reshape asset prices and portfolio allocation.

**Unique Features:**
- 🔴🔵 Two-Shocks Radar
- 📈 Sentiment Trajectory
- 🌊 Capital Flow Sankey
- 📋 Classic Paper Replication
- 🔗 Real-time FRED Data
""")

# ── Route to Pages ──
if page == "🏠 Dashboard":
    from modules import dashboard
    dashboard.render()
elif page == "⚡ Event Study Engine":
    from modules import event_study
    event_study.render()
elif page == "🎯 Two-Shocks Decomposition":
    from modules import two_shocks
    two_shocks.render()
elif page == "💬 FOMC Sentiment Analysis":
    from modules import sentiment
    sentiment.render()
elif page == "🔄 Capital Flow Analysis":
    from modules import capital_flow
    capital_flow.render()
elif page == "📚 Paper Replication Lab":
    from modules import replication
    replication.render()
elif page == "⚙️ Data Explorer":
    from modules import data_explorer
    data_explorer.render()
elif page == "🔬 Phase 1 Research":
    from modules import research
    research.render()
