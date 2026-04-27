# Abigail-Eileen Zhang’s Monetary Policy Research Lab

A distinctive research platform for studying how Federal Reserve announcements affect asset prices and portfolio reallocation.

**Live Demo**: Connect FRED API → Analyze → Export

## 🚀 Quick Start (3 ways)

### Option A: Local (no Docker)
```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

### Option B: Docker
```bash
cp .env.example .env   # Edit FRED_API_KEY (optional)
docker compose up -d
# Open http://localhost:8501
```

### Option C: Streamlit Community Cloud (Free, shareable URL)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect repo → deploy
4. Get `https://your-app.streamlit.app`

## 🔗 FRED API Setup (Free)

1. Get API key: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Enter it in the app (Data Explorer page) — **never stored, session-only**
3. Or set `FRED_API_KEY` env var for server deployment

**No API key?** The platform works fully with built-in demo data.

## 📊 Research Modules

| Module | What It Does | Data Source |
|--------|-------------|-------------|
| ⚡ Event Study | CAR around FOMC, multi-asset comparison | FRED / CSV / Demo |
| 🎯 Two-Shocks | Policy vs Information shock decomposition | FRED / Demo |
| 💬 Sentiment | NLP analysis of FOMC text | Fed website / Manual |
| 🔄 Capital Flow | Portfolio rebalancing, risk regime detection | FRED / Demo |
| 📚 Replication | One-click classic paper replication | Built-in |
| ⚙️ Data Explorer | FRED API, CSV import, data preview | FRED / CSV |
| 🧠 联邦智能分析 | HNSW向量库 · 联邦RAG/CoT/FL · 幻觉防御 · 审计链 | Built-in |

## 🎯 What Makes This Different

- **Two-Shocks Radar**: Decompose FOMC into policy vs information shocks
- **FOMC Sentiment Trajectory**: NLP scoring over time
- **Capital Flow Sankey**: Visualize portfolio rebalancing
- **Classic Paper Replication**: Kuttner 2001, Bernanke-Kuttner 2005, etc.
- **Real-time FRED**: 32 economic indicators, one-click fetch
- **联邦智能分析**: 自研HNSW向量库 · 五层幻觉防御 · 区块链审计（整合自 federated-ai-platform）

## 📁 Project Structure

```
monetary-policy-lab/
├── app.py                    # Streamlit entry point
├── Dockerfile                # Docker image
├── docker-compose.yml        # Docker orchestration
├── deploy.sh                 # One-click deploy script
├── requirements.txt          # Python dependencies
├── analysis/                 # Core analysis engines
│   ├── event_study.py        # Market model event study
│   ├── two_shocks.py         # Policy vs information decomposition
│   ├── nlp_engine.py         # FOMC sentiment (rule-based + FinBERT)
│   └── capital_flow.py       # Portfolio rebalancing analysis
├── data/                     # Data connectors
│   ├── fred_connector.py     # FRED API (32 series, caching, fallback)
│   └── fomc_scraper.py       # FOMC statement scraper
├── visualization/            # Plotly charts
│   └── charts.py             # All interactive visualizations
├── pages/                    # Streamlit pages
│   ├── dashboard.py          # Overview dashboard
│   ├── event_study.py        # Event study UI
│   ├── two_shocks.py         # Two-shocks UI
│   ├── sentiment.py          # Sentiment analysis UI
│   ├── capital_flow.py       # Capital flow UI
│   ├── replication.py        # Paper replication UI
│   └── data_explorer.py      # Data management UI
└── utils/                    # Utilities
    ├── constants.py          # FOMC dates, paper metadata, series map
    └── helpers.py            # Data generation, formatting
```

## 🔧 Data Sources

| Data | Source | Cost | Series |
|------|--------|------|--------|
| Interest rates, yields | FRED API | Free | DGS2, DGS10, DFF, SOFR... |
| Equity indices | FRED / Yahoo Finance | Free | SP500, NASDAQ... |
| FX, Commodities | FRED | Free | DEXUSEU, DCOILWTICO... |
| Inflation, Employment | FRED | Free | CPIAUCSL, UNRATE... |
| FOMC statements | federalreserve.gov | Free | Scraped |
| Fund holdings | CRSP/WRDS | University | Production only |

## 📚 Theoretical Foundation

- Rational Expectations (Lucas, 1972)
- Efficient Market Hypothesis (Fama, 1970)
- Present Value Decomposition (Campbell & Shiller, 1988)
- Portfolio Balance Channel (Tobin, 1969)
- Risk-Taking Channel (Borio & Zhu, 2012)
- Two-Shocks Framework (Jarociński & Karadi, 2020)

## 👤 Built For

Abigail-Eileen Zhang 's research on monetary policy announcements, asset prices, and portfolio reallocation.
