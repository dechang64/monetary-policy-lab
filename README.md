# Eileen Zhang's Monetary Policy Research Lab

A distinctive research platform for studying how Federal Reserve announcements affect asset prices and portfolio reallocation.

**Live Demo**: Connect FRED API → Analyze → Export

## 🚀 Quick Start (3 ways)

### Option A: Local (no Docker)
```bash
pip install -r requirements.txt
streamlit run app.py          # Chinese module
streamlit run app_en.py      # English module
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
| 🧠 Federated AI Intelligence | HNSW · FL · RAG · CoT · Hallucination Defense · Audit | Built-in |

## 🎯 What Makes This Different

- **Two-Shocks Radar**: Decompose FOMC into policy vs information shocks
- **FOMC Sentiment Trajectory**: NLP scoring over time
- **Capital Flow Sankey**: Visualize portfolio rebalancing
- **Classic Paper Replication**: Kuttner 2001, Bernanke-Kuttner 2005, etc.
- **Real-time FRED**: 32 economic indicators, one-click fetch
- **Federated AI Intelligence**: Self-built HNSW · Five-layer hallucination defense · Blockchain audit (integrated from federated-ai-platform)

## 🏛️ Federated AI Intelligence Module (NEW)

Built on the [federated-ai-platform](https://github.com/dechang64/federated-ai-platform):

| Sub-Module | Description |
|------------|-------------|
| 🔍 Vector Search | Self-built Rust HNSW vector DB for Fed statement retrieval, kNN < 1ms |
| 📚 Federated RAG | Cross-institution privacy-preserving document retrieval |
| 🧠 Federated CoT | Distributed chain-of-thought reasoning with multi-node voting |
| 🤝 Federated Learning | FedAvg collaborative modeling, data stays local |
| 🛡️ Hallucination Defense | 5-layer defense: Retrieval · CROWN · Multi-node vote · Self-consistency |
| 🤖 Agent | ReAct task orchestration for autonomous research |
| 🔗 Audit Chain | SHA-256 hash chain, tamper-proof operation log |

**Five-Layer Hallucination Defense:**
```
Layer 1 → Retrieval Consistency (Vector DB)
Layer 2 → Vector Fact-Check
Layer 3 → CROWN Conformity Defense (NeuroSync Original)
Layer 4 → Multi-Node Consensus Vote
Layer 5 → LLM Self-Consistency
```

## 📁 Project Structure

```
monetary-policy-lab/
├── app.py                    # Streamlit entry (Chinese)
├── app_en.py                 # Streamlit entry (English) ← NEW
├── Dockerfile                # Docker image
├── docker-compose.yml         # Docker orchestration
├── deploy.sh                 # One-click deploy script
├── requirements.txt          # Python dependencies
├── analysis/                 # Core analysis engines
│   ├── event_study.py        # Market model event study
│   ├── two_shocks.py         # Policy vs information decomposition
│   ├── nlp_engine.py         # FOMC sentiment (rule-based + FinBERT)
│   └── capital_flow.py       # Portfolio rebalancing analysis
├── modules/
│   ├── fed_intelligence.py    # 🇨🇳 Chinese version
│   ├── fed_intelligence_en.py # 🇺🇸 English version ← NEW
│   ├── dashboard.py
│   ├── research.py
│   └── ...
├── visualization/             # Plotly chart library
├── data/                     # FRED connector & scraper
└── utils/                    # Constants & helpers
```

## 📄 License

Apache-2.0
