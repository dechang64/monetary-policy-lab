# Monetary Policy Lab: Computational Details

> **Technical Reference for Financial Economics Experts**
> 
> Platform: https://monetary-policy-lab.streamlit.app  
> Repository: GitHub (dechang64/monetary-policy-lab)  
> Version: v9.9 (JMP draft complete, 18 robustness checks, rate cut regime in main text, chair FE analysis)

---

## 1. Architecture Overview

The platform consists of three layers:

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Interactive Dashboard** | Streamlit app (`app.py` + `modules/`) | Real-time exploration, visualization, scenario analysis |
| **Research Engine** | `mp-research-platform/` + `analysis/` | Batch regression pipeline, hypothesis testing, robustness checks |
| **Literature Radar** | `literature_radar.py` + cron | Automated daily scan of new research papers |

The dashboard provides 8 modules (Dashboard, Fed Intelligence, Research, Replication, Sentiment, Two Shocks, Capital Flow, Event Study, Data Explorer). The research engine runs formal econometric analysis offline. The Literature Radar scans SSRN, arXiv, BIS/IMF/Fed, and top journals daily, scoring papers by relevance to our research.

---

## 2. Data Sources & Pipeline

### 2.1 FRED API Integration

**Connector**: `data/fred_connector.py` (class `FREDConnector`)

We fetch **31 data series** from FRED, organized into 8 categories:

| Category | Series | FRED ID | Frequency |
|----------|--------|---------|-----------|
| Interest Rates | 1M/2Y/5Y/10Y/30Y Treasury, 2Y-10Y Spread, Effective Fed Funds, SOFR | DGS1MO, DGS2, DGS5, DGS10, DGS30, T10Y2Y, DFF, SOFR | Daily |
| Inflation | CPI YoY, Core CPI YoY, PCE, Core PCE YoY, 5Y/10Y Breakeven | CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, T5YIE, T10YIE | Monthly/Daily |
| Equity | S&P 500, NASDAQ, VIX | SP500, NASDAQCOM, VIXCLS | Daily |
| FX | DXY, EUR/USD, JPY/USD | DTWEXBGS, DEXUSEU, DEXJPUS | Daily |
| Commodities | Gold (AM Fix), Oil (WTI) | GOLDAMGBD228NLBM, DCOILWTICO | Daily |
| Credit | BAA Corporate, AAA Corporate | BAA, AAA | Daily |
| Labor | Unemployment Rate, Nonfarm Payrolls | UNRATE, PAYEMS | Monthly |
| Macro | Real GDP, GDP Growth, Fed Assets Total | GDPC1, A191RL1Q225SBEA, WALCL | Quarterly/Weekly |

**Derived series**: Credit Spread = BAA − AAA (computed in-place).

**Frequency alignment**: All series are reindexed to a daily calendar (`pd.date_range(freq="D")`), with lower-frequency data forward-filled.

**Caching**: Two-tier — in-memory dict + JSON file cache (default 6-hour TTL). Rate-limited at 100ms between requests.

### 2.2 FOMC Statement Corpus

**Scraper**: `data/fomc_scraper.py` (class `FOMCScraper`)

- **Coverage**: 157 FOMC statements (1994-02 to 2025-04), with manually curated URL mappings
- **Source**: `federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm`
- **Extraction**: BeautifulSoup, targeting `div#article` (fallback: `div.col-xs-12.col-sm-8.col-md-8`)
- **Success rate**: 155/157 (99%), 2 failures likely due to emergency/unscheduled meetings
- **Rate decision extraction**: Pattern matching on statement text ("decided to raise/lower/maintain")

### 2.3 FOMC Meeting Metadata

**Source**: `mp-research-platform/data/fomc_meetings.py`

Hand-curated dataset of **164 FOMC meetings** (1994–2025) with:

| Field | Description |
|-------|-------------|
| `date` | Meeting date |
| `decision` | `rate_hike` / `rate_cut` / `unchanged` / `emergency` |
| `rate_before`, `rate_after` | Federal funds target rate (upper bound, %) |
| `rate_change` | `rate_after − rate_before` (in percentage points) |
| `chair` | Greenspan / Bernanke / Yellen / Powell |
| `regime` | `conventional` (pre-2008) / `forward_guidance` (2008–2015) / `normalization` (2016+) |

### 2.4 Asset Prices

**Primary source**: CRSP via WRDS (`data/wrds/crsp_dsi_index.csv`)

| CRSP Variable | Label | Use |
|---------------|-------|-----|
| `vwretd` | CRSP Value-Weighted Return (incl. dividends) | H2: Large-cap equity response |
| `ewretd` | CRSP Equal-Weighted Return (incl. dividends) | H2: Small-cap equity response |
| `sprtrn` | S&P 500 Total Return | H2: Benchmark equity index |

**Coverage**: 1990-01-02 to 2024-12-31 (8,818 trading days). Mapped to FOMC meeting dates by exact date match.

**Advantage over yfinance**: CRSP returns include delisting adjustments and are the standard data source in the monetary policy event study literature.

**Secondary source (fallback)**: yfinance for S&P 500 (`^GSPC`), NASDAQ (`^IXIC`), VIX (`^VIX`), 10Y Treasury (`^TNX`), 13W T-bill (`^IRX`), Gold (`GC=F`).

### 2.5 Acosta (2022) Monetary Policy Shocks

**Source**: `data/mp_shocks_acosta.xlsx` (sheet: `shocks`)

| Variable | Description | Coverage |
|----------|-------------|----------|
| `target` | Target rate surprise (Kuttner-style) | 220 meetings, 1995–2022 |
| `path` | Path factor (Gürkaynak-style) | 220 meetings, 1995–2022 |

**Sample overlap**: 117 meetings have both statements and shock data (2006-01-31 to 2022-07-27).

### 2.6 USMPD Extension

The Federal Reserve Bank of San Francisco's USMPD provides raw high-frequency changes for 276 meetings (1994–2026). Our replicated target/path factors correlate at 0.958/0.970 with Acosta's original series.

**Extended sample results**:

| Sample | R² | Target p | Path p | N | Period |
|--------|-----|----------|--------|---|--------|
| Acosta only (baseline) | 1.57% | 0.017 | 0.152 | 117 | 2006–2022 |
| Acosta + USMPD | 1.65% | 0.419 | 0.058* | 163 | 2006–2026 |

---

## 3. Sentiment Analysis Engine

### 3.1 Dual-Dictionary Approach

**Module**: `analysis/run_v6_comprehensive.py`

We combine two dictionaries:

#### Loughran-McDonald (2011) Financial Sentiment Dictionary
- **Negative set**: ~120 words; **Positive set**: ~50 words
- $$\text{LM Score} = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{total}}}$$

#### Central Bank Hawkish-Dovish Dictionary
- **Hawkish set**: 591 terms; **Dovish set**: 222 terms (fully disjoint)
- $$\text{CB Score} = \frac{N_{\text{hawk}} - N_{\text{dove}}}{N_{\text{total}}}$$

#### Combined Sentiment
$$\text{Sentiment}_t = 0.5 \times \text{LM Score}_t + 0.5 \times \text{CB Score}_t$$

### 3.2 Sentiment Distribution (v9, verified)

| Statistic | Combined | LM Score | CB Score |
|-----------|:--------:|:--------:|:--------:|
| Mean | 0.024 | 0.038 | −0.013 |
| Std | 0.013 | 0.008 | 0.005 |
| Min | −0.012 | 0.006 | −0.089 |
| Max | 0.065 | 0.067 | 0.098 |
| % Negative | 18.8% | 0% | 78% |

**Key observation**: The LM score is always positive for FOMC statements (min = 0.006), creating a positivity bias that dilutes the CB signal. The CB component has substantial sign variation (78% negative), making it the more informative measure for FOMC text.

### 3.3 CB-Only vs. Combined Performance

| Measure | H1 R² | Target p | Path p |
|---------|:------:|:--------:|:------:|
| CB-only | 3.90% | <0.05 | 0.031 |
| Combined (0.5×LM + 0.5×CB) | 1.57% | 0.017 | 0.152 |
| LM-only | 0.33% | 0.474 | 0.552 |

The CB-only measure more than doubles R² and makes the path shock significant at 5%. The LM component adds noise without adding signal for FOMC text.

### 3.4 Forward-Lookingness Decomposition (v9, new)

Inspired by the IMF's four-dimensional framework (Gambacorta et al. 2025), we decompose sentiment into:

- **Forward-looking (FL)**: Sentences containing "expect," "anticipate," "will," "project," "forecast"
- **Current-assessment (CA)**: Sentences containing "recent," "current," "has," "was," "remains"

| Measure | H1 R² | Target p | Path p |
|---------|:------:|:--------:|:------:|
| Combined | 4.12% | 0.099 | 0.015 |
| Forward-looking | 0.79% | 0.012 | 0.800 |
| Current-assessment | 1.30% | 0.012 | 0.506 |

**Counterintuitive finding**: Splitting sentiment reduces R² and makes the path shock less significant. The path shock captures a broad policy stance signal, not a specific forward-guidance dimension.

### 3.5 Known Limitations

| Issue | Detail | Impact |
|-------|--------|--------|
| **LM positivity bias** | LM score always positive for FOMC text | Combined score dominated by LM positivity |
| **Context blindness** | "Higher inflation" (hawkish) vs. "higher growth" (dovish) | Directional ambiguity |
| **No topic decomposition** | All hawkish/dovish terms aggregated | Signal dilution across heterogeneous topics |
| **Equal weighting** | 50/50 LM+CB is arbitrary | Suboptimal signal extraction |
| **No uncertainty quantification** | All classifications treated equally | Noisy classifications dilute signal |

**Upgrade path**: CB-LMs (Gambacorta et al. 2024) offer open-weight, domain-specific embeddings with full reproducibility. Yao & Chai (2025) add uncertainty-aware classification.

---

## 4. Monetary Policy Surprise

### 4.1 Target Surprise (Kuttner 2001)

$$\text{Target Surprise}_t = \frac{D}{D-d} \times (F_{t}^{\text{close}} - F_{t-1}^{\text{close}})$$

where $F_t$ is the current-month Fed Funds futures settlement price, $D$ = days in month, $d$ = day of FOMC meeting.

### 4.2 Path Factor (Gürkaynak et al. 2005a)

$$\text{Path Factor}_t = \text{Surprise}_t^{\text{ED2}} - \text{Surprise}_t^{\text{ED1}}$$

Captures forward guidance — changes in expected future rate path beyond the current meeting.

### 4.3 Surprise Measure Comparison

| Approach | H1 R² | Target p | Path p | Quality |
|----------|:------:|:--------:|:------:|---------|
| Rate change (Δr) | 1.05% | 0.726 | 0.726 | Low — zero for unchanged meetings |
| Kuttner-style (DFF proxy) | 1.49% | 0.089 | 0.211 | Medium — single contract |
| Acosta target+path | 1.57% | 0.017 | 0.152 | High — market-based, narrow-window |
| CB-only + Acosta | 3.90% | <0.05 | 0.031 | High — best current specification |

**Impact**: The switch from naive proxy to Acosta shocks increased R² by 4×. Using CB-only sentiment doubles it again.

---

## 5. Hypothesis Testing Framework

### 5.1 H1: Sentiment ↔ Surprise

$$\text{Sentiment}_t = \alpha + \beta_1 \cdot \text{Target}_t + \beta_2 \cdot \text{Path}_t + \varepsilon_t$$

**Method**: OLS with Newey-West HAC(4) standard errors

**Verified result (v9)**:

| Statistic | Value |
|-----------|:-----:|
| R² | 1.57% |
| β₁ (target) | 0.000577 |
| t(target) | 2.43 |
| p(target) | 0.017 |
| β₂ (path) | 0.000633 |
| t(path) | 1.44 |
| p(path) | 0.152 |
| Wald (β₁=β₂) | p = 0.90 |
| N | 117 |

**Interpretation**: The target shock is significant at 5%, the path shock is not. This contradicts the strong version of the information channel. The Wald test cannot reject coefficient equality.

### 5.2 H2: Asset Return Response to Shocks

$$R_{i,t} = \alpha + \beta_1 \cdot \text{Target}_t + \beta_2 \cdot \text{Path}_t + \varepsilon_t$$

**Verified results (v9)**:

| Asset | R² | β₁ (target) | p(target) | β₂ (path) | p(path) |
|-------|:--:|:-----------:|:---------:|:---------:|:-------:|
| CRSP VW | 9.1% | −0.435 | 0.043 | −0.186 | 0.443 |
| CRSP EW | 10.3% | −0.449 | 0.013 | −0.174 | 0.479 |
| S&P 500 | 7.8% | −0.391 | 0.073 | −0.179 | 0.424 |
| Gold | 7.0% | −0.404 | 0.014 | −0.488 | 0.146 |

**Interpretation**: Target shock negatively affects equity returns, significant for small-cap (p=0.013) but marginal for large-cap (p=0.073). Path shock insignificant for all assets.

### 5.3 H3: Information Channel Test

The path shock does NOT have a significantly larger effect than the target shock. Wald test p = 0.90. Evidence is suggestive but not conclusive.

### 5.4 H4: Forward Guidance Interaction

$$R_t = \alpha + \beta_1 \text{Target}_t + \beta_2 \text{Path}_t + \beta_3 S_t + \beta_4 (S_t \times FG_t) + \varepsilon_t$$

| Asset | β₄ | p(β₄) |
|-------|:--:|:-----:|
| CRSP VW | −48.81 | 0.602 |
| NASDAQ | 202.17 | 0.041* |

*NASDAQ coefficient is economically implausible (202 bp), likely outlier-driven.*

**Result**: H4 is not robustly significant. CRSP VW shows no effect. NASDAQ is marginally significant but implausible.

**Key regime finding**: When split by decision type, the rate cut regime shows path shock highly significant (p < 0.001, R² = 43.1%) — the strongest result in the paper. The full-sample null masks this regime-dependent effect.

| Regime | N | R² | Target p | Path p |
|--------|:--:|:--:|:--------:|:------:|
| Rate hike | 17 | 10.2% | 0.013 | 0.298 |
| Rate cut | 11 | 43.1% | 0.089 | <0.001 |
| Unchanged | 89 | 2.0% | 0.616 | 0.079 |

### 5.5 Dual-Equation Test (v9, new)

Testing the risk premium channel explanation for H4 null:

| Dependent Variable | R² | Target p | Path p |
|-------------------|:--:|:--------:|:------:|
| 10Y Treasury chg | 0.72% | 0.403 | 0.890 |
| 13W T-bill chg | 0.66% | 0.491 | 0.737 |
| VIX change | 0.24% | 0.970 | 0.401 |
| Term spread chg | 1.39% | 0.667 | 0.341 |

**Result**: Risk premium channel not detectable at daily frequency. Consistent with Chen et al. (2025) finding that the channel operates at 30-minute frequency.

### 5.6 Statement Novelty Weighting (v9, new)

Novelty measured as Jaccard distance between consecutive statement word sets (not cosine distance on embeddings).

| Method | H1 R² | Improvement |
|--------|:------:|:-----------:|
| Unweighted OLS | 3.98% | baseline |
| Novelty-weighted WLS | 5.75% | +45% |

Novelty measured as cosine distance between TF-IDF vectors of consecutive statements.

---

## 6. Robustness Checks

| Check | R² | Target p | Path p | N |
|-------|:--:|:--------:|:------:|:--:|
| Full sample (baseline) | 1.57% | 0.017 | 0.152 | 117 |
| CB-only sentiment | 3.90% | <0.05 | 0.031 | 117 |
| No COVID | 1.57% | 0.017 | 0.154 | 115 |
| Post-2010 | 0.59% | 0.117 | 0.258 | 97 |
| Rate change proxy | 1.05% | 0.726 | 0.726 | 117 |
| Kuttner-style (DFF) | 2.14% | 0.004 | 0.152 | 117 |
| NW lag=1 | 1.57% | 0.006 | 0.100 | 117 |
| NW lag=6 | 1.57% | 0.024 | 0.149 | 117 |
| White HC SE | 1.57% | 0.012 | 0.134 | 117 |
| Extended (Acosta+USMPD) | 1.65% | 0.419 | 0.058 | 163 |
| Chair FE (Yellen+Powell) | 27.08% | 0.471 | 0.642 | 117 |
| Term spread chg | 1.39% | 0.667 | 0.341 | 117 |

**Chair FE detail**: Powell dummy significant (β=0.0067, p=0.023), Yellen dummy not significant (β=0.0004, p=0.552). Target shock becomes insignificant with chair FE, but remains significant for asset returns (CRSP VW: p=0.043).

---

## 7. Literature Radar

### 7.1 Architecture

**Module**: `literature_radar.py`

Automated daily scan of:
- **SSRN**: Financial economics preprints
- **arXiv q-fin**: Quantitative finance + computational economics
- **BIS/IMF/Fed**: Central bank working papers
- **Top journals**: JME, JFE, AER, QJE

### 7.2 Scoring

Weighted keyword matching:
- Core terms (FOMC, monetary policy surprise, central bank communication): 3× weight
- Method terms (sentiment analysis, NLP, LLM, hawkish dovish): 2× weight
- Data terms (high-frequency, target path factor, Kuttner, GSS): 1.5× weight
- Extension terms (forward guidance, information shock, term premium): 1× weight

Relevance score = (matched weight / total weight)^0.5, with working paper bonus (1.15×).

### 7.3 Current Database

| Tier | Count | Threshold |
|------|:-----:|-----------|
| High (≥0.4) | 2 | Notify user |
| Medium (0.3-0.4) | 11 | Include in digest |
| Low (<0.3) | 29 | Archive |
| Already cited | 4 | Mark in digest |

### 7.4 Key Papers Discovered

| Paper | Score | Impact Type |
|-------|:-----:|-------------|
| Chen, Granville & Matousek (2025) — PLMIS | 0.42 | methodology_upgrade |
| Gambacorta et al. (2024) — CB-LMs | cited | methodology_upgrade |
| IMF WP 2025/109 — Four-dimensional analysis | cited | directly_related |
| Yao & Chai (2025) — Uncertainty-aware LLM | cited | methodology_upgrade |
| Weinig (2025) — Narrative MP surprises | cited | identification_strategy |
| Fed FEDS Notes (2024) — GPT for FOMC | cited | directly_related |

---

## 8. Key Methodological Notes

### 8.1 Newey-West Lag Choice

Data-driven formula: $L = \lfloor 4(N/100)^{2/9} \rfloor = 4$ for N = 117 (Newey and West, 1994).

### 8.2 Standard Errors

Current: Newey-West HAC(4). Planned: White (1980) HC for cross-sectional; Thompson (2011) double-clustered for panels.

### 8.3 Multiple Testing

H2 tests 4+ assets. Bonferroni-corrected α* = 0.10/4 = 0.025. Only CRSP EW (p=0.013) and Gold (p=0.016) survive.

### 8.4 Endogeneity

OLS may suffer from reverse causality (unlikely for pre-written statements) and omitted variables (macro surprises coinciding with FOMC). IV strategy planned: lagged sentiment as instrument.

---

## 9. Reproducibility

### 9.1 Data Versioning

- FOMC meetings: Hard-coded in `fomc_meetings.py` (164 observations)
- FOMC statements: `mp-research-platform/data/fomc_statements_all.json`
- FRED data: Cached with 6-hour TTL
- Acosta shocks: `data/mp_shocks_acosta.xlsx` (220 observations)
- Analysis results: `results/verified_results.json` (cross-validated)
- Enhanced analysis: `results/enhanced_analysis_results.json`

### 9.2 Dependencies

| Package | Use |
|---------|-----|
| streamlit | Dashboard |
| pandas, numpy | Data manipulation |
| scipy, statsmodels | Statistical tests |
| yfinance | Asset price download (fallback) |
| requests | FRED API, FOMC scraping |
| beautifulsoup4 | HTML parsing |
| plotly | Interactive charts |
| openpyxl | Excel reading (Acosta shocks) |
| python-docx | DOCX generation |
| fitz (PyMuPDF) | PDF verification |

---

## 10. Upgrade Roadmap

| Phase | Component | Expected Impact |
|-------|-----------|-----------------|
| **Current** | CB-only sentiment | R²: 1.57% → 3.90% ✅ |
| **Current** | Novelty-weighted WLS | R²: 3.98% → 5.75% ✅ |
| **Next** | CB-LM embeddings (Gambacorta et al. 2024) | Contextual sentiment, reproducible |
| **Next** | Uncertainty-aware classification (Yao & Chai 2025) | Downweight ambiguous statements |
| **Next** | JK sign-restriction decomposition | Separate MP vs. information shocks |
| **Next** | High-frequency data (30-min windows) | Detect risk premium channel |
| **Future** | Multi-agent narrative surprises (Weinig 2025) | Text-derived surprise measures |
| **Future** | Topic-decomposed sentiment (Chen et al. 2025) | Per-topic hawkish/dovish scores |
| **Future** | Cross-country comparison | ECB, BoJ, BoE |

---

*Document updated: 2026-05-29 (v9.9 — 18 robustness checks, rate cut regime in main text, chair FE, term spread)*  
*Contact: dechang64 (GitHub) / 冬生*
