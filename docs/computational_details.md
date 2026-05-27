# Monetary Policy Lab: Computational Details

> **Technical Reference for Financial Economics Experts**
> 
> Platform: https://monetary-policy-lab.streamlit.app  
> Repository: GitHub (dechang64/monetary-policy-lab)  
> Version: v1.0 (Phase 1 complete, WRDS integration pending)

---

## 1. Architecture Overview

The platform consists of two layers:

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Interactive Dashboard** | Streamlit app (`app.py` + `modules/`) | Real-time exploration, visualization, scenario analysis |
| **Research Engine** | `mp-research-platform/` | Batch regression pipeline, hypothesis testing, robustness checks |

The dashboard provides 8 modules (Dashboard, Fed Intelligence, Research, Replication, Sentiment, Two Shocks, Capital Flow, Event Study, Data Explorer), while the research engine runs the formal econometric analysis offline and stores results in `results/`.

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

**Frequency alignment**: All series are reindexed to a daily calendar (`pd.date_range(freq="D")`), with lower-frequency data forward-filled. This means monthly CPI/PCE values are carried forward within each month — appropriate for event-window analysis where the timing precision is daily.

**Caching**: Two-tier — in-memory dict + JSON file cache (default 6-hour TTL). Rate-limited at 100ms between requests.

### 2.2 FOMC Statement Corpus

**Scraper**: `data/fomc_scraper.py` (class `FOMCScraper`)

- **Coverage**: 157 FOMC statements (1994-02 to 2025-04), with manually curated URL mappings
- **Source**: `federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm`
- **Extraction**: BeautifulSoup, targeting `div#article` (fallback: `div.col-xs-12.col-sm-8.col-md-8`)
- **Success rate**: 155/157 (99%), 2 failures likely due to emergency/unscheduled meetings with non-standard URLs
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

### 2.4 Asset Prices (yfinance)

**Source**: `mp-research-platform/run_analysis_v4.py`

For the research pipeline, we use yfinance to download:

| Ticker | Label | Use |
|--------|-------|-----|
| `^GSPC` | S&P 500 | Equity return |
| `^IXIC` | NASDAQ | Tech equity return |
| `^VIX` | VIX | Volatility control |
| `^TNX` | 10Y Treasury | Long-rate change |
| `^IRX` | 13W T-bill | Short-rate change |
| `GC=F` | Gold | Safe-haven asset |

**Note**: yfinance returns MultiIndex columns in newer versions; we flatten via `df.columns.get_level_values(0)`. The 13W T-bill (`^IRX`) has lower precision than FRED's DGS3MO — a known limitation documented in TOOLS.md.

### 2.5 WRDS (Planned, Not Yet Active)

**Connector**: `data/wrds_connector.py` (class `WRDSConnector`)

Designed for:
- **CME Fed Funds futures** (`cme.ff`): Kuttner (2001) surprise calculation
- **CME Eurodollar futures** (`cme.ef`): Gürkaynak et al. (2005) path factor
- **CRSP daily stock returns** (`crsp.dsf`): Delisted-return-adjusted equity data
- **TAQ intraday trades** (`taqmsec.ctm_*`): High-frequency identification
- **OptionMetrics** (`optionm.opprcd*`): Implied volatility surface

**Current status**: WRDS requires Duo MFA authentication. Connection flow: `wrds.Connection()` → Duo Approve → 30-day MFA exemption. Not yet operational.

---

## 3. Sentiment Analysis Engine

### 3.1 Dictionary-Based Approach

**Module**: `mp-research-platform/data/sentiment.py` (function `compute_lm_sentiment`)

We use a **dual-dictionary** approach combining:

#### Loughran-McDonald (2011) Financial Sentiment Dictionary

- **Negative set**: ~120 words (e.g., "adverse", "concern", "deteriorate", "recession", "volatile")
- **Positive set**: ~130 words (e.g., "achieve", "boost", "confident", "growth", "stable")

$$\text{LM Score} = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{total}}}$$

#### Central Bank Hawkish-Dovish Dictionary (Henry 2008 + custom)

- **Hawkish set**: ~18 words/bigrams (e.g., "tighten", "hike", "inflation", "vigilant", "restrictive")
- **Dovish set**: ~18 words/bigrams (e.g., "accommodative", "patient", "gradual", "easing", "transitory")

$$\text{CB Score} = \frac{N_{\text{hawk}} - N_{\text{dove}}}{N_{\text{total}}}$$

#### Combined Sentiment

$$\text{Sentiment}_t = 0.5 \times \text{LM Score}_t + 0.5 \times \text{CB Score}_t$$

**Interpretation**: Higher values → more hawkish tone.

**Preprocessing**: Lowercase → strip punctuation → filter words with length ≤ 1.

### 3.2 Known Limitations

| Issue | Detail | Impact |
|-------|--------|--------|
| **Sparsity** | LM dictionary designed for 10-K filings, not FOMC statements | std(sentiment) ≈ 0.003 vs. literature benchmark ≈ 0.035 |
| **Context blindness** | "higher" appears in both LM_POSITIVE and CB_HAWKISH — but "higher inflation" is hawkish while "higher growth" is dovish | Directional ambiguity |
| **No bigram/trigram** | Only unigram matching; "strongly committed to returning inflation to 2%" misses the commitment signal | Information loss |
| **Equal weighting** | 50/50 LM+CB is arbitrary; no empirical optimization | Suboptimal signal extraction |

**Planned improvements** (per WRDS_UPGRADE_PLAN.md):
1. **FinBERT** (Huang et al. 2022): Contextual sentiment, requires GPU
2. **Expanded CB dictionary**: Add FOMC-specific phrases from Apel & Blix Grimaldi (2012)
3. **Weight optimization**: Cross-validate LM/CB weights against asset returns

---

## 4. Monetary Policy Surprise

### 4.1 Current Implementation (Proxy)

**Module**: `mp-research-platform/run_analysis_v4.py` (function `compute_surprises`)

In the absence of CME futures data, we use:

$$\text{Surprise}_t = \Delta r_t = r_t^{\text{after}} - r_t^{\text{before}}$$

where $r_t$ is the Federal Funds target rate. This is a **naive proxy** — it equals zero for all "unchanged" meetings, which constitute the majority of the sample.

### 4.2 Target Implementation: Kuttner (2001)

Once WRDS is operational, the correct surprise measure is:

$$\text{Target Surprise}_t = \frac{F_{t}^{\text{close}} - F_{t-1}^{\text{close}}}{1 - D \cdot F_{t}^{\text{close}} / 100}$$

where:
- $F_t^{\text{close}}$ = settlement price of the current-month Fed Funds futures contract on FOMC day
- $F_{t-1}^{\text{close}}$ = same contract's settlement price on the day before
- $D$ = day-of-month (scaling factor for within-month timing)

For scheduled meetings, this captures the **unexpected component** of the rate decision.

### 4.3 Path Factor: Gürkaynak et al. (2005)

$$\text{Path Factor}_t = \text{Surprise}_t^{\text{ED2}} - \text{Surprise}_t^{\text{ED1}}$$

where $\text{ED1}$, $\text{ED2}$ are the first and second Eurodollar futures surprises. This captures **forward guidance** — changes in expected future rate path beyond the current meeting.

---

## 5. Hypothesis Testing Framework

### 5.1 H1: Sentiment ↔ Surprise

**Test**: Does FOMC statement sentiment predict the monetary policy surprise?

$$\text{Sentiment}_t = \alpha + \beta \cdot \text{Surprise}_t + \varepsilon_t$$

**Method**: OLS via `scipy.stats.linregress`

**Current result**: R² ≈ 0.39% (vs. literature benchmark ~2.76%)

**Diagnosis**: The near-zero R² is primarily driven by the naive surprise proxy (all "unchanged" meetings have Surprise = 0, compressing variance). With Kuttner futures-based surprises, we expect R² to increase substantially.

### 5.2 H2: Incremental Predictive Power of Sentiment

**Test**: Does sentiment explain asset returns beyond what surprise alone captures?

**Model 1** (baseline):
$$R_{i,t} = \alpha + \beta_1 \cdot \text{Surprise}_t + \varepsilon_t$$

**Model 2** (augmented):
$$R_{i,t} = \alpha + \beta_1 \cdot \text{Surprise}_t + \beta_2 \cdot \text{Sentiment}_t + \varepsilon_t$$

**Key statistic**: $\beta_2$ (sentiment coefficient) and $\Delta R^2 = R^2_2 - R^2_1$

**Inference**: OLS with heteroskedasticity-robust standard errors (HC0 via $(X'X)^{-1} \hat{\sigma}^2$):

$$\text{Var}(\hat{\beta}) = \hat{\sigma}^2 (X'X)^{-1}, \quad \hat{\sigma}^2 = \frac{\sum \hat{\varepsilon}_i^2}{n - k}$$

$$t = \frac{\hat{\beta}_2}{\text{SE}(\hat{\beta}_2)}, \quad p = 2 \cdot (1 - F_{t_{n-k}}(|t|))$$

**Assets tested**: S&P 500, NASDAQ, Gold (percentage returns); 10Y Yield, 13W T-bill (basis point changes)

**Current results**:

| Asset | $\hat{\beta}_2$ | $p$-value | $\Delta R^2$ | Significant? |
|-------|:---:|:---:|:---:|:---:|
| S&P 500 | — | — | — | No |
| NASDAQ | — | — | — | No |
| Gold | — | 0.087 | — | Marginal* |
| 10Y Yield | — | — | — | No |
| 13W T-bill | — | — | — | No |

*Gold at p=0.087 is a potentially novel finding not reported in the reference paper.

### 5.3 H3: Two-Shocks Decomposition

**Test**: Decompose the FOMC announcement effect into a **policy shock** and an **information shock**.

**Method** (following Jarociński & Karadi 2020; Nakamura & Steinsson 2018):

1. **Standardize** all variables via z-score:
$$z_t = \frac{x_t - \bar{x}}{s_x}$$

2. **Policy shock** = standardized rate change:
$$\text{Policy}_t = z(\Delta r_t)$$

3. **Information shock** = residual from regressing surprise on rate change:
$$\text{Info}_t = z(\text{Surprise}_t) - \hat{\gamma} \cdot z(\Delta r_t)$$
where $\hat{\gamma}$ is from OLS: $z(\text{Surprise}) = \gamma \cdot z(\Delta r) + u$

4. **Loadings**: Regress sentiment on each shock separately:
$$\text{Sentiment}_t = \alpha + \lambda_P \cdot \text{Policy}_t + e_t$$
$$\text{Sentiment}_t = \alpha + \lambda_I \cdot \text{Info}_t + e_t$$

5. **Share decomposition**:
$$\text{Policy Share} = \frac{|\lambda_P|}{|\lambda_P| + |\lambda_I|}, \quad \text{Info Share} = \frac{|\lambda_I|}{|\lambda_P| + |\lambda_I|}$$

**Current result**: Information shock dominates (~99.6% of sentiment variation), consistent with the reference paper's finding of 97.2%.

**Critical caveat**: The current implementation uses `Surprise = rate_change` (naive proxy), which means the "information shock" residual captures essentially all variation for unchanged-rate meetings. This inflates the information share. With Kuttner surprises, the decomposition will be more meaningful.

### 5.4 H4: Regime-Dependent Effects

**Test**: Does the sentiment-return relationship vary across monetary policy regimes?

**Regimes**:
- **Conventional** (pre-2008): Standard rate adjustments
- **Forward Guidance** (2008–2015): ZLB period with explicit guidance
- **Normalization** (2016+): Rate hikes from ZLB

**Method**: Run H2 regression (13W T-bill return ~ Surprise + Sentiment) separately within each regime subsample. Compare $|\hat{\beta}_2|$ across regimes.

**Hypothesis**: Forward guidance regime should exhibit the strongest sentiment effect, because when rates are at ZLB, the statement's forward guidance content is the primary policy tool.

---

## 6. Event Study Engine

### 6.1 Market Model

**Module**: `analysis/event_study.py` (class `EventStudyEngine`)

For each asset $i$ and FOMC event $t$:

**Step 1 — Parameter estimation** (non-event days, trailing 250 trading days):

$$R_{i,\tau} = \hat{\alpha}_i + \hat{\beta}_i \cdot R_{m,\tau} + \varepsilon_{i,\tau}$$

via OLS (`scipy.stats.linregress`), where $R_m$ is the market return (S&P 500 or NASDAQ as alternative market proxy).

**Step 2 — Abnormal returns** (event window $[t-k, t+l]$):

$$\text{AR}_{i,\tau} = R_{i,\tau} - (\hat{\alpha}_i + \hat{\beta}_i \cdot R_{m,\tau})$$

**Step 3 — Cumulative abnormal return**:

$$\text{CAR}_i = \sum_{\tau = t-k}^{t+l} \text{AR}_{i,\tau}$$

**Step 4 — Significance test**:

$$t = \frac{\text{CAR}_i}{\hat{\sigma} \cdot \sqrt{L}}$$

where $\hat{\sigma}$ is the residual standard deviation from the estimation window, and $L$ is the number of days in the event window.

**Default windows**: Estimation = 250 days; Event = $[-1, +5]$ (6 trading days).

### 6.2 Cross-Sectional Aggregation

$$\text{AAR}_t = \frac{1}{N} \sum_{i=1}^{N} \text{AR}_{i,t}, \quad \text{CAAR}_t = \sum_{\tau} \text{AAR}_\tau$$

Aggregated across all assets for each FOMC date.

### 6.3 Asset-Level Summary

For each asset, average CAR across all FOMC events:

| Statistic | Definition |
|-----------|------------|
| avg_CAR | Mean CAR across events |
| median_CAR | Median CAR |
| std_CAR | Standard deviation of CAR |
| pct_positive | % of events with CAR > 0 |

---

## 7. Capital Flow Analysis

### 7.1 Flow Estimation

**Module**: `analysis/capital_flow.py` (class `CapitalFlowAnalyzer`)

Following Ciminelli et al. (2022), we use **return differentials** as a proxy for capital flow direction:

$$\text{Flow}_{i,t} = \bar{R}_{i,\text{post}} - \bar{R}_{i,\text{pre}}$$

where:
- $\bar{R}_{i,\text{pre}}$ = mean daily return for asset $i$ over $[t-5, t-1]$
- $\bar{R}_{i,\text{post}}$ = mean daily return for asset $i$ over $[t+1, t+10]$

**Asset class mapping**: 9 classes (US Large Cap, US Tech, US Small Cap, EM, Treasuries, Corporate Bonds, Commodities, FX, Crypto) mapped to individual series.

### 7.2 Risk Regime Detection

$$\text{Risk Spread}_t = \bar{R}_{\text{Risk-On},t} - \bar{R}_{\text{Risk-Off},t}$$

- **Risk-On**: US Large Cap, US Tech, US Small Cap, EM, Crypto
- **Risk-Off**: US Treasuries, Corporate Bonds

Regime = "Risk-On" if Risk Spread > 0, "Risk-Off" otherwise. A **regime change** occurs when the post-FOMC regime differs from the pre-FOMC regime.

### 7.3 Cross-Asset Correlation Change

$$\bar{\rho}_{\text{pre}} = \frac{\sum_{i \neq j} \rho_{ij}^{\text{pre}}}{N(N-1)}, \quad \bar{\rho}_{\text{post}} = \frac{\sum_{i \neq j} \rho_{ij}^{\text{post}}}{N(N-1)}$$

An increase in average pairwise correlation post-FOMC suggests **herding behavior** — investors move in the same direction across asset classes in response to the announcement.

**Window**: 30 trading days pre/post.

---

## 8. Robustness Checks

### 8.1 Chair Fixed Effects

Run H2 regression separately for each Fed Chair's tenure (Greenspan, Bernanke, Yellen, Powell). Compare $\hat{\beta}_2$ across chairs to test whether the sentiment effect is driven by a particular chair's communication style.

### 8.2 Subsample: Post-2010

Restrict to meetings from 2010 onward, when FOMC statement communication became more standardized (longer statements, explicit forward guidance language, press conferences).

### 8.3 Exclude COVID Period

Drop meetings from March 2020 – December 2021 to remove the extreme volatility and emergency actions during the pandemic.

### 8.4 Full Asset Coverage

Run H2 across all 5 assets (S&P 500, NASDAQ, Gold, 10Y Yield, 13W T-bill) with the full sample, reporting $\hat{\beta}_2$, SE, $t$, $p$, and $R^2$ for each.

---

## 9. Dashboard-Specific Modules

### 9.1 Fed Intelligence (NLP)

**Module**: `modules/analyzers.py` (class `NLPEngine`)

- **Sentiment**: Hawkish/dovish word counting with partial string matching (`any(h in w for h in HAWKISH_WORDS)`)
- **Readability**: Simplified Flesch Reading Ease score:
$$\text{Flesch} = 206.835 - 1.015 \cdot \frac{\text{words}}{\text{sentences}} - 84.6 \cdot \frac{\text{syllables}}{\text{words}}$$
- **Diff analysis**: Set-difference of word tokens between two statements, filtered to words > 4 characters

### 9.2 Two Shocks (Dashboard Visualization)

**Module**: `modules/analyzers.py` (class `TwoShocksEngine`)

The dashboard version uses **simulated** variance decomposition with empirical priors:

| Asset | Policy % | Info % |
|-------|:--------:|:------:|
| S&P 500 | 55 | 45 |
| 10Y Treasury | 70 | 30 |
| DXY | 65 | 35 |
| VIX | 40 | 60 |
| Gold | 50 | 50 |

With Gaussian noise ($\sigma = 3$) added and renormalized to 100%. These are **illustrative** and will be replaced by actual regression-based decomposition once WRDS data is available.

### 9.3 Portfolio Rebalancing Simulation

**Module**: `modules/analyzers.py` (class `PortfolioEngine`)

5 investor types × 6 shock scenarios, with pre-specified allocation shifts:

- **Base allocations**: Mutual Funds, Hedge Funds, Pension Funds, Foreign Investors, Retail
- **Shock scenarios**: ±25bp, ±50bp rate surprise; strong/weak economy information shock
- **Method**: Add shock effect to base allocation, clip at 0, renormalize to 100%

This is a **heuristic simulation**, not estimated from data. It serves as a pedagogical tool for understanding directional capital flow patterns.

---

## 10. Key Methodological Notes & Limitations

### 10.1 Surprise Measurement

| Approach | Data Required | Status | Quality |
|----------|---------------|--------|---------|
| $\Delta r_t$ (rate change) | FOMC records | ✅ Active | Low — zero for unchanged meetings |
| Kuttner (2001) target surprise | CME FF futures | ⏳ WRDS pending | High — market-based expectation |
| Gürkaynak et al. (2005) path factor | CME ED futures | ⏳ WRDS pending | High — captures forward guidance |

**Impact**: The naive surprise proxy is the single largest source of weak results in H1–H3. With futures-based surprises, we expect:
- H1 R² to increase from ~0.4% to ~2–5%
- H2 significance to improve across assets
- H3 decomposition to become more balanced (less inflated info share)

### 10.2 Sentiment Measurement

The dictionary approach has known limitations (see §3.2). The most impactful upgrade path:

1. **Short-term**: Expand CB dictionary with FOMC-specific bigrams from Apel & Blix Grimaldi (2012)
2. **Medium-term**: Implement FinBERT for contextual sentiment (requires GPU)
3. **Long-term**: Fine-tune a BERT model on FOMC statements with hawkish/dovish labels

### 10.3 Standard Errors

Current implementation uses **homoskedastic** standard errors (OLS variance formula). For financial time series with well-documented heteroskedasticity and potential autocorrelation, the appropriate upgrade is:

- **Newey-West** HAC standard errors for time-series regressions
- **White (1980)** heteroskedasticity-robust SE for cross-sectional regressions
- **Thompson (2011)** double-clustered SE (by time and by asset) for panel regressions

### 10.4 Multiple Testing

H2 tests 5 assets simultaneously. Without correction, the probability of at least one false positive at $\alpha = 0.10$ is $1 - 0.90^5 = 41\%$. Appropriate corrections:
- **Bonferroni**: $\alpha^* = 0.10 / 5 = 0.02$
- **Holm-Bonferroni**: Step-down procedure, less conservative
- **Benjamini-Hochberg**: Controls FDR at 10%

### 10.5 Endogeneity

The OLS regression $R_t = \alpha + \beta_1 \text{Surprise}_t + \beta_2 \text{Sentiment}_t + \varepsilon_t$ may suffer from:
- **Reverse causality**: Market reactions could feed back into statement drafting (unlikely for pre-written statements)
- **Omitted variables**: Macro surprises (NFP, CPI releases) coinciding with FOMC meetings
- **Measurement error**: Dictionary sentiment is a noisy proxy for true policy tone

**IV strategy** (planned): Use the previous meeting's sentiment as an instrument for current sentiment, exploiting the autocorrelation in communication style while assuming past sentiment doesn't directly affect current returns.

---

## 11. Reproducibility

### 11.1 Data Versioning

- FOMC meetings: Hard-coded in `fomc_meetings.py` (164 observations, 1994–2025)
- FOMC statements: Cached as individual `.txt` files in `data/cache/fomc/`
- FRED data: Cached as JSON in `data/cache/` with 6-hour TTL
- Analysis results: Saved as CSV (`analysis_dataset_expanded.csv`) and JSON (`regression_results_expanded.json`)

### 11.2 Randomness

- `TwoShocksEngine.variance_decomposition()`: Uses `np.random.normal(0, 3)` for simulation noise — **not seeded**. Results vary across runs.
- All other modules are deterministic given the same input data.

### 11.3 Dependencies

| Package | Version | Use |
|---------|---------|-----|
| streamlit | — | Dashboard |
| pandas, numpy | — | Data manipulation |
| scipy | — | Statistical tests |
| yfinance | — | Asset price download |
| requests | — | FRED API, FOMC scraping |
| beautifulsoup4 | — | HTML parsing |
| plotly | — | Interactive charts |

---

## 12. Upgrade Roadmap

| Phase | Component | Data Source | Expected Impact |
|-------|-----------|-------------|-----------------|
| **Phase 2** | Kuttner surprise | CME FF futures (WRDS) | H1 R²: 0.4% → 2–5% |
| **Phase 2** | Path factor | CME ED futures (WRDS) | H3 decomposition validity |
| **Phase 2** | HAC standard errors | — | Correct inference |
| **Phase 3** | FinBERT sentiment | GPU compute | Sentiment std: 0.003 → 0.02+ |
| **Phase 3** | High-frequency identification | TAQ (WRDS) | Intraday event windows |
| **Phase 3** | IV estimation | — | Address endogeneity |
| **Phase 4** | Sign restriction (JK style) | — | Structural shock identification |
| **Phase 4** | Panel regression with double-clustering | — | Efficient estimation |

---

*Document generated: 2025-05-25*  
*Contact: dechang64 (GitHub) / 冬生*
