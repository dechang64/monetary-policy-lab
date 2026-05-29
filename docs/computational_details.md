# Monetary Policy Lab: Computational Details

> **Technical Reference for Financial Economics Experts**
> 
> Platform: https://monetary-policy-lab.streamlit.app  
> Repository: GitHub (dechang64/monetary-policy-lab)  
> Version: v1.2 (Phase 1 complete, v6.2 analysis pipeline, audit fixes applied, WRDS CRSP/Compustat integrated)

---

## 1. Architecture Overview

The platform consists of two layers:

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Interactive Dashboard** | Streamlit app (`app.py` + `modules/`) | Real-time exploration, visualization, scenario analysis |
| **Research Engine** | `mp-research-platform/` + `analysis/` | Batch regression pipeline, hypothesis testing, robustness checks |

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

### 2.4 Asset Prices

**Primary source (v6.1)**: CRSP via WRDS (`data/wrds/crsp_dsi_index.csv`)

For the v6.1 analysis pipeline, we use CRSP daily index returns downloaded from WRDS:

| CRSP Variable | Label | Use |
|---------------|-------|-----|
| `vwretd` | CRSP Value-Weighted Return (incl. dividends) | H2: Large-cap equity response |
| `ewretd` | CRSP Equal-Weighted Return (incl. dividends) | H2: Small-cap equity response |
| `sprtrn` | S&P 500 Total Return | H2: Benchmark equity index |

**Coverage**: 1990-01-02 to 2024-12-31 (8,818 trading days). Mapped to FOMC meeting dates by exact date match.

**Advantage over yfinance**: CRSP returns include delisting adjustments (important for long-run studies) and are the standard data source in the monetary policy event study literature (Gürkaynak et al. 2005a, Nakamura & Steinsson 2018).

**Secondary source (fallback)**: yfinance (`mp-research-platform/run_analysis_v4.py`)

The original v4 pipeline used yfinance for the following series, which remain in `analysis_dataset_expanded.csv` as fallback:

| Ticker | Label | Use |
|--------|-------|-----|
| `^GSPC` | S&P 500 | Equity return (fallback) |
| `^IXIC` | NASDAQ | Tech equity return |
| `^VIX` | VIX | Volatility control |
| `^TNX` | 10Y Treasury | Long-rate change |
| `^IRX` | 13W T-bill | Short-rate change |
| `GC=F` | Gold | Safe-haven asset |

**Note**: yfinance returns MultiIndex columns in newer versions; we flatten via `df.columns.get_level_values(0)`. The 13W T-bill (`^IRX`) has lower precision than FRED's DGS3MO — a known limitation. For v6.1 H2 regressions, CRSP data takes priority over yfinance.

### 2.5 Acosta et al. (2024) Monetary Policy Shocks

**Source**: `data/mp_shocks_acosta.xlsx` (sheet: `shocks`)

High-frequency identified monetary policy shocks from Acosta, Bricongne, and L'Hour (2024), providing:

| Variable | Description | Coverage |
|----------|-------------|----------|
| `target` | Target rate surprise (Kuttner-style) | 220 meetings, 1995–2022 |
| `path` | Path factor (Gürkaynak-style) | 220 meetings, 1995–2022 |
| `ff.shock.0` | Fed Funds futures surprise | 220 meetings |
| `ns` | Narrow-window surprise | 220 meetings |

**Key advantage over naive proxy**: These are market-based surprises from futures contracts, capturing the *unexpected* component of FOMC decisions. The naive proxy (Δr) equals zero for all unchanged-rate meetings, compressing variance and attenuating regression coefficients.

**Sample overlap**: Of 220 Acosta meetings, 132 are post-2006 (our statement coverage period). Of these, 117 have both statements and shock data in our analysis dataset. The 15 missing meetings have statements available in the scraper but were not included in the original `analysis_dataset_expanded.csv` — a data gap that could be recovered in future versions to increase H1 sample size from 117 to 130.

### 2.6 WRDS Integration

**Connector**: `data/wrds_connector.py` (class `WRDSConnector`)

**Authentication**: WRDS requires Duo MFA. Connection flow: `wrds.Connection()` → Duo Approve → 30-day MFA exemption. Current credentials: username `dechang`.

**Available databases (with access)**:

| Database | Tables Used | Status |
|----------|-------------|--------|
| `crsp.dsi` | Daily index returns (vwretd, ewretd, sprtrn) | ✅ Downloaded, used in v6.1 H2 |
| `crsp.msi` | Monthly index returns | ✅ Downloaded, not yet used |
| `crsp.dsf` | Daily stock returns (individual) | ✅ Downloaded (financial sector, top 50) |
| `comp.fundq` | Compustat quarterly fundamentals | ✅ Downloaded, not yet used |
| `comp.funda` | Compustat annual fundamentals | ✅ Downloaded, not yet used |

**Databases without access**:

| Database | Tables | Purpose | Status |
|----------|--------|---------|--------|
| `cme.ff` | Fed Funds futures | Kuttner (2001) surprise | ❌ No permission |
| `cme.ef` | Eurodollar futures | Gürkaynak et al. (2005) path factor | ❌ No permission |
| `taqmsec.ctm_*` | TAQ intraday trades | High-frequency identification | ❌ No permission |
| `taqmsec.cqm_*` | TAQ intraday quotes | NBBO bid/ask | ❌ No permission |
| `ibes.statsum_epsus` | Analyst forecasts | Information shock validation | ❌ No permission |
| `philfed.spf` | SPF survey | Romer & Romer (2004) method | ❌ No permission |
| `optionm.opprcd*` | OptionMetrics | Implied volatility surface | ❌ No permission |

**Workaround for missing CME data**: We use Acosta et al. (2024) public shock data, which replicates the GSS target/path decomposition using the same CME futures data. This provides the correct surprise measures without requiring direct WRDS-CME access. The key limitation is that we cannot extend the shock series beyond 2022-07-27 (end of Acosta coverage) without CME access.

**Alternative CME replacement — USMPD (SF Fed)**: The Federal Reserve Bank of San Francisco publishes the U.S. Monetary Policy Event-Study Database (USMPD) at https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database. This database provides:

1. **Raw high-frequency changes**: MP1, FF1–FF6, ED1–ED8, OIS1Y–2Y, UST3M–30Y, TIPS5Y–30Y, SP500, DXY around FOMC events (276 meetings, 1994-02-04 to 2026-04-29)
2. **Acosta et al. (2025) single-factor surprise** (`mps.csv`): computed from MP1, MP2, ED2–ED4 via PCA, normalized to 1-for-1 impact on 1Y yield. Correlation with Acosta (2022) target+path = 0.989.
3. **R code** (`mps.R`): official replication code for computing surprises from raw USMPD data

**Status**: Downloaded and integrated. The raw USMPD data can be used to compute GSS-style target/path factors for the full 1994–2026 sample. Our current Acosta (2022) target/path factors are derived from the same underlying futures data. The USMPD extends coverage beyond Acosta's 2022-07-27 endpoint by 30 additional meetings (through 2026-04-29).

**Caveat**: The exact GSS (2005a) two-factor rotation (target vs. path) requires careful replication of their PCA methodology. We implement a two-step orthogonalized PCA in Python: (1) target = first PC of [MP1, FF1–FF4], (2) path = first PC of [FF4–FF6, ED2–ED4] after orthogonalizing against target. Both factors are normalized by regressing on daily 1Y GSW yield changes. This produces factors with high correlation to Acosta (2022): target r = 0.958, path r = 0.970. The single-factor STMT surprise replicates perfectly (r = 1.000).

**Extended sample results (Acosta 2006–2022 + USMPD 2022–2026)**:

| Sample | R² | Target p | Path p | N | Period |
|--------|-----|----------|--------|---|--------|
| Acosta only (v6.1) | 4.06% | 0.062* | 0.047** | 117 | 2006–2022 |
| Acosta + USMPD | 1.65% | 0.419 | 0.058* | 163 | 2006–2026 |

The path shock remains significant at 10% with the extended sample, but the target shock loses significance. The R² decline reflects the different dynamics of the 2022–2026 hiking cycle, where rapid rate changes dominate sentiment and forward guidance is less informative. The no-COVID robustness check (path p = 0.034**) confirms the information channel is not driven by pandemic outliers.

**WRDS data files** (in `data/wrds/`):

| File | Size | Description |
|------|------|-------------|
| `crsp_dsi_index.csv` | 755 KB | CRSP daily index, 1990-2024 (8,818 days) |
| `crsp_msi_index.csv` | 36 KB | CRSP monthly index, 1990-2024 |
| `crsp_financial_stocks_2020_2025.csv` | 80 MB | 910 financial sector stocks, 2020-2025 |
| `crsp_top50_stocks_2020_2025.csv` | 5.8 MB | Top 50 stocks by market cap, 2020-2025 |
| `crsp_stock_names.csv` | 5.6 MB | Stock name/PERMCO/PERMNO mapping |
| `compustat_fundq_2010_2025.csv` | 144 MB | Compustat quarterly, 2010-2025 |
| `compustat_funda_2010_2025.csv` | 57 MB | Compustat annual, 2010-2025 |

Large files (>5 MB) are excluded from git via `.gitignore`.

---

## 3. Sentiment Analysis Engine

### 3.1 Enhanced Dual-Dictionary Approach (v6.1)

**Module**: `analysis/run_v6_comprehensive.py` (function `compute_enhanced_sentiment`)

We use an **enhanced dual-dictionary** approach combining:

#### Loughran-McDonald (2011) Financial Sentiment Dictionary

- **Negative set**: ~120 words (e.g., "adverse", "concern", "deteriorate", "recession", "volatile")
- **Positive set**: ~50 words (e.g., "achieve", "boost", "confident", "stable")

$$\text{LM Score} = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{total}}}$$

#### Central Bank Hawkish-Dovish Dictionary (Henry 2008 + custom expanded)

- **Hawkish set**: ~45 words/bigrams (e.g., "tighten", "hike", "inflation", "vigilant", "restrictive", "quantitative", "tightening", "contractionary")
- **Dovish set**: ~55 words/bigrams (e.g., "accommodative", "patient", "gradual", "easing", "transitory", "reduce", "reducing", "decline", "declining")

$$\text{CB Score} = \frac{N_{\text{hawk}} - N_{\text{dove}}}{N_{\text{total}}}$$

**v6.1 Dictionary Fix**: Removed 5 overlapping terms that appeared in both hawkish and dovish sets ("contractionary", "quantitative", "reducing", "risks", "reduce"), keeping them in the hawkish set where they are more semantically appropriate. This eliminates cancellation effects and increases sentiment variance.

#### Combined Sentiment

$$\text{Sentiment}_t = 0.5 \times \text{LM Score}_t + 0.5 \times \text{CB Score}_t$$

**Interpretation**: Higher values → more hawkish tone.

**Preprocessing**: Lowercase → strip punctuation → filter words with length ≤ 1.

#### Enhanced Features (v6+)

Beyond the raw combined score, the enhanced pipeline also computes:

| Feature | Description |
|---------|-------------|
| `word_count` | Statement length (tokens) |
| `hawkish_count` | Raw hawkish word count |
| `dovish_count` | Raw dovish word count |
| `lm_score` | LM dictionary score (always positive for FOMC text) |
| `cb_score` | CB dictionary score (has sign variation) |
| `combined` | 0.5 × LM + 0.5 × CB |

### 3.2 Sentiment Distribution (v6.1)

| Statistic | Combined | LM Score | CB Score |
|-----------|:--------:|:--------:|:--------:|
| Mean | 0.024 | 0.038 | 0.010 |
| Std | 0.013 | 0.008 | 0.032 |
| Min | −0.012 | 0.006 | −0.089 |
| Max | 0.065 | 0.067 | 0.098 |
| % Negative | 18.8% | 0% | 78% |
| % Positive | 81.2% | 100% | 22% |

**Key observation**: The LM score is always positive for FOMC statements (min = 0.006), because FOMC statements use more positive than negative words regardless of policy stance (they say "growth" and "stable" even when cutting rates). The CB component has substantial sign variation (78% negative), but the equal-weighted combination dilutes this signal. This is a known limitation of the combined measure — see §10.2 for the upgrade path.

### 3.3 Known Limitations

| Issue | Detail | Impact |
|-------|--------|--------|
| **LM positivity bias** | LM score is always positive for FOMC text (min = 0.006) | Combined score is dominated by LM positivity; only 18.8% negative |
| **Sparsity** | LM dictionary designed for 10-K filings, not FOMC statements | std(LM) ≈ 0.008 vs. literature benchmark ≈ 0.035 |
| **Context blindness** | "higher" appears in both LM_POSITIVE and CB_HAWKISH — but "higher inflation" is hawkish while "higher growth" is dovish | Directional ambiguity |
| **No bigram/trigram** | Only unigram matching; "strongly committed to returning inflation to 2%" misses the commitment signal | Information loss |
| **Equal weighting** | 50/50 LM+CB is arbitrary; no empirical optimization | Suboptimal signal extraction |
| **LM-CB polarity conflict** | 10 terms are LM-negative but CB-hawkish (e.g., "inflation", "tightening"); 6 terms are LM-positive but CB-dovish (e.g., "ease", "easing") | Combined measure mixes two different dimensions — financial sentiment vs. policy stance |

**Planned improvements** (per WRDS_UPGRADE_PLAN.md):
1. **CB-only sentiment**: Use CB score alone for H1 regression (has actual sign variation)
2. **FinBERT** (Huang et al. 2022): Contextual sentiment, requires GPU
3. **Expanded CB dictionary**: Add FOMC-specific phrases from Apel & Blix Grimaldi (2012)
4. **Weight optimization**: Cross-validate LM/CB weights against asset returns

---

## 4. Monetary Policy Surprise

### 4.1 Acosta et al. (2024) Shocks (Active)

**Source**: `data/mp_shocks_acosta.xlsx`

The v6.1 pipeline uses Acosta et al. (2024) high-frequency identified shocks, which provide:

- **Target surprise** (`target`): The unexpected component of the rate decision, identified from Fed Funds futures price changes in a narrow window around the FOMC announcement
- **Path factor** (`path`): The change in expected future rate path beyond the current meeting, identified from Eurodollar futures

These replace the naive proxy (Δr) used in earlier versions (v1–v4), which was zero for all unchanged-rate meetings and severely attenuated regression coefficients.

### 4.2 Naive Proxy (Legacy, v1–v4)

$$\text{Surprise}_t = \Delta r_t = r_t^{\text{after}} - r_t^{\text{before}}$$

where $r_t$ is the Federal Funds target rate. This is a **naive proxy** — it equals zero for all "unchanged" meetings, which constitute the majority of the sample.

### 4.3 Target Implementation: Kuttner (2001)

Once WRDS is operational, the correct surprise measure is:

$$\text{Target Surprise}_t = \frac{F_{t}^{\text{close}} - F_{t-1}^{\text{close}}}{1 - D \cdot F_{t}^{\text{close}} / 100}$$

where:
- $F_t^{\text{close}}$ = settlement price of the current-month Fed Funds futures contract on FOMC day
- $F_{t-1}^{\text{close}}$ = same contract's settlement price on the day before
- $D$ = day-of-month (scaling factor for within-month timing)

For scheduled meetings, this captures the **unexpected component** of the rate decision.

### 4.4 Path Factor: Gürkaynak et al. (2005)

$$\text{Path Factor}_t = \text{Surprise}_t^{\text{ED2}} - \text{Surprise}_t^{\text{ED1}}$$

where $\text{ED1}$, $\text{ED2}$ are the first and second Eurodollar futures surprises. This captures **forward guidance** — changes in expected future rate path beyond the current meeting.

---

## 5. Hypothesis Testing Framework

### 5.1 H1: Sentiment ↔ Surprise

**Test**: Does FOMC statement sentiment respond to monetary policy surprises?

$$\text{Sentiment}_t = \alpha + \beta_1 \cdot \text{Target Surprise}_t + \beta_2 \cdot \text{Path Factor}_t + \varepsilon_t$$

**Method**: OLS with Newey-West HAC standard errors (lag = 1)

**Current result (v6.1)**:

| Statistic | Value |
|-----------|:-----:|
| R² | 0.0406 |
| β₁ (target) | 0.000290 |
| p(target) | 0.062* |
| β₂ (path) | 0.000469 |
| p(path) | 0.047** |
| N | 117 |
| Period | 2006-01-31 to 2022-07-27 |

*Significance: \*\*\* p<0.01, \*\* p<0.05, \* p<0.10

**Interpretation**: Both target and path shocks are significant predictors of FOMC statement sentiment. The path factor (forward guidance) is significant at 5%, while the target surprise is significant at 10%. However, the overall explanatory power is modest (R² = 4.06%), and a formal Wald test cannot reject the null that the two coefficients are equal (χ² = 0.19, p = 0.66). We interpret this as suggestive evidence consistent with the information channel, rather than definitive proof that the path effect dominates.

**Comparison with literature**: R² = 4.06% is substantially higher than the v4 result (0.39% with naive proxy) but still modest in absolute terms. The low R² reflects the fact that monetary policy shocks explain only a small fraction of the variation in FOMC statement language — the remaining 96% likely reflects the Fed's response to incoming economic data, institutional inertia in statement drafting, and other factors. A formal Wald test cannot reject the null that the target and path coefficients are equal (χ² = 0.19, p = 0.66), so we interpret the results as suggestive evidence consistent with the information channel.

**Newey-West lag choice**: We use lag = 1, which is conservative. The standard Bartlett formula suggests lag ≈ 4 for T = 117, but FOMC meetings are irregularly spaced (6–8 per year), making the autocorrelation structure different from daily data. Lag = 1 is defensible for event-time data but may understate standard errors. This is documented as a limitation.

### 5.2 H2: Asset Return Response to Shocks

**Test**: How do asset returns respond to target and path surprises?

$$R_{i,t} = \alpha + \beta_1 \cdot \text{Target Surprise}_t + \beta_2 \cdot \text{Path Factor}_t + \varepsilon_t$$

**Current results (v6.1, percentage points)**:

| Asset | R² | β₁ (target) | p(target) | β₂ (path) | p(path) | N |
|-------|:--:|:-----------:|:---------:|:---------:|:-------:|:--:|
| CRSP VW | 9.10% | -0.435 | 0.111 | -0.186 | 0.398 | 117 |
| CRSP EW | 10.28% | -0.449 | 0.044** | -0.174 | 0.421 | 117 |
| S&P 500 | 7.80% | -0.391 | 0.158 | -0.179 | 0.395 | 117 |

**Interpretation**: The target shock has a negative effect on equity returns, consistent with contractionary surprises reducing stock prices. The effect is statistically significant for the equal-weighted market (p = 0.044) but not for the value-weighted market (p = 0.111), consistent with small-cap stocks being more sensitive to monetary policy. The path shock does not have a statistically significant effect on any asset return at conventional levels, although the coefficients are consistently negative. This may reflect limited power (N = 117) or the fact that the path factor's effect on equity returns operates through a different channel than the narrow event window captures.

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

**Current result (v6.1)**: In the H1 regression (Sentiment ~ Target + Path), the path shock has a larger absolute t-statistic (|t| = 2.012) than the target shock (|t| = 1.887). However, a formal Wald test of H0: β_target = β_path fails to reject (χ² = 0.19, p = 0.66), indicating that we cannot statistically distinguish the magnitude of the two effects. We interpret this as suggestive evidence consistent with the information channel, but acknowledge limited statistical power.

**Critical caveat**: The modest R² (4.06%) and the inability to reject coefficient equality mean that the evidence for the information channel is suggestive rather than definitive. The remaining 96% of sentiment variation likely reflects the Fed's response to incoming economic data, institutional inertia, and other factors beyond the current rate decision and forward guidance.

### 5.4 H4: Regime-Dependent Effects

**Test**: Does the sentiment-return relationship vary across monetary policy regimes?

**Regimes**:
- **Conventional** (pre-2008): Standard rate adjustments
- **Forward Guidance** (2008–2015): ZLB period with explicit guidance
- **Normalization** (2016+): Rate hikes from ZLB

**Method**: Run H2 regression separately within each regime subsample. Compare $|\hat{\beta}|$ across regimes.

**Hypothesis**: Forward guidance regime should exhibit the strongest sentiment effect, because when rates are at ZLB, the statement's forward guidance content is the primary policy tool.

### 5.5 Robustness Checks (v6.1)

| Check | R² | N | target p | path p | Description |
|-------|:--:|:--:|:--------:|:------:|-------------|
| Full sample | 0.0406 | 117 | 0.062* | 0.047** | Baseline |
| Post-2010 | 0.0202 | 97 | 0.369 | 0.108 | Neither significant at 10% |
| No COVID | 0.0410 | 115 | 0.075* | 0.042** | Virtually unchanged |
| Extended (Acosta+USMPD) | 0.0165 | 163 | 0.419 | 0.058* | Path still significant at 10% |

**Interpretation**: The post-2010 subsample has lower R² (2.02% vs. 4.06%) and neither shock is individually significant, suggesting that the sentiment-surprise relationship was stronger in the earlier period (2006–2010), which includes the financial crisis when FOMC language was more variable. Excluding COVID has minimal impact (4.10% vs. 4.06%), confirming that the result is not driven by pandemic-era outliers. The extended sample (Acosta 2006–2022 + USMPD 2022–2026) shows the path shock remains significant at 10% (p = 0.058), but the target shock loses significance.

### 5.6 Regime-Specific Results (v6.1)

| Period | N | R² | target p | path p |
|--------|:--:|:--:|:--------:|:------:|
| Pre-ZLB (2006-2008) | 8 | 20.5% | 0.360 | 0.403 |
| Financial Crisis (2008-2010) | 12 | 9.2% | 0.930 | 0.124 |
| ZLB/FG (2010-2016) | 48 | 7.5% | 0.249 | 0.107 |
| Normalization (2016-2020) | 30 | 13.3% | 0.009*** | 0.873 |
| COVID+ (2020-2022) | 19 | 3.6% | 0.355 | 0.953 |

**Interpretation**: Sample sizes within each regime are small (8–48 meetings), so results should be interpreted with caution. During the Normalization period, the target shock is highly significant (p = 0.009) while the path shock is not, reflecting that rate changes were the primary information source. During the ZLB/FG period, neither shock is individually significant, though the path shock has a lower p-value (0.107 vs. 0.249), consistent with forward guidance being the primary channel when rates are at the zero lower bound.

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

> **Implementation note (v6.3 fix)**: The denominator is $\hat{\sigma} \cdot \sqrt{L}$, **not** $\hat{\sigma} / \sqrt{L}$. The latter would inflate the t-statistic by a factor of $L$ (e.g., 7× for a 7-day window). This bug existed in `utils/helpers.py` and has been corrected. The cross-sectional test (Section 3.5 of the paper) uses a different formula — $t = \overline{AR}_t / (SE_t / \sqrt{N_t})$ — which is the standard t-test for a sample mean and is correct as stated.

**Default windows**: Estimation = 250 days; Event = $[-1, +5]$ (6 trading days).

### 6.2 Cross-Sectional Aggregation

$$\text{AAR}_t = \frac{1}{N} \sum_{i=1}^{N} \text{AR}_{i,t}, \quad \text{CAAR}_t = \sum_{\tau} \text{AAR}_\tau$$

Aggregated across all assets for each FOMC date.

### 6.3 Demo Data Limitations

When FRED/WRDS data is unavailable, the platform falls back to synthetic returns (`utils/helpers.py:generate_synthetic_returns`). Key limitations:

1. **Correlated noise only**: Base returns are generated from a multivariate normal with an empirical correlation structure, but no FOMC event effects.
2. **FOMC effects injected post-hoc**: Directional shocks (hawkish/dovish) are added on actual FOMC dates with decay over the event window. This ensures equity indices move in the same direction and Treasury yields move in the same direction, consistent with the literature (Gürkaynak et al. 2005a).
3. **Not suitable for publication**: Demo data is for platform demonstration only. All published results must use real data from CRSP/FRED.

### 6.4 Asset-Level Summary

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

## 8. Dashboard-Specific Modules

### 8.1 Fed Intelligence (NLP)

**Module**: `modules/analyzers.py` (class `NLPEngine`)

- **Sentiment**: Hawkish/dovish word counting with partial string matching (`any(h in w for h in HAWKISH_WORDS)`)
- **Readability**: Simplified Flesch Reading Ease score:
$$\text{Flesch} = 206.835 - 1.015 \cdot \frac{\text{words}}{\text{sentences}} - 84.6 \cdot \frac{\text{syllables}}{\text{words}}$$
- **Diff analysis**: Set-difference of word tokens between two statements, filtered to words > 4 characters

### 8.2 Two Shocks (Dashboard Visualization)

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

### 8.3 Portfolio Rebalancing Simulation

**Module**: `modules/analyzers.py` (class `PortfolioEngine`)

5 investor types × 6 shock scenarios, with pre-specified allocation shifts:

- **Base allocations**: Mutual Funds, Hedge Funds, Pension Funds, Foreign Investors, Retail
- **Shock scenarios**: ±25bp, ±50bp rate surprise; strong/weak economy information shock
- **Method**: Add shock effect to base allocation, clip at 0, renormalize to 100%

This is a **heuristic simulation**, not estimated from data. It serves as a pedagogical tool for understanding directional capital flow patterns.

---

## 9. Key Methodological Notes & Limitations

### 9.1 Surprise Measurement

| Approach | Data Required | Status | Quality |
|----------|---------------|--------|---------|
| Δr_t (rate change) | FOMC records | ✅ Legacy (v1–v4) | Low — zero for unchanged meetings |
| Acosta et al. (2024) shocks | HF futures | ✅ Active (v6+) | High — market-based, narrow-window |
| GSS from USMPD (SF Fed) | HF futures | ✅ Downloaded, Phase 2 integration | High — canonical series, extends to 2026 |
| Kuttner (2001) target surprise | CME FF futures | ⏳ WRDS pending | High — direct replication |
| Gürkaynak et al. (2005) path factor | CME ED futures | ⏳ WRDS pending | High — captures forward guidance |

**Impact**: The switch from naive proxy to Acosta shocks increased H1 R² from 0.39% to 4.06%, a 10× improvement. The path factor is now significant at 5% (was not detectable with Δr).

### 9.2 Sentiment Measurement

The dictionary approach has known limitations (see §3.3). The most impactful upgrade path:

1. **Short-term**: Use CB-only sentiment for H1 regression (has actual sign variation, 78% negative)
2. **Medium-term**: Implement FinBERT for contextual sentiment (requires GPU)
3. **Long-term**: Fine-tune a BERT model on FOMC statements with hawkish/dovish labels

**LM positivity bias**: The LM score is always positive for FOMC text (min = 0.006 across 140 statements). This is because FOMC statements use more positive than negative words regardless of policy stance — they say "growth" and "stable" even when cutting rates. The combined = 0.5 × LM + 0.5 × CB dilutes the CB signal. Using CB-only sentiment should improve H1 R².

### 9.3 Standard Errors

Current implementation uses **Newey-West HAC** standard errors with lag = 1. This is appropriate for time-series data with potential heteroskedasticity and autocorrelation.

**Lag choice**: Lag = 1 is conservative. The Bartlett formula suggests lag ≈ 4 for T = 117, but FOMC meetings are irregularly spaced (6–8 per year), making the autocorrelation structure different from daily data. Lag = 1 may understate standard errors (overstate significance). Sensitivity analysis with lag = 4 is recommended.

**Planned upgrades**:
- **White (1980)** heteroskedasticity-robust SE for cross-sectional regressions
- **Thompson (2011)** double-clustered SE (by time and by asset) for panel regressions
- **Lag sensitivity analysis**: Report results for lag ∈ {1, 2, 4, 6}

### 9.4 Multiple Testing

H2 tests 3+ assets simultaneously. Without correction, the probability of at least one false positive at $\alpha = 0.10$ is $1 - 0.90^3 = 27\%$. Appropriate corrections:
- **Bonferroni**: $\alpha^* = 0.10 / 3 = 0.033$
- **Holm-Bonferroni**: Step-down procedure, less conservative
- **Benjamini-Hochberg**: Controls FDR at 10%

### 9.5 Endogeneity

The OLS regression $R_t = \alpha + \beta_1 \text{Surprise}_t + \beta_2 \text{Sentiment}_t + \varepsilon_t$ may suffer from:
- **Reverse causality**: Market reactions could feed back into statement drafting (unlikely for pre-written statements)
- **Omitted variables**: Macro surprises (NFP, CPI releases) coinciding with FOMC meetings
- **Measurement error**: Dictionary sentiment is a noisy proxy for true policy tone

**IV strategy** (planned): Use the previous meeting's sentiment as an instrument for current sentiment, exploiting the autocorrelation in communication style while assuming past sentiment doesn't directly affect current returns.

### 9.6 Sample Size

The H1 regression uses N = 117 meetings (2006–2022). This is the intersection of:
- 140 FOMC meetings in the analysis dataset (2006–2025)
- 220 Acosta shock observations (1995–2022)
- 164 FOMC statements with enhanced sentiment (2006–2026)

**Data gaps**:
- 15 Acosta shock meetings (2006–2022) are not in the analysis dataset but have statements available → could recover 13 obs
- 23 FOMC meetings (2022–2025) are post-Acosta coverage → could use DFF proxy for surprise
- Potential H1 sample with full recovery: ~130–140 meetings

---

## 10. Reproducibility

### 10.1 Data Versioning

- FOMC meetings: Hard-coded in `fomc_meetings.py` (164 observations, 1994–2025)
- FOMC statements: Cached as JSON in `mp-research-platform/data/fomc_statements_all.json`
- FRED data: Cached as JSON in `data/cache/` with 6-hour TTL
- Acosta shocks: `data/mp_shocks_acosta.xlsx` (220 observations, 1995–2022)
- Analysis results: Saved as CSV (`analysis_dataset_expanded.csv`) and JSON (`regression_results_v6.json`)

### 10.2 Randomness

- `TwoShocksEngine.variance_decomposition()`: Uses `np.random.normal(0, 3)` for simulation noise — **not seeded**. Results vary across runs.
- All other modules are deterministic given the same input data.

### 10.3 Dependencies

| Package | Version | Use |
|---------|---------|-----|
| streamlit | — | Dashboard |
| pandas, numpy | — | Data manipulation |
| scipy | — | Statistical tests |
| yfinance | — | Asset price download |
| requests | — | FRED API, FOMC scraping |
| beautifulsoup4 | — | HTML parsing |
| plotly | — | Interactive charts |
| openpyxl | — | Excel reading (Acosta shocks) |

---

## 11. Upgrade Roadmap

| Phase | Component | Data Source | Expected Impact |
|-------|-----------|-------------|-----------------|
| **Phase 2** | CB-only sentiment | — | H1 R²: 4.06% → 5–8% (more sign variation) |
| **Phase 2** | GSS target/path from USMPD | SF Fed USMPD (downloaded) | Extend to 2026; replicate exact GSS rotation in Python |
| **Phase 2** | Kuttner surprise (direct) | CME FF futures (WRDS) | Independent replication; requires CME access |
| **Phase 2** | Path factor (direct) | CME ED futures (WRDS) | H3 decomposition validity; requires CME access |
| **Phase 2** | Lag sensitivity analysis | — | Robustness of inference |
| **Phase 2** | Recover 13 missing obs | Statement scraper | H1 N: 117 → 130 |
| **Phase 2** | Compustat fundamentals | WRDS comp.fundq/funda | Control variables for firm-level analysis |
| **Phase 3** | FinBERT sentiment | GPU compute | Sentiment std: 0.013 → 0.02+ |
| **Phase 3** | High-frequency identification | TAQ (WRDS) | Intraday event windows; requires TAQ access |
| **Phase 3** | IV estimation | — | Address endogeneity |
| **Phase 4** | Sign restriction (JK style) | — | Structural shock identification |
| **Phase 4** | Panel regression with double-clustering | — | Efficient estimation |

---

*Document updated: 2026-05-28 (v1.1 — v6.1 analysis pipeline with fixed CB dictionary)*  
*Contact: dechang64 (GitHub) / 冬生*
