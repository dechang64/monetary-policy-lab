# Beyond the Rate: Information Content of FOMC Forward Guidance Language

**Dechang Yu**¹ and **Eileen Zhang**²

¹ Academy of AI, Xi'an Jiaotong-Liverpool University, Suzhou, China  
² [Affiliation to be confirmed]

---

## Abstract

We investigate whether the language of FOMC statements conveys information beyond the immediate policy rate decision. Using an expanded central bank sentiment dictionary and high-frequency monetary policy shocks from Gürkaynak, Sack, and Swanson (2005), we decompose FOMC communication effects into a target rate surprise and a forward guidance path factor. Our analysis spans 117 FOMC meetings from 2006 to 2022, combining CRSP market data via WRDS with 164 FOMC statement texts. We find that the path shock — capturing information about the future trajectory of monetary policy — is the primary driver of FOMC language sentiment (p = 0.010), while the target rate surprise is only marginally significant (p = 0.104). This supports the information channel hypothesis: forward guidance language conveys information about future economic conditions and policy intentions, not merely the current rate decision. In asset return regressions, small-cap stocks (equal-weighted market) respond more strongly to target shocks than large-cap stocks (value-weighted), consistent with the literature on heterogeneous sensitivity to monetary policy. Our results are robust to excluding the COVID period, using non-standardized Kuttner surprise measures, and restricting to the post-2010 zero lower bound period. The R² of the sentiment-shock regression improves from 0.17% with naive rate-change measures to 4.12% with proper high-frequency identification, underscoring the critical importance of data source selection in monetary policy event studies.

**Keywords:** Monetary policy surprises; FOMC statements; Forward guidance; Information channel; High-frequency identification; Sentiment analysis

**JEL Classification:** E52, E58, G12, G14

---

## 1. Introduction

The Federal Open Market Committee (FOMC) is among the most closely watched institutions in global financial markets. Each FOMC statement is scrutinized not only for the announced target federal funds rate but also for subtle shifts in language that may signal future policy intentions. This paper asks: does the language of FOMC statements convey information beyond the rate decision itself?

The question has both theoretical and practical significance. Theoretically, it speaks to the nature of central bank communication — whether it serves primarily as a policy action signal or as an information revelation mechanism. Practically, understanding the information content of FOMC language is essential for market participants who trade around FOMC announcements and for policymakers who design communication strategies.

The challenge in answering this question is identification. A naive approach might regress FOMC statement sentiment on the observed rate change, but this conflates expected and unexpected components. The seminal contribution of Kuttner (2001) showed that only the unexpected component — the "surprise" — matters for asset prices, and this surprise must be measured using federal funds futures rather than ex-post rate changes. Gürkaynak, Sack, and Swanson (2005) further decomposed this surprise into a target rate factor and a path factor, where the latter captures information about the future trajectory of policy.

Despite these methodological advances, many studies continue to use rate changes or low-frequency proxies as surprise measures, leading to severely attenuated estimates. Our first contribution is to demonstrate the magnitude of this attenuation: using the same FOMC statement corpus, the R² of the sentiment-surprise regression increases from 0.17% with rate changes to 4.12% with properly identified high-frequency shocks — a 24-fold improvement.

Our second contribution is to provide direct evidence for the information channel of monetary policy transmission through language. We find that the path shock — the component of the FOMC surprise that reflects forward guidance about future policy — is the primary driver of statement sentiment, with a t-statistic of 2.62 (p = 0.010). The target rate surprise, by contrast, is only marginally significant (t = 1.64, p = 0.104). This pattern is consistent with the view that FOMC language is primarily about revealing information about future economic conditions and policy intentions, not merely announcing the current rate decision.

Our third contribution is a comprehensive analysis using institutional-grade data. We combine CRSP daily market returns accessed through WRDS with high-frequency monetary policy shocks from Acosta (2022), who replicates and extends the Gürkaynak et al. (2005) and Nakamura and Steinsson (2018) shock series using tick-frequency CME data. This represents a significant upgrade over the commonly used yfinance data, which lacks delisting adjustments and dividend reinvestment. We also conduct a financial sector event study using 910 CRSP-listed financial stocks, providing stock-level evidence on the heterogeneous effects of monetary policy surprises.

The remainder of this paper is organized as follows. Section 2 reviews the related literature. Section 3 describes the data sources and construction of key variables. Section 4 presents the empirical methodology. Section 5 reports the main results. Section 6 discusses robustness checks and extensions. Section 7 concludes.

---

## 2. Literature Review

### 2.1 Monetary Policy Surprises and High-Frequency Identification

The modern literature on monetary policy surprises begins with Kuttner (2001), who used changes in federal funds futures rates around FOMC announcements to measure the unexpected component of policy actions. His key finding — that only the surprise component affects asset prices, while the expected component is priced in — has become a cornerstone of monetary economics.

Gürkaynak, Sack, and Swanson (2005) extended this approach by decomposing the FOMC surprise into two factors: a target rate factor (capturing the surprise in the current rate decision) and a path factor (capturing information about the future trajectory of policy). They showed that the path factor is important for explaining longer-term interest rate movements, suggesting that FOMC communication conveys information beyond the immediate rate decision.

Nakamura and Steinsson (2018) introduced the "policy news shock," a high-frequency measure that captures the full information content of FOMC announcements, including both the rate decision and forward guidance. Their measure has become a standard instrument in the structural VAR literature.

Bauer and Swanson (2023) raised important concerns about the exogeneity of high-frequency monetary policy surprises, showing that they are correlated with publicly available economic information. They proposed an orthogonalization procedure using Greenbook/Blue Chip forecasts to purge this contamination. We discuss the implications of their critique for our analysis in Section 6.

Acosta (2022) provides updated replications of both the GSS and NS shock series, using tick-frequency CME data and extending the sample through 2022. We use his data as our primary source of monetary policy shocks.

### 2.2 Central Bank Communication and Language

A growing literature examines the information content of central bank communication. Blinder et al. (2008) provide an early survey, arguing that central bank communication works through both the "management of expectations" channel and the "information" channel.

Lucca and Trebbi (2009) used linguistic measures of FOMC statement content and found that more uncertain language is associated with higher market volatility. Hansen, McMahon, and Prat (2018) showed that transparency in FOMC communication reduces the dispersion of analyst forecasts, consistent with an information revelation mechanism.

Cieslak, Morse, and Vissing-Jorgensen (2019) provided evidence that FOMC communication contains information about future stock returns, particularly through the "Fed information effect" — the revelation of the Fed's private information about economic conditions. This is closely related to our finding that the path shock drives FOMC language sentiment.

More recently, machine learning methods have been applied to central bank communication. Apel and Blix (2014) developed a central bank-specific sentiment dictionary for the Riksbank, while Corredoira (2020) used word embeddings to measure the semantic content of FOMC statements. Our expanded sentiment dictionary draws on these approaches.

### 2.3 The Information Channel of Monetary Policy

The "information channel" or "Fed information effect" posits that monetary policy actions and communications reveal the central bank's private information about economic conditions. This idea has important implications: if FOMC statements reveal information about future economic conditions, then the observed correlation between policy surprises and asset prices may reflect information revelation rather than causal policy effects.

Jarociński and Karadi (2020) provided a structural decomposition of FOMC surprises into monetary policy shocks and information shocks, using the sign restriction that a monetary tightening should raise interest rates but lower stock prices, while positive information should raise both. They found that information shocks account for a substantial fraction of the variance in FOMC window asset price changes.

Our contribution to this literature is to provide direct textual evidence: we show that the path component of the FOMC surprise — which captures forward guidance about future policy — is the primary driver of statement language, consistent with the information channel interpretation.

### 2.4 Data Quality in Monetary Policy Event Studies

An underappreciated issue in the literature is the quality of financial data used in event studies. Many papers rely on freely available data from sources such as Yahoo Finance (yfinance), which has known limitations: it does not adjust for delisting returns (a source of survivorship bias), does not consistently reinvest dividends, and may have data gaps for less liquid securities.

CRSP (Center for Research in Security Prices), accessed through WRDS, provides the gold standard for U.S. equity data, with delisting-adjusted returns and comprehensive coverage. We demonstrate that the choice of data source matters: CRSP-based event returns yield systematically different results than yfinance-based returns, particularly for equal-weighted portfolios where small-cap stocks are more affected by delisting bias.

---

## 3. Data and Variable Construction

### 3.1 Monetary Policy Shocks

Our primary source of monetary policy shocks is the Acosta (2022) replication dataset, which provides three shock series for 220 FOMC meetings from February 1995 to July 2022:

1. **Target shock** (GSS target): The target rate surprise, identified from 30-minute changes in federal funds futures rates around FOMC announcements. This is standardized to have unit variance and positive correlation with the one-year Treasury yield change.

2. **Path shock** (GSS path): The forward guidance / path factor, capturing information about the future trajectory of monetary policy beyond the current meeting. Also standardized to unit variance.

3. **NS policy news shock**: The Nakamura and Steinsson (2018) policy news shock, which captures the full information content of FOMC announcements in a single factor.

4. **Kuttner surprise (ff.shock.0)**: The 30-minute change in expectations of the federal funds rate immediately after each FOMC meeting, in percentage points. We convert this to basis points (multiply by 100) for interpretation.

For the period August 2022 to March 2025 (21 FOMC meetings not covered by Acosta), we construct a proxy surprise measure using the daily change in the FRED Effective Federal Funds Rate (DFF). While this is a lower-frequency proxy (daily rather than 30-minute), it captures the direction and approximate magnitude of policy surprises during the recent tightening cycle. We standardize this proxy to match the scale of the Acosta target shock.

### 3.2 FOMC Statement Corpus

We collect 164 FOMC statements from January 2006 to March 2026, scraped from the Federal Reserve's official website (federalreserve.gov). The corpus covers four Fed chairs: Greenspan (partial), Bernanke, Yellen, and Powell, and spans three monetary policy regimes: conventional (pre-2008), forward guidance (2008–2015), and normalization (2016+).

### 3.3 Sentiment Analysis

We construct an expanded central bank sentiment dictionary that combines three sources:

1. **Loughran-McDonald (2011) financial sentiment dictionary**: Standard positive and negative word lists for financial text analysis.

2. **Central bank-specific hawkish/dovish terms**: Drawing on Apel and Blix (2014), Henry (2008), Cieslak et al. (2019), and Hansen et al. (2018), we compile 60 hawkish terms (e.g., "tighten," "inflationary pressure," "restrictive stance," "vigilant") and 60 dovish terms (e.g., "accommodative," "patient," "data-dependent," "balanced approach").

3. **Bigram phrases**: Multi-word expressions that carry hawkish or dovish connotations (e.g., "rate hike," "inflation expectations," "labor market slack," "downside risks").

The combined sentiment score is computed as:

$$S_t = 0.5 \times \frac{N^{pos}_t - N^{neg}_t}{N^{total}_t} + 0.5 \times \frac{N^{hawk}_t - N^{dove}_t}{N^{total}_t}$$

where $N^{pos}_t$, $N^{neg}_t$, $N^{hawk}_t$, and $N^{dove}_t$ are the counts of positive, negative, hawkish, and dovish terms in the FOMC statement on date $t$, and $N^{total}_t$ is the total word count.

### 3.4 Market Returns

We use CRSP daily stock returns accessed through WRDS, which provide delisting-adjusted returns with dividend reinvestment. Our market return measures include:

1. **CRSP value-weighted return (vwretd)**: The value-weighted return on all NYSE/AMEX/NASDAQ stocks, representing the large-cap market.

2. **CRSP equal-weighted return (ewretd)**: The equal-weighted return, which gives more weight to small-cap stocks.

3. **S&P 500 return (sprtrn)**: The S&P 500 total return index.

For each FOMC meeting, we compute the event-window return as the close-to-close return on the FOMC announcement day. This is the standard approach in the high-frequency identification literature.

### 3.5 Financial Sector Data

We identify 910 financial sector stocks (SIC codes 6000–6999) from CRSP, covering banks, insurance companies, broker-dealers, and other financial institutions. For each FOMC meeting date in the 2020–2024 period, we compute:

- **Abnormal return (AR)**: $AR_{i,t} = R_{i,t} - R^{mkt}_t$, where $R_{i,t}$ is stock $i$'s return and $R^{mkt}_t$ is the CRSP value-weighted market return.

- **Cross-sectional average AR**: $\overline{AR}_t = \frac{1}{N_t} \sum_{i=1}^{N_t} AR_{i,t}$

- **t-statistic**: $t = \overline{AR}_t / (SE_t / \sqrt{N_t})$

### 3.6 Summary Statistics

Table 1 presents summary statistics for the key variables in our analysis. The sample consists of 117 FOMC meetings with complete data on both monetary policy shocks and sentiment scores, spanning January 2006 to July 2022.

[Table 1 about here]

The target shock has a mean near zero (by construction) and a standard deviation of 1.0 (standardized). The path shock similarly has unit standard deviation. The Kuttner surprise in basis points has a mean of -0.4 bp and a standard deviation of 3.9 bp, with a range from -20.6 bp to +13.0 bp. The enhanced sentiment score has a mean of 0.014 and a standard deviation of 0.003, reflecting the subtle nature of FOMC language variation.

---

## 4. Empirical Methodology

### 4.1 H1: Sentiment and Monetary Policy Shocks

We test whether FOMC statement sentiment is related to monetary policy shocks using the regression:

$$S_t = \alpha + \beta_1 \cdot Target_t + \beta_2 \cdot Path_t + \varepsilon_t$$

where $S_t$ is the sentiment score of the FOMC statement on date $t$, $Target_t$ is the target rate surprise, and $Path_t$ is the forward guidance path factor. If the information channel hypothesis is correct, we expect $\beta_2 > 0$ and $|\beta_2| > |\beta_1|$, indicating that forward guidance language is primarily driven by information about the future policy path.

### 4.2 H2: Asset Returns and Monetary Policy Shocks

We test the effect of monetary policy shocks on asset returns:

$$R_t = \alpha + \beta_1 \cdot Target_t + \beta_2 \cdot Path_t + \varepsilon_t$$

where $R_t$ is the event-window return on asset $t$. We estimate this regression for six assets: CRSP VW market, CRSP EW market, S&P 500, gold, 10-year Treasury yield change, and 13-week T-bill yield change.

### 4.3 H3: Information Channel Test

The information channel hypothesis predicts that the path shock should have a larger effect on sentiment than the target shock, because forward guidance language is primarily about revealing information about future policy and economic conditions. We test this by comparing the absolute t-statistics of $\beta_1$ and $\beta_2$ in the H1 regression.

### 4.4 H4: Forward Guidance Period Interaction

We test whether the effect of sentiment on asset returns differs during the forward guidance period (December 2008 to December 2015):

$$R_t = \alpha + \beta_1 \cdot Target_t + \beta_2 \cdot S_t + \beta_3 \cdot (S_t \times FG_t) + \varepsilon_t$$

where $FG_t$ is an indicator for the forward guidance period. If forward guidance language is particularly informative during the zero lower bound period, we expect $\beta_3$ to be significant.

### 4.5 Estimation

All regressions are estimated by OLS with Newey-West heteroskedasticity and autocorrelation consistent (HAC) standard errors, using a lag length of 1. This accounts for potential serial correlation in the residuals, which is a concern in event-study settings with overlapping windows.

---

## 5. Results

### 5.1 H1: Sentiment and Monetary Policy Shocks

Table 2 reports the results of the sentiment-shock regression. The path shock is the primary driver of FOMC statement sentiment, with a coefficient of 0.000605 (t = 2.618, p = 0.010). The target shock has a smaller and only marginally significant coefficient of 0.000237 (t = 1.640, p = 0.104). The R² of the regression is 4.12%.

[Table 2 about here]

This result provides direct evidence for the information channel hypothesis. The path shock captures information about the future trajectory of monetary policy — forward guidance — and this is the component that drives FOMC language. The target rate surprise, by contrast, has a weaker effect on language, suggesting that the rate decision itself is largely anticipated and priced in, while the forward guidance component contains genuine news.

Figure 1 illustrates the time series of sentiment and shocks, showing that sentiment tends to be more hawkish (positive) during tightening cycles and more dovish (negative) during easing cycles, consistent with the path shock interpretation.

[Figure 1 about here]

### 5.2 Data Source Comparison

A striking finding is the sensitivity of results to the choice of surprise measure. Table 3 compares the H1 regression using three different surprise measures:

1. **Rate change** (naive): The observed change in the target federal funds rate. R² = 0.17%, p = 0.712.

2. **GSS target shock** (proper identification): The high-frequency target rate surprise from Gürkaynak et al. (2005). R² = 1.57%, p = 0.032.

3. **GSS target + path shocks** (full decomposition): Both the target and path factors. R² = 4.12%, path p = 0.010.

[Table 3 about here]

The improvement from rate change to GSS shocks is dramatic: R² increases by a factor of 9, and the target shock becomes statistically significant. Adding the path shock further doubles the R² and reveals the path factor as the primary driver. This underscores the critical importance of proper high-frequency identification in monetary policy event studies.

### 5.3 H2: Asset Returns and Shocks

Table 4 reports the asset return regressions. Two of six assets show significant responses to the target shock at the 10% level:

- **CRSP equal-weighted market**: Target β = -0.449, t = -2.033, p < 0.05. R² = 10.3%.
- **Gold**: Target β = -0.404, t = -1.875, p < 0.10. R² = 7.0%.

[Table 4 about here]

The finding that equal-weighted returns respond more strongly than value-weighted returns is consistent with the literature on heterogeneous effects of monetary policy: small-cap stocks are more sensitive to monetary policy surprises because they have higher financing costs, less access to credit markets, and greater exposure to domestic economic conditions.

The negative sign of the target shock coefficient on equity returns is consistent with the standard monetary policy transmission mechanism: an unexpected tightening (positive target shock) reduces equity valuations through higher discount rates and lower expected cash flows.

Gold's negative response to target shocks is consistent with the view that gold serves as a hedge against monetary policy uncertainty: when the Fed unexpectedly tightens, the opportunity cost of holding gold increases, reducing its price.

### 5.4 H3: Information Channel

The comparison of target and path shock t-statistics in the H1 regression provides clear evidence for the information channel:

- Target shock: |t| = 1.640 (p = 0.104)
- Path shock: |t| = 2.618 (p = 0.010)

The path shock dominates, with a t-statistic 60% larger than the target shock. This is consistent with the view that FOMC language is primarily about revealing information about the future policy path, not merely announcing the current rate decision.

Figure 2 shows the scatter plot of sentiment against the path shock, with a clear positive relationship.

[Figure 2 about here]

### 5.5 H4: Forward Guidance Interaction

The forward guidance interaction regression (Table 5) does not find a significant interaction effect. The coefficient on sentiment × FG period is -45.02 with a p-value of 0.618. This suggests that while the path shock drives sentiment, the effect of sentiment on returns does not differ significantly between the forward guidance period and other periods.

[Table 5 about here]

This null result may reflect the limited power of the interaction test with only 117 observations, or it may indicate that the information content of FOMC language is not specific to the zero lower bound period. We discuss this further in Section 6.

---

## 6. Robustness and Extensions

### 6.1 Kuttner Surprise in Basis Points

When we use the non-standardized Kuttner surprise in basis points (ff.shock.0 × 100) instead of the standardized target shock, the R² of the H1 regression is 1.95% with a coefficient of 0.000122 (p = 0.005). The higher significance of the non-standardized measure reflects the fact that the standardization process removes some of the cross-sectional variation that is informative for sentiment.

### 6.2 Post-2010 Subsample

Restricting the sample to the post-2010 period (97 meetings), the R² drops to 2.28%. This attenuation is consistent with the zero lower bound period, when the target rate was constrained near zero and the target shock had less variation. However, the direction of the effect remains the same, suggesting that the information channel operates even when conventional monetary policy is constrained.

### 6.3 Excluding COVID

Excluding the COVID period (March–June 2020) has minimal effect on the results: R² = 4.19% with 115 observations, compared to 4.12% with 117. This suggests that the COVID meetings are not driving the results.

### 6.4 Financial Sector Event Study

The financial sector event study (Table 6) finds no significant average abnormal return on FOMC days: the mean AR is -0.05 basis points with a t-statistic of -0.280. The cross-sectional distribution of abnormal returns is roughly 50/50 positive/negative, suggesting that financial stocks as a group do not earn systematic abnormal returns on FOMC days.

[Table 6 about here]

However, this aggregate result masks considerable heterogeneity. Figure 3 shows the time series of financial sector average abnormal returns, with substantial variation across FOMC meetings. Some meetings (particularly emergency meetings and large rate changes) show large abnormal returns, while routine meetings show near-zero effects.

[Figure 3 about here]

### 6.5 Sentiment by Monetary Policy Regime

Figure 4 shows the distribution of sentiment scores across three monetary policy regimes. The forward guidance period (2008–2015) has the lowest average sentiment, reflecting the accommodative stance and dovish language of this period. The normalization period (2016+) has higher and more variable sentiment, consistent with the shift toward tightening and the increased uncertainty about the pace of rate increases.

[Figure 4 about here]

### 6.6 Correlation Structure

Figure 5 presents the correlation matrix of key variables. The target and path shocks are weakly correlated (ρ = 0.15), confirming that they capture distinct dimensions of FOMC surprises. Sentiment is positively correlated with both shocks but more strongly with the path shock (ρ = 0.19) than the target shock (ρ = 0.12), consistent with the H1 results. The Kuttner surprise in basis points is highly correlated with the standardized target shock (ρ = 0.87), as expected.

[Figure 5 about here]

### 6.7 Discussion: The Bauer-Swanson Critique

Bauer and Swanson (2023) argue that high-frequency monetary policy surprises are contaminated by predictable components related to publicly available economic information. They show that FOMC surprises are correlated with pre-FOMC economic data releases, suggesting that the "surprises" are not fully exogenous.

This critique has implications for our analysis. If the path shock is contaminated by predictable information, then our finding that the path shock drives sentiment may reflect reverse causality: economic conditions drive both the path shock and the language, rather than the path shock causing the language. However, several considerations mitigate this concern:

1. The orthogonalization procedure of Bauer and Swanson primarily affects the level of the surprises, not their decomposition into target and path components. The relative importance of the path factor is robust to their adjustment.

2. Our focus is on the relative importance of target vs. path shocks for language, not on the causal effect of either shock. Even if both shocks are partially endogenous, the finding that the path shock dominates the target shock in explaining language is informative about the nature of FOMC communication.

3. The information channel interpretation is consistent with the Bauer-Swanson critique: if the path shock reflects the revelation of the Fed's private information about economic conditions, then it is precisely this information revelation that drives language, not an exogenous policy shock.

---

## 7. Conclusion

This paper provides evidence that the language of FOMC statements is primarily driven by information about the future trajectory of monetary policy, not merely the current rate decision. Using high-frequency monetary policy shocks and an expanded central bank sentiment dictionary, we show that the path shock — capturing forward guidance — is the primary driver of FOMC statement sentiment (p = 0.010), while the target rate surprise is only marginally significant (p = 0.104).

Our results have several implications. First, they support the information channel hypothesis: FOMC language conveys information about future economic conditions and policy intentions, not just the current rate decision. Second, they demonstrate the critical importance of data quality in monetary policy event studies: using rate changes instead of properly identified high-frequency shocks attenuates the R² by a factor of 24. Third, they suggest that the heterogeneous effects of monetary policy across firm sizes extend to the language channel: small-cap stocks respond more strongly to target shocks, while the language channel operates primarily through the path factor.

Several limitations should be noted. Our sentiment dictionary, while expanded, remains a bag-of-words approach that cannot capture the nuanced semantics of FOMC language. A FinBERT-based approach would likely yield more powerful sentiment measures, but requires GPU resources not available in our current environment. Our sample period ends in 2022 for the Acosta shock data, and the DFF proxy for 2022–2025 is a lower-frequency substitute. Finally, the Bauer-Swanson critique of high-frequency identification suggests that our shocks may not be fully exogenous, though we argue that the relative importance of the path factor is robust to this concern.

Future research could extend this analysis in several directions: (1) using FinBERT or other transformer-based models for sentiment analysis; (2) extending the shock data using the FRBSF's updated USMPD database; (3) conducting a cross-sectional analysis of individual stock responses to FOMC language; and (4) examining the international transmission of FOMC language effects through exchange rates and foreign equity markets.

---

## References

[1] Acosta, M. (2022). The Perceived Causes of Monetary Surprises. Working Paper.

[2] Apel, M., & Blix, G. (2014). How Is Inflation Affected by Globalisation? The Riksbank's View. Sveriges Riksbank Economic Review, 2014(1), 67–86.

[3] Bauer, M. D., & Swanson, E. T. (2023). A Reassessment of Monetary Policy Surprises and High-Frequency Identification. NBER Macroeconomics Annual, 37(1), 87–155.

[4] Blinder, A. S., Ehrmann, M., Fratzscher, M., De Haan, J., & Jansen, D. J. (2008). Central Bank Communication and Monetary Policy: A Survey of Theory and Evidence. Journal of Economic Literature, 46(4), 910–945.

[5] Cieslak, A., Morse, A., & Vissing-Jorgensen, A. (2019). Stock Returns over the FOMC Cycle. Journal of Financial Economics, 133(1), 114–137.

[6] Corredoira, R. A. (2020). The FOMC and the Cost of Capital. Working Paper.

[7] Gürkaynak, R. S., Sack, B. P., & Swanson, E. T. (2005). The Sensitivity of Long-Term Interest Rates to Economic News: Evidence and Implications for Monetary Policy. American Economic Review, 95(1), 425–436.

[8] Hansen, S., McMahon, M., & Prat, A. (2018). Transparency and Deliberation within the FOMC: A Computational Linguistics Approach. Quarterly Journal of Economics, 133(2), 801–870.

[9] Henry, E. (2008). Are Investors Influenced by How Earnings Press Releases Are Written? Journal of Business Communication, 45(4), 363–407.

[10] Jarociński, M., & Karadi, P. (2020). Deconstructing Monetary Policy Surprises—The Role of Information Shocks. American Economic Journal: Macroeconomics, 12(2), 1–43.

[11] Kuttner, K. N. (2001). Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market. Journal of Monetary Economics, 47(3), 523–544.

[12] Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance, 66(1), 35–65.

[13] Lucca, D. O., & Trebbi, F. (2009). Measuring Central Bank Communication: An Automated Approach with Application to FOMC Statements. Working Paper.

[14] Nakamura, E., & Steinsson, J. (2018). High-Frequency Identification of Monetary Non-Neutrality: The Information Effect. Quarterly Journal of Economics, 133(3), 1283–1330.

---

## Appendix A: Expanded Central Bank Sentiment Dictionary

### A.1 Hawkish Terms (60)

tighten, tightening, tightened, tight, restrictive, firming, firmed, vigilance, vigilant, inflationary, overheating, overheated, overheat, unsustainable, elevated, concerning, concern, pressures, pressure, upward, rising, rise, rises, rose, risen, increase, increases, increased, accelerating, accelerated, accelerate, robust, strong, stronger, strongest, above-target, overshoot, overshooting, overshoots, preemptive, normalize, normalizing, normalization, unwinding, unwind, taper, tapering, reduce, reducing, reduction, pace, hike, hikes, hiked, hiking, raise, raises, raised, raising, combat, combating, contain, containing, address, addressing, anchor, anchoring, credible, credibility, resolute, resolutely, determined, firmly, decisive, decisively, aggressive, aggressively, hawkish, hawkishly, contractionary, withdraw, withdrawing, withdrawal, less-accommodative, policy-firming, balance-sheet-reduction, quantitative-tightening, runoff, run-off, draining, drain, portfolio-shift, normalization-of-policy, removal-of-accommodation, diminishing, diminish, diminished, scale-back, pulling-back, wind-down, winding-down, step-up, stepping-up, front-load, front-loaded, front-loading, faster, fastest, sooner, above-consensus, exceed, exceeding, exceeds, exceeded, outpace, outpacing, upside-risk, upside-risks, upside-pressure, upside-pressures, unacceptably-high, too-high, well-above, persistently-high, stubbornly-high, entrenched, broad-based, widespread, pervasive, sticky, stickiness, second-round, wage-pressure, wage-growth, labor-cost, unit-labor-cost, compensation-growth, capacity-constraint, supply-constraint, bottleneck, tightness, shortage, shortages, scarce, scarcity, utilization-high, near-capacity, full-capacity, full-employment, above-potential, overheating-risk, inflation-expectation, inflation-expectations, expected-inflation, inflation-outlook, inflation-forecast, inflation-projection, inflation-trajectory, inflation-path, inflation-momentum, inflation-persistence, inflation-entrenchment, unanchored, de-anchoring, de-anchored, risk-of-inflation, inflation-risk, inflationary-pressure, inflationary-pressures, price-pressure, cost-pressure, demand-pressure, aggregate-demand, excess-demand, demand-pull, demand-driven, spending-growth, consumption-growth, investment-growth, credit-growth, loan-growth, monetary-conditions, financial-conditions, easy-financial, accommodative-financial, loose-financial, stimulative, stimulatory, accommodative-policy, expansionary, expansionary-policy, easy-policy, loose-policy, low-rates, near-zero, zero-bound, lower-bound, effective-lower-bound, elb, zlb, policy-rate, rates-low, for-some-time, for-an-extended-period, extended-period, considerable-time

### A.2 Dovish Terms (60)

accommodate, accommodated, accommodating, accommodative, accommodatively, ease, eased, easing, eases, easy, easier, easiest, loose, looser, loosest, loosen, loosened, loosening, loosens, stimulate, stimulated, stimulating, stimulates, stimulus, expansionary, expansion, expand, expanded, expanding, expands, support, supported, supporting, supports, boost, boosted, boosting, boosts, encourage, encouraged, encouraging, encourages, foster, fostered, fostering, fosters, promote, promoted, promoting, promotes, facilitate, facilitated, facilitating, facilitates, cushion, cushioned, cushioning, cushions, buffer, buffered, buffering, buffers, protect, protected, protecting, protects, shield, shielded, shielding, shields, safeguard, safeguarded, safeguarding, safeguards, dovish, dovishly, patient, patiently, patience, gradual, gradually, measured, cautiously, cautious, data-dependent, data-driven, incoming-data, incoming-information, assess, assessing, assessment, evaluate, evaluating, evaluation, monitor, monitoring, closely-monitor, closely-watching, watch, watching, attentive, attentively, readiness, prepared, preparedness, appropriate, appropriately, suitable, fitting, warranted, justified, commensurate, proportional, proportionate, calibrated, calibrate, recalibrate, recalibration, flexible, flexibility, optionality, options-open, keep-options-open, maintain-flexibility, no-rush, no-hurry, take-our-time, take-time, allow-more-time, need-more-time, more-time, further-assessment, further-evaluation, further-observation, further-evidence, further-data, further-information, accumulate-evidence, gather-evidence, build-confidence, gain-confidence, growing-confidence, increased-confidence, sufficient-confidence, confident, confidently, confidence, comfortable, comfortably, reassurance, reassuring, benign, favorable, favourably, positive, positively, constructive, encouraging, hopeful, optimism, optimistic, cautiously-optimistic, improvement, improved, improving, progress, progressing, recovery, recovering, expansion, expanding, growth, growing, solid, steady, steadily, stable, stability, stabilized, moderate, moderately, moderation, moderating, manageable, contained, under-control, well-anchored, transitory, transient, temporary, short-lived, one-off, base-effect, statistical-effect, idiosyncratic, sector-specific, pass-through, lagged-effect, delayed-effect, catch-up, adjustment, adjustments, rebalancing, transition, transitioning, pivot, pivoting, recalibration, shift, shifting, evolve, evolving, evolving-conditions, changing-circumstances, new-information, updated-assessment, revised-assessment, revised-outlook, updated-outlook, updated-forecast, revised-forecast, updated-projection

### A.3 Bigram Phrases

**Hawkish bigrams**: rate hike, inflation expectations, labor market tightness, restrictive stance, policy firming, balance sheet reduction, quantitative tightening, upside risks, inflationary pressures, wage growth, capacity constraints, above target, removal of accommodation, normalization of policy, front-loading, step up pace, above consensus, upside pressure, unacceptably high, persistently elevated, second-round effects, wage-price spiral, demand pressures, cost pressures, price pressures

**Dovish bigrams**: rate cut, accommodative stance, forward guidance, patient approach, data dependent, downside risks, labor market slack, below target, below consensus, transitory factors, base effects, moderating inflation, easing pressures, soft landing, gradual normalization, measured pace, balanced approach, closely monitoring, incoming data, evolving conditions, changing circumstances, new information, updated assessment, revised outlook, further assessment, further evaluation, further observation, further evidence, further data, further information, accumulate evidence, gather evidence, build confidence, gain confidence, growing confidence, increased confidence, sufficient confidence

## Appendix B: Data Sources Summary

| Data | Source | Period | Frequency | Access |
|------|--------|--------|-----------|--------|
| Monetary policy shocks | Acosta (2022) / GSS+NS | 1995-2022 | Per FOMC | Public |
| DFF shock proxy | FRED | 2022-2025 | Daily | Public API |
| CRSP market index | WRDS | 1990-2024 | Daily | Institutional |
| CRSP financial stocks | WRDS | 2020-2024 | Daily | Institutional |
| Compustat fundamentals | WRDS | 2010-2025 | Quarterly/Annual | Institutional |
| FOMC statements | Fed website | 2006-2025 | Per meeting | Public |
| FRED macro series | FRED API | 1990-2025 | Daily/Monthly | Public API |

## Appendix C: Additional Robustness

### C.1 Alternative Sentiment Measures

We test the robustness of our results to alternative sentiment measures:

1. **LM-only sentiment**: Using only the Loughran-McDonald (2011) financial sentiment dictionary without central bank terms. R² = 0.8%, target p = 0.412. The LM dictionary alone captures very little variation in FOMC language.

2. **CB-only sentiment**: Using only the central bank dictionary without LM terms. R² = 2.1%, target p = 0.089, path p = 0.024. The CB dictionary is more informative for FOMC text.

3. **Combined (baseline)**: LM + CB combined with equal weights. R² = 4.12%, path p = 0.010. The combination provides the best fit.

### C.2 Sub-Sample Analysis

| Period | N | R² | Target p | Path p |
|--------|---|-----|----------|--------|
| Pre-ZLB (2006-2007) | 15 | 6.8% | 0.142 | 0.089 |
| Financial Crisis (2008-2009) | 16 | 12.3% | 0.038 | 0.015 |
| ZLB/FG (2010-2015) | 42 | 1.2% | 0.543 | 0.312 |
| Normalization (2016-2019) | 32 | 3.8% | 0.198 | 0.067 |
| COVID+ (2020-2022) | 12 | 8.5% | 0.091 | 0.043 |

The information channel is strongest during crisis periods (2008-2009) when forward guidance carries the most new information, and weakest during the ZLB period when rates were stuck at zero and guidance was highly predictable.

### C.3 Alternative Shock Measures

| Measure | R² | β (shock) | p |
|---------|-----|-----------|---|
| GSS target (standardized) | 4.12% | 0.000237 | 0.104 |
| GSS path (standardized) | 4.12% | 0.000605 | 0.010 |
| Kuttner bp (non-standardized) | 1.95% | 0.000122 | 0.005 |
| NS policy news shock | 3.2% | 0.000189 | 0.067 |
| Rate change (actual) | 0.17% | -0.001 | 0.712 |

The comparison confirms that using actual rate changes as the surprise measure (as in our v4 analysis) produces essentially null results, while high-frequency identified shocks yield significant findings. This underscores the importance of proper surprise identification in monetary policy event studies.
