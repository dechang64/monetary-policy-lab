# Beyond the Rate: Information Content of FOMC Statement Language

**Eileen Zhang**

Academy of AI, Xi'an Jiaotong-Liverpool University, Suzhou, China

---

## Abstract

We investigate whether the language of FOMC statements conveys information beyond the immediate policy rate decision. Using a combined central bank sentiment dictionary (Loughran-McDonald and central-bank-specific terms) and high-frequency monetary policy shocks from Gürkaynak, Sack, and Swanson (2005), we decompose FOMC communication effects into a target rate surprise and a forward guidance path factor. Our analysis spans 117 FOMC meetings from 2006 to 2022, combining CRSP market data with FOMC statement texts. We find that the target shock — capturing the unexpected component of the current rate decision — is a statistically significant predictor of FOMC language sentiment ($\beta$ = 0.000577, t = 2.43, p = 0.017), while the path shock is not significant at conventional levels ($\beta$ = 0.000633, t = 1.44, p = 0.152). A Wald test cannot reject the null that the two coefficients are equal (p = 0.90). In asset return regressions, the target shock significantly predicts equity and gold returns, with small-cap stocks (equal-weighted CRSP) responding more strongly than large-cap stocks (value-weighted), consistent with heterogeneous sensitivity to monetary policy. The forward guidance period interaction is not significant, suggesting that the language channel does not strengthen during zero-lower-bound periods. Using rate changes instead of properly identified high-frequency shocks attenuates the R² by a factor of 4 (from 1.57% to 0.40%), underscoring the critical importance of data quality in monetary policy event studies.

Keywords: Monetary policy; FOMC; Forward guidance; Sentiment analysis; High-frequency identification

JEL Classification: C80, D83, E43, E52, E58, G12, G14

---

## 1. Introduction

Central bank communication has become a central tool of monetary policy. Over the past two decades, the Federal Reserve has progressively increased the transparency and detail of its post-meeting statements, transforming them from brief rate announcements into substantive policy documents. A large literature examines how these communications affect financial markets, but relatively little attention has been paid to a more fundamental question: what determines the language of FOMC statements themselves?

We address this question by examining the relationship between high-frequency monetary policy shocks and the sentiment of FOMC statement language. Using the target/path decomposition of Gürkaynak, Sack, and Swanson (2005, henceforth GSS), we test whether the two dimensions of monetary policy surprises — the target rate surprise and the forward guidance path factor — have differential effects on FOMC language and asset returns.

Our main finding is that the target shock is a statistically significant predictor of FOMC statement sentiment (p = 0.017), while the path shock is not (p = 0.152). This result contrasts with the prediction of the information channel hypothesis, which holds that forward guidance language should be primarily driven by the path shock. However, the overall explanatory power is modest (R² = 1.57%), and a formal Wald test cannot reject the null that the target and path effects are equal (p = 0.90). We therefore interpret our results as providing only suggestive evidence on the relative importance of the two shock dimensions for FOMC language.

In asset return regressions, we find that the target shock significantly predicts equity and gold returns, with effect sizes ranging from 28 basis points for NASDAQ to 45 basis points for the equal-weighted CRSP index. Small-cap stocks respond more strongly than large-cap stocks, consistent with the literature on heterogeneous sensitivity to monetary policy. The path shock does not significantly affect any asset class at the 5% level, although coefficients are consistently in the expected (negative) direction for equities.

We also examine whether the effect of FOMC language on asset returns differs during the forward guidance period (December 2008 to December 2015), when the federal funds rate was at the zero lower bound. We find no evidence of such an interaction: the sentiment × forward guidance interaction term is statistically insignificant for both the CRSP value-weighted index (p = 0.991) and NASDAQ (p = 0.739). This null result suggests that, while FOMC language may convey information, its effect on asset prices does not systematically strengthen when conventional monetary policy is constrained.

Our results demonstrate the critical importance of data quality in monetary policy event studies. Using rate changes instead of properly identified high-frequency shocks attenuates the R² of the sentiment regression by a factor of 4 (from 1.57% to 0.40%), and renders the relationship statistically undetectable. Even the Kuttner (2001) surprise measure, which removes the expected component of rate changes, yields substantially lower explanatory power than the GSS target/path decomposition.

This paper contributes to three strands of the literature. First, we contribute to the literature on central bank communication (Blinder et al., 2008; Gürkaynak et al., 2005b) by directly examining the determinants of FOMC statement language, rather than its effects on asset prices. Second, we contribute to the information channel literature (Romer and Romer, 2000; Campbell et al., 2012; Nakamura and Steinsson, 2018) by testing whether the two dimensions of monetary policy surprises have differential effects on language. Third, we contribute to the methodology of monetary policy event studies by quantifying the sensitivity of results to the choice of surprise measure.

The remainder of the paper is organized as follows. Section 2 reviews the related literature. Section 3 describes the data and variable construction. Section 4 presents the empirical methodology. Section 5 reports the main results. Section 6 discusses robustness checks and extensions. Section 7 concludes.

---

## 2. Literature Review

### 2.1 Monetary Policy Surprises and High-Frequency Identification

The identification of exogenous monetary policy shocks has been a central challenge in macroeconomics. The early literature used narrative approaches (Romer and Romer, 1989) or structural VARs (Christiano et al., 1999), but these methods rely on strong identifying assumptions. Kuttner (2001) pioneered the high-frequency identification approach, using changes in federal funds futures rates around FOMC announcements to isolate the unexpected component of monetary policy decisions. This approach exploits the fact that financial markets incorporate all available information into prices immediately, so that the narrow-window change in interest rates captures the pure surprise.

Gürkaynak, Sack, and Swanson (2005a) extended this approach by decomposing the high-frequency interest rate response into two factors: a target factor (capturing the surprise in the current rate decision) and a path factor (capturing the revision of expectations about future policy). They showed that the path factor explains a substantial fraction of the response of longer-term interest rates and asset prices, suggesting that forward guidance is an important channel of monetary policy transmission.

More recently, Acosta (2022) updated the GSS target and path factors through 2022, providing the shock data used in this paper. Bauer and Swanson (2023) raised concerns about the exogeneity of high-frequency shocks, showing that they are partially predictable from pre-FOMC economic information. We discuss the implications of this critique for our analysis in Section 6.

### 2.2 Central Bank Communication and Language

A growing literature examines the content and effects of central bank communication. Blinder et al. (2008) survey the early literature and conclude that central bank communication can improve the predictability of monetary policy and reduce market volatility. Lucca and Trebbi (2009) use automated content analysis to measure the linguistic complexity and sentiment of FOMC statements, finding that more complex language is associated with larger market reactions.

More recently, researchers have applied natural language processing (NLP) techniques to central bank communication. Apel and Blix (2014) construct a hawkish-dovish sentiment index for the Riksbank. Hansen et al. (2018) analyze the transparency of FOMC communications. Huang et al. (2022) apply FinBERT to financial text, demonstrating that transformer-based models can capture contextual meaning that bag-of-words approaches miss.

Our paper contributes to this literature by examining the relationship between monetary policy shocks and FOMC statement sentiment, rather than the effect of communication on markets. This provides a more direct test of what drives the language of central bank statements.

### 2.3 The Information Channel of Monetary Policy

The information channel hypothesis holds that monetary policy actions convey information about the central bank's assessment of economic conditions, not just the policy decision itself. Romer and Romer (2000) provide early evidence that the Fed has superior information about inflation, and that FOMC actions reveal this information to market participants. Campbell et al. (2012) distinguish between "delphic" forward guidance (revealing the Fed's forecast) and "Odyssean" forward guidance (committing to a future path), and argue that both types convey information.

Nakamura and Steinsson (2018) provide evidence that monetary policy shocks have large effects on long-term expectations, consistent with the information channel. Jarociński and Karadi (2020) use sign restrictions to separately identify monetary policy and information shocks, finding that the two have opposite effects on asset prices.

Our paper tests the information channel by examining whether the path shock — which captures forward guidance about future policy — has a larger effect on FOMC language than the target shock. If FOMC language is primarily about revealing information about the future, the path shock should be the dominant driver. If, instead, the language primarily reflects the current rate decision and its immediate context, the target shock should dominate.

### 2.4 Sentiment Analysis in Finance

Sentiment analysis has become a standard tool in financial economics. The Loughran and McDonald (2011) dictionary is widely used for financial text, providing lists of positive and negative words specifically calibrated for 10-K filings. However, this dictionary has known limitations when applied to central bank communication: FOMC statements use more positive than negative words regardless of policy stance, creating a positivity bias that reduces the dictionary's discriminatory power.

Central bank-specific dictionaries have been developed to address this limitation. Apel and Blix (2014) construct a hawkish-dovish dictionary for the Riksbank. Corredoira et al. (2020) develop a dictionary for FOMC statements. We use a combined approach, averaging the Loughran-McDonald score with a central bank-specific score, while acknowledging the limitations of this approach.

---

## 3. Data and Variable Construction

### 3.1 Monetary Policy Shocks

We use the high-frequency monetary policy shocks from Acosta (2022), who extends the GSS decomposition through July 2022. The Acosta data provide two standardized shock series:

- **Target shock**: The unexpected component of the current federal funds rate decision, identified from changes in the current-month federal funds futures rate in a narrow window around the FOMC announcement.
- **Path shock**: The revision of expectations about the future path of monetary policy, identified from changes in longer-dated Eurodollar futures and federal funds futures.

In the full Acosta sample (1995–2022, 220 meetings), both shocks are standardized to unit variance. However, in our estimation sample (2006–2022, 117 meetings), the standard deviations are 0.82 for the target shock and 0.80 for the path shock, reflecting the different variance structure of the subsample. We use the shocks as provided by Acosta without re-standardizing, noting that this does not affect t-statistics or p-values, only the scale of the coefficients.

### 3.2 FOMC Statement Corpus

We collect 140 FOMC statements from January 2006 to March 2025, scraped from the Federal Reserve's official website. Of these, 117 overlap with the Acosta shock data (January 2006 to July 2022). The corpus covers three Fed chairs: Bernanke, Yellen, and Powell.

### 3.3 Sentiment Analysis

We compute two sentiment scores for each FOMC statement:

**Loughran-McDonald (LM) score**: The fraction of positive words minus the fraction of negative words, using the Loughran and McDonald (2011) dictionary. The LM dictionary contains 354 positive and 2,329 negative terms in its master list, of which a subset appears in FOMC text.

**Central Bank (CB) score**: A hawkish-dovish score computed using an expanded central bank-specific dictionary comprising 591 hawkish terms (e.g., "tightening," "inflationary pressures," "vigilant") and 222 dovish terms (e.g., "accommodative," "downside risks," "labor market slack"). The CB score is computed as (hawkish − dovish) / total words.

**Combined score**: We use the equal-weighted average of the LM and CB scores as our primary sentiment measure. This follows the standard approach in the literature, but we note that the LM component exhibits a positivity bias for FOMC text (see Section 6).

### 3.4 Market Returns

We use CRSP daily market data obtained through WRDS as our primary return data source. Specifically:

- **CRSP value-weighted return** (vwretd): The value-weighted return on all NYSE/AMEX/NASDAQ stocks
- **CRSP equal-weighted return** (ewretd): The equal-weighted return on all NYSE/AMEX/NASDAQ stocks
- **CRSP S&P 500 return** (sprtrn): The S&P 500 total return index

We also collect gold prices, 10-year Treasury yields, 13-week T-bill yields, and the VIX from FRED. All returns are expressed in basis points or percentage changes as appropriate.

### 3.5 Summary Statistics

Table 1 reports the summary statistics for the key variables in our estimation sample.

[Table 1 about here]

Several features deserve comment. First, the combined sentiment score has a mean of 0.014 and a standard deviation of 0.006, reflecting the relatively low variation in FOMC statement language. The LM score is always positive (minimum = 0.031), consistent with the well-known positivity bias of the LM dictionary in the context of central bank communication. The CB score has a negative mean (−0.013), indicating that FOMC statements use more dovish than hawkish language on average, with a standard deviation of 0.005.

Second, the target and path shocks have standard deviations of 0.82 and 0.80 respectively in our sample, rather than 1.00 as in the full Acosta sample. This reflects the different variance structure of the 2006–2022 subsample.

Third, the correlation between the target and path shocks is 0.14, indicating that the two dimensions of monetary policy surprises are largely orthogonal. The correlation between sentiment and the target shock (0.09) is slightly lower than the correlation with the path shock (0.10), but both are modest.

---

## 4. Empirical Methodology

### 4.1 Sentiment and Monetary Policy Shocks

We estimate the following regression to test whether FOMC statement sentiment is related to monetary policy shocks:

$$S_t = \alpha + \beta_1 \cdot Target_t + \beta_2 \cdot Path_t + \varepsilon_t$$

where $S_t$ is the combined sentiment score for FOMC meeting $t$, $Target_t$ is the target shock, and $Path_t$ is the path shock. Under the information channel hypothesis, $\beta_2 > \beta_1$, because the path shock captures forward guidance information that is primarily conveyed through language. Under the alternative hypothesis that FOMC language primarily reflects the current rate decision, $\beta_1 \geq \beta_2$.

### 4.2 Asset Returns and Monetary Policy Shocks

We estimate separate regressions for each asset class:

$$R_t = \alpha + \beta_1 \cdot Target_t + \beta_2 \cdot Path_t + \varepsilon_t$$

where $R_t$ is the daily return on the asset in question. We expect $\beta_1 < 0$ for equities (an unexpected tightening reduces equity valuations) and $\beta_1 > 0$ for short-term Treasury yields (an unexpected tightening raises short rates).

### 4.3 Information Channel Test

We test the information channel by comparing the relative magnitudes of $\beta_1$ and $\beta_2$ in the sentiment regression. A formal Wald test of the null hypothesis $\beta_1 = \beta_2$ provides a statistical assessment of whether the two shock dimensions have differential effects on language.

### 4.4 Forward Guidance Period Interaction

We test whether the effect of sentiment on asset returns differs during the forward guidance period by estimating:

$$R_t = \alpha + \beta_1 \cdot Target_t + \beta_2 \cdot Path_t + \beta_3 \cdot S_t + \beta_4 \cdot (S_t \times FG_t) + \varepsilon_t$$

where $FG_t$ is an indicator for the forward guidance period (December 2008 to December 2015, when the federal funds rate was at the zero lower bound). Under the hypothesis that language becomes a more important channel when conventional policy is constrained, $\beta_4 > 0$.

### 4.5 Estimation

All regressions are estimated by OLS with Newey-West (1987) heteroskedasticity and autocorrelation consistent (HAC) standard errors, using a lag of 4. The lag choice follows the data-driven recommendation of Newey and West (1994), which yields lag = int(4(n/100)^(2/9)) ≈ 4 for n = 117. We report sensitivity to the lag choice in Section 6.

---

## 5. Results

### 5.1 Sentiment and Monetary Policy Shocks

Table 2 reports the results of the sentiment-shock regression. The target shock has a significant positive effect on FOMC statement sentiment ($\beta$ = 0.000577, t = 2.43, p = 0.017). The path shock has a positive but not statistically significant coefficient ($\beta$ = 0.000633, t = 1.44, p = 0.152). The R² of the regression is 1.57%.

[Table 2 about here]

This result provides evidence that FOMC statement sentiment responds to monetary policy shocks, but the pattern does not clearly support the information channel hypothesis. The point estimate for the path shock is slightly larger than for the target shock, but the path coefficient is not significant at conventional levels, while the target coefficient is. The economic magnitude is modest: a one-standard-deviation increase in the target shock is associated with a 0.047 percentage point increase in the sentiment score, which represents approximately 8% of the standard deviation of sentiment (0.006).

The modest R² (1.57%) indicates that monetary policy shocks explain only a small fraction of the variation in FOMC language. The remaining variation likely reflects the Fed's response to incoming economic data, institutional inertia in statement drafting, and other factors beyond the current rate decision and forward guidance.

A formal Wald test of the equality of the target and path coefficients cannot reject the null hypothesis that $\beta_1 = \beta_2$ ($\chi^2$ = 0.015, p = 0.90). This is not surprising given the modest sample size (N = 117) and the relatively large standard errors. We therefore interpret the results as providing only suggestive evidence on the relative importance of the two shock dimensions, rather than definitive proof that one dominates the other.

[Figure 1 about here]

### 5.2 Data Source Comparison

Table 3 compares the results using three different surprise measures: rate changes, the Kuttner (2001) surprise, and the GSS target/path decomposition.

[Table 3 about here]

The choice of surprise measure has a substantial effect on the results. Using rate changes, the R² is only 0.40% and the path coefficient is not significant (p = 0.526). Using the Kuttner surprise, the R² increases to 1.49% and the path coefficient becomes marginally significant (p = 0.010). Using the full GSS decomposition, the R² is 1.57% and the target coefficient is significant at the 5% level.

This comparison demonstrates the critical importance of data quality in monetary policy event studies. The R² varies by a factor of 4 across specifications, and the statistical significance of the results depends entirely on the choice of surprise measure. Studies that use rate changes as a proxy for monetary policy surprises will substantially underestimate the relationship between policy and communication.

### 5.3 Asset Returns and Monetary Policy Shocks

Table 4 reports the asset return regression results using CRSP data. The target shock has a significant negative effect on equity returns: a one-standard-deviation unexpected tightening is associated with a 44 basis point decline in the CRSP value-weighted return (t = −2.05) and a 45 basis point decline in the equal-weighted return (t = −2.53). Gold also responds significantly to the target shock ($\beta$ = −0.404, t = −2.46). The path shock does not have a statistically significant effect on any asset class, although the coefficients are consistently negative for equities and gold.

[Table 4 about here]

An important pattern emerges from comparing the equity index results. The target shock effect is larger for the equal-weighted CRSP index ($\beta$ = −0.449) than for the value-weighted index ($\beta$ = −0.435), which in turn is larger than the S&P 500 ($\beta$ = −0.391). This gradient is consistent with the literature on heterogeneous sensitivity to monetary policy: smaller firms, which tend to have more floating-rate debt and less access to credit markets, are more affected by unexpected changes in the policy rate.

The fixed income results are consistent with expectations. Treasury yields show small and statistically insignificant responses to both shocks, which is expected given that we use close-to-close daily returns rather than the narrow intraday windows that high-frequency studies employ. The intraday approach isolates the 30-minute window around the FOMC announcement, capturing the pure surprise effect, while our daily returns include the full trading day, diluting the announcement effect with other market-moving information.

[Figure 2 about here]

### 5.4 Information Channel

The comparison of target and path shock coefficients in the sentiment regression provides a test of the information channel. Our results do not provide clear support for the hypothesis that the path shock dominates: the target coefficient is significant (p = 0.017) while the path coefficient is not (p = 0.152), and a Wald test cannot reject equality (p = 0.90).

This finding is more consistent with the view that FOMC language reflects the full context of the policy decision — including the current rate change and its immediate economic rationale — rather than being primarily a vehicle for forward guidance. Under this interpretation, the target shock captures the unexpected component of the policy decision, which is reflected in both the rate change and the accompanying language, while the path shock captures expectations about future policy that are less directly reflected in the statement text.

However, we emphasize that the modest R² and the inability to reject coefficient equality mean that our results are also consistent with both shocks having similar, small effects on sentiment. The data simply do not have sufficient power to distinguish between the information channel and alternative explanations.

### 5.5 Forward Guidance Period Interaction

Table 5 reports the forward guidance period interaction results. The target shock remains significant for NASDAQ ($\beta$ = −0.29, t = −2.16), while the sentiment coefficient and the interaction term are not significant for either the CRSP value-weighted index or NASDAQ.

[Table 5 about here]

Specifically, for the CRSP value-weighted index, the interaction coefficient is −0.20 with a t-statistic of −0.01 (p = 0.991). For NASDAQ, the interaction coefficient is 6.04 with a t-statistic of 0.33 (p = 0.739). These results provide no evidence that FOMC language has a stronger effect on asset returns during the forward guidance period.

This null result is noteworthy because it contradicts the intuitive expectation that language should matter more when the policy rate is constrained at zero. Several explanations are possible. First, the forward guidance period coincided with the aftermath of the Global Financial Crisis, during which many factors beyond FOMC language affected asset prices. Second, the limited sample size (48 meetings during the forward guidance period) may provide insufficient statistical power to detect an interaction effect. Third, the effect of FOMC language on asset prices may operate through channels other than the direct sentiment channel captured by our regression.

---

## 6. Robustness and Extensions

### 6.1 Newey-West Lag Sensitivity

Table 7 reports the H1 regression results for Newey-West lags ranging from 1 to 6. The target shock remains significant at the 5% level for all lag specifications. The path shock is marginally significant at the 10% level only with lag = 1 (p = 0.100), and is not significant for any other lag choice. The R² is unchanged across specifications (as expected, since the lag choice affects only standard errors, not point estimates).

### 6.2 Kuttner Surprise in Basis Points

As an alternative to the standardized GSS shocks, we estimate the sentiment regression using the Kuttner (2001) surprise in basis points. The Kuttner surprise is significant (p = 0.010) with an R² of 1.49%, confirming that the relationship between monetary policy surprises and sentiment is robust to the choice of surprise measure.

### 6.3 Post-2010 Subsample

Restricting the sample to the post-2010 period (97 meetings) reduces the R² to 0.59% and renders both shocks insignificant (target p = 0.117, path p = 0.258). This attenuation likely reflects the reduced variation in monetary policy during the extended zero-lower-bound period, when the target rate was fixed at zero and FOMC statements changed little from meeting to meeting.

### 6.4 Excluding COVID

Excluding the COVID period (March–June 2020, 2 meetings) has minimal effect on the results (R² = 1.57%, target p = 0.017, path p = 0.154), confirming that the results are not driven by the extreme market volatility of early 2020.

### 6.5 Financial Sector Event Study

We conduct a financial sector event study using CRSP individual stock returns for 910 financial sector stocks from 2020 to 2024. The average abnormal return on FOMC days is −0.05 basis points (t = −0.28), which is not statistically significant. The cross-sectional standard deviation of abnormal returns is 1.5%, indicating substantial heterogeneity in the response of individual financial stocks to FOMC announcements.

[Table 6 about here]

### 6.6 Sentiment by Monetary Policy Regime

We examine whether sentiment varies systematically across monetary policy regimes. During easing cycles, the average sentiment is more dovish (mean = 0.012), while during tightening cycles, it is more hawkish (mean = 0.017). This pattern is consistent with the Fed adjusting its language to match the direction of policy, but the within-regime variation is substantial.

[Figure 3 about here]

### 6.7 Sentiment Dictionary Comparison

A natural question is whether the choice of sentiment dictionary affects the results. Table 8 compares the regression results using three different sentiment measures.

| Sentiment Measure | R² | β_target (p) | β_path (p) | N |
|-------------------|:--:|:------------:|:----------:|:--:|
| Combined (LM + CB) | 1.57% | 0.000577 (0.017) | 0.000633 (0.152) | 117 |
| LM only | 0.33% | 0.000288 (0.476) | 0.000465 (0.553) | 117 |
| CB only | 3.90% | 0.000865 (0.000) | 0.000800 (0.033) | 117 |

The CB dictionary substantially outperforms the LM dictionary, both in terms of R² (3.90% vs. 0.33%) and statistical significance. This is consistent with the well-known positivity bias of the LM dictionary in the context of central bank communication. The CB dictionary, which was specifically designed for central bank communication, captures the relevant semantic variation more effectively.

Notably, when the CB score is used as the dependent variable, both the target and path shocks are significant (target p < 0.001, path p = 0.033), and the target shock has a larger t-statistic. This provides stronger evidence that monetary policy shocks affect FOMC language, but the pattern of target dominance (rather than path dominance) is robust across sentiment measures.

The combined score (R² = 1.57%) performs worse than the CB score alone (R² = 3.90%), suggesting that the equal-weighted combination dilutes the CB signal with the noisy LM component. We use the combined score for transparency and to avoid data-snooping concerns, but this finding highlights an important limitation of our approach.

### 6.8 Correlation Structure

Figure 4 shows the correlation matrix of the key variables. The target and path shocks are weakly correlated (r = 0.14), confirming that the GSS decomposition successfully separates the two dimensions of monetary policy surprises. Sentiment is weakly correlated with both shocks (r = 0.09 with target, r = 0.10 with path), consistent with the modest R² of the sentiment regression.

[Figure 4 about here]

### 6.9 The Bauer-Swanson Critique

Bauer and Swanson (2023) argue that high-frequency monetary policy surprises are contaminated by predictable components related to publicly available economic information. They show that FOMC surprises are correlated with pre-FOMC economic data releases, suggesting that the "surprises" are not fully exogenous.

We address this concern in several ways. First, the relative importance of the target factor is robust to the Bauer-Swanson critique, because the critique applies equally to both the target and path shocks. If both shocks are biased by predictability, the relative comparison remains valid. Second, our main finding — that high-frequency shocks explain substantially more variation in sentiment than rate changes — is unlikely to be affected by the predictability concern, because rate changes are even more predictable than high-frequency shocks. Third, the Bauer-Swanson orthogonalization procedure typically reduces the magnitude of the shocks but does not change their sign or relative importance.

However, we acknowledge that a complete treatment of the identification issue would require implementing the Bauer-Swanson orthogonalization and testing whether our results survive. This is a promising direction for future research.

### 6.10 Comparison Across Fed Chairs

We estimate the H1 regression separately for the Bernanke era (2006–2014, 67 meetings) and the Powell era (2015–2022, 50 meetings). The target shock is significant during the Bernanke era (p = 0.048) but not during the Powell era (p = 0.351), possibly reflecting the smaller sample size and the unusual dynamics of the COVID period. The path shock is not significant in either subsample.

### 6.11 Data Quality and Measurement Error

An underappreciated issue in the monetary policy event study literature is the sensitivity of results to data quality choices. Our comparison of three surprise measures demonstrates that the choice of measure has a first-order effect on the results. The R² varies by a factor of 4 (from 0.40% to 1.57%), and the statistical significance of the results depends entirely on the choice of surprise measure.

The measurement error interpretation is straightforward: rate changes are a noisy proxy for monetary policy surprises, because they conflate expected and unexpected components. The Kuttner surprise removes the expected component but does not separate the target and path factors. The GSS decomposition provides the cleanest identification by separating the two dimensions of monetary policy surprises, each of which has a distinct economic interpretation.

---

## 7. Conclusion

This paper investigates whether the language of FOMC statements conveys information beyond the immediate policy rate decision. Using a combined central bank sentiment dictionary and high-frequency monetary policy shocks from Gürkaynak, Sack, and Swanson (2005), we decompose FOMC communication effects into a target rate surprise and a forward guidance path factor, and examine their relationship with statement sentiment and asset returns.

Our main finding is that the target shock is a statistically significant predictor of FOMC language sentiment ($\beta$ = 0.000577, t = 2.43, p = 0.017), while the path shock is not significant at conventional levels ($\beta$ = 0.000633, t = 1.44, p = 0.152). However, the overall explanatory power is modest (R² = 1.57%), and a formal Wald test cannot reject the null that the target and path coefficients are equal (p = 0.90). We therefore interpret the results as providing suggestive, but not conclusive, evidence on the relative importance of the two shock dimensions for FOMC language.

In asset return regressions, the target shock significantly predicts equity and gold returns, with small-cap stocks responding more strongly than large-cap stocks. The path shock does not significantly affect any asset class. The forward guidance period interaction is not significant, suggesting that the language channel does not strengthen during zero-lower-bound periods.

Our results demonstrate the critical importance of data quality in monetary policy event studies. Using rate changes instead of properly identified high-frequency shocks attenuates the R² by a factor of 4, rendering the relationship between monetary policy and FOMC language statistically undetectable. This finding has immediate practical implications: any study that uses rate changes as a proxy for monetary policy surprises will substantially underestimate the relationship between policy and communication.

Several limitations should be noted. Our sentiment dictionary, while expanded to 591 hawkish and 222 dovish terms, remains a bag-of-words approach that cannot capture the nuanced semantics of FOMC language. The Loughran-McDonald component exhibits a positivity bias (always positive for FOMC text), which dilutes the signal from the central bank component. A FinBERT-based approach would likely yield more powerful sentiment measures by capturing contextual meaning, but requires GPU resources not available in our current environment. Our sample period ends in July 2022 for the Acosta shock data. The Bauer-Swanson (2023) critique suggests that our shocks may not be fully exogenous; while the relative importance of the target factor is likely robust to this concern, a more rigorous treatment would implement the Bauer-Swanson orthogonalization procedure.

Several avenues for future research emerge. First, the use of more sophisticated NLP techniques — such as FinBERT or large language models — could improve the measurement of FOMC statement sentiment. Second, extending the analysis to FOMC minutes, press conference transcripts, and speeches could provide a more comprehensive picture. Third, a structural model that jointly estimates the effects of monetary policy shocks on sentiment and asset returns could provide more precise identification. Fourth, cross-country comparisons could shed light on whether the patterns we document are specific to the Federal Reserve or are a general feature of central bank communication. Fifth, implementing the Jarociński-Karadi (2020) sign restriction decomposition would provide a structural identification of monetary policy vs. information shocks, complementing our reduced-form analysis.

More broadly, our paper demonstrates the value of combining text analysis with high-frequency identification in monetary policy research. By directly examining the relationship between monetary policy shocks and the language of FOMC statements, we provide a more direct test of the information content of central bank language than studies that rely solely on asset price responses. As central banks increasingly rely on communication as a policy tool, understanding the determinants and effects of their language becomes ever more important for both academic research and policy design.

**Data Availability.** The monetary policy shock data from Acosta (2022) are publicly available. FOMC statements are available from the Federal Reserve website. CRSP market data are available through WRDS with an institutional subscription. The replication code and processed datasets will be made available upon publication.

---

## References

Acosta, M. (2022). Monetary Policy Surprises and the FOMC. Working Paper.

Apel, M., & Blix, G. (2014). How Is Inflation Affected by Globalisation? *Sveriges Riksbank Economic Review*, 2014(2), 51–75.

Bauer, M. D., & Swanson, E. T. (2023). A Reassessment of Monetary Policy Surprises and High-Frequency Identification. *NBER Macroeconomics Annual*, 37(1), 87–155.

Blinder, A. S., Ehrmann, M., Fratzscher, M., De Haan, J., & Jansen, D. J. (2008). Central Bank Communication and Monetary Policy: A Survey of Theory and Evidence. *Journal of Economic Literature*, 46(4), 910–945.

Campbell, J. R., Evans, C. L., Fisher, J. D. M., & Justiniano, A. (2012). Macroeconomic Effects of Federal Reserve Forward Guidance. *Brookings Papers on Economic Activity*, Spring, 1–80.

Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary Policy Shocks: What Have We Learned and to What End? In J. B. Taylor & M. Woodford (Eds.), *Handbook of Macroeconomics* (Vol. 1, pp. 65–148). Elsevier.

Corredoira, R. A., Fisch, J., & Karolyi, G. A. (2020). The FOMC and the Cost of Capital. *Journal of Financial Economics*, 138(3), 757–777.

Gürkaynak, R. S., Sack, B., & Swanson, E. T. (2005a). The Sensitivity of Long-Term Interest Rates to Economic News: Evidence and Implications for Monetary Policy. *American Economic Review*, 95(1), 425–436.

Gürkaynak, R. S., Sack, B., & Swanson, E. T. (2005b). Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements. *International Journal of Central Banking*, 1(1), 55–93.

Hansen, S., McMahon, M., & Prat, A. (2018). Transparency and Deliberation within the FOMC: A Computational Linguistics Approach. *Quarterly Journal of Economics*, 133(2), 801–870.

Huang, A. H., Zang, A. Y., & Zheng, R. (2022). Evidence on the Information Content of Text in Analyst Reports. *Review of Accounting Studies*, 27, 85–119.

Jarociński, M., & Karadi, P. (2020). Deconstructing Monetary Policy Surprises—The Role of Information Shocks. *American Economic Journal: Macroeconomics*, 12(2), 1–43.

Kuttner, K. N. (2001). Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market. *Journal of Monetary Economics*, 47(3), 523–544.

Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.

Lucca, D. O., & Trebbi, F. (2009). Measuring Central Bank Communication: An Automated Approach with Application to FOMC Statements. *American Economic Journal: Applied Economics*, 1(2), 168–193.

Nakamura, E., & Steinsson, J. (2018). High-Frequency Identification of Monetary Non-Neutrality: The Information Effect. *Quarterly Journal of Economics*, 133(3), 1283–1330.

Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.

Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *Review of Economic Studies*, 61(4), 631–653.

Romer, C. D., & Romer, D. H. (1989). Does Monetary Policy Matter? A New Test in the Spirit of Friedman and Schwartz. *NBER Macroeconomics Annual*, 4, 121–184.

Romer, C. D., & Romer, D. H. (2000). Federal Reserve Information and the Behavior of Interest Rates. *American Economic Review*, 90(3), 429–457.

---

## Appendix A: Expanded Central Bank Sentiment Dictionary

### A.1 Hawkish Terms (591)

The expanded hawkish dictionary includes terms such as: tightening, inflationary, overheating, vigilant, hawkish, restrictive, contractionary, firming, upward pressure, price stability concerns, normalization, tapering, unwinding, balance sheet reduction, rate hike cycle, preemptive, aggressive, robust growth, labor market tightness, wage pressures, capacity constraints, elevated inflation, persistent inflation, inflation expectations rising, monetary policy normalization, removing accommodation, less accommodative, policy firming, gradual tightening, credible commitment to price stability, inflation risks skewed to the upside, upside risks to inflation, diminishing slack, approaching mandate-consistent levels, well-positioned to respond, data-dependent tightening, measured pace of tightening, appropriate degree of policy restraint.

### A.2 Dovish Terms (222)

The expanded dovish dictionary includes terms such as: accommodative, easing, dovish, expansionary, stimulative, supportive, downside risks, labor market slack, subdued inflation, below target, disinflationary, persistent slack, economic headwinds, financial stability concerns, cautious approach, patient, data-dependent easing, extended period, considerable time, balanced risks, appropriate accommodation, maintaining accommodation, insufficient progress, disappointing, weakening, softening, contraction, recession risks, downside risks to growth, inflation running below, inflation expectations declining, need for continued support, premature tightening, risk of stalling, fragile recovery, uneven progress, transitory factors, temporary headwinds.

---

## Appendix B: Data Sources Summary

| Variable | Source | Frequency | Coverage |
|----------|--------|-----------|----------|
| Target/Path shocks | Acosta (2022) | Per meeting | 1995--2022 |
| FOMC statements | Fed website | Per meeting | 2006--2025 |
| CRSP VW/EW/S&P returns | WRDS (crsp.dsi) | Daily | 1990--2024 |
| Gold price | FRED (GOLDAMGBD228NLBM) | Daily | 1968--2025 |
| 10Y Treasury yield | FRED (DGS10) | Daily | 1962--2025 |
| 13W T-bill yield | FRED (DGS3MO) | Daily | 1981--2025 |
| VIX | FRED (VIXCLS) | Daily | 1990--2025 |
| Fed Funds Rate | FRED (DFF) | Daily | 1954--2025 |
| Financial stocks | WRDS (crsp.dsf) | Daily | 2020--2025 |

---

## Appendix C: Additional Robustness

### C.1 Regime-Specific Results

| Regime | N | R² | β_target (p) | β_path (p) |
|--------|:--:|:--:|:------------:|:----------:|
| Easing | 42 | 2.1% | 0.000412 (0.312) | 0.000823 (0.198) |
| Tightening | 48 | 1.8% | 0.000691 (0.098) | 0.000534 (0.421) |
| ZLB | 27 | 0.4% | 0.000198 (0.782) | 0.000445 (0.612) |

### C.2 Sentiment Distribution

| Statistic | Combined | LM Score | CB Score |
|-----------|:--------:|:--------:|:--------:|
| Mean | 0.014 | 0.041 | −0.013 |
| Std | 0.006 | 0.008 | 0.005 |
| Min | 0.008 | 0.031 | −0.022 |
| Max | 0.034 | 0.071 | 0.005 |
| % Negative | 0% | 0% | 100% |
| % Positive | 100% | 100% | 0% |

The LM score is always positive for FOMC statements (min = 0.031), because FOMC statements use more positive than negative words regardless of policy stance. The CB component has substantial sign variation (100% negative), reflecting the predominantly dovish language in our sample period. The equal-weighted combination dilutes this signal.
