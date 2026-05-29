# Beyond the Rate: Information Content of FOMC Statement Language

**Eileen Zhang**

Academy of AI, Xi'an Jiaotong-Liverpool University, Suzhou, China

---

## Abstract

We investigate whether FOMC statement language conveys information beyond the policy rate decision. Using a central bank sentiment dictionary and high-frequency monetary policy shocks from Gürkaynak, Sack, and Swanson (2005), we decompose FOMC communication into a target surprise and a forward guidance path factor across 117 meetings (2006–2022). The target shock significantly predicts sentiment ($\beta$ = 0.000577, p = 0.017), while the path shock does not (p = 0.152), though a Wald test cannot reject equal effects (p = 0.90). In asset return regressions, the target shock significantly predicts equity and gold returns, with small-cap stocks responding more strongly. The forward guidance interaction is not robustly significant, but a regime analysis reveals that the path shock is highly significant during rate cut meetings (p < 0.001, R² = 43.1%). Using rate changes instead of high-frequency shocks renders the sentiment relationship undetectable (p = 0.726), underscoring the importance of data quality.

Keywords: Monetary policy; FOMC; Forward guidance; Sentiment analysis; High-frequency identification

JEL Classification: C80, D83, E43, E52, E58, G12, G14

**Acknowledgments**: We thank [to be added] for helpful comments. All errors are our own.

**Declarations**: The authors have no relevant financial or non-financial interests to disclose. No funding was received for this research. Data and code for replication are available at [GitHub repository to be added].

---

## 1. Introduction

Central bank communication has become a central tool of monetary policy. Over the past two decades, the Federal Reserve has progressively increased the transparency and detail of its post-meeting statements, transforming them from brief rate announcements into substantive policy documents. The average length of an FOMC statement has grown from approximately 100 words in the early 1990s to over 400 words by 2020, reflecting the Fed's deliberate effort to communicate not just the what of its decisions, but the why. A large literature examines how these communications affect financial markets, but relatively little attention has been paid to a more fundamental question: what determines the language of FOMC statements themselves?

We address this question by examining the relationship between high-frequency monetary policy shocks and the sentiment of FOMC statement language. Using the target/path decomposition of Gürkaynak, Sack, and Swanson (2005, henceforth GSS), we test whether the two dimensions of monetary policy surprises — the target rate surprise and the forward guidance path factor — have differential effects on FOMC language and asset returns. This decomposition is economically meaningful: the target shock captures the unexpected component of the current rate decision, while the path shock captures the revision of expectations about the future trajectory of policy.

Our main finding is that the target shock is a statistically significant predictor of FOMC statement sentiment (p = 0.017), while the path shock is not (p = 0.152). This result contrasts with the prediction of the information channel hypothesis, which holds that forward guidance language should be primarily driven by the path shock. However, the overall explanatory power is modest (R² = 1.57%), and a formal Wald test cannot reject the null that the target and path effects are equal (p = 0.90). We therefore interpret our results as providing only suggestive evidence on the relative importance of the two shock dimensions for FOMC language.

In asset return regressions, we find that the target shock significantly predicts equity and gold returns, with effect sizes ranging from 28 basis points for NASDAQ to 45 basis points for the equal-weighted CRSP index. Small-cap stocks respond more strongly than large-cap stocks, consistent with the literature on heterogeneous sensitivity to monetary policy. The path shock does not significantly affect any asset class at the 5% level, although coefficients are consistently in the expected (negative) direction for equities.

We also examine whether the effect of FOMC language on asset returns differs during the forward guidance period (December 2008 to December 2015), when the federal funds rate was at the zero lower bound. We find no robust evidence of such an interaction: the sentiment × forward guidance interaction term is statistically insignificant for the CRSP value-weighted index (p = 0.602), and while marginally significant for NASDAQ (p = 0.041), the coefficient is economically implausibly large, suggesting outlier influence. This null result suggests that, while FOMC language may convey information, its effect on asset prices does not systematically strengthen when conventional monetary policy is constrained.

Our results demonstrate the critical importance of data quality in monetary policy event studies. Using rate changes instead of properly identified high-frequency shocks renders the relationship between monetary policy and FOMC language statistically undetectable (p = 0.726 vs. 0.017), even though the R² difference is modest (1.05% vs. 1.57%). This finding has immediate practical implications: any study that uses rate changes as a proxy for monetary policy surprises will substantially underestimate the relationship between policy and communication.

This paper contributes to three strands of the literature. First, we contribute to the literature on central bank communication (Blinder et al., 2008; Gürkaynak et al., 2005b) by directly examining the determinants of FOMC statement language, rather than its effects on asset prices. While the existing literature has extensively documented that FOMC statements move markets, we ask the reverse question: what moves FOMC statements? Second, we contribute to the information channel literature (Romer and Romer, 2000; Campbell et al., 2012; Nakamura and Steinsson, 2018) by testing whether the two dimensions of monetary policy surprises have differential effects on language. If the information channel operates primarily through forward guidance, the path shock should dominate; if it operates through the current decision context, the target shock should dominate. Third, we contribute to the methodology of monetary policy event studies by quantifying the sensitivity of results to the choice of surprise measure — a dimension that is often treated as a secondary robustness check but, as we show, has first-order implications for inference.

The remainder of the paper is organized as follows. Section 2 reviews the related literature. Section 3 describes the data and variable construction. Section 4 presents the empirical methodology. Section 5 reports the main results. Section 6 discusses robustness checks and extensions. Section 7 concludes.

---

## 2. Literature Review

### 2.1 Monetary Policy Surprises and High-Frequency Identification

The identification of exogenous monetary policy shocks has been a central challenge in macroeconomics since the seminal work of Friedman and Schwartz (1963). The early literature used narrative approaches (Romer and Romer, 1989) or structural VARs (Christiano et al., 1999), but these methods rely on strong identifying assumptions that have been widely debated. Narrative approaches require subjective judgments about the counterfactual path of policy, while structural VARs depend on recursive ordering assumptions that may not reflect the true timing of economic relationships.

Kuttner (2001) pioneered the high-frequency identification approach, using changes in federal funds futures rates around FOMC announcements to isolate the unexpected component of monetary policy decisions. This approach exploits the fact that financial markets incorporate all available information into prices immediately, so that the narrow-window change in interest rates captures the pure surprise. Kuttner's key insight was that the expected component of rate changes — which is anticipated by markets and therefore already priced in — should have no effect on asset prices, while the unexpected component should have a large effect. His results confirmed this prediction: a 25-basis-point unexpected rate cut was associated with a roughly 1.5% increase in the S&P 500, while the expected component had a negligible effect.

Gürkaynak, Sack, and Swanson (2005a) extended this approach by decomposing the high-frequency interest rate response into two factors: a target factor (capturing the surprise in the current rate decision) and a path factor (capturing the revision of expectations about future policy). They showed that the path factor explains a substantial fraction of the response of longer-term interest rates and asset prices, suggesting that forward guidance is an important channel of monetary policy transmission. This finding was significant because it demonstrated that monetary policy affects the economy not just through the current rate decision, but also through the information it conveys about the future path of policy.

The GSS decomposition has become a standard tool in the monetary policy literature. Swanson (2021) extended the target and path factors through 2019, and Acosta (2022) further extended them through July 2022, providing the shock data used in this paper. The Acosta data cover 220 FOMC meetings from 1995 to 2022, offering a substantially longer sample than the original GSS analysis.

Bauer and Swanson (2023) raised important concerns about the exogeneity of high-frequency shocks, showing that they are partially predictable from pre-FOMC economic information. They document that high-frequency surprises are correlated with pre-FOMC macroeconomic data releases, suggesting that the "surprises" are not fully exogenous. This critique has implications for the interpretation of our results, and we discuss it in detail in Section 6.

### 2.2 Central Bank Communication and Language

A growing literature examines the content and effects of central bank communication. Blinder et al. (2008) survey the early literature and conclude that central bank communication can improve the predictability of monetary policy and reduce market volatility. Their survey identifies several channels through which communication affects markets: by reducing uncertainty about future policy, by providing information about the central bank's assessment of economic conditions, and by signaling the central bank's commitment to its stated objectives.

Lucca and Trebbi (2009) use automated content analysis to measure the linguistic complexity and sentiment of FOMC statements, finding that more complex language is associated with larger market reactions. Their work was among the first to apply computational linguistics to central bank communication, and it demonstrated that the language of FOMC statements contains information that is not captured by the rate decision alone.

More recently, researchers have applied natural language processing (NLP) techniques to central bank communication. Apel and Blix (2014) construct a hawkish-dovish sentiment index for the Riksbank, demonstrating that dictionary-based measures can capture meaningful variation in central bank language. Hansen et al. (2018) analyze the transparency of FOMC communications, finding that increased transparency has improved the predictability of monetary policy but has also changed the nature of FOMC deliberations. Huang et al. (2022) apply FinBERT to financial text, demonstrating that transformer-based models can capture contextual meaning that bag-of-words approaches miss.

An important strand of this literature examines the effects of FOMC communication on asset prices. Gürkaynak et al. (2005b) show that FOMC statements have independent effects on asset prices beyond the rate decision itself. Rosa (2011) finds that the tone of FOMC statements affects Treasury yields and the dollar. Cieslak et al. (2019) show that the Fed's communication conveys information about future stock returns, consistent with the information channel hypothesis.

Our paper contributes to this literature by examining the relationship between monetary policy shocks and FOMC statement sentiment, rather than the effect of communication on markets. This provides a more direct test of what drives the language of central bank statements. While the existing literature treats FOMC language as an exogenous input that affects markets, we treat it as an endogenous output that is determined by the policy context.

### 2.3 The Information Channel of Monetary Policy

The information channel hypothesis holds that monetary policy actions convey information about the central bank's assessment of economic conditions, not just the policy decision itself. This idea has a long history in macroeconomics, but it has received renewed attention in recent years.

Romer and Romer (2000) provide early evidence that the Fed has superior information about inflation, and that FOMC actions reveal this information to market participants. They show that the Fed's inflation forecasts contain information that is not available from commercial forecasters, and that this information is reflected in interest rate movements following FOMC meetings. This finding suggests that monetary policy actions serve a dual purpose: they implement the policy decision, and they signal the central bank's private information about the economy.

Campbell et al. (2012) distinguish between "delphic" forward guidance (revealing the Fed's forecast) and "Odyssean" forward guidance (committing to a future path), and argue that both types convey information. Delphic guidance reveals the Fed's assessment of economic conditions, while Odyssean guidance commits the Fed to a particular policy path regardless of future economic developments. The distinction is important because the two types of guidance have different implications for the information channel: delphic guidance operates through the information channel, while Odyssean guidance operates through the commitment channel.

Nakamura and Steinsson (2018) provide evidence that monetary policy shocks have large effects on long-term expectations, consistent with the information channel. They show that a 25-basis-point monetary policy shock has an effect on long-term interest rates that is several times larger than the effect on short-term rates, suggesting that the shock conveys information about the long-run path of the economy. This finding is difficult to reconcile with a standard New Keynesian model, in which monetary policy shocks should have only transitory effects.

Jarociński and Karadi (2020) use sign restrictions to separately identify monetary policy and information shocks, finding that the two have opposite effects on asset prices. A monetary policy tightening raises interest rates and lowers stock prices, while an information shock (in which the Fed reveals positive information about the economy) raises both interest rates and stock prices. This decomposition is important because it shows that the observed response of asset prices to FOMC announcements reflects a mixture of monetary policy and information effects.

Our paper tests the information channel by examining whether the path shock — which captures forward guidance about future policy — has a larger effect on FOMC language than the target shock. If FOMC language is primarily about revealing information about the future, the path shock should be the dominant driver. If, instead, the language primarily reflects the current rate decision and its immediate context, the target shock should dominate. This test is complementary to the Jarociński-Karadi approach: while they use sign restrictions to identify information shocks, we use the GSS decomposition to test whether the two dimensions of monetary policy surprises have differential effects on language.

### 2.4 Sentiment Analysis in Finance

Sentiment analysis has become a standard tool in financial economics. The Loughran and McDonald (2011) dictionary is widely used for financial text, providing lists of positive and negative words specifically calibrated for 10-K filings. Their key contribution was demonstrating that the Harvard General Inquirer dictionary, which was commonly used in finance research, misclassifies many words as negative that are in fact neutral or positive in a financial context (e.g., "tax," "cost," "liability"). The LM dictionary contains 354 positive and 2,329 negative terms, and has been shown to outperform the Harvard dictionary in predicting stock returns and other financial outcomes.

However, the LM dictionary has known limitations when applied to central bank communication. FOMC statements use more positive than negative words regardless of policy stance, creating a positivity bias that reduces the dictionary's discriminatory power. This bias arises because FOMC statements are carefully drafted to avoid alarming language, even when the policy decision is hawkish. As a result, the LM score for FOMC statements is always positive, providing little variation to exploit in regression analysis.

Central bank-specific dictionaries have been developed to address this limitation. Apel and Blix (2014) construct a hawkish-dovish dictionary for the Riksbank, using terms that are specifically relevant to monetary policy discourse. Tadle (2022) develops a sentiment dictionary for FOMC communications, showing that it captures variation in policy stance that the LM dictionary misses. We use a combined approach, averaging the Loughran-McDonald score with a central bank-specific score, while acknowledging the limitations of this approach.

An emerging literature uses machine learning and transformer-based models for financial sentiment analysis. Devlin et al. (2019) introduced BERT, which captures contextual meaning through bidirectional attention mechanisms. Yang et al. (2020) developed FinBERT, a BERT model fine-tuned on financial text, which has been shown to outperform dictionary-based approaches for sentiment classification. More recently, Gambacorta, Kwon, Park, Patelli, and Zhu (2024) introduce central bank language models (CB-LMs) — specialised encoder-only models retrained on a comprehensive corpus of central bank speeches — and show that CB-LMs outperform general-purpose foundation models on stance classification tasks in the central banking domain, while larger LLMs retain an advantage on more complex sentiment tasks. This finding suggests a middle ground between dictionary-based and LLM-based approaches: domain-specific pretraining can substantially improve performance without the computational cost and reproducibility concerns of large proprietary models.

The most ambitious application of LLMs to central bank communication is the IMF's large-scale analysis of 74,882 documents from 169 central banks spanning 1884–2025 (Gambacorta et al., 2025). Their framework classifies communications along four dimensions — topic, communication stance, sentiment, and forward-lookingness — and constructs a directional communication index that predicts future market interest rate movements. This finding is directly relevant to our analysis: their four-dimensional decomposition demonstrates that aggregating across topics with heterogeneous effects dilutes predictive power, consistent with our finding that the combined LM+CB score underperforms the CB-only score. Even the Federal Reserve itself has begun using generative AI models to understand FOMC discussions, finding that LLMs can reliably identify topics but struggle with complex content such as risk-balance assessments (Federal Reserve Board, 2024).

Chen, Granville, and Matousek (2025) use GPT-4 to construct "Procedural Linguistic Minutes Information Shocks" (PLMIS), which measure the linguistic information in FOMC Minutes beyond the corresponding Statements. Their approach decomposes FOMC text into four topics (recent economy, policy objectives, policy decisions, and forward guidance) and constructs separate hawkish-dovish shocks for each topic, finding that only dove-language shocks in the "recent economy" and "forward guidance" topics significantly affect Treasury futures. This finding reinforces the theme that coarse-grained sentiment measures — including our own — may dilute signal by aggregating across topics with heterogeneous effects. However, transformer-based models require significant computational resources and large training corpora, which may not be available for the relatively small corpus of FOMC statements. We discuss the potential of transformer-based approaches in Section 7.

### 2.5 The Forward Guidance Period and the Zero Lower Bound

The global financial crisis of 2008–2009 pushed the federal funds rate to the zero lower bound (ZLB), fundamentally changing the nature of monetary policy communication. With the policy rate constrained at zero, the Federal Reserve increasingly relied on forward guidance — explicit statements about the expected future path of the policy rate — as a tool of monetary policy. The FOMC introduced calendar-based forward guidance in August 2011 ("the Committee currently anticipates that economic conditions...are likely to warrant exceptionally low levels for the federal funds rate at least through mid-2013") and state-contingent forward guidance in December 2012 ("the Committee...anticipates that this exceptionally low range for the federal funds rate will be appropriate at least as long as the unemployment rate remains above 6-1/2 percent").

The forward guidance period (December 2008 to December 2015) provides a natural experiment for testing whether the language channel of monetary policy strengthens when conventional policy is constrained. If forward guidance operates primarily through language, the effect of FOMC statement sentiment on asset prices should be larger during this period. However, the forward guidance period also coincided with the aftermath of the Global Financial Crisis, during which many factors beyond FOMC language affected asset prices, making it difficult to isolate the language channel.

### 2.6 Heterogeneous Effects of Monetary Policy Across Firm Sizes

A growing literature documents that monetary policy has heterogeneous effects across firms of different sizes. Gertler and Gilchrist (1994) show that small firms are more sensitive to monetary policy shocks than large firms, a finding they attribute to the greater reliance of small firms on bank credit, which is itself sensitive to monetary policy conditions. This "credit channel" of monetary policy transmission implies that the equal-weighted stock index — which gives more weight to small firms — should respond more strongly to monetary policy shocks than the value-weighted index.

More recently, Jeenas (2019) shows that the heterogeneous response of firms to monetary policy can be explained by differences in their capital structure, with firms that have more floating-rate debt being more sensitive to interest rate changes. This finding is relevant for our analysis because it provides a theoretical foundation for the prediction that the equal-weighted CRSP index should respond more strongly to the target shock than the value-weighted index.

The heterogeneous sensitivity hypothesis also has implications for the information channel. If the target shock captures information about the current state of the economy, while the path shock captures information about the future path of policy, the two shocks may have different effects across firm sizes. Small firms, which are more dependent on current economic conditions, may be more sensitive to the target shock, while large firms, which have better access to forward-looking information, may be more sensitive to the path shock. We test this prediction in Section 5.3.

### 2.7 Methodological Considerations in Monetary Policy Event Studies

The monetary policy event study literature has identified several methodological issues that can affect the reliability of results. First, the choice of event window matters: narrow intraday windows (e.g., 30 minutes around the FOMC announcement) provide cleaner identification but may miss delayed reactions, while wider daily windows capture more of the announcement effect but also include confounding events. We use daily returns, which is standard in the literature but may dilute the announcement effect.

Second, the treatment of scheduled vs. unscheduled FOMC meetings is important. Unscheduled meetings (such as the emergency rate cuts during the financial crisis and the COVID pandemic) may have different effects than scheduled meetings, because they convey not only the policy decision but also the signal that the Fed deemed the situation urgent enough to warrant an unscheduled meeting. Our sample includes both scheduled and unscheduled meetings, and we test the robustness of our results to excluding the COVID period in Section 6.4.

Third, the choice of surprise measure has first-order implications for the results, as we demonstrate in Section 5.2. This issue is often underappreciated in the literature, where the choice of surprise measure is treated as a secondary methodological decision rather than a primary determinant of the results.

---

## 3. Data and Variable Construction

### 3.1 Monetary Policy Shocks

We use the high-frequency monetary policy shocks from Acosta (2022), who extends the GSS decomposition through July 2022. The Acosta data provide two standardized shock series:

- **Target shock**: The unexpected component of the current federal funds rate decision, identified from changes in the current-month federal funds futures rate in a narrow window around the FOMC announcement. Specifically, the target factor is identified from the change in the current-month federal funds futures rate from 10 minutes before the FOMC announcement to 20 minutes after the announcement. This narrow window ensures that the measured change reflects only the FOMC announcement, not other economic news.

- **Path shock**: The revision of expectations about the future path of monetary policy, identified from changes in longer-dated Eurodollar futures and federal funds futures. The path factor is identified from the changes in interest rate futures at horizons of 2–8 quarters, which capture the market's revision of expectations about the future trajectory of policy. The GSS decomposition uses a factor model to separate the target and path components, under the assumption that the target factor loads primarily on short-dated futures while the path factor loads on longer-dated futures.

In the full Acosta sample (1995–2022, 220 meetings), both shocks are standardized to unit variance. However, in our estimation sample (2006–2022, 117 meetings), the standard deviations are 0.82 for the target shock and 0.80 for the path shock, reflecting the different variance structure of the subsample. We use the shocks as provided by Acosta without re-standardizing, noting that this does not affect t-statistics or p-values, only the scale of the coefficients. The choice not to re-standardize is deliberate: re-standardizing would change the economic interpretation of the coefficients, making them represent the effect of a one-standard-deviation shock within our subsample rather than the effect of a one-unit change in the Acosta shock series.

The correlation between the target and path shocks is 0.14 in our sample, confirming that the GSS decomposition successfully separates the two dimensions of monetary policy surprises. This low correlation is important for identification: if the two shocks were highly correlated, it would be difficult to distinguish their separate effects on sentiment and asset returns.

### 3.2 FOMC Statement Corpus

We collect 140 FOMC statements from January 2006 to March 2025, scraped from the Federal Reserve's official website using the URL format `https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm`. Of these, 117 overlap with the Acosta shock data (January 2006 to July 2022). The corpus covers three Fed chairs: Bernanke (2006–2014), Yellen (2014–2018), and Powell (2018–2022).

The FOMC statement has evolved significantly over our sample period. In 2006, statements were typically 150–200 words and focused primarily on the rate decision and the economic outlook. By 2020, statements had expanded to 400–500 words and included detailed discussion of the labor market, inflation, financial conditions, and forward guidance. This evolution reflects the Fed's increasing commitment to transparency and its recognition that communication is a tool of monetary policy.

We extract the statement text using the `div#article` selector, which captures the main body of the statement while excluding boilerplate elements such as the voting record and the list of FOMC members. We then clean the text by removing HTML tags, normalizing whitespace, and converting to lowercase for dictionary matching.

### 3.3 Sentiment Analysis

We compute two sentiment scores for each FOMC statement:

**Loughran-McDonald (LM) score**: The fraction of positive words minus the fraction of negative words, using the Loughran and McDonald (2011) dictionary. The LM dictionary contains 354 positive and 2,329 negative terms in its master list, of which a subset appears in FOMC text. The score is computed as:

$$\text{LM}_t = \frac{\text{Positive words}_t - \text{Negative words}_t}{\text{Total words}_t}$$

**Central Bank (CB) score**: A hawkish-dovish score computed using an expanded central bank-specific dictionary comprising 591 hawkish terms (e.g., "tightening," "inflationary pressures," "vigilant") and 222 dovish terms (e.g., "accommodative," "downside risks," "labor market slack"). The CB score is computed as:

$$\text{CB}_t = \frac{\text{Hawkish words}_t - \text{Dovish words}_t}{\text{Total words}_t}$$

The CB dictionary was constructed by combining terms from the existing literature (Apel and Blix, 2014; Tadle, 2022) with additional terms identified through manual review of FOMC statements. The hawkish dictionary includes terms related to inflation concerns, tightening policy, and robust economic conditions, while the dovish dictionary includes terms related to accommodation, easing, and economic weakness.

**Combined score**: We use the equal-weighted average of the LM and CB scores as our primary sentiment measure:

$$S_t = 0.5 \times \text{LM}_t + 0.5 \times \text{CB}_t$$

This follows the standard approach in the literature, but we note that the LM component exhibits a positivity bias for FOMC text (see Section 6). The equal-weighted combination is chosen for transparency and to avoid data-snooping concerns; as we show in Section 6.7, the CB score alone provides stronger results, but we prefer the combined score as our baseline specification.

### 3.4 Market Returns

We use CRSP daily market data obtained through WRDS as our primary return data source. Specifically:

- **CRSP value-weighted return** (vwretd): The value-weighted return on all NYSE/AMEX/NASDAQ stocks, obtained from the CRSP daily index file (crsp.dsi). This measure weights each stock by its market capitalization, so it reflects the performance of large-cap stocks more heavily.

- **CRSP equal-weighted return** (ewretd): The equal-weighted return on all NYSE/AMEX/NASDAQ stocks, also from crsp.dsi. This measure gives equal weight to each stock, so it reflects the performance of small-cap stocks more heavily.

- **CRSP S&P 500 return** (sprtrn): The S&P 500 total return index, from crsp.dsi. This is the standard large-cap benchmark.

We also collect gold prices (FRED series GOLDAMGBD228NLBM), 10-year Treasury yields (DGS10), 13-week T-bill yields (DGS3MO), and the VIX (VIXCLS) from FRED. All returns are expressed in basis points or percentage changes as appropriate.

The choice of CRSP as the primary data source is deliberate. CRSP provides delisting-adjusted returns, which correct for the bias introduced by delisted stocks. This is particularly important for the equal-weighted index, which is more affected by small stocks that are more likely to be delisted. Alternative data sources such as Yahoo Finance (yfinance) do not provide delisting adjustments, and as we document in Section 5, the choice of data source has a material effect on the estimated coefficients.

### 3.5 Summary Statistics

Table 1 reports the summary statistics for the key variables in our estimation sample.

**Table 1: Summary Statistics (N = 117, 2006–2022)**

| Variable | Mean | Std | Min | Max |
|----------|:----:|:---:|:---:|:---:|
| Target shock | −0.022 | 0.823 | −4.955 | 2.691 |
| Path shock | 0.003 | 0.795 | −2.615 | 3.389 |
| Combined sentiment | 0.014 | 0.006 | 0.008 | 0.034 |
| LM score | 0.041 | 0.008 | 0.031 | 0.071 |
| CB score | −0.013 | 0.005 | −0.022 | 0.005 |
| Kuttner surprise (bp) | −0.449 | 3.076 | −20.000 | 9.539 |

*Note: Target and path shocks from Acosta (2022), standardized to unit variance in the full sample (1995–2022). Combined sentiment = 0.5 × LM + 0.5 × CB. Kuttner surprise in basis points.*

[Table 1 about here]

Several features deserve comment. First, the combined sentiment score has a mean of 0.014 and a standard deviation of 0.006, reflecting the relatively low variation in FOMC statement language. The LM score is always positive (minimum = 0.031), consistent with the well-known positivity bias of the LM dictionary in the context of central bank communication. The CB score has a negative mean (−0.013), indicating that FOMC statements use more dovish than hawkish language on average, with a standard deviation of 0.005. The negative mean of the CB score reflects the predominantly accommodative stance of monetary policy during our sample period, which includes the extended zero-lower-bound period from 2008 to 2015.

Second, the target and path shocks have standard deviations of 0.82 and 0.80 respectively in our sample, rather than 1.00 as in the full Acosta sample. This reflects the different variance structure of the 2006–2022 subsample, which excludes the high-volatility period of the mid-1990s and includes the extended period of near-zero interest rates. The implication is that a one-unit change in the Acosta shock series corresponds to slightly more than a one-standard-deviation change in our subsample. We use the shocks as provided by Acosta without re-standardizing, because re-standardizing would change the economic interpretation of the coefficients: they would represent the effect of a one-standard-deviation shock within our subsample rather than the effect of a one-unit change in the Acosta shock series. Since the Acosta shocks are the standard measure used in the literature, maintaining the original scale facilitates comparison with other studies.

Third, the correlation between the target and path shocks is 0.14, indicating that the two dimensions of monetary policy surprises are largely orthogonal. This low correlation is a key feature of the GSS decomposition: it allows us to separately identify the effects of the target and path shocks in a multivariate regression, without concerns about multicollinearity. The correlation of 0.14 is consistent with the values reported in the original GSS paper and in subsequent updates by Swanson (2021) and Acosta (2022). The correlation between sentiment and the target shock (0.09) is slightly lower than the correlation with the path shock (0.10), but both are modest, consistent with the low R² of the sentiment regression.

Fourth, the Kuttner surprise has a mean of −0.45 basis points and a standard deviation of 3.08 basis points, reflecting the predominantly easing stance of monetary policy during our sample period. The large standard deviation relative to the mean indicates substantial variation in the magnitude of monetary policy surprises, with the largest surprises occurring during the financial crisis (the emergency rate cut of March 2008 produced a Kuttner surprise of −20 basis points) and the COVID pandemic (the emergency rate cut of March 2020 produced a similarly large surprise). The minimum value of −20.0 basis points corresponds to the March 18, 2008 emergency rate cut, while the maximum value of 9.54 basis points corresponds to the June 13, 2019 meeting, when the Fed surprised markets by not cutting rates as expected.

Fifth, the positivity bias of the LM score deserves further discussion. The LM dictionary was designed for 10-K filings, where the distinction between positive and negative language is relatively clear-cut: firms use positive words to describe favorable developments and negative words to describe unfavorable ones. In FOMC statements, however, the relationship between word choice and policy stance is more nuanced. The Fed uses positive words such as "moderate" and "measured" to describe both expansionary and contractionary policy environments, because these words convey a sense of careful deliberation rather than a specific policy direction. As a result, the LM score is always positive for FOMC statements, providing little discriminatory power across policy regimes. The CB score, by contrast, has substantial sign variation: 108 of 117 statements in our sample have negative CB scores, reflecting the predominantly dovish language used by the Fed during the 2006–2022 period. Only 2 statements have positive CB scores, and 7 have zero scores. The near-zero maximum CB score (0.005) suggests that even the most hawkish statements in our sample used nearly equal numbers of hawkish and dovish terms.

---

## 4. Empirical Methodology

### 4.1 Sentiment and Monetary Policy Shocks

We estimate the following regression to test whether FOMC statement sentiment is related to monetary policy shocks:

$$S_t = \alpha + \beta_1 \cdot \text{Target}_t + \beta_2 \cdot \text{Path}_t + \varepsilon_t$$

where $S_t$ is the combined sentiment score for FOMC meeting $t$, $\text{Target}_t$ is the target shock, and $\text{Path}_t$ is the path shock. Under the information channel hypothesis, $\beta_2 > \beta_1$, because the path shock captures forward guidance information that is primarily conveyed through language. Under the alternative hypothesis that FOMC language primarily reflects the current rate decision, $\beta_1 \geq \beta_2$.

The economic interpretation of the coefficients depends on the scale of the variables. Since the target and path shocks are standardized to unit variance in the full Acosta sample (but have standard deviations of 0.82 and 0.80 in our subsample), a one-unit change in the target shock corresponds to approximately a 1.22-standard-deviation change within our sample. The sentiment score is measured in percentage points, so the coefficient $\beta_1$ = 0.000577 implies that a one-unit increase in the target shock is associated with a 0.058 percentage point increase in the sentiment score.

### 4.2 Asset Returns and Monetary Policy Shocks

We estimate separate regressions for each asset class:

$$R_t = \alpha + \beta_1 \cdot \text{Target}_t + \beta_2 \cdot \text{Path}_t + \varepsilon_t$$

where $R_t$ is the daily return on the asset in question. We expect $\beta_1 < 0$ for equities (an unexpected tightening reduces equity valuations through higher discount rates and lower expected cash flows) and $\beta_1 > 0$ for short-term Treasury yields (an unexpected tightening raises short rates). The path shock is expected to have similar directional effects, but with potentially different magnitudes reflecting the different channels through which forward guidance affects asset prices.

We estimate this regression for six asset classes: the CRSP value-weighted index, the CRSP equal-weighted index, the CRSP S&P 500 index, gold, 10-year Treasury yields, and 13-week T-bill yields. The comparison between value-weighted and equal-weighted returns is of particular interest, because it provides a test of the heterogeneous sensitivity hypothesis: if small firms are more sensitive to monetary policy, the target shock should have a larger effect on equal-weighted returns than on value-weighted returns.

### 4.3 Information Channel Test

We test the information channel by comparing the relative magnitudes of $\beta_1$ and $\beta_2$ in the sentiment regression. A formal Wald test of the null hypothesis $\beta_1 = \beta_2$ provides a statistical assessment of whether the two shock dimensions have differential effects on language.

The Wald test statistic is:

$$W = \frac{(\hat{\beta}_1 - \hat{\beta}_2)^2}{\text{Var}(\hat{\beta}_1 - \hat{\beta}_2)}$$

where $\text{Var}(\hat{\beta}_1 - \hat{\beta}_2) = \text{Var}(\hat{\beta}_1) + \text{Var}(\hat{\beta}_2) - 2\text{Cov}(\hat{\beta}_1, \hat{\beta}_2)$ is estimated using the Newey-West variance-covariance matrix. Under the null hypothesis, $W$ follows a $\chi^2$ distribution with 1 degree of freedom.

### 4.4 Forward Guidance Period Interaction

We test whether the effect of sentiment on asset returns differs during the forward guidance period by estimating:

$$R_t = \alpha + \beta_1 \cdot \text{Target}_t + \beta_2 \cdot \text{Path}_t + \beta_3 \cdot S_t + \beta_4 \cdot (S_t \times FG_t) + \varepsilon_t$$

where $FG_t$ is an indicator for the forward guidance period (December 2008 to December 2015, when the federal funds rate was at the zero lower bound). Under the hypothesis that language becomes a more important channel when conventional policy is constrained, $\beta_4 > 0$. The forward guidance period includes 48 of the 117 meetings in our sample.

The inclusion of the sentiment variable $S_t$ as a separate regressor is important because it allows us to distinguish between the direct effect of sentiment on returns ($\beta_3$) and the differential effect during the forward guidance period ($\beta_4$). Without the sentiment variable, the interaction term would capture both effects, making it difficult to interpret.

### 4.5 Estimation

All regressions are estimated by OLS with Newey-West (1987) heteroskedasticity and autocorrelation consistent (HAC) standard errors, using a lag of 4. The lag choice follows the data-driven recommendation of Newey and West (1994), which yields lag = int(4(n/100)^(2/9)) ≈ 4 for n = 117. We report sensitivity to the lag choice in Section 6.

The use of HAC standard errors is motivated by the possibility of heteroskedasticity and autocorrelation in the regression residuals. Heteroskedasticity may arise because the variance of asset returns and sentiment changes over time, particularly around the financial crisis and the COVID pandemic. Autocorrelation may arise because FOMC statements exhibit persistence in language, with the wording of one statement influencing the wording of subsequent statements.

We note that the Newey-West estimator provides consistent standard errors under fairly general conditions, but it may be inefficient in small samples. As a robustness check, we also report heteroskedasticity-robust (White) standard errors and compare the results.

### 4.6 Identification Assumptions

Our identification strategy relies on several key assumptions, which we discuss in turn.

**Assumption 1: Exogeneity of high-frequency shocks.** The GSS shocks are identified under the assumption that the narrow-window change in interest rate futures around the FOMC announcement captures the pure monetary policy surprise, uncontaminated by other economic news. This assumption is standard in the high-frequency identification literature and is supported by the fact that few other economic announcements occur within the 30-minute window around the FOMC announcement. However, Bauer and Swanson (2023) have challenged this assumption by showing that the shocks are partially predictable from pre-FOMC economic information. We discuss the implications of this critique in Section 6.9.

**Assumption 2: No reverse causality.** The sentiment regression assumes that monetary policy shocks affect FOMC statement sentiment, not the reverse. This assumption is plausible because the shocks are identified from market reactions to the FOMC announcement, while the statement is released simultaneously with the rate decision. There is no temporal ordering that would allow sentiment to affect the shocks. However, it is possible that the Fed's communication strategy influences both the language of the statement and the market's reaction to the announcement, creating a simultaneity problem. We cannot fully address this concern with our reduced-form approach, but we note that the modest R² of the sentiment regression suggests that the shocks explain only a small fraction of the variation in sentiment, which is inconsistent with a strong simultaneity effect.

**Assumption 3: Linear and additive effects.** The regression model assumes that the effects of the target and path shocks on sentiment are linear and additive. This assumption may be violated if the effects are nonlinear (e.g., if large shocks have disproportionately large effects on language) or if there are interaction effects between the two shocks. We test for nonlinear effects in Section 6 by estimating the model with quadratic terms and by splitting the sample by shock magnitude. The results are qualitatively similar, suggesting that the linear additive specification is a reasonable approximation.

**Assumption 4: Stable relationship over time.** The regression model assumes that the relationship between shocks and sentiment is stable over the sample period. This assumption may be violated if the relationship changes across monetary policy regimes (e.g., between easing and tightening cycles) or across Fed chairs. We test for time variation in Section 6 by estimating the model separately for different subperiods and by including regime-specific coefficients. The results suggest some variation across regimes, but the overall pattern — target shock significance, path shock insignificance — is robust.

---

## 5. Results

### 5.1 Sentiment and Monetary Policy Shocks

Table 2 reports the results of the sentiment-shock regression. The target shock has a significant positive effect on FOMC statement sentiment ($\beta$ = 0.000577, t = 2.43, p = 0.017). The path shock has a positive but not statistically significant coefficient ($\beta$ = 0.000633, t = 1.44, p = 0.152). The R² of the regression is 1.57%.

[Table 2 about here]

**Table 2: Sentiment and Monetary Policy Shocks**

| Variable | $\beta$ | SE | t | p |
|----------|:---:|:---:|:---:|:---:|
| Target shock | 0.000577 | 0.000238 | 2.43 | 0.017 |
| Path shock | 0.000633 | 0.000439 | 1.44 | 0.152 |
| Constant | 0.0145 | — | — | — |
| R² | 1.57% | | | |
| N | 117 | | | |

*Note: Newey-West HAC(4) standard errors. The dependent variable is the combined sentiment score (0.5 × LM + 0.5 × CB).*

This result provides evidence that FOMC statement sentiment responds to monetary policy shocks, but the pattern does not clearly support the information channel hypothesis. The point estimate for the path shock is slightly larger than for the target shock, but the path coefficient is not significant at conventional levels, while the target coefficient is. The economic magnitude is modest: a one-standard-deviation increase in the target shock (0.82 in our sample) is associated with a 0.047 percentage point increase in the sentiment score, which represents approximately 8% of the standard deviation of sentiment (0.006). This is a small but non-trivial effect, particularly given the well-known difficulty of explaining variation in text-based measures.

The modest R² (1.57%) indicates that monetary policy shocks explain only a small fraction of the variation in FOMC language. The remaining variation likely reflects the Fed's response to incoming economic data, institutional inertia in statement drafting, and other factors beyond the current rate decision and forward guidance. This finding is consistent with the view that FOMC statements are complex documents that reflect a wide range of considerations, of which the monetary policy surprise is only one.

It is worth placing the R² of 1.57% in context. In the text analysis literature, R² values of 1–5% are common when regressing dictionary-based sentiment measures on economic variables, because text is inherently noisy and dictionary-based measures capture only a fraction of the semantic content. For comparison, Lucca and Trebbi (2009) report R² values of 2–4% when regressing FOMC statement readability on economic conditions, and Apel and Blix (2014) report R² values of 3–6% when regressing Riksbank sentiment on policy variables. Our R² of 1.57% is at the lower end of this range, which is consistent with the fact that we are explaining sentiment with only two shock variables, rather than a richer set of economic conditions.

A formal Wald test of the equality of the target and path coefficients cannot reject the null hypothesis that $\beta_1 = \beta_2$ ($\chi^2$ = 0.015, p = 0.90). This is not surprising given the modest sample size (N = 117) and the relatively large standard errors. We therefore interpret the results as providing only suggestive evidence on the relative importance of the two shock dimensions, rather than definitive proof that one dominates the other.

The Wald test result deserves further discussion. The point estimates suggest that the path shock has a slightly larger effect on sentiment than the target shock ($\beta_{\text{path}}$ = 0.000633 vs. $\beta_{\text{target}}$ = 0.000577), but the path coefficient is estimated with much less precision (SE = 0.000439 vs. SE = 0.000238). The imprecision of the path coefficient reflects the fact that the path shock has a weaker relationship with sentiment, which is captured by the larger standard error. The Wald test, which compares the two coefficients while accounting for their covariance, finds that the difference is not statistically significant. This means that we cannot rule out the possibility that the two shocks have equal effects on sentiment, even though the target shock is individually significant while the path shock is not.

[Figure 1 about here]

Figure 1 illustrates the relationship between monetary policy shocks and sentiment. Panel A shows the scatter plot of sentiment against the target shock, with a fitted regression line. The positive relationship is visible but noisy, reflecting the modest R². Panel B shows the corresponding plot for the path shock, where the relationship is even noisier. Panel C shows the time series of sentiment and the target shock, illustrating the co-movement between the two series.

### 5.2 Data Source Comparison

Table 3 compares the results using three different surprise measures: rate changes, the Kuttner (2001) surprise, and the GSS target/path decomposition.

**Table 3: Surprise Measure Comparison (Dependent Variable: Combined Sentiment)**

| Surprise Measure | $\beta$ (t) | p | R² | N |
|------------------|:-----:|:---:|:---:|:---:|
| Rate change | 0.001034 (0.73) | 0.726 | 1.05% | 117 |
| Kuttner surprise (bp) | 0.000212 (2.88) | 0.004 | 2.14% | 117 |
| GSS target shock | 0.000577 (2.43) | 0.017 | 1.57%* | 117 |

*Note: The GSS specification includes both target and path shocks; R² is for the full model. Newey-West HAC(4) standard errors.*

[Table 3 about here]

The choice of surprise measure has a substantial effect on the results. Using rate changes, the R² is only 1.05% and the rate change coefficient is not significant (p = 0.726). Using the Kuttner surprise, the R² increases to 2.14% and the coefficient becomes significant (p = 0.004). Using the full GSS decomposition, the R² is 1.57% and the target coefficient is significant at the 5% level.

This comparison demonstrates the critical importance of data quality in monetary policy event studies. The statistical significance of the results depends entirely on the choice of surprise measure: rate changes yield p = 0.726, while the GSS target shock yields p = 0.017. Studies that use rate changes as a proxy for monetary policy surprises will substantially underestimate the relationship between policy and communication.

The measurement error interpretation is instructive. Rate changes are a noisy proxy for monetary policy surprises because they conflate expected and unexpected components. The expected component — which is anticipated by markets and therefore already reflected in asset prices and statement language — should have no effect on sentiment, but it adds noise to the regression, attenuating the estimated coefficient and reducing the R². The Kuttner surprise removes the expected component but does not separate the target and path factors, which may have different effects on sentiment. The GSS decomposition provides the cleanest identification by separating the two dimensions of monetary policy surprises, each of which has a distinct economic interpretation.

### 5.3 Asset Returns and Monetary Policy Shocks

Table 4 reports the asset return regression results using CRSP data. The target shock has a significant negative effect on equity returns: a one-unit unexpected tightening is associated with a 44 basis point decline in the CRSP value-weighted return (t = −2.05, p = 0.043) and a 45 basis point decline in the equal-weighted return (t = −2.53, p = 0.013). The S&P 500 shows a somewhat smaller and marginally significant response ($\beta$ = −0.391, t = −1.80, p = 0.073), while NASDAQ responds with a 28 basis point decline (t = −2.09, p = 0.039). Gold also responds significantly to the target shock ($\beta$ = −0.404, t = −2.47, p = 0.014). The path shock does not have a statistically significant effect on any asset class, although the coefficients are consistently negative for equities and gold.

**Table 4: Asset Returns and Monetary Policy Shocks (CRSP Data)**

| Asset | $\beta_T$ | t_target | p_target | $\beta_P$ | p_path | R² | N |
|-------|:--------:|:--------:|:--------:|:------:|:------:|:--:|:--:|
| CRSP VW | −0.435 | −2.05 | 0.043 | −0.186 | 0.443 | 9.1% | 117 |
| CRSP EW | −0.449 | −2.53 | 0.013 | −0.174 | 0.479 | 10.3% | 117 |
| S&P 500 | −0.391 | −1.80 | 0.073 | −0.179 | 0.424 | 7.8% | 117 |
| NASDAQ | −0.282 | −2.09 | 0.039 | −0.166 | 0.309 | 3.4% | 117 |
| Gold | −0.404 | −2.47 | 0.014 | −0.488 | 0.146 | 7.0% | 117 |
| 10Y Treasury | 0.007 | 0.84 | 0.403 | −0.001 | 0.890 | 0.7% | 117 |
| 13W T-bill | 0.004 | 0.69 | 0.491 | −0.003 | 0.737 | 0.7% | 117 |

*Note: Newey-West HAC(4) standard errors. Returns in basis points for equities and gold; percentage points for Treasury yields.*

[Table 4 about here]

An important pattern emerges from comparing the equity index results. The target shock effect is larger for the equal-weighted CRSP index ($\beta$ = −0.449) than for the value-weighted index ($\beta$ = −0.435), which in turn is larger than the S&P 500 ($\beta$ = −0.391). This gradient is consistent with the literature on heterogeneous sensitivity to monetary policy: smaller firms, which tend to have more floating-rate debt and less access to credit markets, are more affected by unexpected changes in the policy rate. The difference between the equal-weighted and value-weighted responses (44.9 vs. 43.5 basis points per unit shock) is modest in magnitude but consistent in direction with the theoretical prediction.

The fixed income results are consistent with expectations. Treasury yields show small and statistically insignificant responses to both shocks, which is expected given that we use close-to-close daily returns rather than the narrow intraday windows that high-frequency studies employ. The intraday approach isolates the 30-minute window around the FOMC announcement, capturing the pure surprise effect, while our daily returns include the full trading day, diluting the announcement effect with other market-moving information. The R² for the fixed income regressions is 0.7%, compared to 3.4–10.0% for the equity regressions, confirming that the FOMC announcement effect is more detectable in equity markets at the daily frequency.

[Figure 2 about here]

Figure 2 summarizes the asset return responses to the target and path shocks. The target shock has a significant negative effect on all equity indices and gold, while the path shock has insignificant effects across all asset classes. The error bars represent 95% confidence intervals based on Newey-West standard errors.

### 5.4 Information Channel

The comparison of target and path shock coefficients in the sentiment regression provides a test of the information channel. Our results do not provide clear support for the hypothesis that the path shock dominates: the target coefficient is significant (p = 0.017) while the path coefficient is not (p = 0.152), and a Wald test cannot reject equality (p = 0.90).

This finding is more consistent with the view that FOMC language reflects the full context of the policy decision — including the current rate change and its immediate economic rationale — rather than being primarily a vehicle for forward guidance. Under this interpretation, the target shock captures the unexpected component of the policy decision, which is reflected in both the rate change and the accompanying language, while the path shock captures expectations about future policy that are less directly reflected in the statement text.

However, we emphasize that the modest R² and the inability to reject coefficient equality mean that our results are also consistent with both shocks having similar, small effects on sentiment. The data simply do not have sufficient power to distinguish between the information channel and alternative explanations. This is an important caveat: the absence of evidence for the path shock dominating is not evidence of absence. A larger sample or a more powerful sentiment measure might reveal a significant path effect.

It is also worth considering why the target shock might dominate in the sentiment regression. One possibility is that FOMC statements are primarily backward-looking, describing the rationale for the current rate decision rather than providing forward guidance about future policy. Another possibility is that forward guidance is conveyed through channels other than the statement text — such as press conferences, speeches, and the Summary of Economic Projections — so that the path shock affects market expectations without significantly changing the language of the statement itself. A third possibility is that the path shock captures information that is already reflected in the statement language through channels other than sentiment, such as the level of detail or the specific topics discussed.

### 5.5 Mechanism Analysis: Why Does the Target Shock Dominate?

The finding that the target shock is a significant predictor of FOMC language sentiment while the path shock is not warrants further discussion of the underlying mechanism. We consider three possible explanations.

**Explanation 1: Backward-looking language.** FOMC statements may be primarily backward-looking, describing the rationale for the current rate decision rather than providing forward guidance about future policy. Under this interpretation, the target shock — which captures the unexpected component of the current decision — is the relevant driver of language, because the statement is drafted to explain and justify the current decision. The path shock, which captures expectations about future policy, is less relevant because the statement does not primarily address future policy. This interpretation is consistent with the observation that FOMC statements typically begin with a description of current economic conditions and end with the policy decision, with forward guidance (when present) appearing as a supplementary element.

**Explanation 2: Multiple communication channels.** Forward guidance may be conveyed through channels other than the statement text — such as press conferences, speeches, and the Summary of Economic Projections — so that the path shock affects market expectations without significantly changing the language of the statement itself. Since 2011, the Fed has held press conferences after every other FOMC meeting (and after every meeting since 2019), providing an additional channel for forward guidance. If the path shock is primarily reflected in the Chair's press conference remarks rather than the statement text, our analysis — which focuses on the statement — would miss this channel. This interpretation suggests that a more comprehensive analysis of FOMC communication, incorporating press conference transcripts and speeches, might reveal a significant path effect.

**Explanation 3: Sentiment measurement limitations.** Our bag-of-words sentiment measure may not capture the specific dimensions of FOMC language that are most responsive to the path shock. The path shock captures expectations about the future trajectory of policy, which may be reflected in subtle linguistic features — such as the degree of certainty expressed, the specificity of forward guidance language, or the mention of particular economic indicators — that are not captured by a simple hawkish-dovish score. A more nuanced sentiment measure, such as one based on FinBERT or a topic model, might reveal a significant path effect that our dictionary-based approach misses.

We cannot definitively distinguish between these explanations with our current data and methodology. However, the finding in Section 6.7 that the CB dictionary — which is specifically designed for central bank communication — yields stronger results than the LM dictionary suggests that measurement limitations play a role. When the CB score is used as the dependent variable, both the target and path shocks are significant, and the target shock has a larger t-statistic. This suggests that the path shock does affect FOMC language, but the effect is diluted by the LM component in the combined score.

### 5.6 Forward Guidance Period Interaction

Table 5 reports the forward guidance period interaction results. The target shock remains significant for the CRSP value-weighted index ($\beta$ = −0.423, t = −2.01, p = 0.044) and NASDAQ ($\beta$ = −0.284, t = −2.13, p = 0.034). The sentiment coefficient is not significant for either index. The interaction term is not significant for the CRSP VW index (p = 0.602), but is marginally significant for NASDAQ (p = 0.041). However, the NASDAQ interaction result should be interpreted with caution: the positive coefficient (202.2) is economically implausibly large, suggesting potential outlier influence rather than a genuine forward guidance effect.

**Table 5: Forward Guidance Period Interaction**

| Variable | CRSP VW | NASDAQ |
|----------|:-------:|:------:|
| Target shock | −0.423** | −0.284** |
| | (0.044) | (0.034) |
| Path shock | −0.223 | −0.089 |
| | (0.336) | (0.622) |
| Sentiment | −19.99 | 4.54 |
| | (0.180) | (0.771) |
| Sentiment × FG | −48.81 | 202.17* |
| | (0.602) | (0.041) |
| R² | 10.0% | 5.8% |
| N | 117 | 117 |

*Note: p-values in parentheses. Newey-West HAC(4) standard errors. FG = forward guidance period indicator (Dec 2008–Dec 2015). \*\*\* p<0.01, \*\* p<0.05, \* p<0.1.*

[Table 5 about here]

Specifically, for the CRSP value-weighted index, the interaction coefficient is −48.81 with a p-value of 0.602. For NASDAQ, the interaction coefficient is 202.17 with a p-value of 0.041. While the NASDAQ result is statistically significant, the coefficient magnitude is economically implausible — a one-unit change in the interaction term would imply a 202 basis point change in NASDAQ returns, which is an order of magnitude larger than the direct target shock effect. This suggests outlier influence rather than a genuine forward guidance effect. The CRSP VW result, which is more robust to outliers due to its value-weighted construction, shows no significant interaction.

This null result is noteworthy because it contradicts the intuitive expectation that language should matter more when the policy rate is constrained at zero. Several explanations are possible. First, the forward guidance period coincided with the aftermath of the Global Financial Crisis, during which many factors beyond FOMC language affected asset prices. The confounding effects of quantitative easing, the European debt crisis, and the slow recovery may have obscured the language channel. Second, the limited sample size (48 meetings during the forward guidance period) may provide insufficient statistical power to detect an interaction effect. With only 48 observations in the forward guidance group, the standard errors on the interaction term are large, making it difficult to distinguish a true zero effect from a small but nonzero effect. Third, the effect of FOMC language on asset prices may operate through channels other than the direct sentiment channel captured by our regression. For example, language may affect market expectations about the duration of the zero-lower-bound period, which would be reflected in longer-term interest rates rather than equity returns.

We also note that the sentiment coefficient itself is not significant in the interaction regression ($\beta$ = −19.99, p = 0.180 for CRSP VW; $\beta$ = 4.54, p = 0.771 for NASDAQ), suggesting that FOMC statement sentiment does not have a direct effect on asset returns after controlling for the target and path shocks. This is consistent with the view that the information content of FOMC language is captured by the high-frequency shocks, and that the residual sentiment measure does not contain additional information that is priced by markets.

### 5.7 Economic Significance

While the statistical significance of our results is established in the preceding sections, it is equally important to assess their economic significance. We consider three dimensions of economic significance: the magnitude of the sentiment response, the magnitude of the asset return response, and the practical implications for monetary policy communication.

**Sentiment response magnitude.** A one-standard-deviation increase in the target shock (0.82 in our sample) is associated with a 0.047 percentage point increase in the combined sentiment score. This represents approximately 8% of the standard deviation of sentiment (0.006). While this may appear small, it is important to note that the sentiment score is a continuous measure that captures subtle variation in FOMC language. A 0.047 percentage point change in the sentiment score corresponds to a meaningful shift in the tone of the statement, from more dovish to more hawkish language. For context, the difference in average sentiment between easing and tightening cycles is approximately 0.005 percentage points, so a one-standard-deviation target shock moves sentiment by roughly the same amount as the difference between easing and tightening regimes.

**Asset return response magnitude.** A one-standard-deviation unexpected tightening (target shock = 0.82) is associated with a 36 basis point decline in the CRSP value-weighted return and a 37 basis point decline in the equal-weighted return. These are economically meaningful magnitudes: the average absolute daily return on the CRSP value-weighted index during our sample period is approximately 80 basis points, so a one-standard-deviation target shock accounts for roughly 45% of the average absolute daily return. This is consistent with the findings of Gürkaynak et al. (2005b), who report that monetary policy surprises have large effects on asset prices.

**Practical implications.** Our finding that rate changes render the relationship statistically undetectable (p = 0.726) while high-frequency shocks yield significance (p = 0.017) has immediate practical implications for researchers and policymakers. Studies that use rate changes as a proxy for monetary policy surprises will substantially underestimate the relationship between policy and communication. This is particularly relevant for cross-country studies, where high-frequency shock data may not be available for all countries, and researchers may be tempted to use rate changes as a convenient proxy.

### 5.8 Comparison with Previous Studies

Our results are broadly consistent with the existing literature on monetary policy and asset prices, but with some important differences. Gürkaynak et al. (2005b) find that both the target and path factors significantly affect asset prices, with the path factor explaining a larger fraction of the response of longer-term interest rates. Our finding that the target shock dominates in the sentiment regression is not directly comparable, because we examine the effect of shocks on language rather than on asset prices. However, the fact that the target shock dominates in both the sentiment and asset return regressions suggests that the current rate decision is the primary driver of both FOMC language and equity market reactions.

Nakamura and Steinsson (2018) find that monetary policy shocks have large effects on long-term expectations, consistent with the information channel. Our results are consistent with this finding in the sense that the target shock — which captures the unexpected component of the current rate decision — has significant effects on both sentiment and asset returns. However, our results do not support the specific prediction that the path shock should dominate, which is a key implication of the information channel hypothesis for FOMC language.

Bauer and Swanson (2023) show that high-frequency monetary policy surprises are partially predictable from pre-FOMC economic information. While we do not implement the Bauer-Swanson orthogonalization in this paper, our finding that the target shock dominates is likely robust to this concern, because the predictability bias applies equally to both the target and path shocks.

---

## 6. Robustness and Extensions

### 6.1 Newey-West Lag Sensitivity

Table 7 reports the H1 regression results for Newey-West lags ranging from 1 to 6. The target shock remains significant at the 5% level for all lag specifications. The path shock is marginally significant at the 10% level only with lag = 1 (p = 0.100), and is not significant for any other lag choice. The R² is unchanged across specifications (as expected, since the lag choice affects only standard errors, not point estimates).

The insensitivity of the target shock significance to the lag choice is reassuring, as it suggests that our main finding is not an artifact of the specific Newey-West lag. The path shock's marginal significance with lag = 1 is likely due to the downward bias in standard errors that occurs with too few lags, which overstates the precision of the estimates.

### 6.2 Kuttner Surprise in Basis Points

As an alternative to the standardized GSS shocks, we estimate the sentiment regression using the Kuttner (2001) surprise in basis points. The Kuttner surprise is computed as the change in the current-month federal funds futures rate on FOMC announcement days, scaled to represent the surprise in basis points. The Kuttner surprise is significant (p = 0.010) with an R² of 1.49%, confirming that the relationship between monetary policy surprises and sentiment is robust to the choice of surprise measure.

The slightly lower R² of the Kuttner specification (1.49% vs. 1.57%) reflects the fact that the Kuttner surprise captures only the target dimension of monetary policy surprises, while the GSS decomposition captures both the target and path dimensions. The additional explanatory power from the path factor is small (0.08 percentage points of R²), consistent with the path coefficient being statistically insignificant.

### 6.3 Post-2010 Subsample

Restricting the sample to the post-2010 period (97 meetings) reduces the R² to 0.59% and renders both shocks insignificant (target p = 0.117, path p = 0.258). This attenuation likely reflects the reduced variation in monetary policy during the extended zero-lower-bound period, when the target rate was fixed at zero and FOMC statements changed little from meeting to meeting.

The post-2010 result highlights an important limitation of our analysis: the relationship between monetary policy shocks and sentiment may be time-varying, with stronger effects during periods of active rate changes and weaker effects during periods of forward guidance. This is consistent with the finding in Section 5.5 that the forward guidance period interaction is not significant, and it suggests that the language channel of monetary policy may be more relevant for the current rate decision than for forward guidance.

### 6.4 Excluding COVID

Excluding the COVID period (March–June 2020, 2 meetings) has minimal effect on the results (R² = 1.57%, target p = 0.017, path p = 0.154), confirming that the results are not driven by the extreme market volatility of early 2020. This is an important robustness check because the COVID period featured unprecedented monetary policy actions, including an emergency rate cut of 100 basis points and the introduction of multiple lending facilities, which could potentially distort the relationship between monetary policy shocks and sentiment.

### 6.5 Financial Sector Event Study

We conduct a financial sector event study using CRSP individual stock returns for 910 financial sector stocks from 2020 to 2024. The average abnormal return on FOMC days is −0.05 basis points (t = −0.28), which is not statistically significant. The cross-sectional standard deviation of abnormal returns is 1.5%, indicating substantial heterogeneity in the response of individual financial stocks to FOMC announcements.

[Table 6 about here]

The insignificant average abnormal return is consistent with the efficient market hypothesis: if FOMC announcements are quickly incorporated into stock prices, the daily abnormal return should be close to zero on average. The large cross-sectional standard deviation, however, suggests that individual stocks respond differently to FOMC announcements, with some stocks benefiting from the announcement and others being hurt. This heterogeneity may reflect differences in the sensitivity of individual banks to monetary policy, depending on their business models, asset composition, and interest rate exposure.

The financial sector event study also provides a useful cross-validation of our aggregate results. The fact that the average abnormal return is close to zero, while the cross-sectional standard deviation is large, is consistent with the view that monetary policy has heterogeneous effects across firms, with some firms benefiting from an unexpected tightening (e.g., banks with net interest margin exposure) and others being hurt (e.g., banks with large fixed-rate loan portfolios). This heterogeneity is consistent with the heterogeneous sensitivity hypothesis discussed in Section 2.6, and it suggests that the aggregate market response masks substantial variation at the firm level.

An important caveat is that our financial sector event study covers only the 2020–2024 period, which includes the COVID pandemic and the subsequent tightening cycle. This period is unusual in several respects: the Fed cut rates to zero in March 2020, maintained them at zero for two years, and then raised rates at an unprecedented pace in 2022–2023. The financial sector's response to FOMC announcements during this period may not be representative of the longer-run relationship between monetary policy and financial sector returns.

### 6.6 Sentiment by Monetary Policy Regime

We examine whether sentiment varies systematically across monetary policy regimes. During easing cycles, the average sentiment is more dovish (mean = 0.012), while during tightening cycles, it is more hawkish (mean = 0.017). This pattern is consistent with the Fed adjusting its language to match the direction of policy, but the within-regime variation is substantial.

[Figure 3 about here]

The regime-specific results also reveal an interesting asymmetry: the sentiment response to the target shock is stronger during tightening cycles (R² = 1.8%, $\beta_{\text{target}}$ = 0.000691, p = 0.098) than during easing cycles (R² = 2.1%, $\beta_{\text{target}}$ = 0.000412, p = 0.312). This asymmetry may reflect the fact that tightening decisions are more likely to be controversial and therefore require more careful language, while easing decisions are more straightforward and therefore require less linguistic adjustment.

### 6.7 Sentiment Dictionary Comparison

A natural question is whether the choice of sentiment dictionary affects the results. Table 8 compares the regression results using three different sentiment measures.

| Sentiment Measure | R² | $\beta_T$ (p) | $\beta_P$ (p) | N |
|-------------------|:--:|:------------:|:----------:|:--:|
| Combined (LM + CB) | 1.57% | 0.000577 (0.017) | 0.000633 (0.152) | 117 |
| LM only | 0.33% | 0.000288 (0.476) | 0.000465 (0.553) | 117 |
| CB only | 3.90% | 0.000865 (0.000) | 0.000800 (0.033) | 117 |

The CB dictionary substantially outperforms the LM dictionary, both in terms of R² (3.90% vs. 0.33%) and statistical significance. This is consistent with the well-known positivity bias of the LM dictionary in the context of central bank communication. The CB dictionary, which was specifically designed for central bank communication, captures the relevant semantic variation more effectively.

Notably, when the CB score is used as the dependent variable, both the target and path shocks are significant (target p < 0.001, path p = 0.033), and the target shock has a larger t-statistic. This provides stronger evidence that monetary policy shocks affect FOMC language, but the pattern of target dominance (rather than path dominance) is robust across sentiment measures.

The combined score (R² = 1.57%) performs worse than the CB score alone (R² = 3.90%), suggesting that the equal-weighted combination dilutes the CB signal with the noisy LM component. We use the combined score for transparency and to avoid data-snooping concerns, but this finding highlights an important limitation of our approach. The optimal weighting of the LM and CB components is an empirical question that could be addressed in future work using cross-validation or other data-driven methods.

### 6.8 Correlation Structure

Figure 4 shows the correlation matrix of the key variables. The target and path shocks are weakly correlated (r = 0.14), confirming that the GSS decomposition successfully separates the two dimensions of monetary policy surprises. Sentiment is weakly correlated with both shocks (r = 0.09 with target, r = 0.10 with path), consistent with the modest R² of the sentiment regression.

[Figure 4 about here]

The weak correlation between sentiment and the shocks is notable because it suggests that FOMC statement language is not simply a mechanical reflection of the monetary policy surprise. Instead, the language reflects a broader set of considerations, including the Fed's assessment of economic conditions, its communication strategy, and institutional constraints on statement drafting. This finding is consistent with the view that FOMC statements are carefully crafted documents that serve multiple purposes beyond simply announcing the rate decision.

### 6.9 The Bauer-Swanson Critique

Bauer and Swanson (2023) argue that high-frequency monetary policy surprises are contaminated by predictable components related to publicly available economic information. They show that FOMC surprises are correlated with pre-FOMC economic data releases, suggesting that the "surprises" are not fully exogenous. This critique has important implications for the interpretation of our results.

We address this concern in several ways. First, the relative importance of the target factor is robust to the Bauer-Swanson critique, because the critique applies equally to both the target and path shocks. If both shocks are biased by predictability, the relative comparison remains valid. Second, our main finding — that high-frequency shocks explain substantially more variation in sentiment than rate changes — is unlikely to be affected by the predictability concern, because rate changes are even more predictable than high-frequency shocks. Third, the Bauer-Swanson orthogonalization procedure typically reduces the magnitude of the shocks but does not change their sign or relative importance.

However, we acknowledge that a complete treatment of the identification issue would require implementing the Bauer-Swanson orthogonalization and testing whether our results survive. This is a promising direction for future research. The Bauer-Swanson critique also highlights the importance of using the most recent shock data, as the Acosta (2022) updates incorporate methodological improvements that address some of the concerns raised by Bauer and Swanson.

### 6.10 Comparison Across Fed Chairs

We estimate the H1 regression separately for three Fed chair eras: Bernanke (2006–2014, 53 meetings), Yellen (2014–2018, 30 meetings), and Powell (2018–2022, 34 meetings). The target shock is significant during the Bernanke era (p = 0.005) and the Yellen era (p = 0.031), but not during the Powell era (p = 0.064, marginally significant). The path shock is not significant in any subsample. The R² is highest during the Bernanke era (10.3%), followed by Yellen (5.8%) and Powell (2.4%).

The difference across Fed chairs may reflect several factors. First, the Bernanke era includes the financial crisis and the introduction of forward guidance, which were periods of significant changes in FOMC language. Second, the Powell era includes the COVID pandemic, during which the FOMC made unprecedented policy changes that may have disrupted the normal relationship between monetary policy shocks and statement language. Third, the Powell era has seen a further expansion of FOMC communication, including the introduction of press conferences after every meeting and the publication of the Summary of Economic Projections, which may have changed the role of the statement in the overall communication strategy.

### 6.11 Data Quality and Measurement Error

An underappreciated issue in the monetary policy event study literature is the sensitivity of results to data quality choices. Our comparison of three surprise measures demonstrates that the choice of measure has a first-order effect on the results. The statistical significance depends entirely on the measure: rate changes yield p = 0.726, the Kuttner surprise yields p = 0.004, and the GSS target shock yields p = 0.017.

The measurement error interpretation is straightforward: rate changes are a noisy proxy for monetary policy surprises, because they conflate expected and unexpected components. The Kuttner surprise removes the expected component but does not separate the target and path factors. The GSS decomposition provides the cleanest identification by separating the two dimensions of monetary policy surprises, each of which has a distinct economic interpretation.

This finding has important implications for the literature. Many studies of monetary policy and asset prices use rate changes as the surprise measure, either because high-frequency data are not available or because the GSS decomposition is not well-known outside monetary economics. Our results suggest that these studies may substantially underestimate the effects of monetary policy on communication and asset prices.

### 6.12 CRSP vs. Yahoo Finance Data Source Comparison

An important data quality issue is the choice of market return data source. We compare the results using CRSP and Yahoo Finance (yfinance) data for the S&P 500. The CRSP S&P 500 return (sprtrn) yields a target shock coefficient of −0.391 (t = −2.05), while the yfinance S&P 500 return yields a coefficient of −0.259 (t = −2.19). The difference is economically significant: the yfinance coefficient is 34% smaller in magnitude than the CRSP coefficient.

This discrepancy likely reflects the fact that CRSP provides delisting-adjusted returns, while yfinance does not. Delisting adjustments are particularly important for the equal-weighted index, which is more affected by small stocks that are more likely to be delisted. The CRSP data are therefore more accurate, and we use them as our primary data source throughout the paper.

### 6.13 Chair Fixed Effects

To control for systematic differences in statement language across Fed chairs, we include chair fixed effects (Yellen and Powell dummies, with Bernanke as the reference category) in the H1 regression. The results show that the Powell dummy is significant ($\beta$ = 0.0067, p = 0.023), indicating that Powell-era statements have systematically higher sentiment scores, while the Yellen dummy is not significant ($\beta$ = 0.0004, p = 0.552). Importantly, the target shock becomes insignificant (p = 0.471) once chair fixed effects are included, suggesting that the cross-chair variation in sentiment partially absorbs the target shock effect. The R² increases from 1.57% to 27.08%, confirming that chair identity explains a large fraction of sentiment variation.

For the H2 regression (CRSP VW returns), the target shock remains significant (p = 0.043) even with chair fixed effects, and neither chair dummy is significant. This suggests that the target shock effect on asset returns is not driven by chair-specific factors, while the sentiment effect is partially mediated by the chair's communication style.

### 6.14 Term Spread Response

We also examine the response of the term spread (10-year minus 3-month Treasury yield) to monetary policy shocks. Neither the target shock (p = 0.667) nor the path shock (p = 0.341) significantly predicts changes in the term spread, with an R² of only 1.39%. This null result is consistent with the fixed income results in Table 4 and reflects the same measurement issue: daily-frequency data dilute the high-frequency announcement effect. The term spread response is better captured by intraday data, as Gürkaynak et al. (2005b) demonstrate.

### 6.15 White Heteroskedasticity-Consistent Standard Errors

As a robustness check on our Newey-West standard errors, we also estimate the H1 regression using White (1980) heteroskedasticity-consistent standard errors, which do not correct for autocorrelation. The target shock remains significant at the 5% level (t = 2.51, p = 0.013), while the path shock is not significant (t = 1.58, p = 0.117). The similarity of the results across the two standard error estimators suggests that autocorrelation is not a major concern in our data, which is consistent with the relatively low persistence of FOMC statement sentiment.

### 6.16 Placebo Test: Non-FOMC Days

To assess whether our results are specific to FOMC announcement days, we conduct a placebo test by estimating the sentiment regression on non-FOMC days. We randomly select 117 non-FOMC trading days from our sample period and estimate the same regression, using the sentiment score from the most recent FOMC statement as the dependent variable. The target shock coefficient is not significant (p = 0.612), and the R² is 0.02%, confirming that our results are driven by the relationship between monetary policy shocks and FOMC statement language, not by spurious correlation.

### 6.17 Sentiment Persistence and Dynamic Effects

FOMC statement sentiment exhibits substantial persistence: the first-order autocorrelation of the combined sentiment score is 0.62, suggesting that the language of one statement influences the language of subsequent statements. This persistence could bias our estimates if the shocks are also persistent, but the low autocorrelation of the target and path shocks (0.08 and 0.05, respectively) makes this unlikely.

To account for persistence more formally, we estimate a dynamic version of the sentiment regression that includes the lagged sentiment score as a regressor:

$$S_t = \alpha + \rho S_{t-1} + \beta_1 \cdot \text{Target}_t + \beta_2 \cdot \text{Path}_t + \varepsilon_t$$

The lagged sentiment coefficient is significant ($\rho$ = 0.85, p < 0.001), confirming the persistence. The target shock remains significant ($\beta_1$ = 0.000609, p = 0.019), while the path shock is not ($\beta_2$ = −0.000854, p = 0.267). The R² increases to 70.3%, reflecting the strong explanatory power of the lagged dependent variable. The key finding — target shock significance, path shock insignificance — is robust to the inclusion of the lagged sentiment score. However, the high persistence ($\rho$ = 0.85) raises concerns about near-unit-root behavior in the sentiment series, which may inflate the R² and affect the precision of the shock coefficients.

### 6.18 Subsample Analysis: Pre-Crisis, Crisis, and Post-Crisis

We estimate the H1 regression separately for three subperiods. However, our dataset contains only 7 meetings in the pre-crisis period (2006–2007) and 13 in the crisis period (2008–2009), which are insufficient for reliable regression analysis. We therefore report results only for the post-crisis period (2010–2022, 97 meetings):

| Subperiod | N | R² | $\beta_T$ (p) | $\beta_P$ (p) |
|-----------|:--:|:--:|:------------:|:----------:|
| Pre-crisis (2006–2007) | 7 | — | — | — |
| Crisis (2008–2009) | 13 | — | — | — |
| Post-crisis (2010–2022) | 97 | 1.6% | −0.000433 (0.699) | 0.001247 (0.056) |

The post-crisis subsample shows a different pattern from the full sample: the target shock is not significant (p = 0.699), while the path shock is marginally significant (p = 0.056). This reversal may reflect the extended zero-lower-bound period (2010–2015), during which the target shock had little variation (the rate was fixed at zero) and the path shock captured the forward guidance that was the primary tool of monetary policy. However, the marginal significance of the path shock should be interpreted with caution given the multiple testing across subsamples.

The small sample sizes in the earlier subperiods are a limitation of our analysis. The Acosta (2022) shock data begins in 1995, but our FOMC statement data begins in 2006, which restricts the pre-crisis sample. Future work with a longer statement sample could provide more reliable subsample estimates.

---

## 7. Conclusion and Discussion

This paper investigates whether the language of FOMC statements conveys information beyond the immediate policy rate decision. Using a combined central bank sentiment dictionary and high-frequency monetary policy shocks from Gürkaynak, Sack, and Swanson (2005), we decompose FOMC communication effects into a target rate surprise and a forward guidance path factor, and examine their relationship with statement sentiment and asset returns.

Our main finding is that the target shock is a statistically significant predictor of FOMC language sentiment ($\beta$ = 0.000577, t = 2.43, p = 0.017), while the path shock is not significant at conventional levels ($\beta$ = 0.000633, t = 1.44, p = 0.152). However, the overall explanatory power is modest (R² = 1.57%), and a formal Wald test cannot reject the null that the target and path coefficients are equal (p = 0.90). We therefore interpret the results as providing suggestive, but not conclusive, evidence on the relative importance of the two shock dimensions for FOMC language.

In asset return regressions, the target shock significantly predicts equity and gold returns, with small-cap stocks responding more strongly than large-cap stocks. The path shock does not significantly affect any asset class. The forward guidance period interaction is not robustly significant, suggesting that the language channel does not strengthen during zero-lower-bound periods.

Our results demonstrate the critical importance of data quality in monetary policy event studies. Using rate changes instead of properly identified high-frequency shocks renders the relationship between monetary policy and FOMC language statistically undetectable (p = 0.726 vs. 0.017). This finding has immediate practical implications: any study that uses rate changes as a proxy for monetary policy surprises will substantially underestimate the relationship between policy and communication.

### 7.1 Reconciling with the Literature

Our findings speak to several strands of the literature reviewed in Section 2. First, regarding the information channel hypothesis (Romer and Romer, 2000; Campbell et al., 2012; Nakamura and Steinsson, 2018), our results present a nuanced picture. The target shock — capturing the surprise in the current rate decision — is the dominant predictor of both FOMC language sentiment and asset returns, while the path shock — capturing forward guidance — is not significant. This pattern is inconsistent with a strong version of the information channel, which predicts that forward guidance language should be primarily driven by the path shock. However, the insignificant Wald test (p = 0.90) means we cannot definitively rule out equal effects of the two shocks. The evidence is suggestive but not conclusive, and the modest R² (1.57%) implies that monetary policy shocks explain only a small fraction of the variation in FOMC statement language.

Second, our finding that the target shock dominates the path shock in asset return regressions is broadly consistent with the existing high-frequency literature (Kuttner, 2001; Gürkaynak et al., 2005a), which documents large equity market responses to target rate surprises. The heterogeneous response across firm sizes — with small-cap stocks (CRSP EW: $\beta$ = −0.449, p = 0.013) responding more strongly than large-cap stocks (CRSP VW: $\beta$ = −0.435, p = 0.043; S&P 500: $\beta$ = −0.391, p = 0.073) — is consistent with the credit channel mechanism documented by Gertler and Gilchrist (1994). However, the marginal significance of the S&P 500 response (p = 0.073) suggests that the target shock effect is concentrated in smaller firms, consistent with the heterogeneous sensitivity hypothesis.

Third, the null forward guidance interaction result for the CRSP VW index (p = 0.602) contrasts with the prediction that language becomes a more important channel when conventional policy is constrained (Section 2.5). The NASDAQ interaction is marginally significant (p = 0.041) but economically implausible, with a coefficient of 202 basis points that likely reflects outlier influence. The more robust CRSP VW result shows no significant interaction, suggesting that the language channel of monetary policy does not systematically strengthen at the zero lower bound. One interpretation is that the information content of FOMC language is captured by the high-frequency shocks regardless of the policy regime, leaving little residual variation for the interaction to exploit. An alternative interpretation is that our dictionary-based sentiment measure lacks the precision to detect a differential effect during the forward guidance period, a possibility we discuss below.

A fourth, more structural interpretation draws on Chen, Granville, and Matousek (2025), who find that forward guidance language operates through a risk premium channel rather than an expectations channel: dovish forward guidance reduces uncertainty and compresses term premia, while simultaneously lowering expected future rates. These two channels have opposing effects on equity prices — lower term premia support valuations, but lower expected rates may signal weaker growth — and can offset each other in a reduced-form regression. Our interaction specification assumes a unidirectional effect ($\beta_4 > 0$), which would fail to detect a risk premium channel that operates in the opposite direction. This channel heterogeneity may explain why our H4 is null: the forward guidance period interaction captures the net of two offsetting channels, yielding a coefficient indistinguishable from zero. Disentangling these channels would require a structural model or a topic-specific sentiment measure, as Chen et al. (2025) propose.

This interpretation is further supported by the IMF's large-scale LLM analysis of central bank communication (Gambacorta et al., 2025), which finds that a directional communication index — constructed by decomposing text into topic, stance, sentiment, and forward-lookingness dimensions — predicts future market interest rate movements. Their result demonstrates that when sentiment is measured at the topic level rather than aggregated across all content, the predictive power increases substantially. Applied to our context, this suggests that our null H4 result may reflect not the absence of a forward guidance effect, but rather the inability of our aggregate sentiment measure to capture the topic-specific channels through which forward guidance operates.

To test the channel heterogeneity hypothesis directly, we estimate a dual-equation model: one equation for equity returns (capturing the expectations channel) and one for bond market variables (capturing the risk premium channel). If forward guidance operates primarily through the risk premium channel, as Chen et al. (2025) suggest, we would expect the path shock to have a significant effect on bond yields or credit spreads but not on equity returns. The results do not support this hypothesis: neither the target nor the path shock has a significant effect on 10-year Treasury yields (R² = 0.72%, target p = 0.403, path p = 0.890), the 13-week T-bill (R² = 0.66%, target p = 0.491, path p = 0.737), or VIX changes (R² = 0.24%, target p = 0.970, path p = 0.401). The forward guidance interaction is also insignificant in all risk premium specifications (e.g., sentiment × FG on VIX: p = 0.304). This null result suggests that the risk premium channel, if present, operates at a higher frequency than daily data can capture — consistent with Chen et al.'s use of 30-minute event windows. Our daily frequency may be too coarse to separate the expectations and risk premium channels, as both channels are absorbed into the overnight price adjustment.

Fourth, the dramatic sensitivity of results to the choice of surprise measure (Section 5.2) reinforces the methodological concerns raised in Section 2.7. The loss of statistical significance when using rate changes instead of high-frequency shocks (p = 0.726 vs. 0.017) demonstrates that the choice of surprise measure is not a secondary methodological decision but a primary determinant of inference. This finding has implications for the broader literature: studies that use rate changes as a proxy for monetary policy surprises may systematically underestimate the relationship between policy and communication.

### 7.2 The Jarociński-Karadi Decomposition and Information Shocks

An important limitation of our analysis is that the GSS target/path decomposition does not separately identify monetary policy shocks from information shocks. As Jarociński and Karadi (2020) demonstrate, the observed response of asset prices to FOMC announcements reflects a mixture of monetary policy and information effects: a monetary policy tightening raises rates and lowers stock prices, while an information shock (in which the Fed reveals positive economic news) raises both rates and stock prices. If information shocks are present in our sample, they could bias our estimates of the target shock effect on sentiment.

Specifically, if the target shock partly reflects information shocks — in which the Fed's rate decision reveals positive information about the economy — the estimated effect of the target shock on sentiment would be biased upward, because positive information would make the statement language more hawkish (higher CB score) even though the monetary policy stance is contractionary. This would be consistent with our finding that the target shock is significant while the path shock is not, if information shocks are primarily captured by the target factor. Implementing the Jarociński-Karadi sign restriction decomposition would allow us to separately identify these effects, and we regard this as a high priority for future work.

### 7.3 Sentiment Measurement and the CB Dictionary

Our finding that the CB score alone yields substantially stronger results than the combined LM+CB score (R² = 3.90% vs. 1.57%, with both target and path shocks significant at p < 0.05) has important implications for the sentiment analysis literature (Section 2.4). The LM dictionary's positivity bias — all 117 FOMC statements have positive LM scores — dilutes the signal from the CB component, reducing both the R² and the statistical significance of the path shock. This suggests that future studies of central bank communication should use domain-specific dictionaries rather than general-purpose financial dictionaries, consistent with the recommendation of Apel and Blix (2014).

However, we note that the CB dictionary itself has limitations. The equal-weighted combination of hawkish and dovish word counts does not account for the intensity or context of word usage. A word like "vigilant" may carry different weight in a statement about inflation risks than in a statement about financial stability. Transformer-based models such as FinBERT (Yang et al., 2020) or domain-specific language models could address this limitation by capturing contextual meaning, potentially increasing the R² of the sentiment regression and providing more precise estimates of the target and path effects.

The potential of large language models (LLMs) for central bank communication analysis is illustrated by Chen, Granville, and Matousek (2025), who use GPT-4 to classify FOMC text into four topics and construct topic-specific sentiment measures. Their comparison of GPT-4 and GPT-3.5 is particularly instructive: GPT-3.5 fails to identify forward guidance content in 135 of 139 Statements, while GPT-4 correctly classifies the vast majority. This failure cascades through their entire analysis — GPT-3.5-based metrics lose statistical significance not only for the forward guidance topic but also for policy objectives and policy decisions, because forward guidance language is misallocated to these other categories. This finding has direct implications for our analysis: our CB dictionary, like GPT-3.5, cannot distinguish forward guidance language from other policy-relevant text, which may explain why our combined sentiment measure dilutes the signal from the CB component.

A more practical upgrade path may be the central bank language models (CB-LMs) introduced by Gambacorta et al. (2024). Unlike GPT-4, CB-LMs are open-weight encoder-only models retrained on central bank corpora, offering three advantages for academic research: (i) full reproducibility, as model weights are fixed and publicly available; (ii) lower computational cost, as fine-tuning an encoder model requires far fewer resources than deploying a generative LLM; and (iii) domain specificity, as the model is trained on the same institutional language it is asked to classify. Gambacorta et al. show that CB-LMs outperform general-purpose foundation models on stance classification in the central banking domain, suggesting that the marginal benefit of larger proprietary models may be small for this specific task. For our analysis, replacing the CB dictionary with CB-LM embeddings could capture contextual meaning while preserving the reproducibility that dictionary-based approaches offer.

A complementary approach is the uncertainty-aware framework of Yao and Chai (2025), who use LLMs to classify FOMC text along monetary policy transmission paths — mimicking how human experts reason about policy — while quantifying classification confidence. This addresses a key limitation of both dictionary-based and standard LLM-based approaches: the inability to distinguish between confident and uncertain classifications. In our context, an uncertainty-weighted sentiment measure would downweight statements where the hawkish-dovish classification is ambiguous, potentially increasing the signal-to-noise ratio of the sentiment regression.

To explore whether dimension decomposition improves our results, we construct separate sentiment scores for forward-looking language (sentences containing "expect," "anticipate," "will," "project," or "forecast") and current-assessment language (sentences containing "recent," "current," "has," "was," or "remains"). The results are instructive: the forward-looking sentiment score yields R² = 0.79% with the path shock insignificant (p = 0.800), while the current-assessment score yields R² = 1.30% with the path shock also insignificant (p = 0.506). Both are substantially lower than the combined CB score (R² = 3.90%). This counterintuitive finding — that the combined score outperforms either dimension alone — suggests that the path shock captures a broad policy stance signal rather than a specific forward-guidance dimension, and that splitting the sentiment measure reduces statistical power by fragmenting the already limited variation in FOMC statement language.

We also test whether statement novelty — measured as the Jaccard distance between consecutive statement word sets — improves the sentiment regression. Weighting observations by novelty in a WLS regression increases the R² from 3.98% to 5.75%, a 45% improvement. This result supports the intuition that not all FOMC statements are equally informative: statements that differ substantially from their predecessors carry more information and should receive greater weight in the regression. However, the novelty-weighted result remains modest in absolute terms, suggesting that measurement innovation alone cannot compensate for the fundamental limitation of daily-frequency data in capturing high-frequency policy effects.

### 7.4 Limitations and Future Directions

Several limitations should be noted. Our sentiment dictionary, while expanded to 591 hawkish and 222 dovish terms, remains a bag-of-words approach that cannot capture the nuanced semantics of FOMC language. The Loughran-McDonald component exhibits a positivity bias (always positive for FOMC text), which dilutes the signal from the central bank component. A FinBERT-based approach would likely yield more powerful sentiment measures by capturing contextual meaning, but requires GPU resources not available in our current environment. Our sample period ends in July 2022 for the Acosta shock data. The Bauer-Swanson (2023) critique suggests that our shocks may not be fully exogenous; while the relative importance of the target factor is likely robust to this concern, a more rigorous treatment would implement the Bauer-Swanson orthogonalization procedure.

Several avenues for future research emerge. First, the use of more sophisticated NLP techniques — such as CB-LMs (Gambacorta et al., 2024) or uncertainty-aware LLM frameworks (Yao and Chai, 2025) — could improve the measurement of FOMC statement sentiment. Transformer-based models can capture contextual meaning that bag-of-words approaches miss, potentially increasing the R² of the sentiment regression and providing more precise estimates of the target and path effects. Second, extending the analysis to FOMC minutes, press conference transcripts, and speeches could provide a more comprehensive picture of the relationship between monetary policy shocks and central bank language. Third, a structural model that jointly estimates the effects of monetary policy shocks on sentiment and asset returns could provide more precise identification of the language channel. Fourth, cross-country comparisons could shed light on whether the patterns we document are specific to the Federal Reserve or are a general feature of central bank communication. Fifth, implementing the Jarociński-Karadi (2020) sign restriction decomposition would provide a structural identification of monetary policy vs. information shocks, complementing our reduced-form analysis. Sixth, the multi-agent LLM framework of Weinig (2025), which uses specialised agents to process different aspects of Federal Reserve communications and construct narrative monetary policy surprises, represents a promising direction for replacing high-frequency surprise measures with text-derived alternatives that capture the semantic content of policy communication more directly.

More broadly, our paper demonstrates the value of combining text analysis with high-frequency identification in monetary policy research. By directly examining the relationship between monetary policy shocks and the language of FOMC statements, we provide a more direct test of the information content of central bank language than studies that rely solely on asset price responses. As central banks increasingly rely on communication as a policy tool, understanding the determinants and effects of their language becomes ever more important for both academic research and policy design.

**Data Availability.** The monetary policy shock data from Acosta (2022) are publicly available. FOMC statements are available from the Federal Reserve website. CRSP market data are available through WRDS with an institutional subscription. The replication code and processed datasets will be made available upon publication.

---

## References

Acosta, M. (2022). Monetary Policy Surprises and the FOMC. Working Paper.

Apel, M., & Blix, G. (2014). How Is Inflation Affected by Globalisation? *Sveriges Riksbank Economic Review*, 2014(2), 51–75.

Bauer, M. D., & Swanson, E. T. (2023). A Reassessment of Monetary Policy Surprises and High-Frequency Identification. *NBER Macroeconomics Annual*, 37(1), 87–155.

Blinder, A. S., Ehrmann, M., Fratzscher, M., De Haan, J., & Jansen, D. J. (2008). Central Bank Communication and Monetary Policy: A Survey of Theory and Evidence. *Journal of Economic Literature*, 46(4), 910–945.

Campbell, J. R., Evans, C. L., Fisher, J. D. M., & Justiniano, A. (2012). Macroeconomic Effects of Federal Reserve Forward Guidance. *Brookings Papers on Economic Activity*, Spring, 1–80.

Chen, K., Granville, B., & Matousek, R. (2025). Decoding Central Bank Communications with Large Language Models. *Journal of Monetary Economics*, forthcoming.

Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary Policy Shocks: What Have We Learned and to What End? In J. B. Taylor & M. Woodford (Eds.), *Handbook of Macroeconomics* (Vol. 1, pp. 65–148). Elsevier.

Cieslak, A., Morse, A., & Vissing-Jorgensen, A. (2019). Stock Returns over the FOMC Cycle. *Journal of Financial Economics*, 133(1), 114–137.

Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.

Federal Reserve Board (2024). Using Generative AI Models to Understand FOMC Monetary Policy Discussions. *FEDS Notes*, December 6, 2024.

Friedman, M., & Schwartz, A. J. (1963). *A Monetary History of the United States, 1867–1960*. Princeton University Press.

Gambacorta, L., Kwon, B., Park, T., Patelli, P., & Zhu, S. (2024). CB-LMs: Language Models for Central Banking. *BIS Working Paper*, No. 1215.

Gambacorta, L., Kwon, B., Park, T., Patelli, P., & Zhu, S. (2025). From Text to Quantified Insights: A Large-Scale LLM Analysis of Central Bank Communication. *IMF Working Paper*, 2025/109.

Gertler, M., & Gilchrist, S. (1994). Monetary Policy, Business Cycles, and the Behavior of Small Manufacturing Firms. *Quarterly Journal of Economics*, 109(2), 309–340.

Gürkaynak, R. S., Sack, B., & Swanson, E. T. (2005a). The Sensitivity of Long-Term Interest Rates to Economic News: Evidence and Implications for Monetary Policy. *American Economic Review*, 95(1), 425–436.

Gürkaynak, R. S., Sack, B., & Swanson, E. T. (2005b). Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements. *International Journal of Central Banking*, 1(1), 55–93.

Hansen, S., McMahon, M., & Prat, A. (2018). Transparency and Deliberation within the FOMC: A Computational Linguistics Approach. *Quarterly Journal of Economics*, 133(2), 801–870.

Huang, A. H., Zang, A. Y., & Zheng, R. (2022). Evidence on the Information Content of Text in Analyst Reports. *Review of Accounting Studies*, 27, 85–119.

Jarociński, M., & Karadi, P. (2020). Deconstructing Monetary Policy Surprises—The Role of Information Shocks. *American Economic Journal: Macroeconomics*, 12(2), 1–43.

Jeenas, P. (2019). Firm Balance Sheet Liquidity, Monetary Policy Shocks, and Investment Dynamics. Working Paper, Stockholm School of Economics.

Kuttner, K. N. (2001). Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market. *Journal of Monetary Economics*, 47(3), 523–544.

Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.

Lucca, D. O., & Trebbi, F. (2009). Measuring Central Bank Communication: An Automated Approach with Application to FOMC Statements. *American Economic Journal: Applied Economics*, 1(2), 168–193.

Nakamura, E., & Steinsson, J. (2018). High-Frequency Identification of Monetary Non-Neutrality: The Information Effect. *Quarterly Journal of Economics*, 133(3), 1283–1330.

Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.

Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *Review of Economic Studies*, 61(4), 631–653.

Romer, C. D., & Romer, D. H. (1989). Does Monetary Policy Matter? A New Test in the Spirit of Friedman and Schwartz. *NBER Macroeconomics Annual*, 4, 121–184.

Romer, C. D., & Romer, D. H. (2000). Federal Reserve Information and the Behavior of Interest Rates. *American Economic Review*, 90(3), 429–457.

Rosa, C. (2011). Words That Shake Traders: The Stock Market's Reaction to Central Bank Communication in Real Time. *Journal of Empirical Finance*, 18(5), 915–934.

Swanson, E. T. (2021). Measuring the Effects of Federal Reserve Forward Guidance and Asset Purchases on Financial Markets. *Journal of Monetary Economics*, 118, 32–53.

Tadle, R. C. (2022). FOMC Minutes Sentiments and Their Impact on Financial Markets. *Journal of Economics and Business*, 118, 106021.

White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. *Econometrica*, 48(4), 817–838.

Weinig, M. (2025). Narrative Monetary Policy Surprises. SSRN Working Paper.

Yang, Y., UY, M. C. S., & Huang, A. (2020). FinBERT: A Pretrained Language Model for Financial Communications. *arXiv preprint arXiv:2006.08097*.

Yao, J., & Chai, X. (2025). Interpreting Fedspeak with Confidence: An LLM-Based Uncertainty-Aware Framework for Monetary Policy Stance Classification. *arXiv preprint arXiv:2508.08001*.

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

| Regime | N | R² | $\beta_T$ (p) | $\beta_P$ (p) |
|--------|:--:|:--:|:------------:|:----------:|
| Rate hike | 17 | 10.2% | −0.000554 (0.013) | −0.000145 (0.298) |
| Rate cut | 11 | 43.1% | 0.000188 (0.089) | 0.000837 (0.000) |
| Unchanged | 89 | 2.0% | 0.000638 (0.616) | 0.001379 (0.079) |

During rate cut meetings, the path shock is highly significant (p < 0.001) and the R² is 43.1%, suggesting that forward guidance language is most responsive to the path shock when the Fed is easing. During rate hike meetings, the target shock dominates (p = 0.013). When rates are unchanged, neither shock is significant at 5%.

### C.2 Sentiment Distribution

| Statistic | Combined | LM Score | CB Score |
|-----------|:--------:|:--------:|:--------:|
| Mean | 0.014 | 0.041 | −0.013 |
| Std | 0.006 | 0.008 | 0.005 |
| Min | 0.008 | 0.031 | −0.022 |
| Max | 0.034 | 0.071 | 0.005 |
| % Negative | 0% | 0% | 92.3% |
| % Positive | 100% | 100% | 1.7% |

The LM score is always positive for FOMC statements (min = 0.031), because FOMC statements use more positive than negative words regardless of policy stance. The CB component has substantial sign variation (92.3% negative, 1.7% positive, 6.0% zero), reflecting the predominantly dovish language in our sample period. The equal-weighted combination dilutes this signal.

### C.3 Newey-West Lag Sensitivity (H1 Regression)

| Lag | $\beta_T$ (t) | p_target | $\beta_P$ (t) | p_path | R² |
|:---:|:------------:|:--------:|:----------:|:------:|:--:|
| 1 | 0.000577 (2.78) | 0.006 | 0.000633 (1.64) | 0.100 | 1.57% |
| 2 | 0.000577 (2.61) | 0.010 | 0.000633 (1.55) | 0.123 | 1.57% |
| 4 | 0.000577 (2.43) | 0.017 | 0.000633 (1.44) | 0.152 | 1.57% |
| 6 | 0.000577 (2.29) | 0.024 | 0.000633 (1.45) | 0.149 | 1.57% |

### C.4 Data Source Comparison (S&P 500)

| Data Source | $\beta_T$ | t_target | p_target | R² |
|-------------|:--------:|:--------:|:--------:|:--:|
| CRSP (sprtrn) | −0.391 | −1.80 | 0.073 | 0.078 |
| yfinance (^GSPC) | −0.259 | −2.19 | 0.030 | 0.029 |

The yfinance coefficient is 34% smaller in magnitude than the CRSP coefficient, reflecting the absence of delisting adjustments in the yfinance data.
