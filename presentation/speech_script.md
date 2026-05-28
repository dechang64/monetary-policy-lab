# Oral Presentation Script
## Beyond the Rate: Information Content of FOMC Forward Guidance Language

---

### Slide 1: Title (30s)

Good morning/afternoon. I'm Dechang Yu from XJTLU, and today I'll present our work "Beyond the Rate: Information Content of FOMC Forward Guidance Language." This is joint work with Eileen Zhang.

---

### Slide 2: Motivation (1 min)

Let me start with a simple observation. When the FOMC meets, markets react not just to the rate decision, but to the words in the statement. The classic example is the 2013 "taper tantrum" — rates didn't change, but the language shift moved markets dramatically.

This raises a fundamental question: does FOMC statement language convey information beyond the rate decision itself? If so, what kind of information — about the current economy, or about the future path of policy?

This is the information channel hypothesis, and it's been debated in the literature since Romer and Romer 2000 and Campbell et al. 2012.

---

### Slide 3: Research Question (45s)

Our research question is straightforward: can we decompose the information in FOMC statements into a target rate component and a forward guidance component, and test which one drives statement language?

We formalize this as four hypotheses. H1 and H2 test whether shocks predict sentiment and returns. H3 is the key test — does the path shock have a larger effect than the target shock? And H4 asks whether this varies across policy regimes.

---

### Slide 4: Methodology (1 min 30s)

Our methodology combines two tools.

First, we use high-frequency monetary policy shocks from Gürkaynak, Sack, and Swanson 2005a, as replicated by Acosta 2022. These decompose the FOMC announcement effect into a target surprise — the unexpected component of the rate decision — and a path factor — the surprise about the future trajectory. These are identified using fed funds and eurodollar futures in a narrow 30-minute window around the announcement.

Second, we construct an expanded central bank sentiment dictionary. The standard Loughran-McDonald dictionary was designed for 10-K filings, not central bank communications. We add 97 hawkish terms like "front-load" and "unanchored," and 106 dovish terms like "patient" and "data-dependent." The two sets are fully disjoint — no word appears in both.

Our sentiment score is the difference between hawkish and dovish word counts, scaled by total words.

---

### Slide 5: Data (45s)

Our sample covers 117 FOMC meetings from 2006 to 2022, combining three data sources: Acosta shock data, 164 FOMC statement texts, and CRSP market returns via WRDS.

We also extend the shock series to 2026 using the San Francisco Fed's USMPD database, which provides the same raw futures data. Our replicated factors correlate at 0.958 for target and 0.970 for path with Acosta's original series.

---

### Slide 6: H1 Results (1 min 30s)

Now the main results. Table 4 shows the H1 regression: sentiment on target and path shocks.

The path shock is significant at the 5% level with a positive coefficient — a contractionary path surprise is associated with more hawkish language. The target shock is significant at 10%.

But I want to be transparent about the explanatory power. R-squared is only 4.06%. Monetary policy shocks explain a small fraction of statement language variation. The remaining 96% reflects the Fed's response to economic data, institutional inertia, and other factors.

Also important: a formal Wald test cannot reject that the two coefficients are equal. So while the path coefficient is larger in magnitude, we cannot statistically distinguish the two effects with our sample size.

---

### Slide 7: H1 Visualization (45s)

This figure shows the raw data. On the left, target shock versus sentiment — a weak positive relationship. On the right, path shock versus sentiment — a somewhat stronger relationship, but with considerable noise. The color coding shows the other shock, illustrating that the two are weakly correlated in our sample.

---

### Slide 8: H2 Results (1 min)

For H2, we look at asset returns. The target shock has a negative effect on equity returns — contractionary surprises reduce stock prices. This is significant for the equal-weighted market at 5%, but not for the value-weighted market.

The path shock does NOT significantly affect daily returns. The coefficients are negative but insignificant. This may reflect limited power with 117 observations, or the fact that the path factor operates through different channels than narrow event windows capture.

The small-cap sensitivity finding is consistent with Gertler and Gilchrist 1994 — smaller firms are more sensitive to monetary policy.

---

### Slide 9: H3 — The Information Channel (1 min)

H3 is the heart of the paper. The path shock has a larger t-statistic than the target shock — 2.01 versus 1.89. But as I mentioned, the Wald test cannot reject equality.

So what can we conclude? The evidence is suggestive but not conclusive. The path shock is significant at 5%, the target at 10%, and the point estimate for path is larger. This is consistent with the information channel — forward guidance language conveys information about the future — but we cannot claim the path effect dominates.

I think this honesty is important. Overclaiming from a 4% R-squared would not survive top-journal scrutiny.

---

### Slide 10: Regime Analysis (1 min)

We also look across policy regimes. The regime-specific results should be interpreted cautiously due to small sample sizes — 8 to 48 meetings per regime.

The most interesting finding is during the Normalization period 2016-2020: the target shock is highly significant while the path shock is not. This makes sense — when the Fed was actively raising rates, the rate decision itself was the primary information source.

During the ZLB period, neither shock is significant individually, but the path shock has a lower p-value, consistent with forward guidance being the main tool when rates are stuck at zero.

---

### Slide 11: Robustness (45s)

Robustness checks confirm the main findings. Excluding COVID has minimal impact. The post-2010 subsample loses significance, reflecting the weaker signal during the ZLB period. And the extended sample through 2026 using USMPD data shows the path shock remains significant at 10%.

---

### Slide 12: Contributions (45s)

Our contributions are threefold. First, we provide a direct test of the information channel using statement text rather than just asset prices. Second, we develop a domain-specific sentiment dictionary for central bank communications. Third, we extend the shock series to 2026 using the USMPD, enabling future research on the recent hiking cycle.

---

### Slide 13: Limitations (1 min)

Let me be upfront about limitations.

First, the low R-squared. Shocks explain only 4% of sentiment variation. The information channel exists, but it's one of many forces.

Second, our dictionary-based sentiment measure cannot capture context. "Patient" is dovish in FOMC language but neutral elsewhere. We haven't validated construct validity against human-coded sentiment.

Third, no control variables. Our regressions are bare-bones — no lagged sentiment, no economic conditions. Adding controls is a priority for the next revision.

Fourth, the sample is US-only. Whether the information channel operates similarly at the ECB or Bank of Japan is an open question.

---

### Slide 14: Conclusion (45s)

To conclude: the path shock is a statistically significant predictor of FOMC statement sentiment, supporting the information channel hypothesis. But the effect is modest, and we cannot formally establish that it dominates the target effect.

The key takeaway: FOMC language is forward-looking, but monetary policy shocks are just one of many forces shaping it. Understanding the full information content of central bank communication requires looking beyond the rate decision — but also beyond the shocks themselves.

Thank you. I welcome your questions.

---

Total speaking time: approximately 15 minutes
