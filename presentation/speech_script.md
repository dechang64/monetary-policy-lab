# Oral Presentation Script
## Beyond the Rate: Information Content of FOMC Statement Language

---

### Slide 1: Title (30s)

Good morning. I'm Eileen Zhang from XJTLU, and today I'll present "Beyond the Rate: Information Content of FOMC Statement Language."

---

### Slide 2: Motivation (1 min)

When the FOMC meets, markets react not just to the rate decision, but to the words in the statement. The 2013 taper tantrum is the classic example — rates didn't change, but the language shift moved markets dramatically.

This raises a fundamental question: does FOMC statement language convey information beyond the rate decision itself? And if so, can we decompose that information into a current-policy component and a forward-guidance component?

This is the information channel hypothesis, debated since Romer and Romer 2000 and Campbell et al. 2012.

---

### Slide 3: Research Question (45s)

We formalize this as four hypotheses. H1: monetary policy shocks predict statement sentiment. H2: shocks predict asset returns. H3: the path shock has a larger effect on sentiment than the target shock — the information channel. H4: the sentiment-return relationship strengthens during the forward guidance period.

Our sample covers 117 FOMC meetings from 2006 to 2022, combining Acosta high-frequency shocks, FOMC statement texts, and CRSP market returns.

---

### Slide 4: Methodology — Shocks (1 min)

We use the Gürkaynak, Sack, and Swanson target/path decomposition, as replicated by Acosta 2022. The target surprise captures the unexpected component of the current rate decision, identified from fed funds futures. The path factor captures the surprise about the future rate trajectory, identified from eurodollar futures. Both are measured in a narrow 30-minute window around the announcement.

This decomposition is crucial: using naive rate changes instead of high-frequency shocks attenuates R² by a factor of four — from 1.57% to 0.40%. The choice of surprise measure is not a secondary methodological decision; it's a primary determinant of inference.

---

### Slide 5: Methodology — Sentiment (1 min)

We construct a combined sentiment dictionary: the Loughran-McDonald financial dictionary plus a central-bank-specific hawkish-dovish dictionary with 591 hawkish and 222 dovish terms. The sentiment score is the difference between hawkish and dovish word counts, scaled by total words.

A key finding: the LM component has a positivity bias — all 117 statements have positive LM scores, because FOMC language uses words like "growth" and "stable" regardless of policy stance. The CB component has actual sign variation: 78% of statements have negative CB scores. When we use CB-only sentiment, R² more than doubles from 1.57% to 3.90%.

---

### Slide 6: H1 Results (1 min 30s)

Table 4 shows the H1 regression: sentiment on target and path shocks.

The target shock is significant at 5% — a contractionary target surprise is associated with more hawkish language. The path shock is NOT significant at conventional levels (p = 0.152). R² is 1.57%.

This is the opposite of what the information channel predicts. The strong version of the hypothesis says forward guidance language should be primarily driven by the path shock. We find the opposite: the current rate decision drives language, not the future path.

However, a Wald test cannot reject that the two coefficients are equal (p = 0.90). So we cannot definitively rule out equal effects. The evidence is suggestive but not conclusive.

---

### Slide 7: H2 Results (1 min)

For H2, the target shock has a negative effect on equity returns — contractionary surprises reduce stock prices. This is significant for small-cap stocks (CRSP EW: p = 0.013) but marginal for large-cap (S&P 500: p = 0.073), consistent with the credit channel mechanism of Gertler and Gilchrist 1994.

The path shock does NOT significantly affect daily returns. This is consistent across all equity indices.

---

### Slide 8: H4 — Forward Guidance Interaction (1 min)

H4 asks whether the language channel strengthens at the zero lower bound. The answer is no. The forward guidance interaction is completely insignificant — p = 0.991 for CRSP VW, p = 0.739 for NASDAQ. This null result is robust across all specifications.

Why? We explore three interpretations. First, the information content of FOMC language is captured by the high-frequency shocks regardless of regime. Second, our dictionary-based measure lacks the precision to detect a differential effect. Third — and this is where recent literature helps — Chen, Granville, and Matousek 2025 find that forward guidance operates through a risk premium channel, not an expectations channel. These two channels have opposing effects on equity prices and can offset each other in a reduced-form regression.

---

### Slide 9: Dual-Channel Test (45s)

We test the risk premium channel directly. If forward guidance operates through risk premia, the path shock should affect bond yields or VIX even if it doesn't affect equity returns. The results: nothing significant. 10-year Treasury: p = 0.89. VIX: p = 0.40. The risk premium channel, if present, operates at a higher frequency than daily data can capture — consistent with Chen et al.'s use of 30-minute windows.

---

### Slide 10: Forward-Lookingness Dimension (45s)

Inspired by the IMF's four-dimensional framework, we decompose sentiment into forward-looking and current-assessment components. The result is counterintuitive: forward-looking sentiment has R² = 0.79% with path p = 0.80, while the combined score has R² = 1.57% with path p = 0.152. Splitting the sentiment measure reduces statistical power by fragmenting the already limited variation in FOMC language. The path shock captures a broad policy stance signal, not a specific forward-guidance dimension.

---

### Slide 11: Statement Novelty (30s)

We also test whether statement novelty — how much a statement differs from its predecessor — improves the regression. Weighting by novelty increases R² from 3.98% to 5.75%, a 45% improvement. Not all FOMC statements are equally informative, and weighting by information content helps.

---

### Slide 12: Literature Context (45s)

Our work sits within a rapidly evolving literature. Chen et al. 2025 use GPT-4 to construct topic-specific PLMIS shocks from FOMC Minutes. Gambacorta et al. 2024 introduce CB-LMs — lightweight, open-weight language models that outperform general models on central bank text. The IMF's 2025 analysis of 75,000 documents shows that topic-decomposed sentiment predicts interest rate movements. Even the Federal Reserve itself is using generative AI to analyze FOMC discussions.

The common thread: coarse-grained sentiment measures dilute signal. Our CB-only result (R² = 3.90% vs. combined 1.57%) is a microcosm of this finding.

---

### Slide 13: Contributions (45s)

Our contributions: First, we provide a direct test of the information channel using statement text, finding that the target shock — not the path shock — is the primary driver of language sentiment. Second, we demonstrate that the choice of surprise measure has first-order implications: high-frequency shocks yield 4× the R² of rate changes. Third, we show that the forward guidance interaction is null, and propose a risk premium channel explanation supported by dual-equation tests. Fourth, we introduce statement novelty weighting as a simple but effective improvement.

---

### Slide 14: Limitations & Future Work (1 min)

Limitations: R² is modest. Our dictionary approach cannot capture context. Daily frequency may be too coarse to separate expectations and risk premium channels.

The most promising upgrade path is CB-LMs — open-weight, domain-specific language models that offer the contextual understanding of LLMs with the reproducibility of dictionaries. Combined with uncertainty-aware classification from Yao and Chai 2025, and multi-agent narrative surprise extraction from Weinig 2025, the next generation of this research could substantially improve explanatory power.

---

### Slide 15: Conclusion (30s)

To conclude: the target shock is a statistically significant predictor of FOMC statement sentiment, but the path shock is not. The information channel, as traditionally formulated, receives only suggestive support. Forward guidance does not strengthen the language channel at the zero lower bound. And the risk premium channel, if present, operates at frequencies our daily data cannot capture.

The key takeaway: FOMC language is informative, but monetary policy shocks are just one of many forces shaping it. Understanding the full information content requires looking beyond the rate decision — and beyond the shocks themselves.

Thank you. I welcome your questions.

---

Total speaking time: approximately 15 minutes
