# Oral Presentation Script — v10.3
## Words Beyond the Rate: High-Frequency Monetary Policy Shocks and FOMC Language

---

### Slide 1: Title (30s)

Good morning. I'm Eileen Zhang from Xi'an Jiaotong-Liverpool University, and today I'll present "Words Beyond the Rate: High-Frequency Monetary Policy Shocks and FOMC Language."

This is joint work with Dechang Yang.

---

### Slide 2: Motivation — The Taper Tantrum (1 min)

When the FOMC meets, markets react not just to the rate decision, but to the words in the statement. The 2013 taper tantrum is the classic example — rates didn't change, but the language shift moved markets dramatically.

This raises a fundamental question: does FOMC statement language convey information beyond the rate decision itself? And if so, can we decompose that information into a current-policy component and a forward-guidance component?

This is the information channel hypothesis, debated since Romer and Romer 2000.

---

### Slide 3: Two Channels of FOMC Influence (45s)

FOMC announcements influence markets through two channels.

First, the rate decision — the implementation channel. Markets adjust to the actual change in the federal funds rate.

Second, the statement language — the information channel. The words may reveal the Committee's assessment of economic conditions and future policy intentions.

The key question is: which channel dominates? Is the language primarily explaining what was done, or signaling what will be done?

---

### Slide 4: Two Competing Interpretations (45s)

Under policy implementation, the statement explains the current decision. The target shock — the surprise in the current rate — should drive sentiment. Language reflects what was done. This is consistent with Bernanke and Kuttner 2005.

Under informational revelation, the statement reveals future policy intent. The path shock — the revision in expected future policy — should drive sentiment. Language signals what will be done. This is consistent with Romer and Romer 2000 and Campbell et al. 2012.

Our empirical strategy tests which interpretation the data supports.

---

### Slide 5: Our Contribution (45s)

We make three contributions.

First, we directly test the information channel using high-frequency monetary policy shocks from the Gürkaynak-Sack-Swanson decomposition, which separates current-rate surprises from future-policy revisions.

Second, we combine this with textual sentiment analysis of FOMC statements using both the Loughran-McDonald and a central-bank-specific dictionary.

Third, we examine whether the forward guidance period strengthened the language channel, as theory would predict.

---

### Slide 6: Data Overview (1 min)

Our sample covers 117 FOMC meetings from 2006 to 2022, spanning three Fed Chairs and three policy regimes.

The key monetary policy shocks come from Acosta 2022, who extends the GSS decomposition. The target shock captures the current-rate surprise, and the path shock captures the revision in expected future policy. Their correlation is only 0.14, confirming successful separation.

For sentiment, we use two dictionaries: Loughran-McDonald, the standard finance dictionary, and a central-bank-specific dictionary that captures hawkish and dovish language. Our baseline is the equal-weighted combination.

Financial market data comes from CRSP via WRDS, plus gold, Treasury yields, and VIX.

---

### Slide 7: Empirical Framework (1 min)

We test four hypotheses.

H1: Do monetary policy shocks predict statement sentiment? We regress sentiment on target and path shocks.

H2: Do the shocks predict asset returns? Same regression structure with different dependent variables.

H3: Are target and path effects statistically different? We use a Wald test.

H4: Does the forward guidance period strengthen the language channel? We add a sentiment-times-FG interaction term.

All regressions use OLS with Newey-West HAC(4) standard errors to account for heteroskedasticity and autocorrelation.

---

### Slide 8: Table 1 — Summary Statistics (30s)

Before the results, a quick look at the data. The target and path shocks are standardized to unit variance. The combined sentiment has a mean of 0.014 with modest variation. The Kuttner surprise averages near zero, as expected for a surprise measure.

The low correlation between target and path shocks — 0.14 — is important. It means the GSS decomposition successfully separates the two dimensions of monetary policy.

---

### Slide 9: Table 2 — Main Result (1 min 30s)

Now the main result. When we regress statement sentiment on target and path shocks, the target shock is significant at the 5 percent level, with a p-value of 0.017. The path shock is not significant, with a p-value of 0.152.

The R-squared is 1.57 percent, which is modest but typical for event-study regressions with high-frequency shocks. For comparison, using naive rate changes instead of GSS shocks gives a p-value of 0.726 — completely insignificant. The market-based identification matters.

This pattern — target significant, path not — is consistent with the policy implementation interpretation. Statement language primarily reflects the current rate decision, not forward guidance about future policy.

---

### Slide 10: Table 3 — Surprise Comparison (45s)

We compare three surprise measures. Simple rate changes: not significant. Kuttner surprises: not significant. GSS target and path shocks: target is significant.

The key takeaway is that identification matters. The GSS decomposition separates the current-policy surprise from the future-policy revision, and only the current-policy component predicts statement language.

---

### Slide 11: Table 4 — Asset Returns (1 min)

Do the same shocks predict asset returns? Yes, for equities. The target shock significantly predicts CRSP value-weighted returns, with a p-value of 0.046. The coefficient is negative, meaning hawkish surprises reduce equity prices.

The path shock is not significant for any asset class. Gold and Treasury yields also show no significant response, though this may reflect the limitations of daily frequency data.

The equal-weighted index is less significant than the value-weighted index, suggesting larger firms are more responsive to monetary policy surprises.

---

### Slide 12: H3 — Wald Test (30s)

Although the target shock is significant and the path shock is not, can we statistically distinguish them? The Wald test gives a p-value of 0.90 — we cannot reject that the two coefficients are equal.

This means we cannot make a strong claim that the target effect is different from the path effect. The evidence favors implementation over revelation, but does not conclusively eliminate information effects.

---

### Slide 13: Table 5 — Forward Guidance Interaction (1 min)

The forward guidance hypothesis predicts that the language channel should be stronger during the zero lower bound period, when the Fed relied more on communication. We test this with a sentiment-times-FG interaction term.

The result: the interaction is not significant. For CRSP VW, the p-value is 0.836. For NASDAQ, it's 0.739. Sentiment does not become more important during the forward guidance period.

This is a null result, but an informative one. It suggests that the FG period did not strengthen the link between monetary policy shocks and statement language, contrary to what the information channel would predict.

---

### Slide 14: Table 6 — Alternative Sentiment Measures (1 min)

Are the results robust to alternative sentiment measures? Yes. Using the central-bank dictionary alone, R-squared doubles to 3.90 percent. The CB dictionary captures hawkish and dovish language that the LM dictionary misses in the central-bank context.

More interestingly, when we apply the same analysis to FOMC Minutes instead of statements, the path shock becomes significant, with a p-value of 0.015. Minutes contain longer, more detailed discussions that include forward-looking content. The combined Minutes model achieves an R-squared of 9.35 percent — six times the statement result.

This suggests that the information channel may operate through Minutes rather than statements, and that the document choice matters for identifying these effects.

---

### Slide 15: Robustness (45s)

The results are robust across several dimensions. Alternative Newey-West lag structures — HAC(2), HAC(4), HAC(6) — and White heteroskedasticity-robust standard errors all give the same qualitative conclusion. Excluding the COVID period doesn't change the results.

Some regime heterogeneity emerges: the target shock is most significant during the forward guidance period, while neither shock is significant during normalization. But these subsample results should be interpreted cautiously.

The Bauer-Swanson 2023 critique — that high-frequency shocks may contain predictable components — applies to both target and path shocks, so the relative comparison remains informative.

---

### Slide 16: JK Sign-Restriction Decomposition (1 min)

We implement a simplified Jarociński-Karadi decomposition to separate pure monetary policy shocks from central bank information shocks. The classification is based on sign restrictions: if rates go up and stocks go down, it's a pure MP shock; if rates go up and stocks also go up, it's a CBI shock — the central bank is revealing positive information.

We find 59% MP shocks and 41% CBI shocks. For sentiment, neither component is individually significant — the decomposition loses power by splitting the target shock. But for asset returns, the results are striking: MP shocks push stocks down significantly, CBI shocks push stocks up significantly, and R-squared jumps from 9% to 36%. This confirms the information effect exists in asset markets, but statement sentiment doesn't differentiate between MP and CBI.

---

### Slide 17: Bauer-Swanson Orthogonalization (45s)

We also address the Bauer-Swanson critique directly by orthogonalizing shocks against pre-FOMC macro information. Both shocks are partially predictable — about 10-14% R-squared in the first stage.

After orthogonalization, the target shock loses significance for sentiment (p goes from 0.012 to 0.108), but path remains insignificant. However, for asset returns, the target shock actually strengthens (p goes from 0.043 to 0.005). This asymmetry is interesting: the predictable component attenuates the sentiment relationship but not the asset-return relationship, suggesting sentiment captures a broader communication channel.

---

### Slide 18: Relation to Fernández-Fuertes 2025 (1 min)

A natural question is how our work relates to Fernández-Fuertes 2025, which also uses LLMs and Fed communications to study monetary policy shocks. He constructs a better shock measure — multi-agent LLM framework, processes Statements, Minutes, Beige Books, press conferences. His R-squared is 12.4% versus our 1.57%.

But here's the key: his Table 32 projects his narrative surprise onto the GSS target/path factors and finds exactly the same pattern — target significant, path not significant. This is independent confirmation of our core finding.

The difference is the question. He asks: can we build better shocks? We ask: what does FOMC language convey? We provide four structured hypotheses, a Wald test, an FG interaction test, and a Statement-versus-Minutes channel comparison — none of which he conducts. And our dictionary-based approach is fully transparent and reproducible without API access.

Complementary, not competing.

---

### Slide 19: Future Directions (30s)

Several extensions remain. LLM-based sentiment measures, building on Chen et al. 2025 and Fernández-Fuertes 2025, could capture nuances that dictionary methods miss. Intraday event windows using TAQ data could capture channels that daily data misses. And extending the GSS series to 2025 using the SF Fed's USMPD data would add the recent rate-cut cycle.

On the data side, press conference transcripts and SEP projections could provide richer text data, and cross-country central bank communication analysis could test the external validity of our findings.

---

### Slide 20: Conclusion (30s)

To conclude: the target shock is a statistically significant predictor of FOMC statement sentiment, but the path shock is not. The information channel, as traditionally formulated, receives only suggestive support. Forward guidance does not strengthen the language channel at the zero lower bound.

The key takeaway: FOMC language is informative, but monetary policy shocks are just one of many forces shaping it. Understanding the full information content requires looking beyond the rate decision — and beyond the shocks themselves.

Thank you. I welcome your questions.

---

Total speaking time: approximately 15 minutes
