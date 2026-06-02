# Q&A Preparation: Anticipated Questions & Responses — v10.3

---

## Q1: "Your R² is only 1.57%. How do you justify claiming anything with such weak explanatory power?"

**A:** You're right that 1.57% is modest, and we're transparent about this. But three points:

1. R² in event-study regressions with high-frequency shocks is typically low — Gürkaynak et al. 2005a report similar magnitudes for individual asset returns. The shocks capture the *surprise* component, which is by definition small relative to total variation.

2. Using naive rate changes renders the relationship statistically undetectable (p = 0.726 vs. 0.017 for GSS shocks), which tells us the market-based identification matters. And using CB-only sentiment doubles R² to 3.90%.

3. We don't claim the information channel is the *primary* driver — we say the target shock is a *statistically significant predictor*. The modest R² is itself a finding: monetary policy shocks explain only a small fraction of statement language variation.

---

## Q2: "The path shock is not significant. Does this mean the information channel doesn't exist?"

**A:** Not necessarily. Three considerations:

1. The path shock *is* significant in FOMC Minutes (p = 0.015), suggesting the information channel may operate through longer, more detailed documents rather than the concise statement.

2. The Wald test cannot reject β_T = β_P (p = 0.90), so we cannot statistically distinguish the two effects. The path effect may exist but be too imprecisely estimated.

3. The statement is a carefully crafted document designed to manage expectations. It may deliberately smooth over forward-looking information, making it harder to detect path effects in statements specifically.

---

## Q3: "Why use the equal-weighted combination of LM and CB dictionaries? Why not just CB, which performs better?"

**A:** Good question. The CB dictionary does outperform LM in our context (R² 3.90% vs. 1.57%). We use the equal-weighted combination for three reasons:

1. **Transparency**: Equal weighting is the simplest, most transparent approach. No data-snooping in choosing the weight.

2. **Robustness**: The combined score is less sensitive to dictionary-specific biases. LM may miss central-bank jargon, while CB may overfit to FOMC-specific language.

3. **Conservatism**: Using the weaker combined score makes our results more conservative. If we used CB alone, we'd have stronger results, but we prefer to err on the side of understating rather than overstating.

We report CB-only results in the robustness section (Table 6) so readers can evaluate both.

---

## Q4: "The Wald test p-value is 0.90. Doesn't this mean target and path effects are essentially the same?"

**A:** The high p-value means we *cannot reject* that they're the same — but this is different from confirming they're the same. With N = 117, we have limited power to distinguish two coefficients that are both small.

The point estimates tell a story: β_T = 0.000577 (significant) vs. β_P = 0.000633 (not significant). The magnitudes are similar, but the precision differs. Target is estimated more precisely.

We interpret this as: the evidence *favors* implementation over revelation, but we cannot make a definitive statistical distinction. This is why we frame our conclusion as "suggestive evidence" rather than "strong evidence."

---

## Q5: "Why doesn't the FG interaction work? Theory says it should."

**A:** This is indeed puzzling from a theoretical perspective. Several possible explanations:

1. **Statement vs. Minutes**: The statement is a concise, formulaic document. During FG, the Fed may have relied more on press conferences and Minutes for forward guidance, not the statement itself.

2. **FG language is more subtle**: During ZIRP, the statement language changes may be more nuanced — shifts in tone rather than explicit guidance — which daily returns cannot capture.

3. **FG was expected**: By the time FG was in full effect, markets may have already priced in the language patterns, reducing the surprise component.

4. **Power**: The interaction test requires detecting a *difference in slopes* between FG and non-FG periods, which needs more statistical power than we have.

---

## Q6: "How do you address the Bauer-Swanson (2023) critique about predictable components in high-frequency shocks?"

**A:** Bauer and Swanson show that high-frequency monetary policy shocks contain predictable components related to macroeconomic data releases and Fed communication. This is a valid concern.

However, two points:

1. The critique applies to *both* target and path shocks equally. If both are contaminated by predictable components, the relative comparison between them remains informative.

2. We acknowledge this limitation explicitly and list Bauer-Swanson orthogonalized shocks as a future extension. Using their cleaned shocks would strengthen the identification but would not change the fundamental question we're asking.

---

## Q7: "Why not use LLM-based sentiment analysis instead of dictionary methods?"

**A:** Dictionary methods have limitations — they miss context, negation, and nuanced language. LLM-based approaches (Chen et al. 2025, Gambacorta et al. 2024) can capture these.

We chose dictionaries for three reasons:

1. **Reproducibility**: Dictionary methods are fully transparent and reproducible. LLM outputs can vary across runs and models.

2. **Comparability**: Our results are directly comparable to the large literature using LM sentiment in finance.

3. **Conservatism**: If dictionaries find significant effects, LLMs would likely find stronger ones. We prefer the more conservative approach.

We discuss LLM extensions in the future directions section and expect they would improve explanatory power.

---

## Q8: "Why daily returns instead of intraday windows?"

**A:** We use daily returns for data availability and comparability with the existing literature (GSS 2005a, Bernanke & Kuttner 2005). Intraday windows would better isolate the FOMC announcement effect but:

1. TAQ data requires WRDS access and substantial data processing.
2. The GSS shocks are identified using daily closing prices of federal funds futures.
3. Some assets (gold, Treasury) have less liquid intraday markets.

We acknowledge this limitation and list intraday analysis as a future extension.

---

## Q9: "Your sample includes COVID. Aren't those observations outliers?"

**A:** We address this directly. Excluding the March-June 2020 meetings, the results are unchanged: target shock p ≈ 0.02, path shock p ≈ 0.16. The COVID meetings are not driving our results.

This makes sense: the GSS shocks already capture the *surprise* component, so even extreme meetings like March 2020 are informative about the relationship between surprises and language.

---

## Q10: "What about the Jarociński-Karadi (2020) decomposition? They separate information shocks from monetary policy shocks."

**A:** This is an excellent suggestion. JK use a sign-restriction VAR approach to separate two types of shocks: (1) monetary policy shocks that move rates and stock prices in opposite directions, and (2) information shocks that move them in the same direction.

Our GSS decomposition is different — it separates current-rate surprises from future-policy revisions, but both could contain information effects. The JK decomposition would provide a cleaner test of the information channel.

We list this as a priority future extension. The challenge is that JK shocks are available at monthly frequency, while our analysis uses FOMC meeting dates, requiring careful alignment.

---

## Q11: "Why not include control variables like VIX, term spread, or macro surprises?"

**A:** We deliberately keep the baseline specification parsimonious for two reasons:

1. The GSS shocks are identified from high-frequency windows around FOMC announcements, so they should be orthogonal to other macroeconomic information released at different times.

2. Adding controls in a small sample (N = 117) risks overfitting and reduces degrees of freedom.

However, we do examine VIX and term spread in the descriptive statistics and note that the results are robust to including VIX as a control (available in the replication package).

---

## Q12: "The Minutes result (path shock significant) is interesting. Why isn't it the main result?"

**A:** The Minutes result is indeed stronger (R² = 9.35%, both shocks significant). We focus on statements because:

1. Statements are the primary communication tool — released immediately, closely watched by markets.
2. Minutes are released with a 2-3 week delay, so they cannot drive the immediate market reaction we study.
3. The statement result is the cleaner test of the information channel: if language reveals future policy, it should be detectable in the statement that markets trade on immediately.

The Minutes result is an important robustness finding that suggests the information channel may operate on a different timescale.

---

## Q13: "How sensitive are results to the Newey-West lag selection?"

**A:** We test HAC(2), HAC(4), and HAC(6). The target shock remains significant across all lag choices (p ranges from 0.012 to 0.024). The path shock remains insignificant across all choices (p ranges from 0.11 to 0.19).

The key sensitivity is in the *level* of the p-values, not the *significance* classification. HAC(4) is our baseline because it's the most common choice in the monetary policy event-study literature.

We also report White heteroskedasticity-robust standard errors as an alternative. The qualitative conclusion is unchanged.

---

## Q14: "What about the normalization period? Neither shock is significant. Why?"

**A:** During normalization (2016-2019, N = 49), the Fed was raising rates in a predictable manner. The "dot plot" provided clear forward guidance about the expected path. This means:

1. Both target and path surprises were small — the Fed was largely doing what markets expected.
2. Statement language was relatively formulaic during this period.
3. The lack of significance may reflect genuine absence of information effects, or simply insufficient variation in the shocks.

This is consistent with the broader finding that monetary policy communication matters most when policy is uncertain or unconventional.

---

## Q15: "Can you rule out that sentiment is causing the shocks rather than vice versa?"

**A:** This is the reverse causality concern. We argue the direction is from shocks to sentiment because:

1. The GSS shocks are identified from federal funds futures in a narrow window around the FOMC announcement. They capture market surprises that are realized *before* the statement is fully processed.

2. The statement is released simultaneously with the rate decision. The language is pre-drafted and reflects the Committee's deliberations, not a real-time response to market reactions.

3. However, we cannot fully rule out that the Committee anticipates market reactions and adjusts language accordingly. This is an inherent limitation of observational data.

---

## Q16: "How does your paper relate to the broader central bank communication literature?"

**A:** Our paper sits at the intersection of two literatures:

1. **Monetary policy shocks**: GSS 2005a, Nakamura & Steinsson 2018, Bauer & Swanson 2023. We use their identification but apply it to *textual analysis* rather than just asset returns.

2. **Central bank text analysis**: Lucca & Trebbi 2009, Hansen et al. 2018, Hansen & McMahon 2016. We add the shock-based identification that this literature typically lacks.

The key innovation is combining high-frequency shock identification with textual sentiment, which allows us to test whether the information channel operates through language specifically.

---

## Q17: "What would change your conclusions?"

**A:** Three things would strengthen the information channel evidence:

1. **Path shock significance in statements**: If we found path shock significant in statement sentiment (not just Minutes), this would support the revelation interpretation.

2. **Significant FG interaction**: If sentiment mattered more during the zero lower bound, this would support the theory that FG strengthens the language channel.

3. **JK decomposition results**: If information shocks (per Jarociński-Karadi) predict sentiment while monetary shocks don't, this would directly support the information channel.

Conversely, if Bauer-Swanson orthogonalized shocks eliminate the target effect, this would weaken even the implementation interpretation.

---

## Q18: "Fernández-Fuertes (2025) already does this with LLMs and gets R² = 12.4%. What does your paper add?"

**A:** This is an important comparison, and we discuss it explicitly in the paper. Three key distinctions:

1. **Different question**: FF constructs a better shock measure — that's his contribution. We ask what FOMC language itself conveys. His Table 32 (GSS decomposition) is a validation exercise for him; for us, the target-dominant pattern IS the research question, and we explore it through four structured hypotheses, a Wald test, an FG interaction test, and a Statement-versus-Minutes channel comparison — none of which he conducts.

2. **Communication channel differentiation**: We show that path shocks become significant in Minutes (p = 0.015) but not in Statements, with R² jumping from 1.57% to 9.35%. This finding — that different Fed communication outlets serve different informational roles — is absent from FF because he processes all documents through a single pipeline.

3. **Transparency and reproducibility**: Our dictionary-based approach can be replicated by any researcher with the data. FF's multi-agent LLM framework depends on GPT-4 API access, specific prompt chains, and probabilistic extraction — multiple layers of black-box processing. In academic contexts where reproducibility matters, this is not a trivial advantage.

Most importantly, FF's Table 32 is **independent confirmation** of our core finding using a completely different methodology. The fact that both approaches converge on the same target-dominant pattern strengthens rather than weakens our contribution.

---

## Q19: "Shouldn't you just use LLMs instead of dictionaries?"

**A:** We acknowledge this as a limitation and a future direction. But two points:

1. For our research question — testing whether target vs. path shocks predict sentiment — the measurement precision of dictionaries is sufficient. The target shock is significant at p = 0.017 with dictionaries; it would likely be even more significant with LLMs. The null result for path shocks is unlikely to reverse with better sentiment measurement, because FF also finds path ≈ 0 with his LLM approach.

2. Dictionary methods provide a transparent baseline. When we eventually upgrade to LLM sentiment, readers can compare results and assess whether the improvement comes from better measurement or from the LLM introducing its own biases.

---

*Document version: v10.3*
*Date: 2026-06-02*
