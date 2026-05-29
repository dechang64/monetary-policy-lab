# Q&A Preparation: Anticipated Questions & Responses

---

## Q1: "Your R² is only 1.57%. How do you justify claiming anything with such weak explanatory power?"

**A:** You're right that 1.57% is modest, and we're transparent about this. But three points:

1. R² in event-study regressions with high-frequency shocks is typically low — Gürkaynak et al. 2005a report similar magnitudes for individual asset returns. The shocks capture the *surprise* component, which is by definition small relative to total variation.

2. The 1.57% is a 4× improvement over using naive rate changes (R² = 0.40%), which tells us the market-based identification matters. And using CB-only sentiment doubles it to 3.90%.

3. We don't claim the information channel is the *primary* driver — we say the target shock is a *statistically significant predictor*. The remaining 98% likely reflects the Fed's response to incoming data, institutional inertia, and other factors.

4. We've added a Wald test showing we cannot reject coefficient equality, and we explicitly say the evidence is "suggestive rather than definitive."

---

## Q2: "Why not use FinBERT or LLM-based sentiment instead of a dictionary approach?"

**A:** Three reasons for the baseline, but we now have a clear upgrade path:

1. **Interpretability**: Dictionary methods are transparent — you can trace every sentiment score to specific words. For a paper testing a specific economic mechanism, this matters.

2. **Replicability**: Our dictionary is fully disclosed (591 hawkish, 222 dovish terms in the appendix). Anyone can replicate our scores. LLM-based methods may produce different results with different model versions.

3. **Temporal consistency**: Dictionary methods treat each meeting identically across time. LLMs may implicitly learn regime-specific patterns.

That said, the CB-LM approach from Gambacorta et al. 2024 offers a middle ground: open-weight, domain-specific models that are reproducible AND contextual. We discuss this as the most promising upgrade path in Section 7.3. And Yao and Chai 2025's uncertainty-aware framework could improve signal-to-noise by downweighting ambiguous classifications.

---

## Q3: "H1 says the target shock is significant but the path shock is not. Doesn't this contradict the information channel?"

**A:** It contradicts the *strong* version of the information channel, which predicts that forward guidance language should be primarily driven by the path shock. But the Wald test (p = 0.90) means we cannot definitively rule out equal effects. The evidence is suggestive but not conclusive.

Moreover, our forward-lookingness experiment shows that the path shock doesn't become significant even when we isolate forward-looking language (p = 0.80). This suggests the GSS target/path decomposition and the FOMC text's forward-looking/current-assessment dimensions don't have a simple correspondence — the path shock captures a broad policy stance signal, not specifically forward guidance content.

---

## Q4: "H4 is not robustly significant. Why is the forward guidance interaction insignificant?"

**A:** The CRSP VW interaction is insignificant (p = 0.602). The NASDAQ interaction is marginally significant (p = 0.041), but the coefficient is 202 basis points — economically implausible and likely driven by outliers. However, the regime analysis reveals a crucial nuance: when we split by decision type, the rate cut regime shows the path shock highly significant (p < 0.001, R² = 43.1%). This is the strongest result in the entire paper. The null H4 for the full sample masks this regime-dependent effect — forward guidance language IS responsive to the path shock, but only during easing cycles. Three interpretations for the full-sample null:

1. **Information captured by shocks**: The high-frequency shocks absorb the information content regardless of regime, leaving no residual for the interaction.

2. **Measurement imprecision**: Our dictionary-based sentiment cannot distinguish forward guidance language from other policy-relevant text. Chen et al. 2025 show that even GPT-3.5 fails to identify forward guidance in 135 of 139 statements — our CB dictionary has the same limitation.

3. **Risk premium channel**: Chen et al. 2025 find that forward guidance operates through a risk premium channel (reducing uncertainty → compressing term premia) rather than an expectations channel. These two channels have opposing effects on equity prices and can offset each other in a reduced-form regression. Our dual-equation test doesn't find the risk premium channel at daily frequency, but Chen et al. use 30-minute windows — the channel may be too fast for daily data.

---

## Q5: "How does your work relate to Jarociński and Karadi 2020?"

**A:** JK decompose the FOMC announcement into a monetary policy shock and an information shock using sign restrictions: a monetary policy shock moves rates and stock prices in opposite directions, while an information shock moves them in the same direction.

Our GSS target/path decomposition is based on the maturity structure of futures responses, not sign restrictions. The target shock likely contains both monetary policy and information components. If information shocks are present, they bias our target shock effect upward — positive economic news makes language more hawkish even though the monetary stance is contractionary.

Implementing the JK sign-restriction decomposition would allow us to separately identify these effects. We regard this as a high priority for future work, as discussed in Section 7.2.

---

## Q6: "Your CB-only R² is 3.90% vs. combined 1.57%. Why not just use CB-only?"

**A:** We report both and agree that CB-only is the stronger measure for this specific application. The combined measure dilutes the CB signal because the LM component is always positive for FOMC text — it adds noise without adding signal.

However, we keep the combined measure as our baseline for two reasons: (1) comparability with the existing literature that uses LM-based sentiment, and (2) the combined measure is a more conservative test — if the target shock is significant even with the diluted measure, it's a stronger result.

The CB-only result is important as a robustness check and as evidence that domain-specific dictionaries outperform general-purpose financial dictionaries for central bank communication analysis.

---

## Q7: "Your dual-equation test finds nothing for the risk premium channel. Doesn't this undermine your explanation for H4?"

**A:** It undermines the *strong* version of the risk premium explanation, but not the *weak* version. The strong version says: "the risk premium channel is present and detectable at daily frequency." Our null result rejects this.

The weak version says: "the risk premium channel operates at high frequency and is absorbed into the overnight price adjustment before we can measure it at daily frequency." This is consistent with our null result AND with Chen et al.'s finding using 30-minute windows.

The key insight: daily data may simply be too coarse to separate the expectations and risk premium channels. This is a data limitation, not a theoretical failure.

---

## Q8: "Your forward-lookingness experiment shows that splitting sentiment reduces R². Doesn't this mean topic decomposition is a bad idea?"

**A:** Not necessarily. Our experiment uses a crude rule-based split (sentences containing "expect"/"will" vs. "recent"/"current"), which is far less sophisticated than the LLM-based topic classification in Chen et al. 2025 or the IMF's four-dimensional framework.

The problem isn't topic decomposition per se — it's that FOMC statements are short (typically 300-500 words), and splitting them into sub-components leaves too few words per topic for reliable sentiment scoring. The IMF's analysis works with 75,000 documents; we have 117 short statements.

The solution may be to use LLM-based topic classification that can assign sentiment at the sentence level rather than splitting the document, as Chen et al. do. This preserves the full word count while still capturing topic heterogeneity.

---

## Q9: "Your novelty weighting improves R² by 45%. Why not make this the baseline?"

**A:** The novelty weighting is a promising improvement, but we present it as an extension rather than the baseline for two reasons:

1. **Comparability**: The existing literature uses unweighted OLS. Changing the baseline makes it harder to compare our results with prior work.

2. **Methodological novelty**: The novelty weighting introduces a new methodological choice (how to measure novelty — cosine distance? Jaccard index? Edit distance?) that could affect results. We use cosine distance between TF-IDF vectors of consecutive statements, but this is just one option.

That said, the 45% improvement suggests that not all FOMC statements are equally informative, and future work should consider information-weighted estimation as a standard practice.

---

## Q10: "What about the Bauer-Swanson 2023 critique? Aren't your shocks contaminated?"

**A:** Bauer and Swanson argue that high-frequency monetary policy surprises are not fully exogenous — they may reflect private information that market participants already had before the FOMC announcement, rather than pure policy surprises.

This is a legitimate concern. However, two points:

1. The relative importance of the target factor is likely robust to this critique. Even if the shocks are partly endogenous, the target shock still captures the surprise component of the rate decision.

2. A more rigorous treatment would implement the Bauer-Swanson orthogonalization procedure, which purges the shocks of predictable components using pre-FOMC survey expectations. We acknowledge this as a priority for the next revision.

---

## Q11: "How does your Literature Radar work, and what has it found?"

**A:** The Literature Radar is an automated daily scan of SSRN, arXiv, BIS/IMF/Fed working papers, and top journals. It uses a weighted keyword matrix to score relevance (core terms like "FOMC" get 3× weight, method terms 2×, data terms 1.5×) and classifies papers by impact type.

In its first scan, it found 42 papers, including 2 high-relevance and 11 medium-relevance papers. The most impactful recent finds include: Chen et al. 2025 (PLMIS shocks from GPT-4), Gambacorta et al. 2024 (CB-LMs), the IMF's 2025 four-dimensional analysis, and Yao & Chai 2025 (uncertainty-aware classification). These directly shaped our Discussion section.

---

## Q12: "What's the most important thing you learned from the new literature?"

**A:** That our null H4 result is not a failure — it's a diagnostic. Chen et al.'s risk premium channel, the IMF's topic decomposition, and CB-LM's domain specificity all point to the same conclusion: coarse-grained, dictionary-based sentiment measures cannot capture the multi-dimensional nature of FOMC communication. The null result tells us where the measurement frontier is, not that the phenomenon doesn't exist.

The second insight is that the GSS target/path decomposition doesn't map cleanly onto the forward-looking/current-assessment dimension of FOMC text. Our forward-lookingness experiment showed this directly. This means the information channel hypothesis needs to be reformulated with more precise correspondence between the shock decomposition and the text dimensions.

---

## Q13: "What would you do differently if you started over?"

**A:** Three things:

1. **Use CB-only sentiment from the start.** The LM component adds noise for FOMC text. We wasted statistical power by using the combined measure as our baseline.

2. **Collect high-frequency data.** The risk premium channel is invisible at daily frequency. With 30-minute windows, we could test the dual-channel hypothesis properly.

3. **Use CB-LM embeddings instead of word counts.** Gambacorta et al. show that domain-specific language models outperform dictionaries on stance classification, and they're open-weight and reproducible. The marginal cost of upgrading from dictionaries to CB-LMs is small compared to the gain in measurement precision.
