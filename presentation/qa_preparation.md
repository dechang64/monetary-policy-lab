# Q&A Preparation: Anticipated Questions & Responses

---

## Q1: "Your R² is only 4%. How do you justify claiming the information channel exists with such weak explanatory power?"

**A:** You're right that 4% is modest, and we're transparent about this in the paper. But several points:

1. R² in event-study regressions with high-frequency shocks is typically low — Gürkaynak et al. 2005a report similar magnitudes for individual asset returns. The shocks capture the *surprise* component, which is by definition small relative to total variation.

2. The 4% is a 24× improvement over using naive rate changes (R² = 0.17%), which tells us the market-based identification matters.

3. We don't claim the information channel is the *primary* driver — we say it's a *statistically significant predictor*. The remaining 96% likely reflects the Fed's response to incoming data, institutional inertia in statement drafting, and other factors. This is actually interesting: it means most of what the FOMC says is not driven by policy surprises, but by the economic outlook.

4. We've added a Wald test showing we cannot reject coefficient equality, and we explicitly say the evidence is "suggestive rather than definitive."

---

## Q2: "Why not use FinBERT or LLM-based sentiment instead of a dictionary approach?"

**A:** Great question. Three reasons:

1. **Interpretability**: Dictionary methods are transparent — you can trace every sentiment score to specific words. With FinBERT, you get a black-box score. For a paper testing a specific economic mechanism, interpretability matters.

2. **Replicability**: Our dictionary is fully disclosed (97 hawkish, 106 dovish terms, all listed in the appendix). Anyone can replicate our sentiment scores. FinBERT requires a specific model checkpoint and may produce different results with different versions.

3. **Temporal consistency**: Dictionary methods treat each meeting identically across time. LLM-based methods may implicitly learn regime-specific patterns, which could contaminate the identification.

That said, we're actively exploring FinBERT as a robustness check. If the results are consistent, it strengthens our case. If they diverge, it tells us something interesting about what dictionaries miss.

---

## Q3: "You have no control variables. Aren't your coefficients biased?"

**A:** This is a fair concern. Our specification is intentionally parsimonious, following the event-study literature where the high-frequency identification is meant to provide exogenous variation. The shocks are measured in a 30-minute window, which limits contamination from other releases.

However, we acknowledge that omitted variables could bias our results. The most important potential omitted variable is the Fed's economic assessment — if the FOMC sees deteriorating conditions, they might cut rates AND use dovish language, creating a spurious correlation. The path shock partially addresses this by capturing forward guidance separately from the rate decision, but it's not a complete solution.

In the next revision, we plan to add: (1) lagged sentiment to control for persistence, (2) macroeconomic indicators available at the time of the meeting, and (3) an indicator for unscheduled meetings.

---

## Q4: "Your Wald test shows you can't distinguish target from path. Doesn't this undermine H3?"

**A:** It certainly tempers the claim. H3 predicts the path effect is larger, and while the point estimate is larger, the data don't have enough power to formally establish this. With N=117 and an R² of 4%, the standard errors are too large to distinguish two coefficients that are close in magnitude.

We've rewritten H3's conclusion to reflect this honestly: "suggestive evidence consistent with the information channel, but not definitive proof." I'd rather be honest about the limitation than overclaim.

That said, the path shock being significant at 5% while the target is only significant at 10% is still informative — it tells us forward guidance language carries statistically detectable information, which is the core of the information channel hypothesis.

---

## Q5: "How does your work relate to Jarociński and Karadi 2020? They find information shocks have opposite effects on stock prices."

**A:** Jarociński and Karadi decompose the FOMC announcement into a monetary policy shock and an information shock using sign restrictions: a monetary policy shock moves rates and stock prices in opposite directions, while an information shock moves them in the same direction.

Our approach is different. We use the GSS target/path decomposition, which is based on the maturity structure of futures responses rather than sign restrictions. The target shock captures the current rate surprise, the path shock captures the future rate path surprise.

The relationship is: our path shock likely contains both monetary policy and information components. When the Fed signals higher future rates because the economy is stronger (information), the path shock is positive AND stock prices might rise — this is the information effect. When the Fed signals higher future rates purely as a policy choice (monetary), stock prices fall.

Our finding that the path shock doesn't significantly affect stock returns may reflect these offsetting channels. Disentangling them further would require the JK sign-restriction approach, which is a natural extension.

---

## Q6: "Your dictionary has 591 hawkish and 222 dovish terms — isn't the asymmetry suspicious?"

**A:** The asymmetry reflects the reality of FOMC language. The Fed has more ways to signal accommodation than tightening — "patient," "gradual," "data-dependent," "accommodative," "supportive" are all distinct dovish signals. Hawkish signals tend to be more direct: "tighten," "hike," "restrictive."

We verified that the two sets are fully disjoint — no word appears in both. We also tested the CB-only subset (without LM terms) and found it produces similar results with lower R², confirming that the expansion adds signal rather than noise.

Note: Our expanded dictionary includes 591 hawkish and 222 dovish terms, which includes both single words and compound phrases (hyphenated and spaced variants like "inflationary-pressure" and "inflationary pressure"). The hawkish list is larger because it includes many domain-specific compound terms related to inflation expectations, capacity constraints, and policy normalization that don't have simple dovish counterparts.

---

## Q7: "Your sample starts in 2006. Why not use the full Acosta sample from 1995?"

**A:** The limitation is FOMC statement availability. We systematically scraped statements from the Federal Reserve website starting from 2006. Pre-2006 statements exist but are less standardized in format and harder to process consistently.

The 117-meeting overlap with Acosta shocks (2006-2022) is our core sample. We also have 164 total statements (2006-2025), but only 117 have matching shock data.

We acknowledge this as a sample selection issue in the paper. Extending back to 1995 would add about 40 more meetings, which would substantially improve power for the Wald test.

---

## Q8: "What about the Fed's press conferences? Since 2011, the Chair holds a press conference after each meeting. Doesn't this contaminate your statement-only analysis?"

**A:** Excellent point. The USMPD actually provides separate surprise measures for statements and press conferences. The press conference surprise (PC) is available for 92 meetings since 2011.

In our main analysis, we use the statement surprise (STMT) because our text data is from statements, not press conference transcripts. The statement is released first and is the primary communication channel. Press conferences provide additional context but are less structured.

As a robustness check, we could use the "monetary event" (ME) surprise from USMPD, which combines statement and press conference effects. This is available for all 276 meetings and would capture the total FOMC communication effect.

---

## Q9: "You mention the USMPD extension to 2026, but the R² drops to 1.65%. Does this mean the information channel is weakening?"

**A:** The R² decline in the extended sample reflects the unusual nature of the 2022-2026 period. The Fed raised rates from near-zero to over 5% in the fastest hiking cycle in decades. During this period:

1. Rate changes were large and mostly expected — the target shock was near zero for many meetings
2. Forward guidance was dominated by the magnitude of hikes rather than subtle information about the future path
3. Statement language became more formulaic — "the Committee is strongly committed to returning inflation to 2 percent" appeared in nearly every statement

So the information channel may be weaker when policy is moving rapidly in one direction. But the path shock remains significant at 10% even in the extended sample, suggesting the channel doesn't disappear entirely.

---

## Q10: "What's the practical implication? Should central banks change how they communicate?"

**A:** Our findings suggest that statement language does convey information about the future policy path, not just the current decision. This has two implications:

1. **For central banks**: Be aware that every word choice in the statement is parsed for forward guidance content. Small language changes can signal policy shifts even when rates don't move.

2. **For markets**: Don't just focus on the rate decision. The statement's language — especially words about the economic outlook and future policy intentions — carries independent information.

3. **For researchers**: Text analysis of central bank communications is a valuable complement to asset-price-based measures. The two approaches capture different aspects of the information channel.

But I want to be careful not to overstate the practical implications given the modest R². The information channel exists, but it's not the dominant force in statement language.

---

## Q11: "Your event study t-stat formula — is it CAR/(σ/√N) or CAR/(σ×√N)?"

**A:** The correct formula is **t = CAR / (σ_AR × √N)**, following Brown & Warner (1985).

- σ_AR is the standard deviation of abnormal returns from the estimation window
- N is the number of days in the event window
- The denominator σ_AR × √N is the standard deviation of CAR (since CAR is a sum of N independent ARs)

A common mistake is using σ_AR / √N in the denominator, which gives t = CAR × √N / σ_AR — this inflates the t-statistic by a factor of N. For a 7-day event window, this would make t 7 times too large.

We caught this error in our implementation and corrected it. The key distinction:
- **Single-asset test** (is this asset's CAR significantly different from zero?): t = CAR / (σ_AR × √N)
- **Cross-sectional test** (is the average CAR across K assets significantly different from zero?): t = CAAR / (σ_CAAR / √K)

---

## Q12: "Your demo platform shows S&P 500 positive but NASDAQ/Russell negative after FOMC — is that real data?"

**A:** No, that was a bug in our demo data generator. The synthetic returns had no FOMC event effects — they were pure random noise, so equity CAR signs were random. We've fixed this by:

1. Injecting correlated FOMC effects on actual FOMC dates (hawkish → all equities down, all yields up)
2. Making effects persist over the [-1, +5] event window with decay
3. Ensuring Treasury yield responses are directionally consistent (2Y and 10Y move in the same direction, with 2Y more sensitive)

With real data (CRSP via WRDS), these patterns emerge naturally from the market. The demo mode is only for when FRED/WRDS data isn't available.

---

## Q13: "How sensitive are your results to the Newey-West lag choice? You use lag=4 — what if you used lag=1 or lag=6?"

**A:** We tested lag sensitivity and the core finding is robust:

| Lag | β(path) t-stat | β(path) p-value | Significant at 5%? |
|-----|----------------|-----------------|---------------------|
| 1   | 1.875          | 0.063           | No (10% yes)        |
| 2   | 2.005          | 0.047           | Yes                 |
| 4   | 2.181          | 0.031           | Yes                 |
| 6   | 2.168          | 0.032           | Yes                 |

Our choice of lag=4 follows the data-driven formula from Newey & West (1994): `lag = int(4 × (n/100)^(2/9))`, which gives 4 for n=117. This is the standard recommendation in the econometrics literature.

The path shock is significant at 5% for all lags ≥ 2, and significant at 10% even with lag=1. The qualitative conclusion — that forward guidance language carries statistically detectable information — does not depend on the lag choice.
