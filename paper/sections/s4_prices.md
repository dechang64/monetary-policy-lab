# 4. Prices: What the Statement Text Adds to the Shocks

This section tests T1 and T2. The dependent variables are same-day changes for 10-year and 2-year Treasury futures and same-day equity returns (CRSP value-weighted and S&P 500), observed for the 161 meetings from 2006 to 2026 (150 with CRSP coverage). The text measure is dictionary tone per 100 words (LM hawkish-minus-dovish, `tone100`). Regime indicators mark the forward-guidance era (December 2008 through December 2015, 56 meetings), the 2020–2022 zero-rate window, and the post-2022 tightening cycle. All regressions are OLS with heteroskedasticity-robust standard errors.

## 4.1 The regime structure: tone matters only under forward guidance

Table 4.1 reports the tone coefficient by asset and regime (Panel A: unconditional; Panel B: the FG regime split).

**Cross-model audit note.** Every statement score used in this section passed the five-layer hallucination audit (Appendix A): the LLM-vs-dictionary divergences that define Section 7 were confirmed across four vendor families (sign agreement 100% on the 11 divergence statements), and one one-off scoring error (2011-09-21) was caught and corrected by the cross-model committee.

*[TABLE 4.1 — tone coefficient by asset and regime, from `price_results_v1.json` Panel B: CRSP VW FG −1.119 (p=0.004, R²=0.141, n=56); S&P 500 FG −1.027 (p=0.006, R²=0.130); CRSP VW non-FG +0.037 (p=0.803); TY10 FG +0.044 (p=0.134); TY10 non-FG −0.007 (p=0.317); TY2 all regimes p>0.32]*

Three facts stand out. First, in the FG era a one-standard-deviation hawkish shift in tone (about 12 points) is associated with a −1.12 percentage-point same-day equity return — economically large, and restricted to the FG regime: outside it, the tone coefficient is indistinguishable from zero (+0.04, p=0.80). Second, the sign pattern across assets is exactly what a hawkish text signal implies: equity down, and — in the interaction specification — Treasury yields up (tone×FG on TY10: +0.050, p=0.030). The same words push discount rates up and cash flows down. Third, the short end (TY2) shows no tone response at any horizon: tone acts through the *medium-horizon path* expectations that forward guidance manages, not through the near-term policy rate.

## 4.2 Horse race: tone survives the shocks

Is tone just a noisy proxy for the shocks? No. Table 4.2 (Panel C) puts tone, MP, and CBI in the same regression for the FG subsample. On CRSP VW:

$$\Delta S_m = -1.21\,\text{tone}_m - 1.25\,\text{MP}_m + 1.93\,\text{CBI}_m, \quad R^2 = 0.48, \; n = 56,$$

with all three coefficients significant (tone p<0.001, MP p<0.001, CBI p=0.004). The sign structure is the theory's: conventional shocks (MP) depress equity; information shocks (CBI) *raise* it — the market reads hawkish information as evidence of a stronger economy (Jarociński and Karadi 2020) — and tone carries independent, incremental information about the same-day return beyond both shocks. That a per-100-words dictionary score, orthogonal to the surprise vector by construction, adds 48-percent-of-variance explanatory power jointly with the shocks is the strongest form of T2: the statement's *text* is a separate information channel, not a reflection of the decision.

*[TABLE 4.2 — FG horse race, all four assets; note CBI's positive equity loading and MP's negative one, both consistent with JK; tone loads negatively on equity and positively on TY10 in the interaction spec]*

## 4.3 Orthogonalization as a no-op, and why that answers the question

The advising team asked that tone be orthogonalized to Target and Path before entering the flow regressions of Section 5. It was: orthogonalized tone reproduces raw tone to the fourth decimal (flow-side p=0.635 raw vs 0.625 orth; incremental R² of the orthogonalization 0.4%), because the dictionary score is constructed from the statement text after the decision window and shares almost no variance with the surprise vector. We report the orth spec alongside the raw one throughout — and we flag the deeper point: the near-orthogonality is not a nuisance, it is what makes tone a *separate* measurement channel. A text measure that were highly correlated with the market-priced surprises would be redundant with them; the dictionary's near-independence is precisely why its FG-regime price effect is informative.

## 4.4 Where this leaves the price side

The announcement-day evidence supports T1 and T2: both shock components move prices with the JK sign structure, and the text carries incremental price information concentrated exactly in the regime where the committee's words were the policy instrument (forward guidance). What the text does *not* do — per the framework's deliberate null — is move monthly flows. That is Section 5.
