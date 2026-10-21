# 5. Flows: The Ruling Null

T3 predicts that neither shock component moves monthly fund flows: prices have already cleared the information by the time a monthly flow window opens. This section tests that prediction — and reports the paper's first deliberate null.

## 5.1 Design

The flow panel covers 1,099 fund-month observations across seven Morningstar asset classes (government bonds, corporate bonds, real assets, large-cap / developed / emerging equity, small-cap equity; 157 meeting months, point-in-time classification, server-side aggregation, flows capped at 50% of TNA, windows sharing no calendar months with the gap rule `<=3` months). The specification follows the flow chapter: class-level net flow rate on the shock pair, fund size, and flow volatility, with fund fixed effects absorbed at the class-month aggregation level.

## 5.2 The null holds everywhere

Adding statement tone to the flow regressions changes nothing (models_v2, October 20 run): tone enters at p = 0.64 raw and p = 0.63 orthogonalized in the same-month window; in the forward-guidance subsample, p = 0.94. The LLM-based score behaves identically (p = 0.17). No statement-text measure — dictionary or model — has flow explanatory power at any horizon we test, while the same measures move announcement-day prices by economically large margins (Section 4). The null is not an artifact of one measure, one window, or one asset class.

We emphasize what this null is *not*. It is not evidence that households ignore the Fed: Section 6 shows the same funds reallocate sharply along the risk ladder in response to the information content of the same announcements. It is evidence about *timing and content*: by the time monthly flows are measured, the words have already been priced, and what remains for flows to respond to is the risk content, not the words.

## 5.3 The shock components' direct flow effects

The shocks themselves are likewise silent in the same-month window: MP enters at p = 0.92 and CBI at p = 0.29 (government bonds; the other six classes are in Appendix Table A3). The only significant flow-level interaction involving MP arrives in the zero-rate regime (Section 6, Table 6.2) — and it is a *risk-taking* effect, not an information effect, consistent with the theory's separation of channels.
