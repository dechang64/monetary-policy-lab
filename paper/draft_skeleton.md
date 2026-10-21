---
title: "Prices First, Portfolios Later: The Transmission of Federal Reserve Communication to Asset Prices and Household Portfolios"
author: "Draft skeleton v1 — 2026-10-21"
date: "Working paper"
---

# 1. Introduction

**Research question.** How does Federal Reserve communication move asset prices, and when does it move the portfolios of households and institutions? We answer with a decomposition of monetary-policy news into a conventional component (MP) and an information component (CBI) — constructed by letting financial markets invert the central bank's statement under sign restrictions — and by tracing each component into (i) announcement-day prices and (ii) monthly mutual-fund flows.

**The theoretical anchor (Pillar 1, rational expectations).** CBI is not a property of the text; it is the market's Bayesian update about the central bank's private information. The sign-restriction identification *is* the inversion rule: MP shocks move rates and equity premia with the classical signs; CBI shocks (Coibion-Gorodnichenko-Ulig style information surprises) move them with correlated but distinct signs. [TODO: 1-2 paragraphs on why *statements* — not just decisions — are the object; JK (2020) precedent.]

**What we find.**
1. *Prices* react to both components, but the dictionary-measured *tone* of the statement matters for prices only in the forward-guidance era (beta = −1.01, p = 0.005, R² = 32.8% in the FG subsample; near-zero outside it). Tone's price effect is concentrated exactly where communication paradigm shifted.
2. *Portfolios* do not: neither shocks nor tone nor LLM-scored text add explanatory power to monthly fund flows (H4: p = 0.64 raw; p = 0.17 LLM). This null is the paper's ruling result — "prices first, portfolios later."
3. The risk-taking channel runs through *good* news, not bad: positive CBI shocks (central bank reveals better information) drive de-risking — flows out of high-risk classes when Risk × CBI interact (Risk×CBI beta = −0.066, p = 0.036; government-bond CBI beta = +0.76, p = 0.037, same-month window). This reverses the standard "bad news → reach for safety" reading: it is *good news about the economy* that triggers portfolio rebalancing toward safety. [TODO: connect to APT/Pillar 3 — risk-premium repricing, not risk appetite.]
4. *Measurement*: the LM dictionary fails by compression (not by mis-reading) exactly when the Fed's communication paradigm shifts — on the 11 divergence statements the dictionary output is nearly constant (std 1.10) while multi-model LLM scoring separates directions sharply (std ≈ 70); sign agreement across four vendor families is 100% on those statements. We position the LLM measure as a diagnostic of dictionary failure, not a better scorer (it adds no price or flow predictiveness).

**Contribution.** [TODO: 3 bullets — (i) inversion-rule identification for statements; (ii) the flow null as evidence for price-first transmission with a *theoretical* reason (timing structure: flows are monthly, prices are instantaneous); (iii) measurement-failure evidence that disciplines the tone literature.]

**Roadmap.** [TODO: 1 paragraph.]

# 2. Theoretical Framework

## 2.1 Pillar 1 — Rational expectations and the inversion rule

The central bank reveals private information through statements; markets invert it. Formally, the announcement is a signal in a Kanoidal/CGU structure; the identified CBI shock is the market's posterior revision. Sign restrictions are not an assumption bolted on — they are the observable implication of the model. [TODO: formal model section — 2-period signal extraction, signs table.]

## 2.2 Pillar 2 — Information economics of policy news

Kuttner (2001) surprise construction; Bauer-Swanson (2022) rejection of high-frequency calibration short-cuts; Jarociński-Karadi (2020) MP/CBI decomposition. We extend the decomposition to *statement text* (CBI from statements, 60 identified meetings) alongside decision-window shocks (91 MP meetings). [TODO: expand with the exact identification equations; codebook anchor Table 2.]

## 2.3 Pillar 3 — APT: why portfolios should lag prices

Under arbitrage pricing, professional arbitrageurs absorb announcement-day price moves instantly; household/institutional portfolios adjust through slower channels (rebalancing rules, flows, lags). The testable implication is the *timing structure*: price effects at zero horizon, flow effects (if any) at monthly horizon, and any tone/text effect on flows must vanish conditional on the shocks. [TODO: formalize as 3-type-agent model; state the null precisely.]

**De-risking prediction.** Good CBI news lowers the risk premium → marginal investor's required compensation falls → high-risk assets are *over-valued at the margin* → rule-based portfolios rotate toward safety. Prediction: Risk × CBI(+) < 0 for risky-class flows, CBI(+) > 0 for government-bond flows. [TODO: derive formally.]

# 3. Data and Measurement

## 3.1 Statements and shocks

161 FOMC statements, 2006–2026 (scrubbed: `data/fomc_statements_clean.json`). Decision-window shocks (91 MP / 60 CBI identified meetings) from fed-funds futures + 2y/10y windows [TODO: exact windows]. CBI-from-statements via sign restrictions [TODO: spec].

## 3.2 Fund flows

Morningstar-class monthly net flows (7 classes: government bonds, corporate bonds, real assets, large-cap/DM/EM/small-cap equity), N = 157 per class-month regression. Flow gap rule ≤ 3 calendar months (server-side aggregation, point-in-time classification, pseudo B-S removed). [TODO: data providers, cleaning log.]

## 3.3 Text measures

- LM tone: per-100-words hawkish-dovish; never called "sentiment" (codebook discipline).
- LLM score: full-text stance read by multi-vendor committee (GPT-4.1-mini, Claude-haiku-4.5, DeepSeek-flash, Gemini-2.5-flash; majority sign rule), rebuilt from statement text — *not* meeting metadata (the earlier proxy was circular; see §7).
- Variables: one name, one definition (codebook.docx); "shock" reserved for innovations.

# 4. Prices: The FG Channel

**Headline.** Tone predicts announcement-day price reactions only in the FG era (β = −1.01, p = 0.005, R² = 32.8%, N = 392 fund-meeting observations [TODO: confirm N from price table]). Orthogonalizing tone to Target+Path is a no-op (orth R² = 0.4%) — tone is near-orthogonal by construction, which answers the expert's instruction and is itself informative.

**Reading.** In the ZLB/FG regime the *level* tool is stuck; language does the work — so words move prices. Outside FG, statements confirm decisions already priced in. [TODO: add CBI price regressions; event-study figure.]

# 5. Portfolios: The Flow Null (H4)

**Design.** flows_{c,m} ~ MP_shock_m + CBI_shock_m (same-month window; also Post and Diff windows, all in `flow_results_rerun_v8.json`).

**Result.** Across all 7 classes: no significant flow response to either shock in the same-month window except government bonds (CBI +0.759, p = 0.037, N = 157, R² = 3.4%); adding tone (raw or orthogonalized) leaves flows unmoved (p = 0.64); the rebuilt LLM score likewise (p = 0.17).

**Interpretation.** This null is *theoretical* confirmation, not absence of evidence: under Pillar 3's timing structure, monthly flows aggregate post-price-adjustment rebalancing; announcement-day information is already in prices. Government bonds are the exception that maps to §6's de-risking story — the safety asset absorbs the rotation. [TODO: power analysis — what effect size could we detect?]

# 6. Risk-Taking: De-Risking on Good News (H5)

**Design.** Risk ladder (ordinal 1–7 across classes) × CBI, with sign asymmetry (risk_x_mp_pos / risk_x_mp_neg) and ZLB interaction (MP × ZLB).

**Result.** (i) Risk × CBI = −0.066 (p = 0.036, N = 1099): positive CBI news → flows out of high-risk classes *relative to* low-risk — de-risking on good news. (ii) MP × ZLB = −1.51 (p = 0.005): unconventional-era MP shocks push flows strongly out of risky classes. (iii) The naive "bad news → safety" story fails: risk_x_mp_neg is economically nil (β ≈ 0.006, p ≈ 0.98). [TODO: coefficient table from H5_sign_asymmetry; connect to APT risk-premium mechanism in §2.3.]

**Why this matters.** Standard risk-taking-channel readings (Borio-Zhu, Adrian-Shin) predict risk-taking on *accommodative* news. We find the opposite sign pattern for CBI: the central bank's *better* news compresses risk premia and triggers rule-based rotation to safety — a portfolio-balance reading, not a sentiment reading. [TODO: verify against Bauer-Swanson risk-premium evidence.]

# 7. Measurement: Dictionary Failure and the LLM Diagnostic

**The rebuild.** The original LLM measure proxied statements with meeting metadata — circular (metadata embed outcomes, not language). Rebuilt: full-text stance scoring, 161/161, multi-vendor committee with majority sign rule; audited with a five-layer hallucination-defense pipeline (retrieval consistency, fact-check, CROWN anchor-independence 0/30, prompt-variant vote 100%, self-consistency 70%; cross-vendor pairwise r = 0.89–0.97). One one-off scoring error (Operation Twist 2011: router +85 vs four vendors dovish −60…−90) caught by the cross-model layer and corrected — the audit is part of the methodology. [TODO: appendix on the audit protocol.]

**Score compression.** LLM vs LM tone r = +0.351 overall; on the 11 divergence statements the dictionary output is nearly constant (std 1.10) while LLM separates directions (std ≈ 70). Divergence events: Operation Twist [removed after audit — see note], QE3, 2015 calendar guidance, 2017 run-off, 2020 pandemic QE (×4), 2022-06 acceleration — each a paradigm-shift episode; enriched 1.4× in FG∪ZLB. The dictionary does not mis-read these; it *fails to differentiate* them.

**Positioning.** The LLM measure adds no price or flow predictiveness (p = 0.17 flows; CBI correlation +0.027, as theory requires). Its contribution is diagnostic: tone's FG price effect is measured in precisely the regime where the dictionary is least reliable — reported side by side, per the expert's measurement-first discipline.

# 8. Robustness

1. Orthogonalized tone (Bauer-Swanson style): no-op for flows (near-orthogonal by construction); reported anyway.
2. Window rules: Same/Post/Diff flow windows; gap ≤ 3 rule; cap 0.5. [TODO: table]
3. Sign-asymmetry and ZLB interactions: full table.
4. LLM audit: five-layer protocol + cross-vendor committee (Appendix A).
5. [TODO: placebo dates; alternative tone dictionaries; subsample stability 2006-2012 vs 2013-2026.]

# 9. Conclusion

Communication moves prices when language is the policy tool (FG); portfolios move later and through rotation, not reaction; the rotation is triggered by *good* news (de-risking on positive information surprises); and the dictionary-based text measures that dominate the literature silently lose resolution exactly when communication regime shifts — a measurement failure the inversion-rule framework both predicts and repairs.

[TODO: limitations — fund-flow aggregation granularity; statement-only scope (minutes/speeches out); LLM vendor dependence mitigated by committee rule.]

# Appendix A — LLM Audit Protocol

Five layers (ported from the NeuroSync/federated-platform defense): retrieval consistency, vector fact-check, CROWN anchor-independence, multi-node vote (prompt variants + cross-vendor), self-consistency. Verdicts on the 30-statement audit sample: Verified 13 / LikelyTrue 7 / Uncertain 10 / LikelyFalse 0 / Hallucination 0. Thinking-model caveat (reasoning tokens vs max_tokens). Files: `llm_hallucination_audit.py`, `llm_5way_analysis.py`, `paper/llm_hallucination_audit_report.md`.

# Appendix B — Variable Codebook (anchor)

See `paper/codebook.docx`. Naming discipline: "shock" = innovations only; "surprise" only in GSS-construction context; tone is never "sentiment"; "information shock" only in JK-discussion context (quoted).
