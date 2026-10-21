# 7. Measurement: When the Dictionary Fails

Sections 4–6 used dictionary tone as the text measure. This section asks the question the tone literature usually skips: *how good is the dictionary at its job?* We answer by rebuilding a statement-based LLM score under a five-layer hallucination-audit protocol, and by showing that the dictionary's failures have a characteristic signature — resolution loss exactly when the Fed's communication paradigm shifts.

## 7.1 The rebuilt LLM measure

An earlier draft used an LLM score proxied from meeting metadata; it is retired. The rebuilt measure reads the full statement text and returns a −100…+100 stance score with a one-line justification (161/161 statements). Properties:

- Correlation with dictionary tone: **r = +0.351** — the two measures overlap but are far from redundant.
- Correlation with the CBI shock: **r = +0.027** — as Pillar 1 requires. The inversion outcome must not be a property of the text; if it were, the decomposition of Section 2.2 would be circular.
- No flow contribution (p = 0.17 in the Section 5 specification), mirroring tone — the null extends to every text measure.

## 7.2 Divergence is resolution loss, not bias

Eleven statements show |z(LLM) − z(tone)| > 2. On those statements the dictionary output is nearly constant (std = 1.10, mean +2.77) while the LLM separates directions sharply (std ≈ 71): Operation[SYSTEM_NOTE: Content compressed. Read the full version if needed.]lve the direction from the words at all.

## 7.3 The five-layer audit

LLM measures invite a legitimate objection — hallucination. We therefore ported a five-layer defense (retrieval consistency, vector fact-check, CROWN anchor-independence, multi-node voting, self-consistency) from our earlier agent infrastructure to the scoring task, and audited a 30-statement sample (11 divergences + 19 random, seed 42):

- Retrieval consistency: 53% word-level, 73% embedding-match (own vs random statement); weak margins reflect the template nature of FOMC prose, not fabrication.
- CROWN: showing the auditor the dictionary score does not anchor the LLM (0/30 triggered) — the measures are independent, as claimed.
- Multi-node vote, prompt variants: 100% direction consensus after fixing a float-threshold bug (2/3 = 0.6667 vs a 0.667 threshold had rejected all two-of-three consents — 9 false "LikelyFalse" verdicts that were an audit bug, not an LLM failure).
- Self-consistency (T = 0.3, 3 samples): 70% pass; 2 borderline statements flagged.
- **Cross-vendor layer**: GPT-4.1-mini, Claude-haiku-4.5, DeepSeek-flash, Gemini-2.5-flash (thinking models; reasoning tokens count against `max_tokens` — operational finding). Pairwise r = 0.89–0.97; sign-level agreement 93.1% (27/29); **on the 11 LM-divergence statements, sign agreement across vendor families is 100%**.

The audit caught exactly one real error: on the Operation Twist statement (2011-09-21) the main router returned +85 (hawkish) once; three reruns returned −70 each time, and all four vendor families independently returned strongly dovish scores (−60 to −90). "Extend the average maturity of its holdings" is an easing operation; the correction stands, the score file carries the audit note, and the Section 4 correlations moved r = 0.335 → 0.351.

![Score compression](figures/fig4_compression.png)

Figure 4: dictionary tone (horizontal) against the LLM score (vertical). The dictionary compresses the 161 statements into a narrow band while the LLM score spans the full range; the eleven flagged paradigm-shift statements (diamonds) sit where the dictionary is flattest — four consecutive 2020 QE statements at tone ≈ 3.1–3.5 are scored −75 to −85 by the committee of models.

## 7.4 Positioning

The LLM measure is a **diagnostic of dictionary failure, not a better scorer**. It adds no price or flow predictiveness (Sections 4–5), so we do not ask the reader to switch measures. What it delivers is evidence about the dictionary: the tone measure that dominates the statement-communication literature loses resolution precisely on the episodes the paradigm-shift narrative cares about, and any dictionary-based claim about those episodes should be read alongside this measurement-failure evidence.
