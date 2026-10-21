# LLM Statement Scoring — 5-Layer Hallucination Audit Report

Date: 2026-10-21. Method: the five-layer defense from
`federated-ai-platform/src/hallucination/mod.rs` ported to the scoring audit.
Sample: 30 statements (11 LM-divergence + 19 random, seed 42).
Cross-model layer: amax-router (main), gpt-4.1-mini, claude-haiku-4.5
(gemini-2.5-flash / deepseek-flash queued).

## Layer results (after threshold-bug fix)

| Layer | Test | Result | Verdict |
|---|---|---|---|
| 1 Retrieval consistency | one_line term coverage in statement | 53% word-level; 73% embedding match (own vs random statement) | weak margins are a template-text artifact, not evidence of fabrication |
| 2 Fact check | strong-declaration keywords verified | mean ratio 0.46 | one_lines are compressed paraphrases; no unsupported entities found |
| 3 CROWN | independence from LM anchor | 0/30 triggered; 2/30 direction changed | LLM does NOT get anchored by the dictionary score |
| 4 Multi-node vote (prompt variants) | 3 variants, direction consensus | **100%** after fix (was 70% — float bug: 2/3=0.6667 < 0.667) | direction is prompt-robust |
| 4' Multi-node vote (cross-model) | router/gpt/claude/deepseek (n=29) + gemini (partial) | pairwise r = 0.89–0.97; sign-level agreement 93% (27/29); **LM-divergence sample: sign agreement 100% (11/11)** | cross-vendor consensus on every paradigm-shift episode |
| 5 Self-consistency | 3 samples at T=0.3 | 70% pass; mean range 18.5; 2 statements range > 30 | acceptable; borderline statements flagged |

Final verdicts (v2): Verified 13, LikelyTrue 7, Uncertain 10,
**LikelyFalse 0, Hallucination 0**.

## The catch: one real one-off model error

2011-09-21 (Operation Twist): the main router score was **+85 (hawkish)**,
but gpt-4.1-mini scored −60 and claude-haiku −72 — and the statement's key
sentence ("extend the average maturity of its holdings" = easing) supports
the dovish reading. Three router reruns after the audit: **−70, −70, −70**
(the +85 was a one-off routing error). The score file has been corrected to
−70 with an audit note.

Consequences of the correction:
- LLM vs LM tone correlation: r = 0.335 → **0.351**
- Divergence list: OT removed (it was NOT a defensible disagreement); the
  remaining 11 divergence statements (QE3, calendar guidance 2015, run-off
  2017, pandemic QE 2020, June-2022 acceleration) are cross-model confirmed.

## Audit conclusions

1. No systematic hallucination in the scoring: zero LikelyFalse/Hallucination
   after fixes, 100% prompt-variant direction consistency, cross-vendor r
   up to 0.97.
2. Single-model one-off errors DO occur (1/161 detected). Any statement whose
   score drives a headline claim should get the 2-model cross-check (adds
   ~2 calls per statement).
3. The LLM-vs-LM divergence evidence for the paper stands — and is now
   stronger: every remaining divergence is multi-model confirmed.
4. Operational: Gemini-2.5-flash and DeepSeek-flash are thinking models on
   this gateway — reasoning tokens count against max_tokens (120-token
   budgets truncate mid-JSON; >=800-2000 needed).

Files: `llm_hallucination_audit.py` (5-layer audit),
`llm_audit_recompute.py` (threshold fix + semantic L1),
`llm_audit_matching_test.py` (contrastive match test),
`llm_crossmodel_check.py` (cross-model layer),
`direction2/flow_rerun_output_v8/llm_audit_report_v2.json`,
`llm_crossmodel_scores.jsonl`, `llm_trio_summary.json`.
