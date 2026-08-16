# Appendix A: AI Conversation Log (Summary)

## Project: Direction 2 — Portfolio Rebalancing and Cross-Asset Contagion
## AFA 2027 GenAI Session

---

## Overview

This appendix documents all AI-assisted interactions in this research project, per AFA GenAI Session requirements. The complete conversation log is stored in the audit chain JSONL files with SHA-256 hash verification.

**AI Model Used:** Claude Sonnet 4 (via OpenClaw), GLM-4.6 (via amax-router)
**Total Interaction Rounds:** 15+ (7 debug rounds, 3 analysis rounds, 5 writing rounds)
**Human Initiation:** All rounds initiated by human (Yang Yang)
**AI Autonomy:** Zero — all actions were human-directed

---

## Round 1: Initial Pipeline Review (2026-07-29 08:49)

**Human Input:** ErrorLog from first pipeline run
**AI Action:** Identified three issues: (1) B-S baseline skipped, (2) Panel results not printed, (3) H1 per-asset not significant
**AI Output:** Modified load_phase1_shocks.py (added B-S orthogonalization), h1_h4_regression.py (added Panel summary printing)
**Human Review:** Approved approach, ran modified code

## Round 2: Panel NaN Fix (2026-07-29 09:24)

**Human Input:** ErrorLog showing Panel regression all NaN
**AI Action:** Identified root cause — time FE collinear with date-level mp_shock/cbi_shock. Removed time FE, kept asset FE + clustered SE.
**AI Output:** Modified h1_panel_regression and h3_panel_wald functions
**Human Review:** Approved, ran code

## Round 3: Formula Audit (2026-07-29 09:29)

**Human Input:** "你要不要好好查查公式？怎么每次都有问题？"
**AI Action:** Complete audit of all regression equations. Found: (1) H2 doesn't support B-S baseline, (2) H3 doesn't support B-S baseline, (3) H4 dead code multicollinearity, (4) H3 fallback hardcoded shock names
**AI Output:** Fixed all 5 issues, added use_bs parameter to H2/H3
**Human Review:** Approved, ran code

## Round 4: Panel NaN Fix (Real Data) (2026-07-29 10:17)

**Human Input:** ErrorLog showing Panel still NaN with real data
**AI Action:** Identified root cause — per-asset H1/H3 do NaN cleaning (dropna) but Panel doesn't. Real data has NaN in net_flow_pct for some asset×date cells.
**AI Output:** Added NaN/inf cleaning to h1_panel_regression and h3_panel_wald
**Human Review:** Approved, ran code — Panel results appeared

## Round 5: Results Analysis (2026-07-29 10:59)

**Human Input:** ErrorLog with full results (Panel no longer NaN)
**AI Action:** Analyzed results. Key finding: H2 supported (δ(Risk×CBI)=+0.153 **), H3 supported (χ²=8.67 ***), H5 supported (corp_bonds -0.533 ***)
**AI Output:** Comprehensive results analysis with economic interpretation
**Human Review:** Approved narrative

## Round 6: Event Window Robustness (2026-07-29 11:24)

**Human Input:** "要跑same/diff window做robustness"
**AI Action:** Modified run_pipeline.py to save results with event_window in filename. Generated 3-window comparison.
**AI Output:** 3×3 results matrix showing different windows capture different transmission channels
**Human Review:** Approved interpretation

## Round 7: LLM Sentiment Analysis (2026-07-29 11:46)

**Human Input:** amax router API key and endpoint
**AI Action:** Wrote llm_sentiment_local.py using OpenAI-compatible API with amax-router model
**Human Output:** Ran locally, generated 117 LLM sentiment scores
**Key Finding:** LM dictionary r=0.000 with LLM hawkish. LM cannot distinguish hike (0.012) from cut (0.010). LLM correctly distinguishes (+0.46 vs -0.68).

## Round 8: Triple Baseline Integration (2026-07-29 12:13)

**Human Input:** LLM sentiment results CSV
**AI Action:** Integrated LLM hawkish into B-S orthogonalization. Added bs_llm_mp_shock/bs_llm_cbi_shock columns. Added [5b/6] LLM B-S baseline to run_all_hypotheses.
**AI Output:** Triple baseline: Raw JK + B-S(LM) + B-S(LLM)
**Human Review:** Ran pipeline, all 3 baselines produced results

## Round 9: 3-Window × 3-Baseline Results (2026-07-29 12:24)

**Human Input:** ErrorLog with same/diff/post window results
**AI Action:** Compiled complete 3×3 matrix. Key finding: H3 government_bonds significant in 9/9 specifications.
**AI Output:** Complete robustness analysis with economic interpretation
**Human Review:** Approved as paper cornerstone

## Round 10: Phase 1 LLM Robustness (2026-07-29 12:30)

**Human Input:** "写Phase 1 LLM robustness check脚本"
**AI Action:** Wrote phase1_llm_robustness.py comparing incremental R² of LM vs LLM sentiment for 5 asset returns
**Key Finding:** LM better for CRSP VW in FG (10.62% vs 6.62%), LLM better for Gold in FG (12.93% vs 1.11%). Complementary, not substitutive.

## Round 11: Paper Writing (2026-07-29 12:35)

**Human Input:** "按照AFA2027的要求写论文并准备材料吧"
**AI Action:** Wrote complete paper draft, all appendices, contribution report, audit chain verification
**Human Review:** Pending

---

## AI Tool Usage Summary

| Tool | Purpose | Rounds |
|------|---------|--------|
| Code generation | Pipeline, regression, LLM scripts | 1-4, 7-8, 10 |
| Code review | Formula audit, bug detection | 3-4 |
| Data analysis | Results interpretation, literature search | 5-6, 9-10 |
| Document generation | Paper draft, appendices | 11 |
| LLM API (amax-router) | FOMC sentiment labeling | 7 |

## Human Decision Points

1. Research design (H1-H5, 7-asset ladder, 3 windows)
2. TNA-weighted aggregation method
3. B-S orthogonalization with LLM sentiment
4. Paper narrative ("CBI drives rebalancing, not MP")
5. Not revising MRS paper, citing instead
6. Triple baseline approach
7. Phase 1 LLM robustness check
8. AFA GenAI Session submission target

---

**Note:** The complete JSONL audit chain with SHA-256 hash verification is available in `audit_chain/direction2_chain.jsonl`. The conversation transcript between the human researcher and AI assistant (Claude Sonnet 4 via OpenClaw) is available upon request.
