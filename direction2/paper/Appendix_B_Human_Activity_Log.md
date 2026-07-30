# Appendix B: Human Activity Log

## Project: Direction 2 — Portfolio Rebalancing and Cross-Asset Contagion
## AFA 2027 GenAI Session Submission

---

## Project Timeline

| Date | Activity | Duration (hrs) | Human/AI |
|------|----------|-----------------|----------|
| 2026-07-25 | Project initiation, literature review planning | 1.0 | Human |
| 2026-07-25 | Direction 2 research design, H1-H5 hypothesis formulation | 2.0 | Human |
| 2026-07-25 | WRDS data exploration, CRSP table structure verification | 1.5 | Human |
| 2026-07-26 | WRDS local connection setup (Windows, Python 3.10) | 0.5 | Human |
| 2026-07-26 | Pipeline code review (h1_h4_regression.py) | 1.0 | Human |
| 2026-07-26 | First pipeline run, error log review | 0.5 | Human |
| 2026-07-27 | Error log analysis, feedback to AI | 0.5 | Human |
| 2026-07-27 | Code review after B-S fix, formula audit | 1.0 | Human |
| 2026-07-28 | TNA-weighted aggregation decision | 0.5 | Human |
| 2026-07-28 | Code review, 7-round iteration feedback | 2.0 | Human |
| 2026-07-29 | Formula audit request, regression specification review | 1.0 | Human |
| 2026-07-29 | Event window robustness decision (same/post/diff) | 0.5 | Human |
| 2026-07-29 | LLM sentiment analysis decision, amax router setup | 0.5 | Human |
| 2026-07-29 | LLM sentiment local run (117 FOMC meetings) | 0.2 | Human |
| 2026-07-29 | Triple baseline results review, narrative decision | 1.0 | Human |
| 2026-07-29 | Phase 1 LLM robustness check review | 0.5 | Human |
| 2026-07-29 | Paper structure decision, AFA submission planning | 0.5 | Human |

**Total human time: ~13.2 hours**

## Key Human Decisions

1. **Research design**: H1-H5 hypotheses formulated by human based on portfolio balance theory (Tobin 1969) and information effect literature (Nakamura-Steinsson 2018)

2. **JK decomposition choice**: Human selected Jarociński-Karadi (2020) sign restriction approach over alternative identification strategies

3. **Asset classification**: Human defined 7-asset risk ladder (government bonds → small-cap equity) based on CRSP objective codes

4. **Event window selection**: Human chose 3 windows (same, post, diff) for robustness after initial post-only results

5. **LLM sentiment decision**: Human initiated LLM-based sentiment analysis after discovering LM dictionary limitations

6. **B-S orthogonalization with LLM**: Human decided to use LLM hawkish score as alternative to LM dictionary in B-S orthogonalization

7. **Paper narrative**: Human decided core story is "CBI drives rebalancing, MP doesn't" rather than "MP affects fund flows"

8. **MRS paper decision**: Human decided not to revise MRS paper, cite it in new paper instead

## AI Contributions

- All pipeline code (Python): AI-generated, human-reviewed
- B-S orthogonalization implementation: AI-generated
- LLM sentiment analysis script: AI-generated
- Phase 1 robustness check script: AI-generated
- Literature search: AI-assisted
- Reviewer simulation: AI-generated
- Paper draft: AI-generated, human-edited

## Efficiency Metric

- **Total human time**: ~13.2 hours
- **Total AI time**: ~40+ hours (including 7 debug rounds, 3 window runs, LLM sentiment analysis)
- **Lines of code**: ~2,676 (AI) + ~200 (human edits)
- **Lines of paper**: ~3,000 (AI draft) + human editing
- **Quality per unit of human effort**: High (9-specification robustness, triple baseline, LLM methodology contribution)
