# Appendix C: Contribution Report

## Human vs. AI Code Contribution

### Summary

| Metric | Human | AI | Total |
|--------|-------|-----|-------|
| Python code lines | ~200 | ~2,676 | ~2,876 |
| Percentage | 7.0% | 93.0% | 100% |
| Markdown/docs lines | ~500 | ~3,500 | ~4,000 |
| Research design decisions | 8 | 0 | 8 |
| Hypothesis formulation | 5 | 0 | 5 |
| Data interpretation decisions | 12 | 0 | 12 |

### File-by-File Breakdown

| File | Human lines | AI lines | AI % | Notes |
|------|-------------|----------|------|-------|
| wrds_connector.py | 10 | 440 | 98% | AI wrote, human reviewed WRDS table structure |
| h1_h4_regression.py | 30 | 620 | 95% | AI wrote, human audited formulas |
| h5_regime_analysis.py | 5 | 215 | 98% | AI wrote |
| h4_substitution_matrix.py | 0 | 275 | 100% | AI wrote |
| load_phase1_shocks.py | 5 | 135 | 96% | AI wrote, human verified JK logic |
| run_pipeline.py | 10 | 150 | 94% | AI wrote, human added --skip-wrds |
| audit_chain.py | 0 | 200 | 100% | AI wrote |
| contribution_tracker.py | 0 | 180 | 100% | AI wrote |
| llm_sentiment_local.py | 0 | 175 | 100% | AI wrote |
| phase1_llm_robustness.py | 0 | 200 | 100% | AI wrote |
| **Total Python** | **~60** | **~2,590** | **98%** | |

### Human Code Contributions

1. **WRDS table structure corrections** (~10 lines): Human identified that `fund_hdr2` table doesn't exist, needs 3-table JOIN (`fund_hdr` + `fund_style` + `fund_summary2`)

2. **TNA-weighted aggregation** (~5 lines): Human decided to use TNA-weighted aggregation instead of simple mean, preventing large/small fund flow cancellation

3. **Event window parameter** (~5 lines): Human added `--event-window` CLI parameter for robustness check

4. **NaN handling feedback** (~10 lines): Human identified that NaN in net_flow_pct was not being cleaned in Panel regression

5. **Formula audit requests** (~30 lines): Human identified multiple specification issues (time FE collinearity, H2/H3 B-S support, H4 multicollinearity)

### AI Code Contributions

1. **Full pipeline architecture**: WRDS connector, flow computation, H1-H5 regression, panel regression, B-S orthogonalization, LLM sentiment analysis, Phase 1 robustness

2. **Debug iterations**: 7 rounds of bug fixing (encoding, inf values, column names, function aliases, HAC covariance, control variable loss, code corruption)

3. **Triple baseline implementation**: Raw JK + B-S(LM) + B-S(LLM) with dynamic shock column references

4. **Audit chain system**: SHA-256 hash chain for tamper-proof AI conversation recording

### Interpretation

The human contribution is concentrated in:
- **Research design** (hypotheses, identification strategy, data selection)
- **Domain expertise** (WRDS table structure, TNA weighting, formula auditing)
- **Decision-making** (event window choice, LLM sentiment, narrative)

The AI contribution is concentrated in:
- **Code implementation** (pipeline, regression, visualization)
- **Debugging** (7 rounds of error fixing)
- **Documentation** (literature review, reviewer simulation, paper draft)

This aligns with the AFA GenAI Session evaluation criterion of "maximal quality per unit of human effort" — the human's 13.2 hours produced a 9-specification robustness analysis with triple baseline and LLM methodology contribution.
