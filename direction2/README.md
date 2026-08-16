# Direction 2: Portfolio Rebalancing via Two-Shocks Framework

## Overview

This directory contains the complete replication package for Direction 2 of the
Monetary Policy Research Lab, investigating how Monetary Policy (MP) and Central
Bank Information (CBI) shocks differentially affect fund flow rebalancing.

**Paper**: *Two Shocks, Two Flows: MP vs CBI Effects on Fund Rebalancing*
**Target**: AFA 2027 GenAI Session (deadline: August 31, 2026)
**Authors**: Eileen Zhang (Rutgers Business School) & Yang Dongsheng

## Directory Structure

```
direction2/
├── code/               # Python pipeline and analysis code
│   ├── run_pipeline.py           # Main pipeline orchestrator
│   ├── h1_h4_regression.py       # H1-H4 regression implementations
│   ├── h4_substitution_matrix.py # H4 asset substitution analysis
│   ├── h5_regime_analysis.py     # H5 ZLB regime analysis
│   ├── llm_sentiment_analysis.py # LLM hawkish/dovish scoring
│   ├── llm_sentiment_local.py    # Local LLM sentiment (amax-router)
│   ├── load_phase1_shocks.py     # Phase 1 shock data loader
│   ├── wrds_connector.py         # WRDS database connector
│   ├── phase1_llm_robustness.py  # Phase 1 robustness checks
│   ├── audit_chain.py            # SHA-256 audit chain for GenAI compliance
│   ├── contribution_tracker.py   # Human vs AI code contribution tracker
│   ├── inspect_tables.py         # WRDS table inspection utility
│   ├── test_wrds_connection.py   # WRDS connection test
│   └── audit_chain/              # Audit chain data directory
├── results/            # Empirical results
│   ├── h1_h4_results_same_window.json    # Same-window regressions
│   ├── h1_h4_results_post_window.json   # Post-window regressions
│   ├── h1_h4_results_diff_window.json    # Diff-window regressions
│   ├── llm_sentiment_results.csv         # LLM hawkish scores (117 meetings)
│   ├── llm_sentiment_comparison.csv      # LM vs LLM comparison
│   ├── phase1_llm_robustness.csv         # Phase 1 robustness data
│   └── phase1_llm_robustness_summary.txt # Summary
├── paper/              # LaTeX paper and appendices
│   ├── D2_AFA2027.tex
│   ├── D2_AFA2027.pdf
│   ├── D2_AFA2027_complete.pdf
│   ├── D2_AFA2027_complete.md
│   ├── D2_AFA2027_paper.md
│   ├── Appendix_Audit_Chain.md
│   ├── Appendix_Contribution_Tracker.md
│   ├── Appendix_LLM_Sentiment.md
│   └── Appendix_WRDS_Tables.md
├── figures/            # Generated figures (PNG)
│   ├── fig1_lm_vs_llm_sentiment.png
│   ├── fig2_h1_coefficients_3windows.png
│   ├── fig3_h3_wald_test_3windows.png
│   ├── fig4_phase1_llm_robustness.png
│   ├── fig5_h5_regime_zlb.png
│   └── fig6_significance_heatmap.png
└── README.md           # This file
```

## Key Findings

### H1: Fund flows respond to FOMC shocks ✅
- **Diff window**: MP drives immediate risk-off (gov bonds → inflow, corp bonds → outflow)
- **Post window**: CBI drives lagged rebalancing

### H3: MP ≠ CBI effects ✅ (Most robust finding)
- **Government bonds**: Differential effects across all 3 windows × 3 baselines = 9/9 significant
- This is the paper's core contribution

### LLM vs LM Sentiment: r = 0.000
- LM dictionary cannot distinguish rate hikes from rate cuts (both ≈ 0.01)
- LLM hawkish score correctly separates: hikes = +0.46, cuts = -0.68
- This validates the use of LLM-based sentiment in monetary policy research

### H5: ZLB regime effects
- Corporate bonds show amplified CBI effects during Zero Lower Bound period
- Consistent with forward guidance channel

## Three-Window Design

| Window | [t-1, t+1] | [t+2, t+22] | [t-1, t+22] |
|--------|------------|-------------|-------------|
| Name   | Same       | Post        | Diff        |
| Captures| Immediate MP effect | Lagged CBI effect | Total combined effect |

## Triple Baseline Design

1. **Raw JK**: Jarociński-Korstvedt (2020) original decomposition
2. **B-S LM**: Baker-Sentiment with LM dictionary hawkish measure
3. **B-S LLM**: Baker-Sentiment with LLM hawkish measure (this paper's contribution)

## Replication

```bash
# Prerequisites: Python 3.10+, WRDS account, amax-router API access
cd direction2/code/
python run_pipeline.py --event-window diff --skip-wrds  # Use cached data
python run_pipeline.py --event-window same --skip-wrds
python run_pipeline.py --event-window post --skip-wrds
```

## Audit Chain (GenAI Session Compliance)

All AI interactions are logged via SHA-256 hash chain in `code/audit_chain/`.
The contribution tracker reports human vs AI code line ratios as required by
AFA 2027 GenAI Session guidelines.

## Citation

```bibtex
@unpublished{zhang2026twoshocks,
  title={Two Shocks, Two Flows: MP vs CBI Effects on Fund Rebalancing},
  author={Zhang, Eileen and Yang, Dongsheng},
  year={2026},
  note={AFA 2027 GenAI Session Submission}
}
```
