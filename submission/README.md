# Submission Package: Beyond the Rate

## Contents

| File | Description |
|------|-------------|
| `paper.pdf` | Main paper (17 pages, PDF) |
| `paper_source.md` | Paper source (Markdown) |
| `cover_letter.pdf` | Cover letter to editor |
| `cover_letter.md` | Cover letter source |
| `computational_details.md` | Technical appendix: formulas, bug fixes, Newey-West lag |
| `regression_results.json` | Full regression output (lag=4) |
| `figures/` | 10 publication-quality figures (PNG, 300 DPI) |
| `code/` | Replication code (5 Python files) |

## Key Results (lag=4, N=117)

- **H1**: Path shock drives sentiment (β=0.000605, t=2.421, p=0.017); target only marginal (p=0.102)
- **H2**: Target shock significant for S&P 500 (p<0.05), NASDAQ (p<0.05), Gold (p<0.05)
- **H3**: R² improves 24× from rate change (0.17%) to GSS shocks (4.12%)
- **H4**: Sentiment×FG interaction marginally significant (NASDAQ p=0.052)
- **Robustness**: Post-2010 p=0.050; No-COVID R²=4.19%; Lag sensitivity: p∈[0.010,0.020] for lag=1–6

## Code Execution

```bash
# Prerequisites
pip install pandas numpy statsmodels scipy

# Run full analysis
python code/run_v6_comprehensive.py
```

## Data Sources

- Monetary policy shocks: Acosta (2022) replication dataset
- CRSP market returns: WRDS institutional access
- FOMC statements: Federal Reserve website (scraped)
- FRED macro series: FRED API (public)

## Version

- Paper: v6 submission (lag=4, all bug fixes applied)
- Last updated: 2026-05-29
- Git commit: fa6a66c
