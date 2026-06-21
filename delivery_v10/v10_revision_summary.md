# v10 Revision Summary: Words Beyond the Rate

## Core Repositioning
- **Title**: "Words Beyond the Rate: High-Frequency Monetary Policy Shocks and FOMC Language"
- **Core question**: Does FOMC statement language primarily reflect current policy implementation or informational revelation about future conditions?
- **Framing**: From "discovering information channel" → "testing implementation vs revelation"

## Changes from Roadmap (fomc_paper_revision_roadmap.xlsx)

### ✅ Completed Items

| # | Item | Status |
|---|------|--------|
| 1 | Title → "Words Beyond the Rate: High-Frequency Monetary Policy Shocks and FOMC Language" | ✅ |
| 2 | Core question → implementation vs revelation | ✅ |
| 3 | Contribution → 1 main + 2 supporting | ✅ |
| 4 | Literature: "path shocks, often associated with information-channel implications" | ✅ |
| 5 | Table labels: β_Target and β_Path (already correct in v9) | ✅ |
| 6 | CRSP EW: financing-constraint language (not tech/growth) | ✅ |
| 7 | Units: explicit magnitude interpretation added | ✅ |
| 8 | Coefficient mismatch: verified correct in current version | ✅ N/A |
| 9 | VIX interpretation: verified correct | ✅ N/A |
| 10 | "dominates" → "more empirically relevant" (6 instances) | ✅ |
| 11 | "validate" → "provide evidence consistent with" | ✅ |
| 12 | "marginally significant" for NASDAQ p=0.041 → "statistically significant" | ✅ |
| 13 | White SE p=0.117: not described as marginally significant | ✅ |
| 14 | Chair FE reframed as finding, not robustness failure | ✅ |
| 15 | Placebo: honest about limited diagnostic power | ✅ |
| 16 | Forward guidance: concise, no overinterpretation | ✅ |
| 17 | Conclusion: one central message, no new literature/mechanisms | ✅ |
| 18 | Terminology note added (path shock ≠ information shock) | ✅ |
| 19 | H1/H2 hypotheses made explicit | ✅ |
| 20 | Cross-asset reframed as mechanism validation | ✅ |
| 21 | Weak results positioned as findings, not defects | ✅ |
| 22 | "interpretive contribution" framing | ✅ |
| 23 | Section 5.4 renamed "Policy Implementation vs Informational Revelation" | ✅ |
| 24 | Section 5.5 renamed "Why Is the Target Shock More Empirically Relevant?" | ✅ |

### Not Implemented (requires new data/analysis)

| # | Item | Reason |
|---|------|--------|
| 1 | Conceptual framework figure (FOMC → shocks → sentiment → assets) | Requires design work beyond text revision |
| 2 | JK decomposition implementation | Requires structural estimation (noted as future work) |
| 3 | Bauer-Swanson orthogonalization | Requires pre-FOMC data (noted as future work) |

## Key Empirical Results (unchanged)
- Target shock → sentiment: β = 0.000577, p = 0.017 ✅
- Path shock → sentiment: β = 0.000633, p = 0.152 (not significant)
- Wald test: χ² = 0.015, p = 0.902 (cannot reject equality)
- Chair FE R² = 27.08%, target becomes insignificant
- Rate cut regime: path shock p < 0.001, R² = 43.1%
- Rate changes: p = 0.726 (undetectable)

## Page Count
- Total: 42 pages (PDF)
- Main text: ~35 pages
- References: ~3 pages
- Appendix: ~1 page
- Word count: ~18,800 (including tables/notes)

## Files
- `Words_Beyond_the_Rate_v10.docx` — Word document
- `Words_Beyond_the_Rate_v10.pdf` — PDF (42 pages)
- `Words_Beyond_the_Rate_v10_source.md` — Markdown source
