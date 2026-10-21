# 8. Robustness

## 8.1 Orthogonalized tone

The expert-facing concern — tone may proxy the shocks — is addressed by orthogonalizing tone to Target+Path before entering it in every specification. The orthogonalization is a near no-op (first-stage R² = 0.4%; raw vs orth tone coefficients in the flow regressions: +0.0231 vs +0.0239), and the price-side horse race (Section 4.3) shows tone surviving *jointly* with both shocks, so the FG-regime text effect is not a shock in disguise.

## 8.2 Flow windows: the null and its dynamics

Table 8.1 splits the flow response by window. Same-month government-bond flows carry the de-risking inflow (CBI +0.759, p = 0.037); the *difference* window (post minus same) reverses it (CBI −0.988, p = 0.014). MP is a precise zero in all three windows (p ≥ 0.76). Two readings, both consistent with the mechanism: households receive the information shock into safe assets first, then rebalance — a reversal pattern, not a drift; and the conventional component never moves flows at any horizon.

```{=latex}
\input{tables/table81.tex}
```

## 8.3 Cross-measure nulls

The flow null is not tied to one text measure: dictionary tone (p = 0.64), its orthogonalized version (p = 0.63), the FG-regime spec (p = 0.94), and the audited LLM score (p = 0.17) all fail to enter. The price-side null outside FG likewise holds for every asset in Panel B of Table 4.1.

## 8.4 Pipeline audit

Every headline number runs through the archival pipeline: the shock file (single CSV, versioned), the dataset builder with sanity checks (statement length, tone–length correlation, CRSP same-day cross-checks), and per-chapter result JSONs committed to the repository. Statement scores carry a five-layer audit (Appendix A) with per-statement verdicts; the two drafting-stage corrections (Operation Twist's router one-off; retirement of two untraceable recalled numbers) are documented in the audit notes rather than silently fixed.
