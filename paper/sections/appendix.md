# Appendix A — LLM Audit Protocol

The statement scores used in Sections 4 and 7 pass a five-layer audit ported from the NeuroSync/federated-platform defense (`federated-ai-platform/src/hallucination/mod.rs`): (1) retrieval consistency — the returned justification must be supported by the statement text (word-level coverage and an embedding contrastive test against random statements); (2) vector fact-check — strong-declaration keywords verified word-by-word; (3) CROWN anchor-independence — the scorer is re-queried with the dictionary score shown; a confidence drop above $\delta = 0.10$ flags anchoring; (4) multi-node vote — prompt-variant voting within a model and cross-vendor voting across four to five model families; (5) self-consistency — repeated sampling at elevated temperature. Per-statement verdicts use the DefenseResult categories: Verified / LikelyTrue / Uncertain / LikelyFalse / Hallucination.

```{=latex}
\input{tables/tablea1.tex}
```

```{=latex}
\input{tables/tablea2.tex}
```

Two drafting-stage fixes are part of the audit record. First, the multi-node consensus rule was initially implemented with a floating-point threshold (0.667) that rejects the intended two-thirds majority (2/3 = 0.6667), suppressing nine verdicts into LikelyFalse; after the fix, all nine resolve as supported. Second, the Operation Twist router one-off (Section 7.3) was caught by the cross-vendor layer and corrected. Both corrections are documented in the audit notes rather than silently applied.

# Appendix B — Seven-Class Flow Effects and Codebook

```{=latex}
\input{tables/tablea3.tex}
```

The variable codebook (`paper/codebook.docx`) enforces one name per definition across the paper: shocks are innovations; "surprise" appears only in the GSS-construction context; tone is never sentiment; "information shock" appears only in the Jarociński-Karadi discussion context, quoted. The audit trail for every headline number runs from the codebook through the archival result files — numbers that could not be traced to an archival source were retired during drafting (two instances, documented in Section 8.4).
