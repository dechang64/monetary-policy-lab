# 3. Data and Measurement

Our sample spans every FOMC meeting with a priced statement from January 2006 through April 2026 — 161 meetings, covering the pre-crisis tightening, the crisis and zero-lower-bound era, the forward-guidance window, the 2020 pandemic response, the 2022–2023 tightening cycle, and the current balance-sheet run-off. Three measurement blocks feed the paper: the shock decomposition, the statement text measures, and the fund-flow panel.

## 3.1 The shock decomposition

Policy surprises come from the Acosta USMPD family of high-frequency surprise measures (target and path shocks around FOMC announcements), extended through 2026 with USMPD-based construction after July 2022 (`combined_shocks_acosta_usmpd.csv`; 250 meeting events from 1995, of which 161 fall in our 2006+ sample window). We apply the market-inversion split of Section 2.2: a target shock is classified as the conventional component (MP) when its sign disagrees with the same-day CRSP value-weighted market return, and as the information component (CBI) when the signs agree. This one-line rule is the sign-restriction identification in its cleanest form — no auxiliary instruments, no high-frequency futures pricing beyond the construction of the underlying surprise — and it inherits the JK logic while remaining auditable from two columns of data.

Regime indicators mark (i) the forward-guidance era (December 2008 – December 2015; 56 of the 161 meetings), (ii) the 2020–2022 zero-rate window, and (iii) the post-2022 tightening cycle.

## 3.2 Statement text measures

Statements are scraped from the Federal Reserve's press-release pages (2006–2026) and cleaned to the canonical statement body (`data/fomc_statements_clean.json`; median length a few hundred words; the 2004–2005 short-statement era predates our window). Two text measures are used:

- **Dictionary tone.** LM hawkish-minus-dovish word counts per 100 words (`tone100 = sentiment × 100`). Naming discipline: this variable is never called "sentiment" (the codebook reserves that word out of the paper entirely to avoid collision with the investor-sentiment literature).
- **LLM score.** A rebuilt measure that reads the full statement text and returns a −100…+100 stance score with a one-line justification (161/161 statements scored; Section 7 reports its construction and the five-layer audit).

[SYSTEM_NOTE: Content compressed. Read the full version if needed.]on share changes; the flow window must start at the meeting month or later (`gap ≤ 3` calendar months); point-in-time Morningstar classification at the aggregation date; and flows bounded at 50% of TNA to contain outliers. The panel aggregates to 1,099 fund-month observations across seven asset classes (government bonds, corporate bonds, real assets, large-cap equity, developed-market equity, emerging-market equity, small-cap equity) over 157 meeting months.

## 3.4 One name, one definition

The variable codebook (`paper/codebook.docx`) enforces single definitions: shocks are innovations; "surprise" appears only in the GSS-construction context; tone is never sentiment; "information shock" only in JK-discussion context, quoted. Price reactions use one definition (CRSP value-weighted same-day return) unless a table header states otherwise. The audit trail for every headline number runs from the codebook through the archival result files (`price_results_v1.json`, `flow_results_rerun_v8.json`, `models_v2_results.json`) — numbers that could not be traced to an archival source were retired during drafting.

## 3.5 Three episodes

Three meetings anchor the reader's intuition — and, not accidentally, all three are cross-vendor-confirmed members of the Section 7 divergence set.

**Taper tantrum (June 19, 2013).** The statement added one clause — purchases would "likely be appropriate to moderate" — and nothing else changed: target shock +0.14, dictionary tone +2.75, within half a point of the twelve-month median. Markets read a regime change: CRSP VW −1.34%, ten-year futures +0.130 price points (≈13bp, the session the press christened the tantrum). The LLM committee read the statement's *level* as extremely dovish (−85: "massive asset purchases and zero rates") — correct about the level, silent about the change. The episode is our honest exhibit: markets price *changes*, dictionaries price *levels*, and the divergence between the two is not a defect to be averaged away but the object of Section 7's measurement analysis.

**Pandemic QE, four consecutive statements (June–December 2020).** The committee's language settled into a template — "increase holdings at least at the current pace," zero rates, crisis support — and the dictionary scored all four within +3.1 to +3.5, mildly hawkish by word-count arithmetic. The audited LLM committee reads −75 to −85: strongly dovish, on the structure ("at least at the current pace" is a commitment, not a description) rather than the word list. Four statements, one regime, dictionary resolution ≈ 0.4 points of tone; the paradigm-shift set would be invisible without the second measure.

**The 75-basis-point acceleration (June 15, 2022).** The first 75bp move since 1994, with an explicit inflation commitment: target shock +1.03, path shock +1.30 — the largest conventional surprise in the sample. Dictionary tone: +2.10, again indistinguishable from an ordinary meeting. The LLM committee: +75, strongly hawkish. Equities rose +1.50% on the day — certainty about the path, after months of guessing, was itself information.

These three episodes are why Section 7 exists: the dictionary's resolution loss is not a statistical curiosity but the loss of exactly the statements that define the regimes the paper studies.
