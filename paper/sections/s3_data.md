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
