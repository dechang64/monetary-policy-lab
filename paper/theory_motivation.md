# Theoretical Motivation (Paper Section 2 skeleton)

Per Oct-20 expert feedback: **motivate from theory, not only empirical papers.**
Three pillars, then the risk-ladder foundation, then the fund-flow literature map.
Written 2026-10-20; results cited are from the corrected pipeline (161 meetings;
flows gap<=3mo).

## Pillar 1 — Rational expectations: only surprises matter

Lucas (1972); Sargent and Wallace (1975): agents form expectations using the
policy rule itself. A fully anticipated policy action is already priced in;
**only the innovation** moves prices and allocations. This is the reason every
variable in the study is an innovation, not a level:

- Target / Path shocks: ex post futures-implied rate minus ex ante expectations
  (GSS 2005).
- Tone enters as a statement-level deviation, not the statement itself.
- Flow regressions use meeting-month innovations.

*Implication for the empirical design*: the dependent variables and regressors
are all measured against an expectations benchmark. There is no "interest rate
level" regression anywhere in the paper.

## Pillar 2 — Information economics: the Fed reveals private information with noise

Grossman and Stiglitz (1980): prices only partially aggregate information
(noise keeps it from being fully revealing); Kyle (1985): informed traders move
prices and market makers Bayesian-update. The Fed is the quintessential
informed agent: it observes internal staff forecasts (private information) and
its statement is a **noisy public signal** of that information. Markets
*invert* the signal: any market-moving statement is decomposed, in equilibrium,
into (a) news about the policy rule (action) and (b) news about the state of
the economy (information).

**This is the theoretical foundation of the MP/CBI split** (JK 2020 are the
empirical implementers). Our sign restriction is the inversion rule:

- Statement hawkish **and** prices down => the market read it as an action
  (discount-rate channel) => MP shock.
- Statement hawkish **and** prices up => the market read it as news about the
  economy (information channel) => CBI shock.

91 MP / 60 CBI meetings out of 161 (2006-2026). Under Pillar 2, CBI shocks are
the states where the noise term is small enough that the market successfully
inverts the signal.

**New result feeding this pillar (models_v2, sign asymmetry)**: RiskxCBI is
driven by *positive* CBI shocks (-0.088, p=0.020) and not negative ones
(-0.055, p=0.100). Good news is also *news*: revealing favourable private
information still raises the incentive to trade on it, and the resulting
uncertainty triggers de-risking. A pure risk-channel story would predict the
opposite asymmetry.

## Pillar 3 — APT: why linear specifications

Ross (1976); Huberman (1982): absence of arbitrage makes asset (excess)
returns approximately **linear in factors**, with loadings on
asset characteristics. Our price and flow regressions are linear in the
factors (MP, CBI, Tone) with characteristic-scaled interactions (Risk ladder,
regime dummies). The interaction term RiskxCBI is not an ad-hoc extension: it
is the APT statement that factor loadings are themselves functions of asset
characteristics. (If the expert prefers, a single-index nonparametric check
can be added in the appendix; the baseline stays linear.)

## The risk ladder: what theory orders these seven classes?

The ordinal ranking (1 = government bonds ... 7 = small-cap equity) rests on
three mutually consistent orderings, each with its own literature:

1. **Cash-flow seniority**: government claims are safest, corporate credit is
   junior to them, equity is the residual claimant. Within equity,
   market-capitalisation sorts by aggregate risk (Banz 1981; Fama-French
   1993 size factor).
2. **Market beta / duration**: systematic-factor exposure rises along the same
   axis (CAPM/ICAPM logic; intertemporal hedging demand, Campbell-Viceira).
3. **Behavioral portfolio layers**: Shefrin and Statman (2000) build
   portfolio choice as layers (protection -> core -> aspiration). The ladder
   is the cross-sectional analogue: shocks that change risk appetite move
   money *between layers*, not just within one asset.

Test implication (Shefrin-Statman + uncertainty channel): a shock that raises
perceived uncertainty shifts the portfolio toward the protection layer. That
is exactly RiskxCBI < 0 — the de-risking we estimate (-0.066, p=0.036;
gap-free -0.114, p=0.002). The behavioral layering literature supplies the
"why the ladder moves as a ladder" logic; APT (Pillar 3) supplies the linear
form.

## Fund-flow literature: where we sit

- Warther (1998): aggregate fund flows and returns are jointly determined;
  flows follow returns.
- Edelen (1999): flows measure *investor* (not manager) timing.
- Froot, O'Connell and Seasholes (2001): flows forecast returns in some
  markets; a portfolio-allocation response to information is plausible.
- Kumar and Lee (2006): retail-flow portfolios co-move with returns — flows
  carry information about investor positioning.
- Ben-Rephael, Kandel and Whaley (2010): aggregate investor attention and
  sentiment through flows.
- Fecht and Kellers (2026): fund flows respond to monetary policy shocks
  (closest predecessor).

**Contribution (per Oct-19 feedback, theme 3)**: the paper documents that
*rate news* and *economy news* — two shocks that are contemporaneously
orthogonal by construction — move money in **different directions along the
risk ladder** (MP: no ladder pattern; CBI: de-risking). Eileen's survey found
only one other paper linking FOMC *statements* (not just decisions) to
*allocation*; the tone-add-on to flows we estimate is null (see below), which
sharpens rather than dilutes the shock-based contribution.

## LLM vs dictionary: divergence is score compression, not noise

The rebuilt LLM measure reads the full statement text (161/161 scored,
Oct 21). Headline statistics:

- Correlation with LM tone: r = +0.335 (the old metadata-proxy was r = 0.139;
  the rebuilt one is a genuinely different, statement-based measure).
- Correlation with CBI shock: r = +0.03 (as theory requires — CBI is the
  market's inversion outcome, not a property of the text).
- In the H4 flow regressions the LLM score adds nothing (p = 0.17), mirroring
  tone: statement variables have no monthly-flow contribution.

**The substantive finding is in the disagreements.** 11 statements show
|z(LLM) - z(tone)| > 2, and on those statements the LM dictionary output is
nearly constant (std = 1.10, mean +2.77) while the LLM separates directions
sharply (std = 71): Operation Twist (2011-09), QE3 (2012-09), the 2015
calendar-guidance statement, the 2017 balance-sheet run-off, four consecutive
2020 pandemic QE statements (LM +3.1 to +3.5 = mildly hawkish; LLM -75 to -85
= strongly dovish), and the June 2022 acceleration. **The dictionary does not
mis-read these statements slightly; it fails to differentiate them at all**
(score compression: resolution 1/65 of the LLM's on exactly the statements
where communication paradigm shifted — structural operations and
forward-guidance language). Divergence events are enriched 1.4x in the
FG∪ZLB regimes and each is a textbook paradigm-shift episode.

Positioning (consistent with the expert's contribution discipline): the LLM
comparison is NOT "a better scorer" — it adds no price or flow predictiveness.
Its value is diagnostic: the dictionary loses information precisely when the
Fed's communication paradigm shifts, and the tone price-effect (FG headline)
must therefore be read alongside this measurement-failure evidence.

## Variables: one name, one definition (codebook anchor)

| Name | Symbol | Definition | Source |
|---|---|---|---|
| Target shock | t_t | bp innovation in FF futures around 2:00pm release | CME via WRDS (GSS 2005) |
| Path shock | p_t | bp innovation in ED futures (future-path surprise) | CME via WRDS (GSS 2005) |
| MP shock | m_t | part of t_t where sign(t) != sign(R_t): policy action | JK 2020 logic on CRSP VW |
| CBI shock | c_t | part of t_t where sign(t) == sign(R_t): information | JK 2020 logic on CRSP VW |
| Tone | tau_t | (hawkish - dovish LM words)/total x 100, body-only text | Loughran-McDonald on our corpus |
| Price | R_t | CRSP value-weighted same-day return, % (meeting close vs prior close) | CRSP |
| Flow f(c,w) | f_c,w | class net flow / lagged class TNA, %; w = Same, Post, Diff | CRSP MF, gap<=3mo, cap 0.5 |
| Risk rank | k_c | 1 = gov bonds, 2 = corp bonds, 3 = real assets, 4 = large-cap eq, 5 = developed eq, 6 = emerging eq, 7 = small-cap eq | construction above |
| FG regime | FG_t | Dec 16 2008 - Dec 15 2015 | calendar |
| ZLB regime | ZLB_t | Mar 15 2020 - Jan 26 2022 | calendar |

Rule: every symbol appears with exactly one name in the paper. "Shock" is
reserved for innovations; "surprise" only in the GSS construction context;
tone is never called sentiment.

## Status of queued runs

1. ~~LLM rebuild~~ **DONE (Oct 21)**: 161/161 statements scored from full
   text; see the divergence section above. Results in
   `direction2/flow_rerun_output_v8/llm_statement_scores.jsonl`.
2. Credit-premium interaction (Ben's idea): Risk x CBI x (Baa-Treasury
   spread) — FRED unreachable for two days (read timeouts); `fetch_spread.py`
   in repo, run when the line recovers. Spec: flows ~ MP + CBI + triple
   interaction, spread at the meeting month.
