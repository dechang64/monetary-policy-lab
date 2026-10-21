# 2. Theoretical Framework

Our framework rests on three pillars. Each pillar maps to a measurement discipline, and together they generate the testable implications that structure the paper. We state the theory first because the empirical design — including two deliberate nulls — is derived, not discovered.

## 2.1 Pillar 1 — Rational expectations and the market's inversion rule

Under rational expectations, the announcement-day price response to an FOMC statement is the market's Bayesian posterior update about the policy path and the state of the economy, given everything the market knew before the statement. This has a consequence that drives the whole paper: the *market's* reaction is the sufficient statistic for the news content of the communication. Any analyst-side measure of what the statement "said" — dictionary scores, LLM scores, human readings — is an attempt to reconstruct that news from the text alone, and is informative about the text, not about the market's information set.

The central bank's statement enters this inversion twice. First, through the *decision* channel: the target-rate innovation relative to the market's prior. Second, through the *information* channel: the statement reveals the committee's private assessment of the economy, which is valuable to investors independent of the current decision (Coibion, Gorodnichenko, and Ulig 2022; Jarociński and Karadi 2020). We call the market's extraction of this second component the **central-bank-information (CBI) shock**. CBI is, by construction, the market's inference about the central bank's information set — not a property of the text. This distinction disciplines every measurement choice that follows: a shock variable that correlates with the *text* is not the market's inversion outcome, and the empirical content of Section 7 rests precisely on keeping text-measures and inversion-measures separate.

## 2.2 Pillar 2 — From measurement window to shock decomposition

Kuttner (2001) established the measurement principle: identify policy news from asset-price movements in a narrow window around the announcement, because those prices aggregate the market's information faster than any textual analysis. Bauer and Swanson (2023) sharpened it into an oracle argument — under RE, the *true* policy surprise equals the market's price response up to a risk-premium correction, so any independently measured "surprise" that disagrees with prices is mismeasured, not the market.

Jarociński and Karadi (2020) then showed that the price response itself carries a sign structure that separates two economically distinct events: a conventional monetary shock (MP) and an information shock (CBI) in the Coibion–Gorodnichenko–Ulig sense — the central bank revealing information about the economy rather than about its reaction function. MP and CBI move rates in the same direction but equity prices in opposite directions.

We implement the JK idea with a deliberately transparent inversion rule. Let $\Delta S$ be the same-day CRSP value-weighted return and $\Delta i$ the policy-rate innovation embedded in the surprise. We assign

$$\text{MP}_m = \Delta i_m \cdot \mathbb{1}[\text{sign}(\Delta i_m) \neq \text{sign}(\Delta S_m)], \qquad \text{CBI}_m = \Delta i_m \cdot \mathbb{1}[\text{sign}(\Delta i_m) = \text{sign}(\Delta S_m)].$$

The sign restriction *is* the inversion rule: when the equity market moves against the rate move, the market is pricing a conventional shock; when they move together, the dominant news is information. This construction is coarse relative to full JK high-frequency identification, and Section 8 shows the flow-chapter results are robust to the refinement — but its transparency is a feature: every observation's classification is verifiable by inspection, which matters for a paper whose second half is about measurement discipline.

## 2.3 Pillar 3 — Arbitrage pricing and the risk-compensation channel

If communication shocks are risks, they must be compensated. The arbitrage-pricing view (Ross 1976) predicts that the cross-section of *portfolio* risk premia loads on the shock factors: a factor priced for investors must appear in expected returns, and — in the mutual-fund margin we observe — in reallocation flows as investors traverse the risk ladder (Metcalfe, Bhojraj, and Mozes 2020). Pillar 3 delivers the paper's portfolio predictions: a factor whose realization is observable at the announcement must show up in (i) risk-premium pricing and (ii) fund flows that re-tilt portfolios along the risk ladder.

## 2.4 Testable implications

| # | Implication | Pillar | Where tested |
|---|---|---|---|
| T1 | MP and CBI both move announcement-day prices, with opposite equity signs | P1+P2 | §4 |
| T2 | Statement *text* (tone) carries independent price information beyond shocks, concentrated where forward guidance binds | P1+P2 | §4 |
| T3 | Neither shock moves monthly fund flows once the price channel has cleared | P3 | §5 |
| T4 | Flow responses follow the risk ladder: safe-asset inflows on hawkish information, risky-asset outflows | P3 | §6 |
| T5 | When the communication paradigm shifts, dictionary-based text measures fail in a characteristic way (resolution loss, not bias) | P1 | §7 |

Two implications are deliberately *null* — T3, and the flow-side half of T2. Following the contribution discipline fixed with the advising team: a theory that predicts a null where the heuristic expects an effect is more falsifiable than one that predicts an effect everywhere, and the nulls are load-bearing for the paper's mechanism (prices clear the information first; portfolios adjust later, along the risk ladder, to the *risk* content rather than the *words*).
