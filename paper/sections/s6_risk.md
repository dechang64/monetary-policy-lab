# 6. The Risk-Taking Channel: De-Risking on Hawkish Information

T4 predicts flow reallocation along the Metcalfe-style risk ladder in response to the information component of policy news. The finding reverses the paper's original H5 hypothesis — announced in the proposal as risk-*taking* on expansionary surprises — and the reversal is itself informative: what households and institutions respond to is the *direction of new information about the economy*, not the sign of the policy move.

## 6.1 Risk ladder interaction

With each fund class mapped to a 1–7 ordinal risk rank (government bonds = 1 … small-cap equity = 7), the interaction of the risk rank with the CBI shock is negative and significant: **Risk × CBI = −0.066 (p = 0.036, n = 1,099)**, while Risk × MP is a precise zero (+0.005, p = 0.88). The CBI main effect is positive (+0.332, p = 0.019) — the low-risk anchor class *receives* flows on information shocks — so the interaction says: the higher the class sits on the risk ladder, the more it loses on the same information event. One unit of hawkish information news shifts roughly 0.07 pp of monthly flow from the riskiest class toward the safest, per unit of shock. This is **de-risking on hawkish information**, exactly the ordering T4 predicts.

The Wald test cannot reject equality of the two interaction coefficients (p = 0.11); we therefore claim the asymmetry in *sign* (MP flows are risk-neutral; CBI flows are risk-ordered), not in magnitude.

```{=latex}
\input{tables/table61.tex}
```

![Flow dynamics by window](figures/fig3_dynamics.png)

Figure 3: flow response per unit shock by window (same month, post, difference), safe classes in blue, risky classes in red. The government-bond inflow (+0.76, same month) reverses by the difference window (−0.99); the conventional component is flat at every window.

## 6.2 Sign asymmetry: positive news does the work

Splitting CBI by sign sharpens the channel (models_v2): **Risk × CBI(+) = −0.088 (p = 0.020)**; Risk × CBI(−) = −0.055 (p = 0.10). Hawkish information — the market telling the investor that the Fed knows something bad — drives the ladder. Dovish information produces a weak, statistically indistinguishable version of the same slope. The economics: funds do not chase the *policy* on good news; they flee *risk* on bad news. The asymmetry is the portfolio-side footprint of the asymmetry that Badarinath–Frydman-style and homestead asset-pricing work has long documented for prices, here appearing one step later, in quantities.

## 6.3 The zero-rate exception

In the 2020–2022 zero-rate window, the *conventional* component acquires a flow effect: **MP × ZLB = −1.505 (p = 0.005, n = 1,099, R² = 0.050)** — a one-unit MP shock at the ZLB pushes roughly 1.5 pp of monthly flow out of the average class, concentrated in the safe classes that normally absorb the CBI response (CBI × ZLB is a precise zero, p = 0.78). The reading follows the exposition in Section 2: at the ZLB a conventional surprise is an *exit option* event — the committee has no room to offset a hawkish surprise with further cuts, so even small conventional news is portfolio-relevant, and it operates through the rescue-capacity channel rather than the information channel (whose interaction is null precisely when the level channel binds).

```{=latex}
\input{tables/table62.tex}
```

## 6.4 Putting Sections 4–6 together

The three sections trace one mechanism in three acts. *Prices* clear the statement's information on the announcement day — shocks with JK signs, tone with incremental FG-regime information (Section 4). *Monthly flows* ignore the words entirely — the ruling null (Section 5). *Flows* respond only to the risk content of the information, along the ladder, asymmetrically, with a ZLB exception that reveals the conventional channel working through rescue capacity rather than news (Section 6). Prices first; portfolios later; and what reaches portfolios is not the sentence — it is the risk.
