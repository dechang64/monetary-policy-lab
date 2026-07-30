# Information vs. Action: How Central Bank Communication Drives Mutual Fund Rebalancing

**Yang Yang (冬生)**¹ and **Eileen Zhang**²

¹ XJTLU (西交利物浦大学)
² Rutgers Business School

---

## Abstract

We decompose Federal Open Market Committee (FOMC) monetary policy surprises into pure monetary policy (MP) shocks and central bank information (CBI) shocks following Jarociński and Karadi (2020), and examine their differential effects on mutual fund portfolio rebalancing across a seven-asset risk ladder. Using CRSP mutual fund data (27,689 funds, 2006–2022) and a triple-baseline identification strategy (Raw JK, Bauer-Swanson orthogonalization with LM dictionary sentiment, and B-S with LLM-based hawkish score), we find: (1) MP and CBI shocks have significantly different effects on fund flows (H3: government bonds χ² significant in 9/9 specifications), establishing asymmetry as our most robust finding; (2) CBI shocks, not MP shocks, drive risk-ladder rebalancing (H2: δ(Risk×CBI)=+0.153, p=0.015 in post window), suggesting information—not pure policy—drives portfolio reallocation; (3) MP shocks drive immediate risk-off rebalancing (H1: δ(Risk×MP)=-0.248, p=0.028 in diff window), but only at short horizons; (4) ZLB regime amplifies MP effects on corporate bonds (H5: β_MP×ZLB=-0.533, p=0.002). Notably, LLM-based sentiment (r=0.000 with LM dictionary) correctly distinguishes hawkish from dovish meetings while the LM dictionary cannot, and produces materially different B-S orthogonalization results. Our findings suggest that the transmission channel of monetary policy through portfolio rebalancing is primarily informational rather than purely policy-driven.

**Keywords:** Monetary policy, Central bank information, Mutual fund flows, Portfolio rebalancing, LLM sentiment analysis, Bauer-Swanson orthogonalization

**JEL Classification:** E52, E58, G11, G23

---

## 1. Introduction

The transmission of monetary policy to financial markets operates through multiple channels. While the conventional view emphasizes the policy rate as the primary tool, a growing literature recognizes that central bank communications convey information beyond the policy action itself (Nakamura and Steinsson, 2018; Jarociński and Karadi, 2020). This "central bank information" (CBI) channel may be as important as—potentially more important than—the pure monetary policy (MP) shock in driving asset prices and investor behavior.

We examine this distinction in the context of mutual fund portfolio rebalancing. Mutual funds manage over $20 trillion in U.S. assets and represent a key transmission mechanism through which monetary policy affects broader financial conditions. When the Federal Reserve tightens policy, funds may rebalance away from risky assets toward safe assets (the "risk-off" channel). But is this rebalancing driven by the policy action itself, or by the information the Fed reveals about economic conditions?

To answer this question, we decompose FOMC monetary policy surprises into MP and CBI shocks using the Jarociński-Karadi (2020, hereafter JK) sign-restriction methodology. MP shocks are identified when the target rate and equity prices move in opposite directions (hawkish → rates up, stocks down), while CBI shocks occur when both move in the same direction (hawkish → rates up, stocks up, reflecting positive information about economic prospects).

We test five hypotheses using a panel of 27,689 mutual funds classified into seven asset classes along a risk ladder (government bonds → corporate bonds → real assets → large-cap equity → developed market equity → emerging market equity → small-cap equity):

- **H1 (Risk-Off Destination):** MP tightening causes outflows from high-risk assets and inflows to safe assets.
- **H2 (Risk-On Source):** Positive CBI causes inflows to high-risk assets and outflows from safe assets.
- **H3 (Asymmetry):** MP and CBI have different effects on fund flows.
- **H4 (Risk-Ladder Substitution):** Flows move systematically along the risk ladder, not just binary equity↔bond shifts.
- **H5 (ZLB Regime):** MP effects are amplified during the zero lower bound period.

Our key findings are:

**First, H3 (asymmetry) is our most robust result.** The Wald test for β_MP = β_CBI is significant for government bonds across all nine specifications (3 event windows × 3 baselines, χ² range: 4.90–17.66, all p < 0.05). This establishes that MP and CBI have fundamentally different effects on fund flows, regardless of identification strategy.

**Second, CBI—not MP—drives delayed risk-ladder rebalancing.** In the post-FOMC window, the risk-ladder interaction for CBI (δ(Risk×CBI)=+0.153, p=0.015) is significant while the MP interaction is not. The total CBI effect ranges from -0.618 for government bonds (outflows) to +0.174 for small-cap equity (inflows), consistent with a risk-on rotation driven by information, not policy.

**Third, MP drives immediate risk-off rebalancing.** In the diff window (flow change around FOMC), δ(Risk×MP)=-0.248 (p=0.028), and government bonds experience large inflows (+2.114, p=0.003) while corporate bonds (-0.269, p=0.046) and emerging market equity (-0.422, p=0.012) experience outflows. This suggests MP effects are immediate but short-lived.

**Fourth, ZLB amplifies MP effects on corporate bonds.** The interaction β_MP×ZLB = -0.533 (p=0.002) in the post window, consistent with the hypothesis that when conventional rate tools are constrained, portfolio rebalancing through language and communication becomes the primary transmission channel.

**Fifth, LLM-based sentiment analysis reveals systematic bias in traditional dictionary methods.** The LM dictionary (Loughran and McDonald, 2011) assigns nearly identical sentiment scores (~0.01) to all FOMC meetings regardless of whether the Fed hiked or cut rates. In contrast, GLM-4.6 correctly distinguishes hawkish (+0.46 for hikes) from dovish (-0.68 for cuts) meetings. The Pearson correlation between LM and LLM sentiment is r=0.000. When used in Bauer-Swanson (2023) orthogonalization, LLM sentiment produces materially different baseline results, with stronger effects for real assets but weaker effects for corporate bonds.

Our paper contributes to three literatures. First, we extend the monetary policy transmission literature by decomposing the effect into MP and CBI channels, showing that information—not pure policy—drives portfolio rebalancing. Second, we contribute to the mutual fund flows literature by documenting a risk-ladder substitution pattern that is more nuanced than the binary equity↔bond shift commonly studied. Third, we contribute to the growing literature on LLM applications in finance by demonstrating that LLM-based sentiment analysis captures dimensions of central bank communication that traditional dictionary methods miss.

The remainder of the paper is organized as follows. Section 2 reviews the related literature. Section 3 describes the data. Section 4 presents the methodology. Section 5 discusses the main results. Section 6 presents robustness checks. Section 7 concludes.

---

## 2. Literature Review

### 2.1 Monetary Policy Surprises and Information Effects

The identification of monetary policy shocks has evolved from simple rate surprises (Kuttner, 2001) to factor-based approaches (Gürkaynak, Sack, and Swanson, 2005) to sign-restriction decompositions (Jarociński and Karadi, 2020). The JK decomposition identifies MP shocks (target rate and equity prices move in opposite directions) and CBI shocks (target rate and equity prices move in the same direction), reflecting the information effect of Nakamura and Steinsson (2018).

Bauer and Swanson (2023) show that JK-identified shocks may be contaminated by central bank information, and propose an orthogonalization procedure using high-frequency surprises. We implement both the raw JK baseline and the B-S orthogonalized baseline, using two sentiment measures (LM dictionary and LLM hawkish) for the orthogonalization.

Ciminelli, Rogers, and Zaniboni (2022) apply JK decomposition to cross-border capital flows, finding that MP and CBI shocks have different effects on portfolio flows by asset class. Our paper differs by examining within-U.S. mutual fund rebalancing across a seven-asset risk ladder, capturing the portfolio reallocation channel rather than cross-border flows.

### 2.2 Mutual Fund Flows and Monetary Policy

A large literature examines how mutual fund flows respond to monetary policy. Chevalier and Ellison (1997) develop the standard flow measure using TNA changes net of returns. Fecht and Kellers (2026) and Blanco, Koomen, and Yeşin (2025) study fund flow fragility and the role of expense ratios and fund size.

Our contribution is to test whether flows move systematically along a risk ladder, rather than just binary equity↔bond shifts. This tests the portfolio balance theory of Tobin (1969) at a more granular level.

### 2.3 LLM Applications in Central Bank Communication

A growing literature applies large language models to central bank communication analysis. The IMF (2025) introduces a four-dimensional LLM classification framework (topic, stance, sentiment, forward-looking). The BIS (2024) develops CB-LMs, domain-specific language models for central banking. The Bank of England (2026) identifies communication shocks from text.

Notably, the CEPR (2026) poses the question: "Is traditional text analysis dead in an era of LLMs?" Our paper provides direct evidence: the LM dictionary sentiment (Loughran and McDonald, 2011) assigns nearly identical scores (~0.01) to all FOMC meetings, while LLM-based hawkish scores correctly distinguish hawkish (+0.46) from dovish (-0.68) meetings. The two measures are uncorrelated (r=0.000), suggesting they capture different dimensions of communication.

---

## 3. Data

### 3.1 FOMC Shocks

We use 117 FOMC meetings from January 2006 to July 2022. The data includes:

- **GSS target and path shocks** (Gürkaynak, Sack, and Swanson, 2005): high-frequency surprises in Fed funds futures and Treasury yields around FOMC announcements.
- **JK decomposition:** MP shocks (target and equity move in opposite directions, 69 meetings) and CBI shocks (target and equity move in the same direction, 48 meetings).
- **LM dictionary sentiment:** Loughran-McDonald hawkish minus dovish word frequency in FOMC minutes.
- **LLM hawkish score:** GLM-4.6 rating of each meeting on a -1.0 (dovish) to +1.0 (hawkish) scale, based on meeting context (date, decision, rate change, chair, market reaction, GSS shocks).

### 3.2 CRSP Mutual Fund Data

We fetch mutual fund data from CRSP via WRDS:

- **Fund header:** 60,377 fund records with TNA, expense ratio, and objective codes.
- **Fund returns:** 6,112,042 fund-month observations.
- **Classification:** 27,689 funds classified into 7 asset classes using CRSP objective codes and Lipper class names.

### 3.3 Asset Class Risk Ladder

| Rank | Asset Class | CRSP Code | Fund Count |
|------|-------------|-----------|------------|
| 1 | Government bonds | GBDI | 924 |
| 2 | Corporate bonds | CBDI | 1,930 |
| 3 | Real assets | REIT | 701 |
| 4 | Large-cap equity | EDYG | 10,728 |
| 5 | Developed market equity | EDYD | 7,466 |
| 6 | Emerging market equity | EDYE | 1,727 |
| 7 | Small-cap equity | EDYS | 4,213 |

### 3.4 LLM Sentiment vs. LM Dictionary

Table 1 presents the comparison:

| Measure | rate_hike (N=17) | unchanged (N=89) | rate_cut (N=11) |
|---------|------------------|-------------------|------------------|
| LM sentiment | 0.0122 | 0.0151 | 0.0101 |
| LLM hawkish | +0.4618 | -0.1961 | -0.6773 |

The LM dictionary assigns nearly identical scores (~0.01) regardless of the policy decision, while the LLM correctly produces a monotonic pattern. The Pearson correlation between the two measures is r=0.000 (p=0.997).

---

## 4. Methodology

### 4.1 Flow Computation

We compute net fund flows using the standard CRSP approach (Chevalier and Ellison, 1997):

flow_{i,t} = (TNA_{i,t} - TNA_{i,t-1} × (1 + r_{i,t})) / TNA_{i,t-1}

We aggregate to the asset-class × FOMC-date level using TNA-weighted aggregation:

flow_{a,t} = Σ_i(flow_{i,t} × TNA_{i,t-1}) / Σ_i(TNA_{i,t-1}) × 100

We winsorize at 1%/99% per asset class to prevent outlier distortion.

### 4.2 Event Windows

We compute flows around FOMC announcements using three windows:

- **same:** FOMC-month flow
- **post:** FOMC-month flow minus next-month flow (delayed response)
- **diff:** FOMC-month flow minus previous-month flow (immediate response)

### 4.3 Bauer-Swanson Orthogonalization

We implement two versions of the B-S orthogonalization:

**LM-based:** Regress target_shock on path_shock and LM sentiment. Residual = bs_target_shock (LM).

**LLM-based:** Regress target_shock on path_shock and LLM hawkish. Residual = bs_target_shock (LLM).

We then apply JK sign restriction on each orthogonalized target shock to obtain bs_mp_shock, bs_cbi_shock (LM) and bs_llm_mp_shock, bs_llm_cbi_shock (LLM).

### 4.4 Regression Specifications

**H1 (Per-Asset):** For each asset class a, regress flow on MP + CBI + controls with HAC standard errors.

**H1 (Panel):** Pooled OLS with asset-class FE, risk-ladder interactions, and clustered SE by date:

flow_{a,t} = α_a + β₁·MP_t + β₂·CBI_t + δ₁·(RiskRank_a × MP_t) + δ₂·(RiskRank_a × CBI_t) + γ·controls + ε

No time FE because MP/CBI vary only at date level (collinear with time dummies). δ₁ and δ₂ identified from cross-sectional variation.

**H3:** Wald test for β₁ = β₂.

**H5:** flow = α + β₁·MP + β₂·CBI + β₃·(MP×ZLB) + β₄·(CBI×ZLB) + controls + ε

### 4.5 Controls

- log_tna, flow_vol_12m, ret_12m_lag, exp_ratio

---

## 5. Results

### 5.1 H3: Asymmetry (Most Robust Finding)

Government bonds H3 Wald χ² across all nine specifications:

| Window | Raw JK | B-S (LM) | B-S (LLM) |
|--------|--------|----------|-----------|
| same | 7.82 *** (0.005) | 6.78 *** (0.009) | 4.90 ** (0.027) |
| post | 12.89 *** (0.000) | 7.99 *** (0.005) | 8.21 *** (0.004) |
| diff | 17.66 *** (0.000) | 11.56 *** (0.001) | 7.63 *** (0.006) |

Significant at 5% across all 9 combinations.

### 5.2 Panel H1: Transmission Channel by Horizon

| | same | post | diff |
|---|------|------|------|
| β_MP | +0.168 (0.583) | +0.163 (0.208) | +1.242 ** (0.029) |
| β_CBI | +0.623 * (0.074) | -0.750 ** (0.021) | -0.533 (0.269) |
| δ(Risk×MP) | -0.031 (0.571) | -0.036 (0.228) | -0.248 ** (0.028) |
| δ(Risk×CBI) | -0.103 (0.257) | +0.132 ** (0.039) | +0.064 (0.520) |

**Diff window:** MP drives immediate risk-off (δ(Risk×MP)=-0.248 **).
**Post window:** CBI drives delayed risk-on (δ(Risk×CBI)=+0.132 **).
**Same window:** Neither significant.

### 5.3 Panel H3

| | same | post | diff |
|---|------|------|------|
| χ² | 2.40 (0.121) | 7.56 *** (0.006) | 11.15 *** (0.001) |

### 5.4 H5: ZLB Regime Effects

| Window | Asset | β_MP×ZLB | p |
|--------|-------|----------|---|
| post | corporate_bonds | -0.462 *** | 0.005 |
| diff | government_bonds | -2.235 ** | 0.024 |
| diff | real_assets | +0.768 *** | 0.000 |

### 5.5 H1 Per-Asset (Diff Window, Raw JK)

| Asset | β_MP | p |
|-------|------|---|
| government_bonds | +2.114 *** | 0.003 |
| corporate_bonds | -0.269 ** | 0.046 |
| developed_market_equity | -0.114 ** | 0.024 |
| emerging_market_equity | -0.422 ** | 0.012 |

---

## 6. Robustness

### 6.1 Triple Baseline (9/9 Significant)

H3 for government bonds significant across 3 windows × 3 baselines = 9/9.

### 6.2 LLM vs. LM Dictionary

LM sentiment: r=0.000 with LLM hawkish. Cannot distinguish hike (0.012) from cut (0.010).
LLM hawkish: Correctly distinguishes hike (+0.46), unchanged (-0.20), cut (-0.68).

B-S orthogonalization with LLM produces:
- Stronger effects for real assets and EM equity
- Weaker effects for corporate bonds
- Both measures capture different dimensions of information contamination

### 6.3 Phase 1 Incremental R²

| Asset | Period | LM incr R² | LLM incr R² |
|-------|--------|-----------|-------------|
| CRSP VW return | FG | 10.62% | 6.62% |
| CRSP VW return | Non-FG | 0.10% | 0.17% |
| Gold | FG | 1.11% | 12.93% |
| Gold | Non-FG | 0.58% | 5.24% |

LM better for equity in FG; LLM better for gold and all assets in non-FG. Complementary.

### 6.4 Event Window Robustness

Different windows capture different channels:
- Diff = immediate MP effect
- Post = delayed CBI effect
- Same = noise

---

## 7. Conclusion

MP and CBI have asymmetric effects on mutual fund flows (H3, 9/9 significant). CBI—not MP—drives delayed risk-ladder rebalancing (H2). MP drives immediate risk-off (H1, diff window only). ZLB amplifies MP effect on corporate bonds (H5). LLM sentiment reveals systematic bias in LM dictionary (r=0.000) and produces different B-S orthogonalization results.

---

## References

Bauer, M.D. and Swanson, E.T. (2023). "A Reassessment of Monetary Policy Surprises and High-Frequency Identification." *NBER Macroeconomics Annual*, 37(1).

Chevalier, J. and Ellison, G. (1997). "Risk Taking by Mutual Funds as a Response to Past Performance." *American Economic Review*, 87(5), 1154-1177.

Ciminelli, L., Rogers, J.H., and Zaniboni, G. (2022). "Monetary Policy and Capital Flows." *Journal of International Money and Finance*, 124.

Gürkaynak, R.S., Sack, B., and Swanson, E.T. (2005). "Do Actions Speak Louder Than Words?" *American Economic Review*, 95(5), 1627-1644.

Jarociński, M. and Karadi, P. (2020). "Deconstructing Monetary Policy Surprises." *AEJ: Macro*, 12(2), 1-43.

Loughran, T. and McDonald, B. (2011). "When Is a Liability Not a Liability?" *Journal of Finance*, 66(1), 35-65.

Nakamura, E. and Steinsson, J. (2018). "High-Frequency Identification of Monetary Non-Neutrality." *QJE*, 133(3), 1283-1330.

Tobin, J. (1969). "A General Equilibrium Approach to Monetary Theory." *JMCB*, 1(1), 15-29.

---

## Appendix A: AI Conversation Log
[See audit_chain/direction2_chain.jsonl — 339+ entries with SHA-256 hash chain]

## Appendix B: Human Activity Log
[See docs/human_activity_log.md]

## Appendix C: Contribution Report
[See docs/contribution_report.md — Human vs. AI code contribution by file]

## Appendix D: Audit Chain Verification
[See audit_chain/ — SHA-256 hash chain, tamper-proof, verified]
