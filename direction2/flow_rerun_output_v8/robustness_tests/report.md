# Robustness tests: results

## Baseline check (classical SEs)

| Version | MP meetings | CBI meetings | Gov CBI (same) | Risk x CBI (same) |
|---|---|---|---|---|
| As published (2025-26 coded MP) | 91 | 60 | +0.759 (se 0.360, p 0.037) | -0.066 (se 0.032, p 0.036) |
| Fixed (2025-26 dropped) | 83 | 60 | +0.763 (se 0.374, p 0.043) | -0.066 (se 0.032, p 0.041) |

## Test 1. Robust standard errors

| Result | N | Classical | HC1 | Other |
|---|---|---|---|---|
| M2 full: R ~ Target + Path + Tone | 150 | -0.145 (se 0.128, p 0.260) | p 0.248 | hac: p 0.277; wild bootstrap p 0.248 |
| M3 full: + FG + Tone x FG | 150 | -1.133 (se 0.381, p 0.003) | p 0.003 | hac: p 0.005; wild bootstrap p 0.003 |
| FG only: R ~ Target + Path + Tone | 56 | -1.013 (se 0.342, p 0.005) | p 0.010 | hac: p 0.011; wild bootstrap p 0.012 |
| H4 gov bonds, same month: CBI | 146 | +0.763 (se 0.374, p 0.043) | p 0.342 | hac: p 0.341 |
| H5 pooled (same): Risk x CBI | 1022 | -0.066 (se 0.032, p 0.041) | p 0.434 | cluster_month: p 0.466; twoway_month_class: p 0.492 |
| H5 pooled (post): Risk x CBI | 1029 | +0.040 (se 0.030, p 0.181) | p 0.507 | cluster_month: p 0.433; twoway_month_class: p 0.461 |

**Influence (leave one meeting out).** Most influential meeting for gov-bond CBI: 2024-09-18 (beta without it +1.516). For Risk x CBI: 2008-09-16 (beta without it +0.017).

- excluding 2008-09-16: gov CBI +0.029 (se 0.311, p 0.925) (HC1); Risk x CBI +0.017 (se 0.043, p 0.684) (clustered)
- excluding Sep-Dec 2008: gov CBI +0.040 (se 0.317, p 0.901) (HC1); Risk x CBI +0.018 (se 0.043, p 0.685) (clustered)

## Test 3. Beta-based risk

| Class | Beta | Downside beta | Upside beta |
|---|---|---|---|
| government_bonds | 0.00 | 0.01 | 0.00 |
| corporate_bonds | 0.25 | 0.35 | 0.23 |
| real_assets | 0.90 | 1.11 | 0.87 |
| large_cap_equity | 1.00 | 0.98 | 1.00 |
| developed_market_equity | 0.87 | 0.90 | 0.84 |
| emerging_market_equity | 0.95 | 1.07 | 0.93 |
| small_cap_equity | 0.97 | 1.00 | 0.96 |

Order by beta: government_bonds < corporate_bonds < developed_market_equity < real_assets < emerging_market_equity < small_cap_equity < large_cap_equity

**same window** (SE clustered by meeting month)

| Risk measure | N | Risk x CBI | x CBI positive | x CBI negative | pos = neg p |
|---|---|---|---|---|---|
| A. asset-class rank (current) | 1022 | -0.066 (se 0.090, p 0.466) | -0.086 (se 0.089, p 0.332) | -0.056 (se 0.084, p 0.509) | 0.076 |
| A'. asset-class rank, excluding gov bonds | 876 | +0.024 (se 0.022, p 0.286) | | | |
| B. full-sample beta rank | 1022 | -0.060 (se 0.088, p 0.498) | -0.081 (se 0.086, p 0.346) | -0.049 (se 0.082, p 0.551) | 0.065 |
| C. continuous beta | 1022 | -0.448 (se 0.618, p 0.470) | -0.663 (se 0.601, p 0.272) | -0.337 (se 0.548, p 0.540) | 0.012 |
| D. downside beta | 1022 | -0.454 (se 0.625, p 0.468) | -0.659 (se 0.609, p 0.281) | -0.348 (se 0.559, p 0.534) | 0.012 |
| E. rolling 36m beta (lagged; from 2008) | 938 | -0.684 (se 0.559, p 0.223) | -0.712 (se 0.528, p 0.179) | -0.639 (se 0.581, p 0.273) | 0.512 |
| F. rolling 36m beta rank (lagged; from 2008) | 938 | -0.104 (se 0.107, p 0.332) | -0.113 (se 0.107, p 0.291) | -0.098 (se 0.102, p 0.339) | 0.262 |

**post window** (SE clustered by meeting month)

| Risk measure | N | Risk x CBI | x CBI positive | x CBI negative | pos = neg p |
|---|---|---|---|---|---|
| A. asset-class rank (current) | 1029 | +0.040 (se 0.051, p 0.433) | +0.029 (se 0.071, p 0.682) | +0.046 (se 0.040, p 0.251) | 0.626 |
| A'. asset-class rank, excluding gov bonds | 882 | +0.046 (se 0.028, p 0.099) | | | |
| B. full-sample beta rank | 1029 | +0.035 (se 0.054, p 0.520) | +0.017 (se 0.071, p 0.806) | +0.043 (se 0.042, p 0.305) | 0.442 |
| C. continuous beta | 1029 | +0.204 (se 0.395, p 0.607) | +0.076 (se 0.530, p 0.887) | +0.269 (se 0.307, p 0.382) | 0.446 |
| D. downside beta | 1029 | +0.189 (se 0.410, p 0.645) | +0.067 (se 0.541, p 0.901) | +0.252 (se 0.324, p 0.439) | 0.452 |
| E. rolling 36m beta (lagged; from 2008) | 938 | +0.108 (se 0.344, p 0.753) | +0.036 (se 0.414, p 0.931) | +0.178 (se 0.250, p 0.478) | 0.488 |
| F. rolling 36m beta rank (lagged; from 2008) | 938 | +0.024 (se 0.058, p 0.683) | +0.015 (se 0.081, p 0.857) | +0.028 (se 0.045, p 0.533) | 0.737 |

## Test 6. Power and multiple testing

Simple regressions first (HC1), full sample:

- 1. Tone only: Tone -0.166 (se 0.125, p 0.185), R2 0.011, N 150
- 2. + Target, Path: Tone -0.145 (se 0.125, p 0.248), R2 0.064, N 150
- 3. + FG, Tone x FG: Tone +0.051 (se 0.134, p 0.704), R2 0.119, N 150; Tone x FG -1.133 (se 0.374, p 0.003)

FG period only:

- 1. Tone only: Tone -1.119 (se 0.355, p 0.003), R2 0.141, N 56
- 2. + Target, Path: Tone -1.013 (se 0.381, p 0.011), R2 0.328, N 56

| Minimum detectable effect (80% power) | Estimate | SE | MDE |
|---|---|---|---|
| Tone on returns, full sample | -0.145 | 0.125 | 0.350 |
| Tone on returns, FG only | -1.013 | 0.381 | 1.068 |
| CBI on government_bonds flow (same) | +0.763 | 0.801 | 2.243 |
| CBI on corporate_bonds flow (same) | -0.202 | 0.099 | 0.278 |
| CBI on large_cap_equity flow (same) | +0.009 | 0.027 | 0.075 |
| CBI on small_cap_equity flow (same) | +0.028 | 0.065 | 0.181 |

Placebo FG regimes: actual Tone x FG = -1.133. Random 56-meeting sets: p = 0.000. Consecutive 56-meeting blocks: 95, of which 1.1% are at least as negative (most negative starts 2008-12-16, -1.133).

- 42 H4 tests (classical SE): 6 with raw p < .05; 0 survive Holm; 0 survive Benjamini-Hochberg.
- 42 H4 tests (hc1 SE): 2 with raw p < .05; 0 survive Holm; 0 survive Benjamini-Hochberg.

| Class | Window | Shock | Beta | p | Holm | BH |
|---|---|---|---|---|---|---|
| large_cap_equity | same | MP | -0.073 | 0.0248 | 1.000 | 0.610 |
| corporate_bonds | same | CBI | -0.202 | 0.0433 | 1.000 | 0.610 |
| large_cap_equity | diff | CBI | -0.098 | 0.0580 | 1.000 | 0.610 |
| government_bonds | diff | CBI | -0.937 | 0.0641 | 1.000 | 0.610 |
| corporate_bonds | post | CBI | -0.335 | 0.0726 | 1.000 | 0.610 |
| real_assets | same | MP | +0.286 | 0.1147 | 1.000 | 0.691 |
| large_cap_equity | post | CBI | -0.087 | 0.1152 | 1.000 | 0.691 |
| large_cap_equity | post | MP | -0.051 | 0.1622 | 1.000 | 0.767 |
