#!/usr/bin/env python3
"""
JK Decomposition & Bauer-Swanson Orthogonalization
for Words Beyond the Rate v10.3

1. Jarociński-Karadi (2020): Decompose MP shocks into
   "information shocks" vs "pure monetary policy shocks"
   using sign restrictions on stock-return and interest-rate responses.

2. Bauer-Swanson (2023): Orthogonalize HF surprises w.r.t.
   pre-FOMC macro information to address predictability critique.

Both extensions test robustness of our core finding:
target shock significant, path shock not.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──
df = pd.read_csv('results/minutes_sentiment_corrected.csv')
df['date'] = pd.to_datetime(df['date'])
N = len(df)
print(f"Sample: {N} FOMC meetings, {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")

results = {}

# ============================================================
# PART 1: JK-Style Decomposition (Simplified Sign Restriction)
# ============================================================
# Jarociński & Karadi (2020) identify two shocks:
#   - Monetary policy (MP) shock: rate↑ + stock↓ (contractionary)
#   - Information (CBI) shock: rate↑ + stock↑ (central bank info)
#
# We implement a simplified version using the sign restriction
# on the co-movement of target shock and equity returns.
# If target shock > 0 (hawkish) AND equity return < 0 → MP shock
# If target shock > 0 (hawkish) AND equity return ≥ 0 → CBI shock
# ============================================================

print("\n" + "="*70)
print("PART 1: JK-Style Sign Restriction Decomposition")
print("="*70)

# Use CRSP VW return as the equity signal
df['vw_ret_pct'] = df['vwretd_day'] * 100  # convert to percentage

# Classify each meeting's target shock
df['target_sign'] = np.sign(df['target_shock'])
df['equity_sign'] = np.sign(df['vw_ret_pct'])

# MP shock indicator: target and equity move in opposite directions
# (hawkish shock → rates up, stocks down; or dovish → rates down, stocks up)
df['is_mp_shock'] = (df['target_sign'] * df['equity_sign'] < 0).astype(int)

# CBI shock indicator: target and equity move in same direction
# (hawkish shock → rates up, stocks up = information revelation)
df['is_cbi_shock'] = (df['target_sign'] * df['equity_sign'] > 0).astype(int)

# Neutral: either shock or return is zero
df['is_neutral'] = ((df['target_sign'] == 0) | (df['equity_sign'] == 0)).astype(int)

n_mp = df['is_mp_shock'].sum()
n_cbi = df['is_cbi_shock'].sum()
n_neutral = df['is_neutral'].sum()

print(f"\nClassification of target shocks:")
print(f"  MP shocks (opposite signs):  {n_mp} ({n_mp/N*100:.1f}%)")
print(f"  CBI shocks (same signs):     {n_cbi} ({n_cbi/N*100:.1f}%)")
print(f"  Neutral (zero shock/return): {n_neutral} ({n_neutral/N*100:.1f}%)")

# Construct MP-only and CBI-only target shock series
df['target_mp'] = df['target_shock'] * df['is_mp_shock']
df['target_cbi'] = df['target_shock'] * df['is_cbi_shock']

# ── H1 with decomposed shocks ──
print("\n--- H1: Sentiment ~ Decomposed Target Shocks ---")

# Baseline (original)
X_base = sm.add_constant(df[['target_shock', 'path_shock']].dropna())
y = df.loc[X_base.index, 'sentiment']
model_base = sm.OLS(y, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"\nBaseline: β_T={model_base.params['target_shock']:.6f} (p={model_base.pvalues['target_shock']:.3f}), "
      f"β_P={model_base.params['path_shock']:.6f} (p={model_base.pvalues['path_shock']:.3f}), "
      f"R²={model_base.rsquared*100:.2f}%")

# With decomposed target shocks
X_decomp = sm.add_constant(df[['target_mp', 'target_cbi', 'path_shock']].dropna())
y_d = df.loc[X_decomp.index, 'sentiment']
model_decomp = sm.OLS(y_d, X_decomp).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"Decomposed: β_MP={model_decomp.params['target_mp']:.6f} (p={model_decomp.pvalues['target_mp']:.3f}), "
      f"β_CBI={model_decomp.params['target_cbi']:.6f} (p={model_decomp.pvalues['target_cbi']:.3f}), "
      f"β_P={model_decomp.params['path_shock']:.6f} (p={model_decomp.pvalues['path_shock']:.3f}), "
      f"R²={model_decomp.rsquared*100:.2f}%")

# F-test: are MP and CBI coefficients equal?
r_matrix = np.array([[0, 1, -1, 0]])  # target_mp = target_cbi
ftest = model_decomp.f_test(r_matrix)
print(f"F-test (β_MP = β_CBI): F={float(ftest.fvalue):.3f}, p={ftest.pvalue:.3f}")

results['jk_decomposition'] = {
    'n_mp_shocks': int(n_mp),
    'n_cbi_shocks': int(n_cbi),
    'n_neutral': int(n_neutral),
    'baseline': {
        'beta_target': float(model_base.params['target_shock']),
        'p_target': float(model_base.pvalues['target_shock']),
        'beta_path': float(model_base.params['path_shock']),
        'p_path': float(model_base.pvalues['path_shock']),
        'r_squared': float(model_base.rsquared * 100)
    },
    'decomposed': {
        'beta_mp': float(model_decomp.params['target_mp']),
        'p_mp': float(model_decomp.pvalues['target_mp']),
        'beta_cbi': float(model_decomp.params['target_cbi']),
        'p_cbi': float(model_decomp.pvalues['target_cbi']),
        'beta_path': float(model_decomp.params['path_shock']),
        'p_path': float(model_decomp.pvalues['path_shock']),
        'r_squared': float(model_decomp.rsquared * 100)
    },
    'ftest_equal': {
        'f_stat': float(ftest.fvalue),
        'p_value': float(ftest.pvalue)
    }
}

# ── H2 with decomposed shocks (CRSP VW) ──
print("\n--- H2: CRSP VW Returns ~ Decomposed Target Shocks ---")
df_vw = df.dropna(subset=['vwretd_day', 'target_mp', 'target_cbi', 'path_shock'])
X_vw = sm.add_constant(df_vw[['target_mp', 'target_cbi', 'path_shock']])
y_vw = df_vw['vwretd_day'] * 100
model_vw = sm.OLS(y_vw, X_vw).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"β_MP={model_vw.params['target_mp']:.3f} (p={model_vw.pvalues['target_mp']:.3f}), "
      f"β_CBI={model_vw.params['target_cbi']:.3f} (p={model_vw.pvalues['target_cbi']:.3f}), "
      f"β_P={model_vw.params['path_shock']:.3f} (p={model_vw.pvalues['path_shock']:.3f}), "
      f"R²={model_vw.rsquared*100:.2f}%")

results['jk_h2_vw'] = {
    'beta_mp': float(model_vw.params['target_mp']),
    'p_mp': float(model_vw.pvalues['target_mp']),
    'beta_cbi': float(model_vw.params['target_cbi']),
    'p_cbi': float(model_vw.pvalues['target_cbi']),
    'beta_path': float(model_vw.params['path_shock']),
    'p_path': float(model_vw.pvalues['path_shock']),
    'r_squared': float(model_vw.rsquared * 100)
}

# ============================================================
# PART 2: Bauer-Swanson (2023) Orthogonalization
# ============================================================
# B&S argue HF surprises are predictable from pre-FOMC macro info.
# We orthogonalize target and path shocks w.r.t. available
# pre-meeting information, then re-run H1.
# ============================================================

print("\n" + "="*70)
print("PART 2: Bauer-Swanson Orthogonalization")
print("="*70)

# Pre-FOMC information set (available in our data):
# - VIX (volatility, proxy for uncertainty)
# - term_spread (yield curve shape)
# - lagged sentiment (persistence)
# - lagged target/path shocks (autocorrelation)

df['lag_sentiment'] = df['sentiment'].shift(1)
df['lag_target'] = df['target_shock'].shift(1)
df['lag_path'] = df['path_shock'].shift(1)
df['lag_vwret'] = df['vwretd_day'].shift(1)

# Drop first row with NaN lags
df_bs = df.dropna(subset=['lag_sentiment', 'lag_target', 'lag_path', 'lag_vwret',
                           'vix', 'term_spread']).copy()

print(f"B-S sample: {len(df_bs)} meetings (after lag adjustment)")

# Step 1: Project target shock on pre-FOMC info
X_pred = sm.add_constant(df_bs[['vix', 'term_spread', 'lag_sentiment', 
                                 'lag_target', 'lag_path', 'lag_vwret']])
y_target = df_bs['target_shock']
model_pred_t = sm.OLS(y_target, X_pred).fit()
df_bs['target_orth'] = model_pred_t.resid  # orthogonalized target

y_path = df_bs['path_shock']
model_pred_p = sm.OLS(y_path, X_pred).fit()
df_bs['path_orth'] = model_pred_p.resid  # orthogonalized path

print(f"\nPredictability of target shock: R²={model_pred_t.rsquared*100:.2f}%")
print(f"Predictability of path shock:   R²={model_pred_p.rsquared*100:.2f}%")

# Step 2: Re-run H1 with orthogonalized shocks
print("\n--- H1: Sentiment ~ Orthogonalized Shocks ---")

# Original (non-orthogonalized) for comparison
X_orig = sm.add_constant(df_bs[['target_shock', 'path_shock']])
y_orig = df_bs['sentiment']
model_orig = sm.OLS(y_orig, X_orig).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"Original:    β_T={model_orig.params['target_shock']:.6f} (p={model_orig.pvalues['target_shock']:.3f}), "
      f"β_P={model_orig.params['path_shock']:.6f} (p={model_orig.pvalues['path_shock']:.3f}), "
      f"R²={model_orig.rsquared*100:.2f}%")

# Orthogonalized
X_orth = sm.add_constant(df_bs[['target_orth', 'path_orth']])
y_orth = df_bs['sentiment']
model_orth = sm.OLS(y_orth, X_orth).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"Orthogonal:  β_T={model_orth.params['target_orth']:.6f} (p={model_orth.pvalues['target_orth']:.3f}), "
      f"β_P={model_orth.params['path_orth']:.6f} (p={model_orth.pvalues['path_orth']:.3f}), "
      f"R²={model_orth.rsquared*100:.2f}%")

# Step 3: Re-run H2 with orthogonalized shocks
print("\n--- H2: CRSP VW Returns ~ Orthogonalized Shocks ---")
X_orth_vw = sm.add_constant(df_bs[['target_orth', 'path_orth']])
y_orth_vw = df_bs['vwretd_day'] * 100
model_orth_vw = sm.OLS(y_orth_vw, X_orth_vw).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"Orthogonal:  β_T={model_orth_vw.params['target_orth']:.3f} (p={model_orth_vw.pvalues['target_orth']:.3f}), "
      f"β_P={model_orth_vw.params['path_orth']:.3f} (p={model_orth_vw.pvalues['path_orth']:.3f}), "
      f"R²={model_orth_vw.rsquared*100:.2f}%")

results['bauer_swanson'] = {
    'predictability_target_r2': float(model_pred_t.rsquared * 100),
    'predictability_path_r2': float(model_pred_p.rsquared * 100),
    'original_h1': {
        'beta_target': float(model_orig.params['target_shock']),
        'p_target': float(model_orig.pvalues['target_shock']),
        'beta_path': float(model_orig.params['path_shock']),
        'p_path': float(model_orig.pvalues['path_shock']),
        'r_squared': float(model_orig.rsquared * 100)
    },
    'orthogonalized_h1': {
        'beta_target': float(model_orth.params['target_orth']),
        'p_target': float(model_orth.pvalues['target_orth']),
        'beta_path': float(model_orth.params['path_orth']),
        'p_path': float(model_orth.pvalues['path_orth']),
        'r_squared': float(model_orth.rsquared * 100)
    },
    'orthogonalized_h2_vw': {
        'beta_target': float(model_orth_vw.params['target_orth']),
        'p_target': float(model_orth_vw.pvalues['target_orth']),
        'beta_path': float(model_orth_vw.params['path_orth']),
        'p_path': float(model_orth_vw.pvalues['path_orth']),
        'r_squared': float(model_orth_vw.rsquared * 100)
    }
}

# ============================================================
# PART 3: Combined JK + B-S Analysis
# ============================================================
print("\n" + "="*70)
print("PART 3: Combined — JK Decomposition with B-S Orthogonalized Shocks")
print("="*70)

# Orthogonalize the decomposed shocks
df_bs['target_mp_orth'] = df_bs['target_mp'] - sm.OLS(df_bs['target_mp'], X_pred).fit().predict(X_pred)
df_bs['target_cbi_orth'] = df_bs['target_cbi'] - sm.OLS(df_bs['target_cbi'], X_pred).fit().predict(X_pred)

X_comb = sm.add_constant(df_bs[['target_mp_orth', 'target_cbi_orth', 'path_orth']])
y_comb = df_bs['sentiment']
model_comb = sm.OLS(y_comb, X_comb).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"β_MP_orth={model_comb.params['target_mp_orth']:.6f} (p={model_comb.pvalues['target_mp_orth']:.3f}), "
      f"β_CBI_orth={model_comb.params['target_cbi_orth']:.6f} (p={model_comb.pvalues['target_cbi_orth']:.3f}), "
      f"β_P_orth={model_comb.params['path_orth']:.6f} (p={model_comb.pvalues['path_orth']:.3f}), "
      f"R²={model_comb.rsquared*100:.2f}%")

results['combined_jk_bs'] = {
    'beta_mp_orth': float(model_comb.params['target_mp_orth']),
    'p_mp_orth': float(model_comb.pvalues['target_mp_orth']),
    'beta_cbi_orth': float(model_comb.params['target_cbi_orth']),
    'p_cbi_orth': float(model_comb.pvalues['target_cbi_orth']),
    'beta_path_orth': float(model_comb.params['path_orth']),
    'p_path_orth': float(model_comb.pvalues['path_orth']),
    'r_squared': float(model_comb.rsquared * 100)
}

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n1. JK Decomposition:")
print(f"   MP shocks: {n_mp} meetings ({n_mp/N*100:.1f}%)")
print(f"   CBI shocks: {n_cbi} meetings ({n_cbi/N*100:.1f}%)")
mp_sig = "***" if results['jk_decomposition']['decomposed']['p_mp'] < 0.01 else \
         "**" if results['jk_decomposition']['decomposed']['p_mp'] < 0.05 else \
         "*" if results['jk_decomposition']['decomposed']['p_mp'] < 0.1 else ""
cbi_sig = "***" if results['jk_decomposition']['decomposed']['p_cbi'] < 0.01 else \
          "**" if results['jk_decomposition']['decomposed']['p_cbi'] < 0.05 else \
          "*" if results['jk_decomposition']['decomposed']['p_cbi'] < 0.1 else ""
print(f"   β_MP = {results['jk_decomposition']['decomposed']['beta_mp']:.6f}{mp_sig} (p={results['jk_decomposition']['decomposed']['p_mp']:.3f})")
print(f"   β_CBI = {results['jk_decomposition']['decomposed']['beta_cbi']:.6f}{cbi_sig} (p={results['jk_decomposition']['decomposed']['p_cbi']:.3f})")
print(f"   F-test (β_MP = β_CBI): p={results['jk_decomposition']['ftest_equal']['p_value']:.3f}")

print("\n2. Bauer-Swanson Orthogonalization:")
print(f"   Target predictability: R²={results['bauer_swanson']['predictability_target_r2']:.2f}%")
print(f"   Path predictability:   R²={results['bauer_swanson']['predictability_path_r2']:.2f}%")
t_sig = "***" if results['bauer_swanson']['orthogonalized_h1']['p_target'] < 0.01 else \
        "**" if results['bauer_swanson']['orthogonalized_h1']['p_target'] < 0.05 else \
        "*" if results['bauer_swanson']['orthogonalized_h1']['p_target'] < 0.1 else ""
p_sig = "***" if results['bauer_swanson']['orthogonalized_h1']['p_path'] < 0.01 else \
        "**" if results['bauer_swanson']['orthogonalized_h1']['p_path'] < 0.05 else \
        "*" if results['bauer_swanson']['orthogonalized_h1']['p_path'] < 0.1 else ""
print(f"   Orthogonalized β_T = {results['bauer_swanson']['orthogonalized_h1']['beta_target']:.6f}{t_sig} (p={results['bauer_swanson']['orthogonalized_h1']['p_target']:.3f})")
print(f"   Orthogonalized β_P = {results['bauer_swanson']['orthogonalized_h1']['beta_path']:.6f}{p_sig} (p={results['bauer_swanson']['orthogonalized_h1']['p_path']:.3f})")

print("\n3. Core finding robustness:")
target_survives = results['bauer_swanson']['orthogonalized_h1']['p_target'] < 0.1
path_still_weak = results['bauer_swanson']['orthogonalized_h1']['p_path'] > 0.1
print(f"   Target shock survives B-S orthogonalization: {'YES' if target_survives else 'NO'}")
print(f"   Path shock remains insignificant after B-S:  {'YES' if path_still_weak else 'NO'}")

# Save results
with open('results/jk_bs_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/jk_bs_results.json")
