# -*- coding: utf-8 -*-
"""
Phase 1 LLM Robustness Check
Compares incremental R² of LM dictionary sentiment vs LLM hawkish score
in predicting FOMC-day asset returns.

Reproduces the MRS 2026 finding (sentiment incremental R² = 30.6% in FG, 5.6% non-FG)
and compares with LLM-based sentiment.

Usage:
    python phase1_llm_robustness.py

Output:
    ../results/phase1_llm_robustness.csv
    ../results/phase1_llm_robustness_summary.txt
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json
import os
import sys

# ============================================================
# Load data
# ============================================================
csv_path = "../results/minutes_sentiment_corrected.csv"
llm_path = "../results/llm_sentiment_results.csv"

df = pd.read_csv(csv_path)
llm = pd.read_csv(llm_path)

# Merge
df['date'] = pd.to_datetime(df['date'])
llm['date'] = pd.to_datetime(llm['date'])
df = df.merge(llm[['date', 'llm_hawkish', 'llm_forward_guidance', 'llm_information']],
              on='date', how='left')

print(f"Loaded {len(df)} FOMC meetings")
print(f"FG period: {df['fg_period'].sum()} meetings")
print(f"Non-FG period: {(1-df['fg_period']).sum()} meetings")
print(f"LLM hawkish: mean={df['llm_hawkish'].mean():.4f}, std={df['llm_hawkish'].std():.4f}")
print(f"LM sentiment: mean={df['sentiment'].mean():.4f}, std={df['sentiment'].std():.4f}")
print()

# ============================================================
# Create interaction terms
# ============================================================
df['sentiment_x_fg'] = df['sentiment'] * df['fg_period']
df['llm_hawkish_x_fg'] = df['llm_hawkish'] * df['fg_period']
df['llm_info_x_fg'] = df['llm_information'] * df['fg_period']

# ============================================================
# Asset returns to test
# ============================================================
asset_returns = {
    'vwretd_day': 'CRSP VW Return',
    'sp500_ret': 'S&P 500',
    'nasdaq_ret': 'NASDAQ',
    'ty10_chg': '10Y Treasury',
    'gold_ret': 'Gold',
}

# ============================================================
# Incremental R² Analysis
# ============================================================
results = []

for ret_col, ret_name in asset_returns.items():
    y = df[ret_col].dropna()
    X_base = df.loc[y.index, ['target_shock', 'path_shock']].fillna(0)

    # --- LM sentiment ---
    X_lm = pd.concat([X_base,
                      df.loc[y.index, ['sentiment', 'sentiment_x_fg']].fillna(0)],
                     axis=1)
    X_lm = sm.add_constant(X_lm)

    # --- LLM hawkish ---
    X_llm = pd.concat([X_base,
                       df.loc[y.index, ['llm_hawkish', 'llm_hawkish_x_fg']].fillna(0)],
                      axis=1)
    X_llm = sm.add_constant(X_llm)

    # --- LLM information ---
    X_llm_info = pd.concat([X_base,
                            df.loc[y.index, ['llm_information', 'llm_info_x_fg']].fillna(0)],
                           axis=1)
    X_llm_info = sm.add_constant(X_llm_info)

    # Baseline model (target + path only)
    X_base_c = sm.add_constant(X_base)
    model_base = sm.OLS(y, X_base_c).fit()
    r2_base = model_base.rsquared

    # LM sentiment model
    model_lm = sm.OLS(y, X_lm).fit()
    r2_lm = model_lm.rsquared
    incr_r2_lm = r2_lm - r2_base

    # LLM hawkish model
    model_llm = sm.OLS(y, X_llm).fit()
    r2_llm = model_llm.rsquared
    incr_r2_llm = r2_llm - r2_base

    # LLM information model
    model_llm_info = sm.OLS(y, X_llm_info).fit()
    r2_llm_info = model_llm_info.rsquared
    incr_r2_llm_info = r2_llm_info - r2_base

    # --- Subsample analysis: FG vs non-FG ---
    fg_mask = df.loc[y.index, 'fg_period'] == 1
    nonfg_mask = df.loc[y.index, 'fg_period'] == 0

    for period, mask in [('FG', fg_mask), ('Non-FG', nonfg_mask)]:
        y_sub = y[mask]
        X_sub_base = df.loc[y_sub.index, ['target_shock', 'path_shock']].fillna(0)
        X_sub_base = sm.add_constant(X_sub_base)

        # LM
        X_sub_lm = pd.concat([X_sub_base,
                              df.loc[y_sub.index, ['sentiment']].fillna(0)], axis=1)
        X_sub_lm = sm.add_constant(X_sub_lm.drop(columns='const'))

        # LLM
        X_sub_llm = pd.concat([X_sub_base,
                               df.loc[y_sub.index, ['llm_hawkish']].fillna(0)], axis=1)
        X_sub_llm = sm.add_constant(X_sub_llm.drop(columns='const'))

        m_base = sm.OLS(y_sub, X_sub_base).fit()
        m_lm = sm.OLS(y_sub, X_sub_lm).fit()
        m_llm = sm.OLS(y_sub, X_sub_llm).fit()

        results.append({
            'asset': ret_name,
            'asset_col': ret_col,
            'period': period,
            'n': len(y_sub),
            'r2_base': m_base.rsquared,
            'r2_lm': m_lm.rsquared,
            'incr_r2_lm': m_lm.rsquared - m_base.rsquared,
            'r2_llm': m_llm.rsquared,
            'incr_r2_llm': m_llm.rsquared - m_base.rsquared,
            'lm_coef': m_lm.params.get('sentiment', np.nan),
            'lm_pval': m_lm.pvalues.get('sentiment', np.nan),
            'llm_coef': m_llm.params.get('llm_hawkish', np.nan),
            'llm_pval': m_llm.pvalues.get('llm_hawkish', np.nan),
        })

    # Full sample with interaction
    results.append({
        'asset': ret_name,
        'asset_col': ret_col,
        'period': 'Full (with interaction)',
        'n': len(y),
        'r2_base': r2_base,
        'r2_lm': r2_lm,
        'incr_r2_lm': incr_r2_lm,
        'r2_llm': r2_llm,
        'incr_r2_llm': incr_r2_llm,
        'lm_coef': model_lm.params.get('sentiment_x_fg', np.nan),
        'lm_pval': model_lm.pvalues.get('sentiment_x_fg', np.nan),
        'llm_coef': model_llm.params.get('llm_hawkish_x_fg', np.nan),
        'llm_pval': model_llm.pvalues.get('llm_hawkish_x_fg', np.nan),
    })

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv("../results/phase1_llm_robustness.csv", index=False)
print(f"Saved to ../results/phase1_llm_robustness.csv")
print()

# ============================================================
# Print summary
# ============================================================
summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("Phase 1 LLM Robustness Check: Incremental R² Comparison")
summary_lines.append("LM Dictionary Sentiment vs LLM Hawkish Score")
summary_lines.append("=" * 80)
summary_lines.append("")

for ret_col, ret_name in asset_returns.items():
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append(f"Asset: {ret_name} ({ret_col})")
    summary_lines.append(f"{'='*60}")

    for period in ['Non-FG', 'FG', 'Full (with interaction)']:
        row = results_df[(results_df['asset_col'] == ret_col) &
                         (results_df['period'] == period)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        summary_lines.append(f"\n  {period} (N={r['n']:.0f}):")
        summary_lines.append(f"    Baseline R² (target+path):     {r['r2_base']:.4f}")
        summary_lines.append(f"    LM Sentiment R²:               {r['r2_lm']:.4f} (incr: {r['incr_r2_lm']:+.4f})")
        summary_lines.append(f"    LLM Hawkish R²:                {r['r2_llm']:.4f} (incr: {r['incr_r2_llm']:+.4f})")
        summary_lines.append(f"    LM coef={r['lm_coef']:.4f} (p={r['lm_pval']:.4f})")
        summary_lines.append(f"    LLM coef={r['llm_coef']:.4f} (p={r['llm_pval']:.4f})")

summary_lines.append(f"\n{'='*80}")
summary_lines.append("Key Comparison: Incremental R² in FG period")
summary_lines.append(f"{'='*80}")
summary_lines.append(f"\n{'Asset':15s} {'LM incr R²':>12s} {'LLM incr R²':>12s} {'Difference':>12s}")
summary_lines.append("-" * 55)
for ret_col, ret_name in asset_returns.items():
    fg_row = results_df[(results_df['asset_col'] == ret_col) &
                        (results_df['period'] == 'FG')]
    if len(fg_row) > 0:
        r = fg_row.iloc[0]
        diff = r['incr_r2_llm'] - r['incr_r2_lm']
        summary_lines.append(f"{ret_name:15s} {r['incr_r2_lm']:12.4f} {r['incr_r2_llm']:12.4f} {diff:+12.4f}")

summary_lines.append(f"\n{'='*80}")
summary_lines.append("Key Comparison: Incremental R² in Non-FG period")
summary_lines.append(f"{'='*80}")
summary_lines.append(f"\n{'Asset':15s} {'LM incr R²':>12s} {'LLM incr R²':>12s} {'Difference':>12s}")
summary_lines.append("-" * 55)
for ret_col, ret_name in asset_returns.items():
    nonfg_row = results_df[(results_df['asset_col'] == ret_col) &
                           (results_df['period'] == 'Non-FG')]
    if len(nonfg_row) > 0:
        r = nonfg_row.iloc[0]
        diff = r['incr_r2_llm'] - r['incr_r2_lm']
        summary_lines.append(f"{ret_name:15s} {r['incr_r2_lm']:12.4f} {r['incr_r2_llm']:12.4f} {diff:+12.4f}")

summary_text = "\n".join(summary_lines)
print(summary_text)

with open("../results/phase1_llm_robustness_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\n✅ Summary saved to ../results/phase1_llm_robustness_summary.txt")
