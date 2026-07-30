# -*- coding: utf-8 -*-
"""
Phase 1 Shock Loader — JK Decomposition for Direction 2

Loads Phase 1 results from the monetary-policy-lab repo:
  - minutes_sentiment_corrected.csv (117 FOMC meetings, 2006-2022)
  - JK sign-restriction decomposition: target_shock → mp_shock + cbi_shock

JK Decomposition Logic (from jk_bs_decomposition.py):
  - MP shock:  target_shock and equity return move in OPPOSITE directions
               (hawkish → rates up, stocks down; dovish → rates down, stocks up)
  - CBI shock: target_shock and equity return move in SAME direction
               (hawkish → rates up, stocks up = central bank information revelation)
  - path_shock: carried through unchanged from Phase 1

Output columns for Direction 2 pipeline:
  date, mp_shock, cbi_shock, path_shock, target_shock,
  sentiment, fg_period, vwretd_day
"""

import pandas as pd
import numpy as np
import os


def load_phase1_shocks(csv_path="../results/minutes_sentiment_corrected.csv",
                       llm_sentiment_path="../results/llm_sentiment_results.csv"):
    """
    Load Phase 1 data and apply JK sign-restriction decomposition.

    Args:
        csv_path: Path to minutes_sentiment_corrected.csv from Phase 1.
        llm_sentiment_path: Path to LLM sentiment results (if exists, used for B-S).

    Returns:
        DataFrame with columns:
            date, mp_shock, cbi_shock, path_shock, target_shock,
            bs_mp_shock, bs_cbi_shock (LM-based B-S),
            bs_llm_mp_shock, bs_llm_cbi_shock (LLM-based B-S),
            sentiment, llm_hawkish, fg_period, vwretd_day
    """
    if not os.path.exists(csv_path):
        print(f"⚠️  Phase 1 data not found at {csv_path}")
        print("   Downloading from GitHub (dechang64/monetary-policy-lab)...")
        url = ("https://raw.githubusercontent.com/dechang64/"
               "monetary-policy-lab/main/results/minutes_sentiment_corrected.csv")
        df = pd.read_csv(url)
        # Cache locally
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Downloaded {len(df)} FOMC meetings, cached to {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} FOMC meetings from {csv_path}")

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # ── JK Sign-Restriction Decomposition ──
    # Equity signal: CRSP VW return (convert to percentage)
    df['vw_ret_pct'] = df['vwretd_day'] * 100

    # Classify each meeting
    df['target_sign'] = np.sign(df['target_shock'])
    df['equity_sign'] = np.sign(df['vw_ret_pct'])

    # MP shock: target and equity move in opposite directions
    df['is_mp_shock'] = (df['target_sign'] * df['equity_sign'] < 0).astype(int)

    # CBI shock: target and equity move in same direction
    df['is_cbi_shock'] = (df['target_sign'] * df['equity_sign'] > 0).astype(int)

    # Construct MP-only and CBI-only shock series
    df['mp_shock'] = df['target_shock'] * df['is_mp_shock']
    df['cbi_shock'] = df['target_shock'] * df['is_cbi_shock']

    # path_shock is carried through unchanged
    # (already in the CSV from Phase 1)

    n_mp = df['is_mp_shock'].sum()
    n_cbi = df['is_cbi_shock'].sum()
    n_neutral = len(df) - n_mp - n_cbi
    print(f"   JK decomposition: {n_mp} MP shocks, {n_cbi} CBI shocks, "
          f"{n_neutral} neutral")

    # ── Bauer-Swanson Orthogonalization ──
    # Following Bauer & Swanson (2023, NBER Macro Annual):
    # Regress target_shock on path_shock (and sentiment as additional control)
    # to remove information contamination from the pure monetary policy signal.
    # The residual = orthogonalized target shock ("clean" MP surprise).
    print("   Performing Bauer-Swanson orthogonalization...")
    from statsmodels.regression.linear_model import OLS as _OLS
    import statsmodels.api as _sm

    bs_exog = _sm.add_constant(df[['path_shock', 'sentiment']].fillna(0))
    bs_model = _OLS(df['target_shock'].fillna(0), bs_exog).fit()
    df['bs_target_shock'] = df['target_shock'].fillna(0) - bs_model.fittedvalues

    # Apply JK sign restriction on orthogonalized target shock
    df['bs_mp_shock'] = df['bs_target_shock'] * df['is_mp_shock']
    df['bs_cbi_shock'] = df['bs_target_shock'] * df['is_cbi_shock']

    n_bs_mp_nonzero = (df['bs_mp_shock'].abs() > 1e-10).sum()
    n_bs_cbi_nonzero = (df['bs_cbi_shock'].abs() > 1e-10).sum()
    print(f"   B-S (LM sentiment): {n_bs_mp_nonzero} MP, {n_bs_cbi_nonzero} CBI shocks")

    # ── LLM-based Bauer-Swanson Orthogonalization ──
    # Uses llm_hawkish instead of LM sentiment for information removal.
    # LLM hawkish captures the hawkish/dovish dimension that LM dictionary misses.
    if os.path.exists(llm_sentiment_path):
        llm_df = pd.read_csv(llm_sentiment_path)
        llm_df['date'] = pd.to_datetime(llm_df['date'])
        df = df.merge(llm_df[['date', 'llm_hawkish']], on='date', how='left')
        print(f"   Loaded LLM sentiment from {llm_sentiment_path}")

        bs_llm_exog = _sm.add_constant(df[['path_shock', 'llm_hawkish']].fillna(0))
        bs_llm_model = _OLS(df['target_shock'].fillna(0), bs_llm_exog).fit()
        df['bs_llm_target_shock'] = df['target_shock'].fillna(0) - bs_llm_model.fittedvalues

        df['bs_llm_mp_shock'] = df['bs_llm_target_shock'] * df['is_mp_shock']
        df['bs_llm_cbi_shock'] = df['bs_llm_target_shock'] * df['is_cbi_shock']

        n_llm_mp = (df['bs_llm_mp_shock'].abs() > 1e-10).sum()
        n_llm_cbi = (df['bs_llm_cbi_shock'].abs() > 1e-10).sum()
        print(f"   B-S (LLM hawkish): {n_llm_mp} MP, {n_llm_cbi} CBI shocks")
    else:
        print("   ⚠️  LLM sentiment not found, skipping LLM-based B-S")
        df['bs_llm_mp_shock'] = np.nan
        df['bs_llm_cbi_shock'] = np.nan
        df['llm_hawkish'] = np.nan

    # Select and rename columns for Direction 2
    result = df[[
        'date', 'mp_shock', 'cbi_shock', 'path_shock', 'target_shock',
        'bs_mp_shock', 'bs_cbi_shock',
        'bs_llm_mp_shock', 'bs_llm_cbi_shock',
        'sentiment', 'llm_hawkish', 'fg_period', 'vwretd_day'
    ]].copy()

    result = result.sort_values('date').reset_index(drop=True)
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Shock Loader — JK Decomposition")
    print("=" * 60)

    shocks = load_phase1_shocks()

    print(f"\nSample: {len(shocks)} FOMC meetings")
    print(f"  Date range: {shocks['date'].min().strftime('%Y-%m-%d')} "
          f"to {shocks['date'].max().strftime('%Y-%m-%d')}")
    print(f"\nColumn summary:")
    for col in ['mp_shock', 'cbi_shock', 'path_shock', 'target_shock']:
        nonzero = (shocks[col] != 0).sum()
        print(f"  {col:15s}: mean={shocks[col].mean():.6f}, "
              f"std={shocks[col].std():.6f}, "
              f"nonzero={nonzero}/{len(shocks)}")

    print(f"\nFirst 5 rows:")
    print(shocks.head().to_string(index=False))

    print(f"\n✅ Phase 1 shocks ready for Direction 2 pipeline")
