# -*- coding: utf-8 -*-
"""
H5: ZLB Regime-Dependent Effect
Direction 2 — Echoes Phase 1 finding (FG sentiment R²=30.6% vs 5.6%)

Tests whether MP and CBI shocks have asymmetric effects on fund flows
across monetary policy regimes:
  - Pre-ZLB (2006-2008)
  - ZLB/Forward Guidance (2008-2015)
  - Post-ZLB Normalization (2015-2022)
  - COVID (2020-2022)

Model:
  Flow_{i,t} = α + β₁·MP_t + β₂·CBI_t + β₃·(MP_t × ZLB_t) 
               + β₄·(CBI_t × ZLB_t) + controls + ε

H5: β₃ < 0 (MP tightening has stronger outflow effect during ZLB)
    β₄ > 0 (CBI has stronger inflow effect during ZLB)

Rationale: When conventional rate tool is constrained, portfolio rebalancing
through language/communication becomes the primary transmission channel.
This echoes Phase 1's finding that sentiment incremental R² = 30.6% in FG
period vs 5.6% in non-FG period.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_chain import AuditChain

RISK_RANKING = {
    'government_bonds': 1,
    'corporate_bonds': 2,
    'real_assets': 3,
    'large_cap_equity': 4,
    'developed_market_equity': 5,
    'emerging_market_equity': 6,
    'small_cap_equity': 7,
}


def define_regimes(date_series):
    """
    Classify FOMC dates into regimes.
    
    Pre-ZLB:     before 2008-12-16
    ZLB/FG:      2008-12-16 to 2015-12-16
    Normalization: 2015-12-17 to 2020-02-29
    COVID:       2020-03-01 onwards
    """
    regimes = pd.Series('pre_zlb', index=date_series.index)
    regimes[date_series >= '2008-12-16'] = 'zlb'
    regimes[date_series >= '2015-12-17'] = 'normalization'
    regimes[date_series >= '2020-03-01'] = 'covid'
    return regimes


def h5_regime_analysis(flows_df, shocks_df, audit_chain=None):
    """
    Test H5: ZLB regime-dependent effect on fund flows.
    
    For each asset class, estimate:
      Flow = α + β₁·MP + β₂·CBI + β₃·(MP×ZLB) + β₄·(CBI×ZLB) + controls + ε
    
    H5: β₃ < 0 (MP effect stronger in ZLB), β₄ > 0 (CBI effect stronger in ZLB)
    """
    if audit_chain:
        audit_chain.log_human_decision(
            "H5 specification: Flow ~ MP + CBI + MP×ZLB + CBI×ZLB + controls. "
            "Test β₃<0 and β₄>0 for regime-dependent amplification.",
            author="ai"
        )
    
    # Merge flows with shocks
    merged = flows_df.merge(shocks_df, on='date', how='inner')
    
    # Define regimes
    merged['regime'] = define_regimes(merged['date'])
    merged['zlb_dummy'] = (merged['regime'] == 'zlb').astype(int)
    
    # Interaction terms
    merged['mp_x_zlb'] = merged['mp_shock'] * merged['zlb_dummy']
    merged['cbi_x_zlb'] = merged['cbi_shock'] * merged['zlb_dummy']
    
    # Controls (if available)
    control_cols = []
    for col in ['log_tna', 'flow_vol_12m', 'ret_12m_lag', 'exp_ratio']:
        if col in merged.columns:
            control_cols.append(col)
    
    results = {}
    
    for asset_class in RISK_RANKING.keys():
        subset = merged[merged['asset_class'] == asset_class].dropna(
            subset=['net_flow_pct', 'mp_shock', 'cbi_shock']
        )
        
        # ── FIX 4: Guard against inf/NaN and winsorize ──
        # Replace inf with NaN, then dropna again
        subset = subset.replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=['net_flow_pct', 'mp_shock', 'cbi_shock',
                                        'mp_x_zlb', 'cbi_x_zlb'])
        # Winsorize net_flow_pct at 1%/99% to prevent extreme outliers
        # from creating spurious significance (e.g., p=0.000000)
        if len(subset) > 10:
            p01, p99 = subset['net_flow_pct'].quantile([0.01, 0.99])
            subset = subset.copy()
            subset['net_flow_pct'] = subset['net_flow_pct'].clip(lower=p01, upper=p99)
        
        if len(subset) < 20:
            continue
        
        X_cols = ['mp_shock', 'cbi_shock', 'mp_x_zlb', 'cbi_x_zlb', 'zlb_dummy'] + control_cols
        X = subset[X_cols].copy()
        X = sm.add_constant(X)
        y = subset['net_flow_pct'].astype(float)
        
        # Fill NaN in controls with median
        for col in control_cols:
            X[col] = X[col].fillna(X[col].median())
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            
            # FIX 4b: Validate p-values (clip to [0,1], replace inf/NaN)
            def _safe_p(key):
                p = model.pvalues.get(key, np.nan)
                if np.isnan(p) or np.isinf(p):
                    return np.nan
                return float(np.clip(p, 0.0, 1.0))
            
            def _safe_beta(key):
                b = model.params.get(key, np.nan)
                if np.isnan(b) or np.isinf(b):
                    return np.nan
                return float(b)
            
            results[asset_class] = {
                'risk_rank': RISK_RANKING[asset_class],
                'n': len(subset),
                'n_zlb': int(subset['zlb_dummy'].sum()),
                'beta_mp': _safe_beta('mp_shock'),
                'p_mp': _safe_p('mp_shock'),
                'beta_cbi': _safe_beta('cbi_shock'),
                'p_cbi': _safe_p('cbi_shock'),
                'beta_mp_x_zlb': _safe_beta('mp_x_zlb'),
                'p_mp_x_zlb': _safe_p('mp_x_zlb'),
                'beta_cbi_x_zlb': _safe_beta('cbi_x_zlb'),
                'p_cbi_x_zlb': _safe_p('cbi_x_zlb'),
                'r_squared': float(model.rsquared) if not np.isnan(model.rsquared) else 0.0,
                'h5_mp_amplified': (
                    _safe_beta('mp_x_zlb') < 0 and
                    _safe_p('mp_x_zlb') < 0.10
                ),
                'h5_cbi_amplified': (
                    _safe_beta('cbi_x_zlb') > 0 and
                    _safe_p('cbi_x_zlb') < 0.10
                ),
            }
        except Exception as e:
            print(f"  Warning: {asset_class} failed: {e}")
    
    # Summary
    n_amplified_mp = sum(1 for r in results.values() if r.get('h5_mp_amplified'))
    n_amplified_cbi = sum(1 for r in results.values() if r.get('h5_cbi_amplified'))
    
    summary = {
        'n_asset_classes': len(results),
        'n_mp_amplified_in_zlb': n_amplified_mp,
        'n_cbi_amplified_in_zlb': n_amplified_cbi,
        'h5_supported': n_amplified_mp >= 3 or n_amplified_cbi >= 3,
        'interpretation': (
            f'H5 supported: {n_amplified_mp} assets show MP amplification, '
            f'{n_amplified_cbi} show CBI amplification in ZLB period'
            if n_amplified_mp >= 3 or n_amplified_cbi >= 3
            else f'H5 not supported: only {n_amplified_mp} MP and {n_amplified_cbi} CBI amplified'
        ),
    }
    
    if audit_chain:
        audit_chain.log_ai_response(
            f"H5 regime analysis complete. {summary['interpretation']}",
            model="analysis_pipeline"
        )
    
    return {'by_asset': results, 'summary': summary}


def print_h5_results(h5_results):
    """Pretty-print H5 results."""
    print(f"\n{'='*80}")
    print("H5: ZLB Regime-Dependent Effect on Fund Flows")
    print(f"{'='*80}")
    
    print(f"\n{'Asset Class':25s} {'β_MP':>8s} {'β_MP×ZLB':>10s} {'β_CBI':>8s} {'β_CBI×ZLB':>10s} {'R²':>6s}")
    print("-" * 75)
    
    for ac, r in h5_results['by_asset'].items():
        sig_mp = '***' if r['p_mp'] < 0.01 else '**' if r['p_mp'] < 0.05 else '*' if r['p_mp'] < 0.10 else ''
        sig_mpz = '***' if r['p_mp_x_zlb'] < 0.01 else '**' if r['p_mp_x_zlb'] < 0.05 else '*' if r['p_mp_x_zlb'] < 0.10 else ''
        sig_cbi = '***' if r['p_cbi'] < 0.01 else '**' if r['p_cbi'] < 0.05 else '*' if r['p_cbi'] < 0.10 else ''
        sig_cbiz = '***' if r['p_cbi_x_zlb'] < 0.01 else '**' if r['p_cbi_x_zlb'] < 0.05 else '*' if r['p_cbi_x_zlb'] < 0.10 else ''
        
        print(f"{ac:25s} {r['beta_mp']:7.4f}{sig_mp:1s} {r['beta_mp_x_zlb']:9.4f}{sig_mpz:1s} "
              f"{r['beta_cbi']:7.4f}{sig_cbi:1s} {r['beta_cbi_x_zlb']:9.4f}{sig_cbiz:1s} {r['r_squared']:5.1f}%")
    
    print(f"\nSummary: {h5_results['summary']['interpretation']}")


if __name__ == "__main__":
    print("H5 ZLB Regime Analysis Module")
    print("Run after fund flow data is available.")
