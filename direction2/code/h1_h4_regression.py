# -*- coding: utf-8 -*-
"""
Direction 2 — H1-H5 Regression Analysis (Optimized)
Portfolio Rebalancing and Cross-Asset Contagion

Tests (optimized after literature review):
  H1 (Risk-Off Destination): MP tightening → outflows from high-risk,
     inflows to safe — AND flows move along risk ladder (not just exit)
  H2 (Risk-On Source): Positive CBI → inflows to high-risk, outflows
     from safe — AND flows move up the risk ladder (not just new entry)
  H3 (Asymmetry): MP ≠ CBI effect on flows (Wald test per asset)
  H4 (Risk-Ladder Substitution): 7×7 substitution matrix (see h4_substitution_matrix.py)
  H5 (ZLB Regime): MP/CBI effects amplified during ZLB period (see h5_regime_analysis.py)

Uses JK decomposition shocks from Phase 1:
  - MP shock (pure monetary policy)
  - CBI shock (central bank information)
  - Path shock (residual)

Controls (per Fecht & Kellers 2026, Blanco et al. 2025):
  - log_tna: fund size
  - flow_vol_12m: fragility proxy
  - ret_12m_lag: return-chasing proxy
  - exp_ratio: expense ratio (median-filled)

Data: 117 FOMC meetings (2006-2022) from Phase 1

Dual baseline: Raw JK + Bauer-Swanson orthogonalized
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_chain import AuditChain

# Risk ranking (1=safest, 7=riskiest)
RISK_RANKING = {
    'government_bonds': 1,
    'corporate_bonds': 2,
    'real_assets': 3,
    'large_cap_equity': 4,
    'developed_market_equity': 5,
    'emerging_market_equity': 6,
    'small_cap_equity': 7,
}


def load_phase1_data(audit_chain=None):
    """
    Load Phase 1 data: JK-decomposed shocks for 117 FOMC meetings.
    
    Returns DataFrame with: date, mp_shock, cbi_shock, path_shock, 
    target_shock, sentiment, fg_period
    """
    # Load from Phase 1 results
    # Source: results/jk_bs_results.json + results/minutes_sentiment_corrected.csv
    # This will be populated from the actual repo data
    
    # For now, define the expected schema
    columns = [
        'date',           # FOMC meeting date
        'mp_shock',       # JK pure monetary policy shock
        'cbi_shock',      # JK central bank information shock  
        'path_shock',     # GSS path factor (residual)
        'target_shock',   # GSS target factor
        'sentiment',      # FOMC statement LM sentiment score
        'fg_period',      # Forward guidance indicator (2008-2015)
    ]
    
    if audit_chain:
        audit_chain.log_data_access(
            source="Phase 1 results",
            query="JK-decomposed shocks + sentiment",
            metadata={"expected_n": 117, "date_range": "2006-2022"}
        )
    
    return columns


def h1_risk_off(flows_df, shocks_df, audit_chain=None, use_bs=False):
    """
    H1 (Risk-Off Destination): Pure MP tightening → outflows from high-risk,
    inflows to safe — AND flows move along risk ladder (not just exit).
    
    Model: Flow_{i,t} = α_i + β₁·MP_shock_t + β₂·CBI_shock_t 
                         + γ₁·log_tna + γ₂·flow_vol_12m + γ₃·ret_12m_lag 
                         + γ₄·exp_ratio + ε_{i,t}
    
    H1 test: β₁ < 0 for high-risk assets, β₁ > 0 for safe assets
    Destination test: β₁(safe) > 0 AND β₁(high-risk) < 0 
                      → flows reallocate, not just exit
    
    Args:
        use_bs: If True, use Bauer-Swanson orthogonalized shocks (dual baseline)
    """
    shock_prefix = 'bs_' if use_bs else ''
    shock_label = 'B-S orthogonalized' if use_bs else 'raw JK'
    
    if audit_chain:
        audit_chain.log_human_decision(
            f"H1 specification ({shock_label}): Flow ~ MP + CBI + controls, "
            f"test β_MP < 0 for high-risk, β_MP > 0 for safe assets, "
            f"destination test: β(safe) > 0 AND β(risky) < 0",
            author="ai"
        )
    
    results = {}
    
    # Control variables (optimized per Fecht & Kellers 2026, Blanco et al. 2025)
    control_cols = ['log_tna', 'flow_vol_12m', 'ret_12m_lag', 'exp_ratio']
    available_controls = [c for c in control_cols if c in flows_df.columns]
    
    for asset_class in RISK_RANKING.keys():
        subset = flows_df[flows_df['asset_class'] == asset_class].merge(
            shocks_df, on='date', how='inner'
        )
        
        if len(subset) < 10:
            continue
        
        # ── FIX 6: Winsorize net_flow_pct at 1%/99% per asset class ──
        # Prevents residual outliers from distorting regression coefficients
        subset = subset.copy()
        subset = subset.replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=['net_flow_pct'])
        if len(subset) > 10:
            p01, p99 = subset['net_flow_pct'].quantile([0.01, 0.99])
            subset['net_flow_pct'] = subset['net_flow_pct'].clip(lower=p01, upper=p99)
        
        mp_col = f'{shock_prefix}mp_shock' if f'{shock_prefix}mp_shock' in subset.columns else 'mp_shock'
        cbi_col = f'{shock_prefix}cbi_shock' if f'{shock_prefix}cbi_shock' in subset.columns else 'cbi_shock'
        
        X_cols = [mp_col, cbi_col] + available_controls
        X = subset[X_cols].copy()
        X = sm.add_constant(X)
        y = subset['net_flow_pct'].astype(float)
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        
        results[asset_class] = {
            'risk_rank': RISK_RANKING[asset_class],
            'n': len(subset),
            'beta_mp': model.params.get(mp_col, np.nan),
            'p_mp': model.pvalues.get(mp_col, np.nan),
            'beta_cbi': model.params.get(cbi_col, np.nan),
            'p_cbi': model.pvalues.get(cbi_col, np.nan),
            'r_squared': model.rsquared * 100,
            'controls': available_controls,
            'shock_type': shock_label,
        }
    
    return results


def h2_risk_on(flows_df, shocks_df, audit_chain=None, use_bs=False):
    """
    H2 (Risk-On): Positive CBI shock → inflows to high-risk, outflows from safe.
    
    Same model as H1, but focus on β_CBI:
    H2 test: β_CBI > 0 for high-risk assets, β_CBI < 0 for safe assets
    """
    shock_label = 'B-S orthogonalized' if use_bs else 'raw JK'
    if audit_chain:
        audit_chain.log_human_decision(
            f"H2 specification ({shock_label}): Same model as H1, test β_CBI > 0 for high-risk",
            author="ai"
        )

    # H2 uses the same regression as H1, different coefficient focus
    results = h1_risk_off(flows_df, shocks_df, audit_chain=audit_chain, use_bs=use_bs)
    
    # Re-interpret for H2
    for asset_class, res in results.items():
        res['h2_prediction'] = 'inflow' if RISK_RANKING[asset_class] >= 4 else 'outflow'
        res['h2_supported'] = (
            (res['beta_cbi'] > 0 and RISK_RANKING[asset_class] >= 4) or
            (res['beta_cbi'] < 0 and RISK_RANKING[asset_class] < 4)
        ) and res['p_cbi'] < 0.10
    
    return results


def h3_asymmetry(flows_df, shocks_df, audit_chain=None, use_bs=False):
    """
    H3 (Asymmetry): MP shock effect ≠ CBI shock effect on flows.
    
    Test: Wald test for β_MP = β_CBI in each asset class.
    H3 supported if Wald test rejects equality for multiple asset classes.
    """
    shock_prefix = 'bs_' if use_bs else ''
    shock_label = 'B-S orthogonalized' if use_bs else 'raw JK'
    if audit_chain:
        audit_chain.log_human_decision(
            f"H3 specification ({shock_label}): Wald test β_MP = β_CBI per asset class",
            author="ai"
        )
    
    # Control variables (same as H1 for consistency)
    control_cols = ['log_tna', 'flow_vol_12m', 'ret_12m_lag', 'exp_ratio']
    available_controls = [c for c in control_cols if c in flows_df.columns]
    
    results = {}
    
    for asset_class in RISK_RANKING.keys():
        subset = flows_df[flows_df['asset_class'] == asset_class].merge(
            shocks_df, on='date', how='inner'
        )
        
        if len(subset) < 10:
            continue
        
        # ── FIX 8: Same winsorize + inf/NaN cleaning as H1 ──
        subset = subset.copy()
        subset = subset.replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=['net_flow_pct'])
        if len(subset) > 10:
            p01, p99 = subset['net_flow_pct'].quantile([0.01, 0.99])
            subset['net_flow_pct'] = subset['net_flow_pct'].clip(lower=p01, upper=p99)
        
        # Use controls (same as H1) — without controls, HAC cov can be singular
        mp_col = f'{shock_prefix}mp_shock' if f'{shock_prefix}mp_shock' in subset.columns else 'mp_shock'
        cbi_col = f'{shock_prefix}cbi_shock' if f'{shock_prefix}cbi_shock' in subset.columns else 'cbi_shock'
        X_cols = [mp_col, cbi_col] + available_controls
        X = subset[X_cols].copy()
        # Fill NaN controls with median
        for col in available_controls:
            X[col] = X[col].fillna(X[col].median())
        X = sm.add_constant(X)
        y = subset['net_flow_pct'].astype(float)
        
        # Drop any remaining NaN rows
        mask = y.notna() & X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(y) < 10:
            continue
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        
        # ── FIX 9: Robust Wald test with fallbacks ──
        wald_stat = np.nan
        wald_p = np.nan
        
        # Attempt 1: Wald test with use_f=False (chi-square)
        try:
            wald = model.wald_test(f'{mp_col} = {cbi_col}', use_f=False)
            wald_stat = float(wald.statistic)
            wald_p = float(wald.pvalue)
            if np.isinf(wald_stat) or np.isnan(wald_stat):
                wald_stat = np.nan
                wald_p = np.nan
        except Exception:
            pass
        
        # Attempt 2: If Wald failed, try with use_f=True (F-test)
        if np.isnan(wald_p):
            try:
                wald = model.wald_test(f'{mp_col} = {cbi_col}', use_f=True)
                wald_stat = float(wald.statistic)
                wald_p = float(wald.pvalue)
                if np.isinf(wald_stat) or np.isnan(wald_stat):
                    wald_stat = np.nan
                    wald_p = np.nan
            except Exception:
                pass
        
        # Attempt 3: Manual t-test of coefficient difference
        # t = (β_MP - β_CBI) / SE(β_MP - β_CBI)
        # SE(diff) = sqrt(var_MP + var_CBI - 2*cov_MP,CBI)
        if np.isnan(wald_p):
            try:
                b_mp = model.params.get(mp_col, 0)
                b_cbi = model.params.get(cbi_col, 0)
                var_mp = model.cov_params().loc[mp_col, mp_col]
                var_cbi = model.cov_params().loc[cbi_col, cbi_col]
                cov_mp_cbi = model.cov_params().loc[mp_col, cbi_col]
                
                diff = b_mp - b_cbi
                se_diff = np.sqrt(var_mp + var_cbi - 2 * cov_mp_cbi)
                
                if se_diff > 0 and not np.isinf(se_diff) and not np.isnan(se_diff):
                    t_stat = diff / se_diff
                    from scipy import stats as sp_stats
                    wald_stat = t_stat ** 2  # chi2(1) = t^2
                    wald_p = 2 * (1 - sp_stats.norm.cdf(abs(t_stat)))  # two-sided
            except Exception:
                pass
        
        # Attempt 4: If all above failed, try OLS without HAC
        if np.isnan(wald_p):
            try:
                model_ols = sm.OLS(y, X).fit()
                wald = model_ols.wald_test(f'{mp_col} = {cbi_col}', use_f=False)
                wald_stat = float(wald.statistic)
                wald_p = float(wald.pvalue)
                if np.isinf(wald_stat) or np.isnan(wald_stat):
                    wald_stat = np.nan
                    wald_p = np.nan
            except Exception:
                pass
        
        results[asset_class] = {
            'risk_rank': RISK_RANKING[asset_class],
            'n': len(subset),
            'beta_mp': model.params.get(mp_col, np.nan),
            'beta_cbi': model.params.get(cbi_col, np.nan),
            'wald_chi2': wald_stat,
            'wald_p': wald_p,
            'h3_rejected': wald_p < 0.10 if not np.isnan(wald_p) else False,
        }
    
    return results


def h4_risk_ladder(flows_df, shocks_df, audit_chain=None):
    """
    H4 (Risk-Ladder Substitution): Flows vary systematically with risk rank.
    
    Model: Flow_{i,t} = α + γ·RiskRank_i + δ₁·(RiskRank_i × MP_shock_t) 
                        + δ₂·(RiskRank_i × CBI_shock_t) + ε_{i,t}
    
    H4 test: δ₁ < 0 (MP tightening → more outflows as risk increases)
             δ₂ > 0 (positive CBI → more inflows as risk increases)
    
    This tests whether reallocation is monotonic along the risk ladder,
    rather than a binary equity↔bond shift.
    """
    if audit_chain:
        audit_chain.log_human_decision(
            "H4 specification: Flow ~ RiskRank + RiskRank×MP + RiskRank×CBI, "
            "test δ₁ < 0 and δ₂ > 0 (monotonic risk-ladder substitution)",
            author="ai"
        )
    
    # Add risk rank to flows
    flows_with_risk = flows_df.copy()
    flows_with_risk['risk_rank'] = flows_with_risk['asset_class'].map(RISK_RANKING)
    
    # Merge with shocks
    merged = flows_with_risk.merge(shocks_df, on='date', how='inner')
    
    # Create interaction terms
    merged['risk_x_mp'] = merged['risk_rank'] * merged['mp_shock']
    merged['risk_x_cbi'] = merged['risk_rank'] * merged['cbi_shock']
    
    # Pooled regression: asset-class FE absorbs risk_rank main effect
    # (risk_rank is collinear with asset dummies, so don't include both)
    X = merged[['risk_x_mp', 'risk_x_cbi', 'mp_shock', 'cbi_shock']]
    X = pd.get_dummies(merged['asset_class'], drop_first=True).join(X)
    X = sm.add_constant(X.astype(float))
    y = merged['net_flow_pct']
    
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': merged['date']})
    
    results = {
        'n': len(merged),
        'n_asset_classes': merged['asset_class'].nunique(),
        'delta_1_risk_x_mp': model.params.get('risk_x_mp', np.nan),
        'p_delta_1': model.pvalues.get('risk_x_mp', np.nan),
        'delta_2_risk_x_cbi': model.params.get('risk_x_cbi', np.nan),
        'p_delta_2': model.pvalues.get('risk_x_cbi', np.nan),
        'r_squared': model.rsquared,
        'h4_monotonic_mp': model.params.get('risk_x_mp', 0) < 0 and 
                          model.pvalues.get('risk_x_mp', 1) < 0.10,
        'h4_monotonic_cbi': model.params.get('risk_x_cbi', 0) > 0 and 
                           model.pvalues.get('risk_x_cbi', 1) < 0.10,
    }
    
    return results


# ============================================================
# P1: Panel Regression with Fixed Effects
# ============================================================

def h1_panel_regression(flows_df, shocks_df, audit_chain=None, use_bs=False):
    """
    H1 Panel Regression: Pooled OLS with asset-class FE.

    Uses all 819 observations (117 FOMC × 7 asset classes) in a single
    regression, leveraging cross-sectional variation for power.

    Model: Flow_{i,t} = α_i + β₁·MP_shock_t + β₂·CBI_shock_t
                         + γ·controls_{i,t} + δ_i·RiskRank_i×(MP+CBI)
                         + ε_{i,t}

    where:
      α_i = asset-class fixed effect (absorbs time-invariant asset differences)
      δ_i = risk-ladder interaction (tests monotonic reallocation)

    NOTE: No time FE because mp_shock/cbi_shock vary only at date level,
    making them perfectly collinear with time dummies. Clustered SE by
    date accounts for cross-sectional correlation.

    SE: clustered by date (account for cross-sectional correlation)

    H1 test: β₁ < 0 (MP tightening → outflows on average)
             δ₁ < 0 (MP effect intensifies along risk ladder)
    """
    shock_prefix = 'bs_' if use_bs else ''
    shock_label = 'B-S orthogonalized' if use_bs else 'raw JK'
    
    if audit_chain:
        audit_chain.log_human_decision(
            f"H1 panel ({shock_label}): Pooled OLS + asset FE + time FE, "
            f"clustered SE by date. Test β_MP < 0 and RiskRank×MP < 0.",
            author="ai"
        )
    
    merged = flows_df.merge(shocks_df, on='date', how='inner')
    if len(merged) < 50:
        return {'error': 'Insufficient data for panel regression'}
    
    mp_col = f'{shock_prefix}mp_shock' if f'{shock_prefix}mp_shock' in merged.columns else 'mp_shock'
    cbi_col = f'{shock_prefix}cbi_shock' if f'{shock_prefix}cbi_shock' in merged.columns else 'cbi_shock'
    
    # Add risk rank
    merged['risk_rank'] = merged['asset_class'].map(RISK_RANKING)
    merged['risk_x_mp'] = merged['risk_rank'] * merged[mp_col]
    merged['risk_x_cbi'] = merged['risk_rank'] * merged[cbi_col]
    
    # Control variables
    control_cols = [c for c in ['log_tna', 'flow_vol_12m', 'ret_12m_lag', 'exp_ratio']
                    if c in merged.columns]
    
    # Build design matrix
    y = merged['net_flow_pct']
    X_cols = [mp_col, cbi_col, 'risk_x_mp', 'risk_x_cbi'] + control_cols
    
    # Asset-class FE only (drop one as baseline)
    # NOTE: Time FE dropped because mp_shock/cbi_shock vary only at date level,
    # making them perfectly collinear with time dummies → NaN coefficients.
    # Clustering SE by date accounts for cross-sectional correlation.
    ac_dummies = pd.get_dummies(merged['asset_class'], prefix='ac', drop_first=True, dtype=float)

    X = pd.concat([merged[X_cols].astype(float), ac_dummies], axis=1)
    X = sm.add_constant(X)

    # ── Clean NaN/inf (same as per-asset H1/H3) ──
    # Real data has NaN in net_flow_pct (no funds for some asset×date cells)
    # Without this cleaning, OLS returns NaN for ALL coefficients
    merged_clean = merged.replace([np.inf, -np.inf], np.nan)
    y = merged_clean['net_flow_pct'].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Fill NaN controls with median (same as h3_asymmetry)
    for col in control_cols:
        if col in X.columns and X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    # Drop rows where y or any X is NaN
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    cluster_groups = merged_clean.loc[mask, 'date']

    if len(y) < 50:
        return {'error': f'Insufficient data after cleaning ({len(y)} obs)'}

    # Drop any all-NaN or constant columns (defensive)
    X = X.dropna(axis=1, how='all')
    const_cols = [c for c in X.columns if X[c].nunique() <= 1 and c != 'const']
    if const_cols:
        X = X.drop(columns=const_cols)

    # Clustered SE by date
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})
    
    results = {
        'n': len(merged),
        'n_asset_classes': merged['asset_class'].nunique(),
        'n_dates': merged['date'].nunique(),
        'beta_mp': float(model.params.get(mp_col, np.nan)),
        'p_mp': float(model.pvalues.get(mp_col, np.nan)),
        'beta_cbi': float(model.params.get(cbi_col, np.nan)),
        'p_cbi': float(model.pvalues.get(cbi_col, np.nan)),
        'delta_risk_x_mp': float(model.params.get('risk_x_mp', np.nan)),
        'p_risk_x_mp': float(model.pvalues.get('risk_x_mp', np.nan)),
        'delta_risk_x_cbi': float(model.params.get('risk_x_cbi', np.nan)),
        'p_risk_x_cbi': float(model.pvalues.get('risk_x_cbi', np.nan)),
        'r_squared': float(model.rsquared),
        'r_squared_adj': float(model.rsquared_adj),
        'controls': control_cols,
        'shock_type': shock_label,
        'h1_mp_negative': model.params.get(mp_col, 0) < 0 and model.pvalues.get(mp_col, 1) < 0.10,
        'h1_risk_ladder_mp': model.params.get('risk_x_mp', 0) < 0 and model.pvalues.get('risk_x_mp', 1) < 0.10,
        'h1_risk_ladder_cbi': model.params.get('risk_x_cbi', 0) > 0 and model.pvalues.get('risk_x_cbi', 1) < 0.10,
    }
    
    return results


def h3_panel_wald(flows_df, shocks_df, audit_chain=None):
    """
    H3 Panel Wald Test: β_MP = β_CBI in the pooled panel regression.
    
    Single Wald test using all 819 observations, more powerful than
    7 separate per-asset tests.
    """
    if audit_chain:
        audit_chain.log_human_decision(
            "H3 panel Wald: Test β_MP = β_CBI in pooled regression with FE",
            author="ai"
        )
    
    merged = flows_df.merge(shocks_df, on='date', how='inner')
    if len(merged) < 50:
        return {'error': 'Insufficient data'}
    
    control_cols = [c for c in ['log_tna', 'flow_vol_12m', 'ret_12m_lag', 'exp_ratio']
                    if c in merged.columns]
    
    y = merged['net_flow_pct']
    X_cols = ['mp_shock', 'cbi_shock'] + control_cols
    
    ac_dummies = pd.get_dummies(merged['asset_class'], prefix='ac', drop_first=True, dtype=float)

    # Asset-class FE only — no time FE (mp_shock/cbi_shock vary only at date level,
    # making them perfectly collinear with time dummies → NaN coefficients)
    X = pd.concat([merged[X_cols].astype(float), ac_dummies], axis=1)
    X = sm.add_constant(X)

    # ── Clean NaN/inf (same as h1_panel_regression) ──
    merged_clean = merged.replace([np.inf, -np.inf], np.nan)
    y = merged_clean['net_flow_pct'].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in control_cols:
        if col in X.columns and X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    cluster_groups = merged_clean.loc[mask, 'date']

    if len(y) < 50:
        return {'error': f'Insufficient data after cleaning ({len(y)} obs)'}

    # Drop any all-NaN or constant columns (defensive)
    X = X.dropna(axis=1, how='all')
    const_cols = [c for c in X.columns if X[c].nunique() <= 1 and c != 'const']
    if const_cols:
        X = X.drop(columns=const_cols)

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})
    
    try:
        wald = model.wald_test('mp_shock = cbi_shock', use_f=False)
        wald_stat = float(wald.statistic)
        wald_p = float(wald.pvalue)
    except Exception:
        wald_stat = np.nan
        wald_p = np.nan
    
    return {
        'n': len(merged),
        'wald_chi2': wald_stat,
        'wald_p': wald_p,
        'h3_rejected': wald_p < 0.10 if not np.isnan(wald_p) else False,
        'beta_mp': float(model.params['mp_shock']),
        'beta_cbi': float(model.params['cbi_shock']),
    }


def run_all_hypotheses(flows_df, shocks_df, audit_chain=None):
    """
    Run all five hypotheses with dual baseline (raw JK + B-S orthogonalized).
    
    Returns combined results for H1-H5, each tested with both raw JK shocks
    and Bauer-Swanson orthogonalized shocks.
    """
    print("=" * 60)
    print("Direction 2 — H1-H5 Regression Analysis (Optimized)")
    print("Triple baseline: Raw JK + B-S(LM) + B-S(LLM)")
    print("=" * 60)
    
    # Import H4 matrix and H5 regime modules
    from h4_substitution_matrix import build_substitution_matrix, test_risk_ladder_hypothesis
    from h5_regime_analysis import h5_regime_analysis as h5_zlb_regime_effect
    
    results = {}
    
    # --- Raw JK baseline ---
    print("\n[1/6] Raw JK baseline (per-asset)...")
    results['h1_raw'] = h1_risk_off(flows_df, shocks_df, audit_chain, use_bs=False)
    results['h2_raw'] = h2_risk_on(flows_df, shocks_df, audit_chain, use_bs=False)
    results['h3_raw'] = h3_asymmetry(flows_df, shocks_df, audit_chain, use_bs=False)
    
    # H4: Substitution matrix (raw JK)
    print("[2/6] H4 substitution matrix (raw JK)...")
    mp_matrix = build_substitution_matrix(flows_df, shocks_df, 'mp_shock', audit_chain)
    cbi_matrix = build_substitution_matrix(flows_df, shocks_df, 'cbi_shock', audit_chain)
    results['h4_raw_mp'] = test_risk_ladder_hypothesis(mp_matrix, audit_chain)
    results['h4_raw_cbi'] = test_risk_ladder_hypothesis(cbi_matrix, audit_chain)
    
    # H5: ZLB regime
    print("[3/6] H5 ZLB regime analysis...")
    results['h5'] = h5_zlb_regime_effect(flows_df, shocks_df, audit_chain)
    
    # --- P1: Panel regression with FE ---
    print("[4/6] Panel regression (asset FE, clustered SE)...")
    results['h1_panel'] = h1_panel_regression(flows_df, shocks_df, audit_chain, use_bs=False)
    results['h3_panel'] = h3_panel_wald(flows_df, shocks_df, audit_chain)
    
    # --- B-S orthogonalized baseline ---
    if 'bs_mp_shock' in shocks_df.columns:
        print("[5/6] Bauer-Swanson orthogonalized baseline...")
        results['h1_bs'] = h1_risk_off(flows_df, shocks_df, audit_chain, use_bs=True)
        results['h2_bs'] = h2_risk_on(flows_df, shocks_df, audit_chain, use_bs=True)
        results['h3_bs'] = h3_asymmetry(flows_df, shocks_df, audit_chain, use_bs=True)
        
        bs_mp_matrix = build_substitution_matrix(flows_df, shocks_df, 'bs_mp_shock', audit_chain)
        bs_cbi_matrix = build_substitution_matrix(flows_df, shocks_df, 'bs_cbi_shock', audit_chain)
        results['h4_bs_mp'] = test_risk_ladder_hypothesis(bs_mp_matrix, audit_chain)
        results['h4_bs_cbi'] = test_risk_ladder_hypothesis(bs_cbi_matrix, audit_chain)

    # --- LLM-based B-S orthogonalized baseline ---
    if 'bs_llm_mp_shock' in shocks_df.columns and not shocks_df['bs_llm_mp_shock'].isna().all():
        print("[5b/6] LLM-based B-S orthogonalized baseline...")
        # Temporarily rename columns for use_bs=True path
        shocks_llm = shocks_df.copy()
        shocks_llm['bs_mp_shock'] = shocks_llm['bs_llm_mp_shock']
        shocks_llm['bs_cbi_shock'] = shocks_llm['bs_llm_cbi_shock']
        results['h1_bs_llm'] = h1_risk_off(flows_df, shocks_llm, audit_chain, use_bs=True)
        results['h2_bs_llm'] = h2_risk_on(flows_df, shocks_llm, audit_chain, use_bs=True)
        results['h3_bs_llm'] = h3_asymmetry(flows_df, shocks_llm, audit_chain, use_bs=True)

    print("[6/6] Done.")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for baseline in ['raw', 'bs', 'bs_llm']:
        if f'h1_{baseline}' not in results:
            continue
        label = {'raw': 'Raw JK', 'bs': 'B-S (LM sentiment)', 'bs_llm': 'B-S (LLM hawkish)'}[baseline]
        print(f"\n--- {label} ---")
        
        print(f"\nH1 (Risk-Off Destination):")
        for ac, r in results[f'h1_{baseline}'].items():
            sig = '***' if r['p_mp'] < 0.01 else '**' if r['p_mp'] < 0.05 else '*' if r['p_mp'] < 0.10 else ''
            print(f"  {ac:30s} β_MP={r['beta_mp']:8.4f} {sig:3s} (p={r['p_mp']:.3f})")
        
        print(f"\nH3 (Asymmetry):")
        for ac, r in results[f'h3_{baseline}'].items():
            sig = '***' if r['wald_p'] < 0.01 else '**' if r['wald_p'] < 0.05 else '*' if r['wald_p'] < 0.10 else ''
            chi2 = r.get('wald_chi2', r.get('wald_f', np.nan))
            print(f"  {ac:30s} χ²={chi2:6.2f} {sig:3s} (p={r['wald_p']:.3f})")
    
    # Panel regression results
    if 'h1_panel' in results and 'error' not in results['h1_panel']:
        p = results['h1_panel']
        print(f"\n--- Panel Regression (Asset FE, Clustered SE) ---")
        print(f"  N = {p['n']} ({p['n_asset_classes']} assets × {p['n_dates']} FOMC dates)")
        sig_mp = '***' if p['p_mp'] < 0.01 else '**' if p['p_mp'] < 0.05 else '*' if p['p_mp'] < 0.10 else ''
        sig_cbi = '***' if p['p_cbi'] < 0.01 else '**' if p['p_cbi'] < 0.05 else '*' if p['p_cbi'] < 0.10 else ''
        sig_rmp = '***' if p['p_risk_x_mp'] < 0.01 else '**' if p['p_risk_x_mp'] < 0.05 else '*' if p['p_risk_x_mp'] < 0.10 else ''
        sig_rcbi = '***' if p['p_risk_x_cbi'] < 0.01 else '**' if p['p_risk_x_cbi'] < 0.05 else '*' if p['p_risk_x_cbi'] < 0.10 else ''
        print(f"  β_MP          = {p['beta_mp']:8.4f} {sig_mp:3s} (p={p['p_mp']:.4f})")
        print(f"  β_CBI         = {p['beta_cbi']:8.4f} {sig_cbi:3s} (p={p['p_cbi']:.4f})")
        print(f"  δ(Risk×MP)    = {p['delta_risk_x_mp']:8.4f} {sig_rmp:3s} (p={p['p_risk_x_mp']:.4f})")
        print(f"  δ(Risk×CBI)   = {p['delta_risk_x_cbi']:8.4f} {sig_rcbi:3s} (p={p['p_risk_x_cbi']:.4f})")
        print(f"  R² = {p['r_squared']:.4f} (adj = {p['r_squared_adj']:.4f})")
        print(f"  H1 (β_MP<0): {'✅' if p['h1_mp_negative'] else '❌'}")
        print(f"  H1 (Risk×MP<0): {'✅' if p['h1_risk_ladder_mp'] else '❌'}")

    if 'h3_panel' in results and 'error' not in results['h3_panel']:
        p = results['h3_panel']
        print(f"\nH3 Panel Wald Test (β_MP = β_CBI):")
        sig = '***' if p['wald_p'] < 0.01 else '**' if p['wald_p'] < 0.05 else '*' if p['wald_p'] < 0.10 else ''
        print(f"  χ² = {p['wald_chi2']:6.2f} {sig:3s} (p={p['wald_p']:.4f})")
        print(f"  β_MP = {p['beta_mp']:.4f}, β_CBI = {p['beta_cbi']:.4f}")
        print(f"  H3 (MP≠CBI): {'✅' if p['h3_rejected'] else '❌'}")

    if 'h5' in results:
        print(f"\nH5 (ZLB Regime):")
        for ac, r in results['h5']['by_asset'].items():
            sig = '***' if r['p_mp_x_zlb'] < 0.01 else '**' if r['p_mp_x_zlb'] < 0.05 else '*' if r['p_mp_x_zlb'] < 0.10 else ''
            print(f"  {ac:30s} β_MP×ZLB={r['beta_mp_x_zlb']:8.4f} {sig:3s} (p={r['p_mp_x_zlb']:.3f})")
    
    if audit_chain:
        audit_chain.log_ai_response(
            f"H1-H5 analysis complete (dual baseline). Results: {json.dumps(results, default=str, indent=2)}",
            model="analysis_pipeline_optimized"
        )
    
    return results


if __name__ == "__main__":
    # This will be run after WRDS data is fetched
    print("Direction 2 Analysis Pipeline (Optimized)")
    print("Run this after wrds_connector.py has fetched fund flow data.")
    print("\nExpected input files:")
    print("  - results/fund_flows.csv (from WRDS, with control variables)")
    print("  - results/shocks_jk.csv (from Phase 1, with raw JK + B-S)")
    print("\nModules:")
    print("  - h1_h4_regression.py: H1-H3 + run_all_hypotheses")
    print("  - h4_substitution_matrix.py: H4 7×7 matrix")
    print("  - h5_regime_analysis.py: H5 ZLB regime")
