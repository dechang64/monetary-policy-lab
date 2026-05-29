#!/usr/bin/env python3
"""
Enhanced analysis inspired by Chen et al. (2025), Gambacorta et al. (2024/2025), Yao & Chai (2025)

Four experiments:
1. Dual-equation regression: equity returns (expectations channel) + credit spreads (risk premium channel)
2. Forward-lookingness dimension: separate sentiment for forward-looking vs current-assessment language
3. Statement novelty weighting: edit distance from previous statement as proxy for information content
4. CB-only vs combined vs forward-looking sentiment comparison

Data: analysis_dataset_v6.csv + FRED credit spread data
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "/home/z/my-project/monetary-policy-lab/results/analysis_dataset_v6.csv"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "d43f3c8a1a5c48c5b2a7e9f0d1a2b3c4")
OUT_DIR = "/home/z/my-project/monetary-policy-lab/results"
FOMC_DIR = "/home/z/my-project/monetary-policy-lab/mp-research-platform/data"

# ── Load data ──
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
# Ensure numeric types
df['fg_period'] = df['fg_period'].astype(float)
df['sentiment_x_fg'] = df['sentiment_x_fg'].astype(float)
print(f"Dataset: {len(df)} meetings, {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# ── Helper: Newey-West OLS ──
def nw_ols(dep_var, indep_vars, data, lag=4):
    """OLS with Newey-West HAC standard errors"""
    y = data[dep_var].values
    X = data[indep_vars].values
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    
    results = {
        'n': int(model.nobs),
        'r2': round(model.rsquared * 100, 2),
        'r2_adj': round(model.rsquared_adj * 100, 2),
    }
    
    var_names = ['const'] + indep_vars
    for i, name in enumerate(var_names):
        results[name] = {
            'beta': round(float(model.params[i]), 6),
            'se': round(float(model.bse[i]), 4),
            't': round(float(model.tvalues[i]), 2),
            'p': round(float(model.pvalues[i]), 3),
        }
    
    # Wald test for target = path
    if 'target_shock' in indep_vars and 'path_shock' in indep_vars:
        ti = indep_vars.index('target_shock') + 1  # +1 for const
        pi = indep_vars.index('path_shock') + 1
        r_matrix = np.zeros((1, len(var_names)))
        r_matrix[0, ti] = 1
        r_matrix[0, pi] = -1
        wald = model.wald_test(r_matrix)
        results['wald_equal'] = {
            'F': round(float(wald.statistic), 2),
            'p': round(float(wald.pvalue), 3),
        }
    
    return results

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Dual-Equation Regression
# Equity returns (expectations channel) + Credit spreads (risk premium channel)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 1: Dual-Equation Regression")
print("="*70)

# We already have ty10_chg (10Y Treasury change) and tb13w_chg (13W T-bill change)
# Credit spread proxy: term_spread is already in the dataset
# Let's also compute BAA-AAA spread change from FRED if possible
# For now, use term_spread change and ty10_chg as the two channels

# Channel 1: Equity returns (expectations channel) - already in paper
# Channel 2: Credit/Term spread (risk premium channel)

# Compute term spread change
df['term_spread_chg'] = df['term_spread'].diff()
# Compute credit spread proxy: 10Y - 13W (already have term_spread which is 10Y-2Y approx)
# Use VIX change as risk premium proxy too
df['vix_chg'] = df['vix'].diff()

# Drop first row with NaN from diff
df_exp1 = df.dropna(subset=['term_spread_chg', 'vix_chg']).copy()

print("\n--- Channel 1: Equity Returns (Expectations Channel) ---")
eq1_equity = nw_ols('crsp_vw_ret', ['target_shock', 'path_shock'], df)
print(f"  CRSP VW: R²={eq1_equity['r2']}%, target p={eq1_equity['target_shock']['p']}, path p={eq1_equity['path_shock']['p']}")

eq1_sp500 = nw_ols('sp500_ret_crsp', ['target_shock', 'path_shock'], df)
print(f"  S&P 500: R²={eq1_sp500['r2']}%, target p={eq1_sp500['target_shock']['p']}, path p={eq1_sp500['path_shock']['p']}")

print("\n--- Channel 2: Bond Market / Risk Premium Channel ---")
eq1_ty10 = nw_ols('ty10_chg', ['target_shock', 'path_shock'], df)
print(f"  10Y Treasury: R²={eq1_ty10['r2']}%, target p={eq1_ty10['target_shock']['p']}, path p={eq1_ty10['path_shock']['p']}")

eq1_tb13w = nw_ols('tb13w_chg', ['target_shock', 'path_shock'], df)
print(f"  13W T-bill: R²={eq1_tb13w['r2']}%, target p={eq1_tb13w['target_shock']['p']}, path p={eq1_tb13w['path_shock']['p']}")

eq1_term = nw_ols('term_spread_chg', ['target_shock', 'path_shock'], df_exp1)
print(f"  Term spread chg: R²={eq1_term['r2']}%, target p={eq1_term['target_shock']['p']}, path p={eq1_term['path_shock']['p']}")

eq1_vix = nw_ols('vix_chg', ['target_shock', 'path_shock'], df_exp1)
print(f"  VIX change: R²={eq1_vix['r2']}%, target p={eq1_vix['target_shock']['p']}, path p={eq1_vix['path_shock']['p']}")

# Now add sentiment to the risk premium channel
print("\n--- Channel 2 + Sentiment ---")
eq1_ty10_s = nw_ols('ty10_chg', ['target_shock', 'path_shock', 'sentiment_use'], df)
print(f"  10Y Treasury + sentiment: R²={eq1_ty10_s['r2']}%, target p={eq1_ty10_s['target_shock']['p']}, path p={eq1_ty10_s['path_shock']['p']}, sent p={eq1_ty10_s['sentiment_use']['p']}")

eq1_term_s = nw_ols('term_spread_chg', ['target_shock', 'path_shock', 'sentiment_use'], df_exp1)
print(f"  Term spread chg + sentiment: R²={eq1_term_s['r2']}%, target p={eq1_term_s['target_shock']['p']}, path p={eq1_term_s['path_shock']['p']}, sent p={eq1_term_s['sentiment_use']['p']}")

eq1_vix_s = nw_ols('vix_chg', ['target_shock', 'path_shock', 'sentiment_use'], df_exp1)
print(f"  VIX change + sentiment: R²={eq1_vix_s['r2']}%, target p={eq1_vix_s['target_shock']['p']}, path p={eq1_vix_s['path_shock']['p']}, sent p={eq1_vix_s['sentiment_use']['p']}")

# KEY TEST: Forward guidance interaction on risk premium channel
print("\n--- FG Interaction on Risk Premium Channel ---")
eq1_ty10_fg = nw_ols('ty10_chg', ['target_shock', 'path_shock', 'sentiment_use', 'fg_period', 'sentiment_x_fg'], df)
print(f"  10Y Treasury + FG interaction: R²={eq1_ty10_fg['r2']}%")
print(f"    target p={eq1_ty10_fg['target_shock']['p']}, path p={eq1_ty10_fg['path_shock']['p']}")
print(f"    sentiment p={eq1_ty10_fg['sentiment_use']['p']}, sent×FG p={eq1_ty10_fg['sentiment_x_fg']['p']}")

eq1_vix_fg = nw_ols('vix_chg', ['target_shock', 'path_shock', 'sentiment_use', 'fg_period', 'sentiment_x_fg'], df_exp1)
print(f"  VIX change + FG interaction: R²={eq1_vix_fg['r2']}%")
print(f"    target p={eq1_vix_fg['target_shock']['p']}, path p={eq1_vix_fg['path_shock']['p']}")
print(f"    sentiment p={eq1_vix_fg['sentiment_use']['p']}, sent×FG p={eq1_vix_fg['sentiment_x_fg']['p']}")

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Forward-Lookingness Dimension
# Separate sentiment for forward-looking vs current-assessment language
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 2: Forward-Lookingness Dimension")
print("="*70)

# Forward-looking keywords
FL_HAWK = {'expect', 'anticipate', 'will', 'forecast', 'project', 'outlook', 
            'forward', 'future', 'ahead', 'upcoming', 'subsequent', 'prospective',
            'likely', 'anticipated', 'projected', 'foresee', 'envision'}
FL_DOVE = {'remain', 'continue', 'maintain', 'sustain', 'persist', 'ongoing',
           'sustained', 'accommodative', 'extended', 'prolonged'}

# Current-assessment keywords
CA_HAWK = {'inflation', 'prices', 'cost', 'wage', 'labor', 'employment', 'tight',
           'pressure', 'rising', 'increased', 'elevated', 'above', 'exceed'}
CA_DOVE = {'slack', 'weak', 'slow', 'decline', 'below', 'subdued', 'modest',
           'muted', 'soft', 'disappointing', 'headwind', 'fragile'}

# Load FOMC statements for text analysis
statements_path = os.path.join(FOMC_DIR, "fomc_statements_all.json")
if os.path.exists(statements_path):
    with open(statements_path, 'r') as f:
        statements = json.load(f)
    print(f"  Loaded {len(statements)} FOMC statements")
    
    # Build statement text lookup (dict: date_str -> text)
    stmt_lookup = {}
    for date_str, text in statements.items():
        if date_str and text:
            stmt_lookup[date_str] = text.lower()
    
    # Compute forward-looking and current-assessment sentiment
    fl_scores = []
    ca_scores = []
    fl_word_counts = []
    ca_word_counts = []
    
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        text = stmt_lookup.get(date_str, '')
        
        if text:
            words = text.split()
            word_set = set(words)
            
            # Forward-looking hawkish/dovish
            fl_hawk = sum(1 for w in words if w in FL_HAWK)
            fl_dove = sum(1 for w in words if w in FL_DOVE)
            fl_total = max(len(words), 1)
            fl_score = (fl_hawk - fl_dove) / fl_total
            fl_wc = fl_hawk + fl_dove
            
            # Current-assessment hawkish/dovish
            ca_hawk = sum(1 for w in words if w in CA_HAWK)
            ca_dove = sum(1 for w in words if w in CA_DOVE)
            ca_score = (ca_hawk - ca_dove) / fl_total
            ca_wc = ca_hawk + ca_dove
        else:
            fl_score = np.nan
            ca_score = np.nan
            fl_wc = 0
            ca_wc = 0
        
        fl_scores.append(fl_score)
        ca_scores.append(ca_score)
        fl_word_counts.append(fl_wc)
        ca_word_counts.append(ca_wc)
    
    df['fl_sentiment'] = fl_scores
    df['ca_sentiment'] = ca_scores
    df['fl_word_count'] = fl_word_counts
    df['ca_word_count'] = ca_word_counts
    
    df_fl = df.dropna(subset=['fl_sentiment', 'ca_sentiment']).copy()
    print(f"  Forward-looking sentiment: {len(df_fl)} meetings with text")
    print(f"  FL sentiment: mean={df_fl['fl_sentiment'].mean():.4f}, std={df_fl['fl_sentiment'].std():.4f}")
    print(f"  CA sentiment: mean={df_fl['ca_sentiment'].mean():.4f}, std={df_fl['ca_sentiment'].std():.4f}")
    
    # H1 with forward-looking sentiment
    print("\n--- H1: Sentiment ~ Target + Path (by dimension) ---")
    h1_combined = nw_ols('sentiment_use', ['target_shock', 'path_shock'], df_fl)
    print(f"  Combined sentiment: R²={h1_combined['r2']}%, target p={h1_combined['target_shock']['p']}, path p={h1_combined['path_shock']['p']}")
    
    h1_fl = nw_ols('fl_sentiment', ['target_shock', 'path_shock'], df_fl)
    print(f"  Forward-looking:    R²={h1_fl['r2']}%, target p={h1_fl['target_shock']['p']}, path p={h1_fl['path_shock']['p']}")
    
    h1_ca = nw_ols('ca_sentiment', ['target_shock', 'path_shock'], df_fl)
    print(f"  Current-assessment: R²={h1_ca['r2']}%, target p={h1_ca['target_shock']['p']}, path p={h1_ca['path_shock']['p']}")
    
    # H2 with forward-looking sentiment
    print("\n--- H2: Returns ~ Shocks + Sentiment (by dimension) ---")
    h2_combined = nw_ols('crsp_vw_ret', ['target_shock', 'path_shock', 'sentiment_use'], df_fl)
    print(f"  Combined sentiment: R²={h2_combined['r2']}%, sent p={h2_combined['sentiment_use']['p']}")
    
    h2_fl = nw_ols('crsp_vw_ret', ['target_shock', 'path_shock', 'fl_sentiment'], df_fl)
    print(f"  Forward-looking:    R²={h2_fl['r2']}%, FL sent p={h2_fl['fl_sentiment']['p']}")
    
    h2_ca = nw_ols('crsp_vw_ret', ['target_shock', 'path_shock', 'ca_sentiment'], df_fl)
    print(f"  Current-assessment: R²={h2_ca['r2']}%, CA sent p={h2_ca['ca_sentiment']['p']}")
    
    # KEY: path shock on forward-looking sentiment
    print("\n--- KEY: Path shock → Forward-looking sentiment ---")
    if h1_fl['path_shock']['p'] < 0.15:
        print(f"  ★ Path shock IS significant for FL sentiment (p={h1_fl['path_shock']['p']})!")
        print(f"    This supports the dimension mismatch hypothesis!")
    else:
        print(f"  Path shock not significant for FL sentiment (p={h1_fl['path_shock']['p']})")
    
else:
    print("  WARNING: FOMC statements not found, skipping Experiment 2")
    df_fl = df.copy()

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Statement Novelty Weighting
# Edit distance from previous statement as proxy for information content
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 3: Statement Novelty Weighting")
print("="*70)

if os.path.exists(statements_path):
    # Compute Jaccard similarity between consecutive statements
    novelties = []
    prev_words = None
    
    for _, row in df_fl.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        text = stmt_lookup.get(date_str, '')
        
        if text and prev_words:
            curr_words = set(text.split())
            # Jaccard distance = 1 - similarity
            intersection = len(curr_words & prev_words)
            union = len(curr_words | prev_words)
            jaccard_sim = intersection / max(union, 1)
            novelty = 1 - jaccard_sim  # higher = more different from previous
        else:
            novelty = np.nan
        
        novelties.append(novelty)
        if text:
            prev_words = set(text.split())
    
    df_fl['novelty'] = novelties
    df_nov = df_fl.dropna(subset=['novelty']).copy()
    
    print(f"  Novelty: mean={df_nov['novelty'].mean():.3f}, std={df_nov['novelty'].std():.3f}")
    print(f"  Novelty range: [{df_nov['novelty'].min():.3f}, {df_nov['novelty'].max():.3f}]")
    
    # Weighted OLS: weight by novelty (high novelty = more informative)
    # Normalize novelty to [0.5, 2.0] range for weights
    nov_min = df_nov['novelty'].min()
    nov_max = df_nov['novelty'].max()
    df_nov['novelty_weight'] = 0.5 + 1.5 * (df_nov['novelty'] - nov_min) / max(nov_max - nov_min, 0.001)
    
    print("\n--- H1: Sentiment ~ Shocks (novelty-weighted) ---")
    y = df_nov['sentiment_use'].values
    X = df_nov[['target_shock', 'path_shock']].values
    X = sm.add_constant(X)
    weights = df_nov['novelty_weight'].values
    
    model_wls = sm.WLS(y, X, weights=weights).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    print(f"  Novelty-weighted H1: R²={model_wls.rsquared*100:.2f}%")
    print(f"    target: beta={model_wls.params[1]:.6f}, t={model_wls.tvalues[1]:.2f}, p={model_wls.pvalues[1]:.3f}")
    print(f"    path:   beta={model_wls.params[2]:.6f}, t={model_wls.tvalues[2]:.2f}, p={model_wls.pvalues[2]:.3f}")
    
    # Compare with unweighted
    h1_unwt = nw_ols('sentiment_use', ['target_shock', 'path_shock'], df_nov)
    print(f"  Unweighted H1:      R²={h1_unwt['r2']}%, target p={h1_unwt['target_shock']['p']}, path p={h1_unwt['path_shock']['p']}")
    
    # H2 with novelty weighting
    print("\n--- H2: Returns ~ Shocks + Sentiment (novelty-weighted) ---")
    y2 = df_nov['crsp_vw_ret'].values
    X2 = df_nov[['target_shock', 'path_shock', 'sentiment_use']].values
    X2 = sm.add_constant(X2)
    
    model_wls2 = sm.WLS(y2, X2, weights=weights).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    print(f"  Novelty-weighted H2: R²={model_wls2.rsquared*100:.2f}%")
    print(f"    target: beta={model_wls2.params[1]:.4f}, t={model_wls2.tvalues[1]:.2f}, p={model_wls2.pvalues[1]:.3f}")
    print(f"    path:   beta={model_wls2.params[2]:.4f}, t={model_wls2.tvalues[2]:.2f}, p={model_wls2.pvalues[2]:.3f}")
    print(f"    sent:   beta={model_wls2.params[3]:.4f}, t={model_wls2.tvalues[3]:.2f}, p={model_wls2.pvalues[3]:.3f}")
    
    # Split sample: high novelty vs low novelty
    median_novelty = df_nov['novelty'].median()
    df_high_nov = df_nov[df_nov['novelty'] >= median_novelty].copy()
    df_low_nov = df_nov[df_nov['novelty'] < median_novelty].copy()
    
    print(f"\n--- Subsample: High Novelty (n={len(df_high_nov)}) vs Low Novelty (n={len(df_low_nov)}) ---")
    h1_high = nw_ols('sentiment_use', ['target_shock', 'path_shock'], df_high_nov)
    h1_low = nw_ols('sentiment_use', ['target_shock', 'path_shock'], df_low_nov)
    print(f"  High novelty: R²={h1_high['r2']}%, target p={h1_high['target_shock']['p']}, path p={h1_high['path_shock']['p']}")
    print(f"  Low novelty:  R²={h1_low['r2']}%, target p={h1_low['target_shock']['p']}, path p={h1_low['path_shock']['p']}")
    
    h2_high = nw_ols('crsp_vw_ret', ['target_shock', 'path_shock', 'sentiment_use'], df_high_nov)
    h2_low = nw_ols('crsp_vw_ret', ['target_shock', 'path_shock', 'sentiment_use'], df_low_nov)
    print(f"  High novelty H2: R²={h2_high['r2']}%, sent p={h2_high['sentiment_use']['p']}")
    print(f"  Low novelty H2:  R²={h2_low['r2']}%, sent p={h2_low['sentiment_use']['p']}")

else:
    print("  Skipping (no statement texts)")

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 4: CB-only vs Combined vs Forward-Looking Comparison
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 4: Sentiment Measure Comparison")
print("="*70)

# CB-only score
print("\n--- H1 with different sentiment measures ---")
h1_cb = nw_ols('cb_score', ['target_shock', 'path_shock'], df)
print(f"  CB score:       R²={h1_cb['r2']}%, target p={h1_cb['target_shock']['p']}, path p={h1_cb['path_shock']['p']}")

h1_lm = nw_ols('lm_score', ['target_shock', 'path_shock'], df)
print(f"  LM score:       R²={h1_lm['r2']}%, target p={h1_lm['target_shock']['p']}, path p={h1_lm['path_shock']['p']}")

h1_comb = nw_ols('sentiment_use', ['target_shock', 'path_shock'], df)
print(f"  Combined:       R²={h1_comb['r2']}%, target p={h1_comb['target_shock']['p']}, path p={h1_comb['path_shock']['p']}")

if 'fl_sentiment' in df_fl.columns:
    h1_fl2 = nw_ols('fl_sentiment', ['target_shock', 'path_shock'], df_fl)
    print(f"  Forward-looking: R²={h1_fl2['r2']}%, target p={h1_fl2['target_shock']['p']}, path p={h1_fl2['path_shock']['p']}")
    
    h1_ca2 = nw_ols('ca_sentiment', ['target_shock', 'path_shock'], df_fl)
    print(f"  Current-assess:  R²={h1_ca2['r2']}%, target p={h1_ca2['target_shock']['p']}, path p={h1_ca2['path_shock']['p']}")

# ════════════════════════════════════════════════════════════════
# Save all results
# ════════════════════════════════════════════════════════════════
all_results = {
    'experiment_1_dual_equation': {
        'equity_channel': {
            'crsp_vw': eq1_equity,
            'sp500': eq1_sp500,
        },
        'risk_premium_channel': {
            'ty10_chg': eq1_ty10,
            'tb13w_chg': eq1_tb13w,
            'term_spread_chg': eq1_term,
            'vix_chg': eq1_vix,
        },
        'risk_premium_with_sentiment': {
            'ty10_chg': eq1_ty10_s,
            'term_spread_chg': eq1_term_s,
            'vix_chg': eq1_vix_s,
        },
        'fg_interaction_risk_premium': {
            'ty10_chg': eq1_ty10_fg,
            'vix_chg': eq1_vix_fg,
        },
    },
    'experiment_4_sentiment_comparison': {
        'cb_score': h1_cb,
        'lm_score': h1_lm,
        'combined': h1_comb,
    },
}

if 'fl_sentiment' in df_fl.columns:
    all_results['experiment_2_forward_lookingness'] = {
        'h1_combined': h1_combined if 'h1_combined' in dir() else None,
        'h1_fl': h1_fl if 'h1_fl' in dir() else None,
        'h1_ca': h1_ca if 'h1_ca' in dir() else None,
        'h2_combined': h2_combined if 'h2_combined' in dir() else None,
        'h2_fl': h2_fl if 'h2_fl' in dir() else None,
        'h2_ca': h2_ca if 'h2_ca' in dir() else None,
    }
    all_results['experiment_4_sentiment_comparison']['forward_looking'] = h1_fl2 if 'h1_fl2' in dir() else None
    all_results['experiment_4_sentiment_comparison']['current_assessment'] = h1_ca2 if 'h1_ca2' in dir() else None

# Clean up None values recursively
def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if v is not None}
    return d

all_results = clean_dict(all_results)

out_path = os.path.join(OUT_DIR, "enhanced_analysis_results.json")
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")

# ════════════════════════════════════════════════════════════════
# Summary of key findings
# ════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY OF KEY FINDINGS")
print("="*70)

print("\n1. DUAL-EQUATION: Risk Premium Channel")
print(f"   10Y Treasury: target p={eq1_ty10['target_shock']['p']}, path p={eq1_ty10['path_shock']['p']}, R²={eq1_ty10['r2']}%")
print(f"   VIX change:   target p={eq1_vix['target_shock']['p']}, path p={eq1_vix['path_shock']['p']}, R²={eq1_vix['r2']}%")
print(f"   FG×sentiment on VIX: p={eq1_vix_fg['sentiment_x_fg']['p']}")

if 'h1_fl' in dir():
    print("\n2. FORWARD-LOOKINGNESS: Dimension Mismatch")
    print(f"   Combined sentiment: R²={h1_combined['r2']}%, path p={h1_combined['path_shock']['p']}")
    print(f"   Forward-looking:    R²={h1_fl['r2']}%, path p={h1_fl['path_shock']['p']}")
    print(f"   Current-assessment: R²={h1_ca['r2']}%, path p={h1_ca['path_shock']['p']}")
    
    if h1_fl['path_shock']['p'] < h1_combined['path_shock']['p']:
        print("   ★ Path shock MORE significant for forward-looking sentiment!")
    if h1_fl['r2'] > h1_combined['r2']:
        print("   ★ Forward-looking sentiment has HIGHER R²!")

if 'df_nov' in dir():
    print("\n3. STATEMENT NOVELTY")
    print(f"   Novelty-weighted H1 R²: {model_wls.rsquared*100:.2f}% (vs unweighted {h1_unwt['r2']}%)")
    print(f"   High novelty subsample R²: {h1_high['r2']}% (vs low {h1_low['r2']}%)")
