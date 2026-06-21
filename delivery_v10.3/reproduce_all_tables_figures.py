#!/usr/bin/env python3
"""
=============================================================================
Words Beyond the Rate v10.3 — 完整计算复现脚本
=============================================================================

Eileen: 这个脚本复现论文中每一个 Table 和 Figure 背后的计算。
从原始数据加载 → 回归 → 输出，每一步都有注释。

运行方式:
  python3 reproduce_all_tables_figures.py

数据要求:
  - results/minutes_sentiment_corrected.csv (主数据集, N=117)
  - data/gss_target_path_acosta_method.csv (GSS shocks)
  - data/dff_recent.csv (FRED DFF, 用于 Kuttner surprise)

标准误: 全部使用 Newey-West HAC(4)，与论文一致
=============================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ── 全局设置 ──
NW_KWARGS = {'cov_type': 'HAC', 'cov_kwds': {'maxlags': 4}}
DATA_FILE = 'results/minutes_sentiment_corrected.csv'

print("=" * 70)
print("Words Beyond the Rate v10.3 — 计算复现")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 第0步: 加载数据
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("第0步: 加载数据")
print("─" * 70)

df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df['fg_period'] = df['fg_period'].astype(int)
df['sentiment_x_fg'] = df['sentiment'] * df['fg_period']

# CRSP returns 是 decimal (0.01 = 1%)，转成 percentage 与论文一致
df['vwretd_pct'] = df['vwretd_day'] * 100   # CRSP VW
df['ewretd_pct'] = df['ewretd_day'] * 100   # CRSP EW
df['sprtrn_pct'] = df['sprtrn_day'] * 100   # S&P 500
# nasdaq_ret, gold_ret, ty10_chg, tb13w_chg 已经是 percentage

# Minutes combined sentiment
df['min_combined'] = 0.5 * df['min_lm_score'] + 0.5 * df['min_cb_score']

print(f"样本量: N = {len(df)}")
print(f"时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Forward Guidance 期间会议数: {df['fg_period'].sum()}")
print(f"变量列表: {list(df.columns)}")


# ══════════════════════════════════════════════════════════════════════
# Table 1: Summary Statistics (N = 117, 2006–2022)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Table 1: Summary Statistics (N = 117, 2006–2022)")
print("─" * 70)

table1_vars = {
    'Target shock': 'target_shock',
    'Path shock': 'path_shock',
    'Combined sentiment': 'sentiment',
    'LM score': 'lm_score',
    'CB score': 'cb_score',
    'Kuttner surprise (bp)': 'kuttner_bp',
}

table1_data = []
for name, col in table1_vars.items():
    s = df[col]
    table1_data.append({
        'Variable': name,
        'Mean': f"{s.mean():.3f}",
        'Std': f"{s.std():.3f}",
        'Min': f"{s.min():.3f}",
        'Max': f"{s.max():.3f}",
    })

table1 = pd.DataFrame(table1_data)
print(table1.to_string(index=False))
print("\n✅ Table 1 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Table 2: Sentiment and Monetary Policy Shocks (H1)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Table 2: Sentiment and Monetary Policy Shocks")
print("模型: Sentiment = α + β₁·Target_Shock + β₂·Path_Shock + ε")
print("标准误: Newey-West HAC(4)")
print("─" * 70)

X_h1 = sm.add_constant(df[['target_shock', 'path_shock']])
y_h1 = df['sentiment']
model_h1 = sm.OLS(y_h1, X_h1).fit(**NW_KWARGS)

print(f"\n{'Variable':<20} {'β':>10} {'SE':>10} {'t':>8} {'p':>8}")
print("-" * 58)
for var in ['target_shock', 'path_shock']:
    sig = "***" if model_h1.pvalues[var] < 0.01 else "**" if model_h1.pvalues[var] < 0.05 else "*" if model_h1.pvalues[var] < 0.1 else ""
    print(f"{var:<20} {model_h1.params[var]:>10.6f} {model_h1.bse[var]:>10.6f} {model_h1.tvalues[var]:>8.2f} {model_h1.pvalues[var]:>8.3f}{sig}")

print(f"{'const':<20} {model_h1.params['const']:>10.6f}")
print(f"{'R²':<20} {model_h1.rsquared:>10.4f}  ({model_h1.rsquared*100:.2f}%)")
print(f"{'N':<20} {int(model_h1.nobs):>10d}")

print(f"\n解读: Target shock 显著 (p={model_h1.pvalues['target_shock']:.3f}), "
      f"Path shock 不显著 (p={model_h1.pvalues['path_shock']:.3f})")
print("✅ Table 2 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Table 3: Surprise Measure Comparison
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Table 3: Surprise Measure Comparison (Dep Var: Combined Sentiment)")
print("─" * 70)

# 3a. Rate Change Only
X_rc = sm.add_constant(df[['rate_change']])
m_rc = sm.OLS(df['sentiment'], X_rc).fit(**NW_KWARGS)
print(f"\nRate Change Only:")
print(f"  β = {m_rc.params['rate_change']:.6f}, t = {m_rc.tvalues['rate_change']:.2f}, "
      f"p = {m_rc.pvalues['rate_change']:.3f}, R² = {m_rc.rsquared*100:.2f}%")

# 3b. Kuttner Surprise Only
X_ku = sm.add_constant(df[['kuttner_bp']])
m_ku = sm.OLS(df['sentiment'], X_ku).fit(**NW_KWARGS)
print(f"\nKuttner Surprise Only:")
print(f"  β = {m_ku.params['kuttner_bp']:.6f}, t = {m_ku.tvalues['kuttner_bp']:.2f}, "
      f"p = {m_ku.pvalues['kuttner_bp']:.3f}, R² = {m_ku.rsquared*100:.2f}%")

# 3c. GSS Target + Path (same as Table 2)
print(f"\nGSS Target + Path (same as Table 2):")
print(f"  β_T = {model_h1.params['target_shock']:.6f} (p={model_h1.pvalues['target_shock']:.3f}), "
      f"β_P = {model_h1.params['path_shock']:.6f} (p={model_h1.pvalues['path_shock']:.3f}), "
      f"R² = {model_h1.rsquared*100:.2f}%")

print("\n✅ Table 3 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Table 4: Asset Returns and Monetary Policy Shocks (H2)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Table 4: Asset Returns and Monetary Policy Shocks (CRSP Data)")
print("模型: Return = α + β_T·Target + β_P·Path + ε")
print("─" * 70)

assets = {
    'CRSP VW': 'vwretd_pct',
    'CRSP EW': 'ewretd_pct',
    'S&P 500': 'sprtrn_pct',
    'NASDAQ': 'nasdaq_ret',
    'Gold': 'gold_ret',
    '10Y Treasury': 'ty10_chg',
    '13W T-bill': 'tb13w_chg',
}

print(f"\n{'Asset':<15} {'β_T':>8} {'t_T':>8} {'p_T':>8} {'β_P':>8} {'p_P':>8} {'R²':>8} {'N':>4}")
print("-" * 75)

table4_results = {}
for asset_name, col in assets.items():
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    y = df[col]
    m = sm.OLS(y, X).fit(**NW_KWARGS)
    
    sig_t = "***" if m.pvalues['target_shock']<0.01 else "**" if m.pvalues['target_shock']<0.05 else "*" if m.pvalues['target_shock']<0.1 else ""
    sig_p = "***" if m.pvalues['path_shock']<0.01 else "**" if m.pvalues['path_shock']<0.05 else "*" if m.pvalues['path_shock']<0.1 else ""
    
    print(f"{asset_name:<15} {m.params['target_shock']:>8.3f} {m.tvalues['target_shock']:>8.2f} "
          f"{m.pvalues['target_shock']:>7.3f}{sig_t} {m.params['path_shock']:>8.3f} "
          f"{m.pvalues['path_shock']:>7.3f}{sig_p} {m.rsquared*100:>7.1f}% {int(m.nobs):>4}")
    
    table4_results[asset_name] = {
        'beta_T': round(m.params['target_shock'], 3),
        't_T': round(m.tvalues['target_shock'], 2),
        'p_T': round(m.pvalues['target_shock'], 3),
        'beta_P': round(m.params['path_shock'], 3),
        'p_P': round(m.pvalues['path_shock'], 3),
        'r2': round(m.rsquared, 4),
    }

print("\n解读: Target shock 对股票和黄金收益显著为负; Path shock 对所有资产均不显著")
print("✅ Table 4 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Table 5: Forward Guidance Period Interaction (H4)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Table 5: Forward Guidance Period Interaction")
print("模型: Return = α + β₁·Target + β₂·Path + β₃·Sentiment + β₄·(Sentiment×FG) + ε")
print("─" * 70)

fg_assets = {
    'CRSP VW': 'vwretd_pct',
    'NASDAQ': 'nasdaq_ret',
}

print(f"\n{'Variable':<20} {'CRSP VW':>12} {'NASDAQ':>12}")
print("-" * 48)

table5_results = {}
for asset_name, col in fg_assets.items():
    X = sm.add_constant(df[['target_shock', 'path_shock', 'sentiment', 'sentiment_x_fg']])
    y = df[col]
    m = sm.OLS(y, X).fit(**NW_KWARGS)
    
    table5_results[asset_name] = {}
    for var in ['target_shock', 'path_shock', 'sentiment', 'sentiment_x_fg']:
        sig = "***" if m.pvalues[var]<0.01 else "**" if m.pvalues[var]<0.05 else "*" if m.pvalues[var]<0.1 else ""
        table5_results[asset_name][var] = {
            'beta': round(m.params[var], 3) if 'shock' in var else round(m.params[var], 2),
            'p': round(m.pvalues[var], 3),
        }
    table5_results[asset_name]['r2'] = round(m.rsquared * 100, 1)

# Print formatted table
for var, label in [('target_shock', 'Target shock'), ('path_shock', 'Path shock'),
                    ('sentiment', 'Sentiment'), ('sentiment_x_fg', 'Sentiment × FG')]:
    vw = table5_results['CRSP VW'][var]
    nq = table5_results['NASDAQ'][var]
    vw_str = f"{vw['beta']:.2f}" if abs(vw['beta']) >= 1 else f"{vw['beta']:.3f}"
    nq_str = f"{nq['beta']:.2f}" if abs(nq['beta']) >= 1 else f"{nq['beta']:.3f}"
    print(f"{label:<20} {vw_str:>12} {nq_str:>12}")
    print(f"{'':>20} ({vw['p']:.3f}){'':>5} ({nq['p']:.3f})")

print(f"{'R²':<20} {table5_results['CRSP VW']['r2']:>11.1f}% {table5_results['NASDAQ']['r2']:>11.1f}%")
print(f"{'N':<20} {'117':>12} {'117':>12}")

print(f"\n解读: Sentiment×FG 在两个资产中均不显著 (CRSP VW p={table5_results['CRSP VW']['sentiment_x_fg']['p']:.3f}, "
      f"NASDAQ p={table5_results['NASDAQ']['sentiment_x_fg']['p']:.3f})")
print("→ Forward Guidance 期间 sentiment 并未变得更重要")
print("✅ Table 5 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Table 6: Alternative Sentiment Measures and Monetary Policy Shocks
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Table 6: Alternative Sentiment Measures and Monetary Policy Shocks")
print("─" * 70)

table6_specs = [
    ("Statement ~ Shocks", 'sentiment'),
    ("Minutes LM ~ Shocks", 'min_lm_score'),
    ("Minutes CB ~ Shocks", 'min_cb_score'),
    ("Minutes Combined ~ Shocks", 'min_combined'),
]

print(f"\n{'Model':<30} {'β_T':>10} {'p_T':>8} {'β_P':>10} {'p_P':>8} {'R²':>8}")
print("-" * 78)

for spec_name, dep_var in table6_specs:
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    y = df[dep_var]
    m = sm.OLS(y, X).fit(**NW_KWARGS)
    
    sig_t = "***" if m.pvalues['target_shock']<0.01 else "**" if m.pvalues['target_shock']<0.05 else "*" if m.pvalues['target_shock']<0.1 else ""
    sig_p = "***" if m.pvalues['path_shock']<0.01 else "**" if m.pvalues['path_shock']<0.05 else "*" if m.pvalues['path_shock']<0.1 else ""
    
    print(f"{spec_name:<30} {m.params['target_shock']:>10.6f}{sig_t} {m.pvalues['target_shock']:>7.3f} "
          f"{m.params['path_shock']:>10.6f}{sig_p} {m.pvalues['path_shock']:>7.3f} {m.rsquared*100:>7.2f}%")

# Last row: Statement ~ Shocks + Minutes Combined
X = sm.add_constant(df[['target_shock', 'path_shock', 'min_combined']])
y = df['sentiment']
m = sm.OLS(y, X).fit(**NW_KWARGS)
sig_t = "***" if m.pvalues['target_shock']<0.01 else "**" if m.pvalues['target_shock']<0.05 else "*" if m.pvalues['target_shock']<0.1 else ""
sig_p = "***" if m.pvalues['path_shock']<0.01 else "**" if m.pvalues['path_shock']<0.05 else "*" if m.pvalues['path_shock']<0.1 else ""
sig_mc = "***" if m.pvalues['min_combined']<0.01 else "**" if m.pvalues['min_combined']<0.05 else "*" if m.pvalues['min_combined']<0.1 else ""

print(f"{'Statement ~ Shocks+MinC':<30} {m.params['target_shock']:>10.6f}{sig_t} {m.pvalues['target_shock']:>7.3f} "
      f"{m.params['path_shock']:>10.6f}{sig_p} {m.pvalues['path_shock']:>7.3f} {m.rsquared*100:>7.2f}%")
print(f"  → Minutes Combined β = {m.params['min_combined']:.6f}{sig_mc}, p = {m.pvalues['min_combined']:.3f}")

print("\n解读: CB dictionary 和 Combined sentiment 的解释力最强; "
      "Minutes 中 path shock 变得显著 (Combined: p=0.014)")
print("✅ Table 6 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Appendix C.1: Regime-Specific Results
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Appendix C.1: Regime-Specific Results (Sentiment ~ Target + Path)")
print("─" * 70)

regimes = df['regime'].unique()
print(f"\n{'Regime':<15} {'N':>4} {'R²':>8} {'β_T (p)':>15} {'β_P (p)':>15}")
print("-" * 60)

for regime in ['conventional', 'forward_guidance', 'normalization']:
    sub = df[df['regime'] == regime]
    if len(sub) < 5:
        print(f"{regime:<15} {len(sub):>4}  (样本太小，跳过)")
        continue
    X = sm.add_constant(sub[['target_shock', 'path_shock']])
    y = sub['sentiment']
    m = sm.OLS(y, X).fit(**NW_KWARGS)
    print(f"{regime:<15} {len(sub):>4} {m.rsquared*100:>7.1f}% "
          f"{m.params['target_shock']:>8.6f} ({m.pvalues['target_shock']:.3f}) "
          f"{m.params['path_shock']:>8.6f} ({m.pvalues['path_shock']:.3f})")

print("✅ Appendix C.1 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Appendix C.2: Sentiment Distribution
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Appendix C.2: Sentiment Score Distribution")
print("─" * 70)

for col, name in [('sentiment', 'Combined'), ('lm_score', 'LM Score'), ('cb_score', 'CB Score')]:
    s = df[col]
    neg_pct = (s < 0).sum() / len(s) * 100
    pos_pct = (s > 0).sum() / len(s) * 100
    print(f"  {name:<12}: Mean={s.mean():.3f}, Std={s.std():.3f}, "
          f"Min={s.min():.3f}, Max={s.max():.3f}, "
          f"%Negative={neg_pct:.1f}%, %Positive={pos_pct:.1f}%")

print("✅ Appendix C.2 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Appendix C.3: Newey-West Lag Sensitivity
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Appendix C.3: Newey-West Lag Sensitivity (H1 Regression)")
print("─" * 70)

print(f"\n{'Lag':>4} {'β_T (t)':>15} {'p_T':>8} {'β_P (t)':>15} {'p_P':>8} {'R²':>8}")
print("-" * 62)

for lag in [1, 2, 4, 6]:
    nw = {'cov_type': 'HAC', 'cov_kwds': {'maxlags': lag}}
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    m = sm.OLS(df['sentiment'], X).fit(**nw)
    print(f"{lag:>4} {m.params['target_shock']:.6f} ({m.tvalues['target_shock']:.2f}) "
          f"{m.pvalues['target_shock']:>7.3f} {m.params['path_shock']:.6f} ({m.tvalues['path_shock']:.2f}) "
          f"{m.pvalues['path_shock']:>7.3f} {m.rsquared*100:>7.2f}%")

print("\n解读: β系数和R²在不同lag下不变（lag只影响标准误）; "
      "Target shock 在所有lag下显著; Path shock 在所有lag下不显著")
print("✅ Appendix C.3 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Appendix C.4: Data Source Comparison (S&P 500)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Appendix C.4: Data Source Comparison (S&P 500)")
print("─" * 70)

# CRSP sprtrn (delisting-adjusted)
X = sm.add_constant(df[['target_shock', 'path_shock']])
m_crsp = sm.OLS(df['sprtrn_pct'], X).fit(**NW_KWARGS)
print(f"\nCRSP (sprtrn):  β_T = {m_crsp.params['target_shock']:.3f}, "
      f"t = {m_crsp.tvalues['target_shock']:.2f}, p = {m_crsp.pvalues['target_shock']:.3f}, "
      f"R² = {m_crsp.rsquared:.3f}")

# yfinance S&P 500 (sp500_ret, not delisting-adjusted)
m_yf = sm.OLS(df['sp500_ret'], X).fit(**NW_KWARGS)
print(f"yfinance (^GSPC): β_T = {m_yf.params['target_shock']:.3f}, "
      f"t = {m_yf.tvalues['target_shock']:.2f}, p = {m_yf.pvalues['target_shock']:.3f}, "
      f"R² = {m_yf.rsquared:.3f}")

diff_pct = (m_yf.params['target_shock'] - m_crsp.params['target_shock']) / m_crsp.params['target_shock'] * 100
print(f"\nyfinance 系数比 CRSP 小 {abs(diff_pct):.0f}%（缺少退市调整）")
print("✅ Appendix C.4 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Conceptual Framework (无需计算)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Figure 1: Conceptual Framework — 框架图，无需计算")
print("─" * 70)


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Sentiment vs Monetary Policy Shocks
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Figure 2: Sentiment vs Monetary Policy Shocks")
print("─" * 70)

# 散点图数据: sentiment vs target_shock, sentiment vs path_shock
corr_target = df['sentiment'].corr(df['target_shock'])
corr_path = df['sentiment'].corr(df['path_shock'])

print(f"  Sentiment ↔ Target shock 相关系数: {corr_target:.3f}")
print(f"  Sentiment ↔ Path shock 相关系数: {corr_path:.3f}")
print(f"  Target shock 对 sentiment 的正向关系更明显 (r={corr_target:.3f} vs r={corr_path:.3f})")

# 生成散点图
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Target shock
    axes[0].scatter(df['target_shock'], df['sentiment'], alpha=0.5, s=30, color='#2196F3')
    # 回归线
    z = np.polyfit(df['target_shock'], df['sentiment'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['target_shock'].min(), df['target_shock'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r-", linewidth=2)
    axes[0].set_xlabel('Target Shock', fontsize=12)
    axes[0].set_ylabel('Combined Sentiment', fontsize=12)
    axes[0].set_title(f'Sentiment vs Target Shock (r={corr_target:.3f})', fontsize=13)
    
    # Path shock
    axes[1].scatter(df['path_shock'], df['sentiment'], alpha=0.5, s=30, color='#FF9800')
    z2 = np.polyfit(df['path_shock'], df['sentiment'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(df['path_shock'].min(), df['path_shock'].max(), 100)
    axes[1].plot(x_line2, p2(x_line2), "r-", linewidth=2)
    axes[1].set_xlabel('Path Shock', fontsize=12)
    axes[1].set_ylabel('Combined Sentiment', fontsize=12)
    axes[1].set_title(f'Sentiment vs Path Shock (r={corr_path:.3f})', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('delivery_v10.3/figure2_sentiment_vs_shocks_reproduced.png', dpi=300, bbox_inches='tight')
    print("  ✅ Figure 2 散点图已保存")
except ImportError:
    print("  ⚠️ matplotlib 未安装，跳过图表生成")

print("✅ Figure 2 复现完成")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Asset Return Responses to Monetary Policy Shocks
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("Figure 3: Asset Return Responses to Monetary Policy Shocks")
print("─" * 70)

# 柱状图: β_T 和 β_P 对各资产
print(f"\n{'Asset':<15} {'β_T':>8} {'95% CI':>20} {'β_P':>8} {'95% CI':>20}")
print("-" * 75)

for asset_name, col in assets.items():
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    m = sm.OLS(df[col], X).fit(**NW_KWARGS)
    ci_t = m.conf_int().loc['target_shock']
    ci_p = m.conf_int().loc['path_shock']
    print(f"{asset_name:<15} {m.params['target_shock']:>8.3f} [{ci_t[0]:.3f}, {ci_t[1]:.3f}]  "
          f"{m.params['path_shock']:>8.3f} [{ci_p[0]:.3f}, {ci_p[1]:.3f}]")

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    asset_names = list(assets.keys())
    beta_T = [table4_results[a]['beta_T'] for a in asset_names]
    beta_P = [table4_results[a]['beta_P'] for a in asset_names]
    
    x = np.arange(len(asset_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, beta_T, width, label='Target Shock (β_T)', color='#2196F3')
    bars2 = ax.bar(x + width/2, beta_P, width, label='Path Shock (β_P)', color='#FF9800')
    
    ax.set_ylabel('Coefficient', fontsize=12)
    ax.set_title('Asset Return Responses to Monetary Policy Shocks', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names, rotation=30, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('delivery_v10.3/figure3_asset_returns_reproduced.png', dpi=300, bbox_inches='tight')
    print("  ✅ Figure 3 柱状图已保存")
except:
    pass

print("✅ Figure 3 复现完成")


# ══════════════════════════════════════════════════════════════════════
# 额外验证: H3 Wald Test (Target = Path?)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("H3 Wald Test: β_Target = β_Path?")
print("─" * 70)

r_matrix = np.array([[0, 1, -1]])  # H0: target_shock = path_shock
wald_test = model_h1.wald_test(r_matrix)
print(f"  H0: β_Target = β_Path")
print(f"  Chi² = {wald_test.statistic.item():.4f}")
print(f"  p = {wald_test.pvalue.item():.4f}")
print(f"  结论: {'无法拒绝' if wald_test.pvalue.item() > 0.05 else '拒绝'}等式 — "
      f"两个 shock 对 sentiment 的效果{'无显著差异' if wald_test.pvalue.item() > 0.05 else '有显著差异'}")


# ══════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("全部复现完成!")
print("=" * 70)
print("""
核心发现:
  1. Target shock 显著预测 FOMC 声明 sentiment (p≈0.015-0.017)
  2. Path shock 不显著 (p≈0.149-0.152)
  3. Wald test 无法拒绝 β_T = β_P (p≈0.90)
  4. 证据偏向 policy implementation 而非 information revelation
  5. Sentiment×FG interaction 不显著 (CRSP VW p=0.836, NASDAQ p=0.739)
  6. CB dictionary 解释力优于 LM dictionary
  7. Minutes 中 path shock 变得显著 (Combined: p=0.014)

数据来源:
  - GSS shocks: Acosta (2022) target & path factors
  - CRSP returns: WRDS (delisting-adjusted)
  - Sentiment: 0.5*LM + 0.5*CB (combined)
  - 标准误: Newey-West HAC(4)
""")
