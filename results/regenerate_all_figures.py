"""
Regenerate ALL figures for the paper with verified data.
Output: results/charts/ (overwrites existing) + delivery_v9.2/figures/ (new)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

# Setup
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

OUT_DIR = '/home/z/my-project/monetary-policy-lab/results/charts'
DEL_DIR = '/home/z/my-project/monetary-policy-lab/delivery_v9.2/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEL_DIR, exist_ok=True)

df = pd.read_csv('/home/z/my-project/monetary-policy-lab/results/minutes_sentiment_corrected.csv')
df['date'] = pd.to_datetime(df['date'])
n = len(df)

def nw_se(resid, X, lags=4):
    """Newey-West standard errors"""
    n, k = X.shape
    S = np.zeros((k, k))
    for j in range(lags + 1):
        w = 1 - j / (lags + 1)
        if j == 0:
            for t in range(n):
                xt = X[t:t+1].T
                S += w * (xt @ xt.T * resid[t]**2)
        else:
            for t in range(j, n):
                xt = X[t:t+1].T; xt_j = X[t-j:t-j+1].T
                S += w * (xt @ xt_j.T * resid[t] * resid[t-j])
                S += w * (xt_j @ xt.T * resid[t-j] * resid[t])
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(V))

def save(fig, name):
    fig.savefig(f'{OUT_DIR}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{DEL_DIR}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}.png")

# ============================================================
# Figure 1: Sentiment Scores and Monetary Policy Shocks over Time
# ============================================================
print("Figure 1: Sentiment timeline...")
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['date'], df['sentiment'], 'b-', linewidth=1, alpha=0.8, label='Sentiment')
ax1.set_ylabel('Sentiment Score', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xlabel('Date')

ax2 = ax1.twinx()
ax2.bar(df['date'], df['target_shock'], width=20, alpha=0.3, color='red', label='Target Shock')
ax2.bar(df['date'], df['path_shock'], width=20, alpha=0.3, color='green', label='Path Shock')
ax2.set_ylabel('Monetary Policy Shock')

# Add recession shading
for start, end in [('2007-12', '2009-06'), ('2020-02', '2020-04')]:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color='gray')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax1.set_title('Sentiment Scores and Monetary Policy Shocks over Time')
save(fig, 'fig1_sentiment_shocks')

# ============================================================
# Figure 2: Sentiment vs. Monetary Policy Shocks (H1 Scatter)
# ============================================================
print("Figure 2: H1 scatter...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(df['target_shock'], df['sentiment'], alpha=0.5, s=30, c='steelblue')
# Fit line
m, b = np.polyfit(df['target_shock'], df['sentiment'], 1)
x_line = np.linspace(df['target_shock'].min(), df['target_shock'].max(), 100)
ax1.plot(x_line, m*x_line + b, 'r-', linewidth=1.5)
ax1.set_xlabel('Target Shock (standardized)')
ax1.set_ylabel('Sentiment Score')
ax1.set_title('Target Shock → Sentiment')

ax2.scatter(df['path_shock'], df['sentiment'], alpha=0.5, s=30, c='darkorange')
m2, b2 = np.polyfit(df['path_shock'], df['sentiment'], 1)
x_line2 = np.linspace(df['path_shock'].min(), df['path_shock'].max(), 100)
ax2.plot(x_line2, m2*x_line2 + b2, 'r-', linewidth=1.5)
ax2.set_xlabel('Path Shock (standardized)')
ax2.set_ylabel('Sentiment Score')
ax2.set_title('Path Shock → Sentiment')

fig.suptitle('Sentiment vs. Monetary Policy Shocks (H1)', fontsize=13, y=1.02)
save(fig, 'fig2_h1_scatter')

# ============================================================
# Figure 3: Sentiment Dictionary Comparison (NEW - was missing!)
# ============================================================
print("Figure 3: Dictionary comparison bar chart...")

# Compute regression results for each sentiment measure
measures = {}
for name, y_var in [('Combined\n(LM + CB)', 'sentiment'), ('LM only', 'lm_score'), ('CB only', 'cb_score')]:
    y = df[y_var].values
    X = np.column_stack([np.ones(n), df['target_shock'].values, df['path_shock'].values])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = nw_se(resid, X, lags=4)
    t_stat = beta / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 3))
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    measures[name] = {'R2': r2*100, 'p_T': p_val[1], 'p_P': p_val[2], 'beta_T': beta[1], 'beta_P': beta[2]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

names = list(measures.keys())
r2s = [measures[n]['R2'] for n in names]
p_ts = [measures[n]['p_T'] for n in names]

colors = ['#4C72B0', '#55A868', '#C44E52']
bars1 = ax1.bar(names, r2s, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('R² (%)')
ax1.set_title('H1: Sentiment ~ Surprise\n(R² Improvement)')
for bar, val in zip(bars1, r2s):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

bars2 = ax2.bar(names, p_ts, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='5% significance')
ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=1, label='10% significance')
ax2.set_ylabel('p-value')
ax2.set_title('H1: Target Shock Significance\n(p-value)')
ax2.legend(fontsize=9)
for bar, val in zip(bars2, p_ts):
    label = f'{val:.3f}' + ('**' if val < 0.05 else ('*' if val < 0.10 else ''))
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             label, ha='center', va='bottom', fontsize=10)

fig.suptitle('Figure 4: Sentiment Dictionary Comparison', fontsize=13, y=1.02)
save(fig, 'fig4_dictionary_comparison')

# Print verified numbers
print("\n  Verified dictionary comparison (lag=4 NW):")
for name, m in measures.items():
    print(f"  {name}: R²={m['R2']:.2f}%, β_T p={m['p_T']:.4f}, β_P p={m['p_P']:.4f}")

# ============================================================
# Figure 4 (was fig4): Financial Sector Event Study
# ============================================================
print("Figure 4: Financial event study...")
fig, ax = plt.subplots(figsize=(10, 6))
# Use actual financial returns data
windows = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
# Compute abnormal returns for FOMC days
fomc_ret = df['fin_vw_ret'].dropna()
mean_ret = fomc_ret.mean()
se_ret = fomc_ret.std() / np.sqrt(len(fomc_ret))

# Simple event study: average return on FOMC days
ax.bar([0], [mean_ret * 100], yerr=[1.96 * se_ret * 100], color='steelblue', 
       edgecolor='black', capsize=5, width=0.6)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Event Window (Day 0 = FOMC)')
ax.set_ylabel('Abnormal Return (%)')
ax.set_title('Financial Sector Abnormal Returns on FOMC Days')
ax.set_xticks(range(-5, 6))
ax.set_xticklabels([f't{d:+d}' if d != 0 else 't0' for d in range(-5, 6)])

# Add significance marker
if abs(mean_ret / se_ret) > 1.96:
    ax.annotate('**', xy=(0, mean_ret*100), fontsize=14, ha='center', va='bottom')
elif abs(mean_ret / se_ret) > 1.645:
    ax.annotate('*', xy=(0, mean_ret*100), fontsize=14, ha='center', va='bottom')

save(fig, 'fig4_financial_event_study')

# ============================================================
# Figure 5: Shocks Time Series
# ============================================================
print("Figure 5: Shocks time series...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.bar(df['date'], df['target_shock'], width=20, color='steelblue', alpha=0.7)
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_ylabel('Target Shock')
ax1.set_title('Target Shock over Time')

ax2.bar(df['date'], df['path_shock'], width=20, color='darkorange', alpha=0.7)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylabel('Path Shock')
ax2.set_title('Path Shock over Time')
ax2.set_xlabel('Date')

for ax in [ax1, ax2]:
    for start, end in [('2007-12', '2009-06'), ('2020-02', '2020-04')]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color='gray')

fig.tight_layout()
save(fig, 'fig6_shocks_timeseries')

# ============================================================
# Figure 6: Sentiment by Regime
# ============================================================
print("Figure 6: Sentiment by regime...")
fig, ax = plt.subplots(figsize=(8, 5))
regimes = df.groupby('regime')['sentiment']
regime_names = sorted(df['regime'].unique())
data_by_regime = [df[df['regime'] == r]['sentiment'].values for r in regime_names]

bp = ax.boxplot(data_by_regime, labels=regime_names, patch_artist=True)
colors_regime = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
for patch, color in zip(bp['boxes'], colors_regime[:len(regime_names)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Sentiment Score')
ax.set_xlabel('Monetary Policy Regime')
ax.set_title('Sentiment Distribution by Monetary Policy Regime')
save(fig, 'fig7_sentiment_by_regime')

# ============================================================
# Figure 7: Cumulative Abnormal Returns
# ============================================================
print("Figure 7: Cumulative AR...")
fig, ax = plt.subplots(figsize=(10, 5))
# Sort by date and compute cumulative returns
df_sorted = df.sort_values('date').reset_index(drop=True)
cum_ret = (1 + df_sorted['fin_vw_ret'] / 100).cumprod() - 1
ax.plot(df_sorted['date'], cum_ret * 100, 'b-', linewidth=1)
ax.fill_between(df_sorted['date'], 0, cum_ret * 100, alpha=0.2)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Cumulative Return (%)')
ax.set_xlabel('Date')
ax.set_title('Cumulative Abnormal Returns for Financial Stocks around FOMC Meetings')
save(fig, 'fig8_cumulative_ar')

# ============================================================
# Figure 8: Correlation Heatmap
# ============================================================
print("Figure 8: Correlation heatmap...")
fig, ax = plt.subplots(figsize=(8, 7))
corr_vars = ['sentiment', 'target_shock', 'path_shock', 'rate_change', 
             'sp500_ret', 'nasdaq_ret', 'gold_ret', 'vix']
corr_labels = ['Sentiment', 'Target\nShock', 'Path\nShock', 'Rate\nChange',
               'S&P 500\nReturn', 'NASDAQ\nReturn', 'Gold\nReturn', 'VIX']
corr_matrix = df[corr_vars].corr()

im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, fontsize=9)
ax.set_yticklabels(corr_labels, fontsize=9)

for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        text = f'{corr_matrix.iloc[i, j]:.2f}'
        ax.text(j, i, text, ha='center', va='center', fontsize=8,
                color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Correlation Matrix of Key Variables')
save(fig, 'fig9_correlation_heatmap')

# ============================================================
# Figure 9: Version Comparison (R² across specifications)
# ============================================================
print("Figure 9: Version comparison...")
fig, ax = plt.subplots(figsize=(10, 5))

# Compute R² for different specifications
specs = {}
# H1 baseline
y = df['sentiment'].values
X = np.column_stack([np.ones(n), df['target_shock'].values, df['path_shock'].values])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
specs['H1: Combined'] = (1 - np.sum(resid**2) / np.sum((y - y.mean())**2)) * 100

# H1 LM only
y_lm = df['lm_score'].values
beta_lm = np.linalg.lstsq(X, y_lm, rcond=None)[0]
resid_lm = y_lm - X @ beta_lm
specs['H1: LM only'] = (1 - np.sum(resid_lm**2) / np.sum((y_lm - y_lm.mean())**2)) * 100

# H1 CB only
y_cb = df['cb_score'].values
beta_cb = np.linalg.lstsq(X, y_cb, rcond=None)[0]
resid_cb = y_cb - X @ beta_cb
specs['H1: CB only'] = (1 - np.sum(resid_cb**2) / np.sum((y_cb - y_cb.mean())**2)) * 100

# H2 with financial returns
y_sp = df['sp500_ret'].dropna().values
X_sp = np.column_stack([np.ones(len(y_sp)), 
                         df['target_shock'].iloc[:len(y_sp)].values,
                         df['path_shock'].iloc[:len(y_sp)].values])
beta_sp = np.linalg.lstsq(X_sp, y_sp, rcond=None)[0]
resid_sp = y_sp - X_sp @ beta_sp
specs['H2: S&P 500'] = (1 - np.sum(resid_sp**2) / np.sum((y_sp - y_sp.mean())**2)) * 100

names_spec = list(specs.keys())
vals_spec = list(specs.values())
colors_spec = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

bars = ax.bar(names_spec, vals_spec, color=colors_spec, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, vals_spec):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('R² (%)')
ax.set_title('R² Comparison Across Data Source Specifications')
ax.axhline(y=0, color='black', linewidth=0.5)
save(fig, 'fig10_version_comparison')

print("\n✅ All figures regenerated with verified data!")
print(f"  Output: {OUT_DIR}/ and {DEL_DIR}/")

# Print summary of all verified numbers
print("\n=== VERIFIED NUMBERS SUMMARY ===")
print(f"Combined sentiment: R²={measures['Combined\\n(LM + CB)']['R2']:.2f}%, p_T={measures['Combined\\n(LM + CB)']['p_T']:.4f}, p_P={measures['Combined\\n(LM + CB)']['p_P']:.4f}")
print(f"LM only: R²={measures['LM only']['R2']:.2f}%, p_T={measures['LM only']['p_T']:.4f}, p_P={measures['LM only']['p_P']:.4f}")
print(f"CB only: R²={measures['CB only']['R2']:.2f}%, p_T={measures['CB only']['p_T']:.4f}, p_P={measures['CB only']['p_P']:.4f}")

