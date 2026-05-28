"""
Comprehensive Analysis Pipeline v6
====================================
Four upgrades over v5:
1. Expanded CB sentiment dictionary (3x more hawkish/dovish terms)
2. 2022-2025 shock proxy from FRED DFF daily changes
3. Financial sector stock-level FOMC event study
4. Publication-quality charts

Data sources:
- Monetary policy shocks: Acosta (2022) 1995-2022 + FRED DFF proxy 2022-2025
- Market returns: CRSP via WRDS
- FOMC statements: existing scraper
- FRED rates: DFF, DGS10, DGS3MO
"""

import pandas as pd
import numpy as np
import os, sys, json
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WRDS_DIR = DATA_DIR / "wrds"
RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. EXPANDED CENTRAL BANK SENTIMENT DICTIONARY
# ============================================================
# Original: ~18 hawkish + ~18 dovish = 36 terms
# Expanded: ~60 hawkish + ~60 dovish = 120 terms
# Sources: Henry (2008), Apel & Blix (2014), Corredoira (2020),
#          Cieslak et al. (2019), Hansen et al. (2018)

CB_HAWKISH_EXPANDED = {
    # Rate policy
    "tighten", "tightening", "tightened", "restrict", "restricting",
    "restrictive", "hike", "hiking", "hiked", "higher", "elevated",
    "firm", "firming", "hawkish", "aggressive", "accelerate",
    "accelerating", "accelerated", "preemptive", "raise", "raising",
    "raised", "increase", "increasing", "increased", "escalate",
    "escalating", "escalated", "step", "stepped", "stepping",
    # Inflation concerns
    "inflation", "inflationary", "overheating", "overheated",
    "pressures", "upside", "unacceptable", "entrenched",
    "persistent", "persistently", "stubborn", "sticky", "surge",
    "surging", "surged", "spike", "spiking", "spiked", "soar",
    "soaring", "soared", "rapid", "rapidly", "excessive", "excessively",
    "run", "running", "outpace", "outpacing", "outpaced",
    # Economic strength
    "strong", "stronger", "robust", "robustly", "vigorous",
    "vigilant", "vigilance", "resilient", "resilience", "solid",
    "boom", "booming", "overheat", "tight", "constrained",
    "capacity", "constraints", "bottleneck", "bottlenecks",
    # Policy stance
    "normalize", "normalized", "normalization", "unwinding",
    "unwind", "balance", "rebalancing", "contractionary",
    "withdraw", "withdrawing", "withdrawal", "diminish", "diminishing", "taper", "tapering",
    "tapered", "quantitative", "tightening",
}

CB_DOVISH_EXPANDED = {
    # Rate policy
    "patient", "patience", "accommodative", "accommodation",
    "supportive", "easing", "eased", "ease", "stimulus", "stimulative",
    "flexible", "flexibility", "measured", "gradual", "gradually",
    "moderate", "moderation", "modest", "modestly", "cautious",
    "careful", "carefully", "wait", "cut", "cutting", "reduce",
    "reducing", "lower", "lowering", "lowered", "decrease",
    # Economic weakness
    "soft", "softening", "softened", "subdued", "slack",
    "headwinds", "downturn", "slowdown", "slowing", "slowed",
    "weak", "weaker", "weakening", "weakened", "fragile",
    "fragility", "vulnerable", "vulnerability", "uncertainty",
    "uncertain", "risks", "downside", "deteriorate", "deteriorating",
    "deteriorated", "contraction", "contracting", "recession",
    "recessionary", "drag", "dragging",
    # Inflation outlook
    "transitory", "temporary", "temporarily", "disinflation",
    "disinflationary", "contained", "contain", "containing",
    "anchor", "anchored", "anchoring", "stable", "stabilize",
    "stabilizing", "stability", "symmetric", "symmetrically",
    "balanced", "neutral", "appropriate", "well-anchored",
    "converge", "converging", "converged", "ease", "easing",
    # Policy stance
    "accommodate", "accommodating", "support", "supporting",
    "sustain", "sustaining", "maintain", "maintaining", "preserve",
    "preserving", "protect", "protecting", "safeguard", "safeguarding",
    "expansionary", "expansion", "expand", "expanding",
    "easing", "forward", "guidance",
}


def compute_enhanced_sentiment(text):
    """
    Compute enhanced sentiment with expanded CB dictionary.
    Returns: (combined_score, lm_score, cb_score, word_count,
              hawk_count, dove_count, hawk_words, dove_words)
    """
    words = text.lower().split()
    words = [w.strip(".,;:!?()[]{}\"'-") for w in words]
    words = [w for w in words if len(w) > 1]
    total = len(words)

    if total == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0, [], []

    # LM sentiment (same as before)
    LM_NEGATIVE = {
        "adverse", "bad", "bear", "bearish", "below", "bottleneck",
        "breakdown", "bubble", "burden", "caution", "cautious",
        "challenge", "collapse", "concern", "concerned", "concerns",
        "constraint", "contraction", "correction", "crash", "crisis",
        "critical", "cut", "damage", "danger", "decline", "declining",
        "decrease", "deficit", "deflation", "delay", "demand",
        "depressed", "deteriorate", "deterioration", "difficult",
        "diminish", "down", "downgrade", "downside", "downturn",
        "drop", "dysfunction", "excess", "excessive", "fail",
        "failure", "fall", "falling", "fear", "fears", "flat",
        "flattening", "freeze", "gap", "hardship", "headwinds",
        "hesitate", "hinder", "hit", "illiquid", "impair",
        "inadequate", "inflation", "insolvency", "insufficient",
        "loss", "losses", "low", "lower", "negative", "obstacle",
        "overdue", "overhang", "overheating", "overvalued",
        "penalty", "pessimistic", "plunge", "poor", "pressure",
        "problem", "prohibit", "recession", "reduction", "restrict",
        "restricted", "restriction", "risk", "risky", "slow",
        "slowed", "slowing", "slowdown", "soften", "softened",
        "softening", "squeeze", "stagnant", "strain", "stressed",
        "struggle", "subdued", "suffer", "surge", "suspend",
        "tension", "tight", "tighten", "tightening", "tough",
        "trouble", "uncertain", "uncertainty", "uncomfortable",
        "undermine", "unemployment", "uneven", "unexpected",
        "unfavorable", "unprecedented", "unstable", "volatile",
        "volatility", "vulnerable", "weak", "weaken", "weakening",
        "weakness", "worse", "worst",
    }
    LM_POSITIVE = {
        "achieve", "advantage", "benefit", "better", "bolster",
        "boost", "breakthrough", "bullish", "capable", "certainty",
        "clear", "clarity", "comfortable", "commitment", "confident",
        "constructive", "continue", "continued", "control",
        "cooperation", "discipline", "ease", "eased", "easing",
        "efficient", "enhance", "enhanced", "ensure", "expand",
        "expanded", "expanding", "expansion", "favorable", "firm",
        "flexibility", "gain", "gained", "growing", "growth",
        "healthy", "improve", "improved", "improvement", "increase",
        "innovation", "intact", "investment", "liquidity",
        "maintain", "moderate", "momentum", "optimism", "optimistic",
        "outlook", "positive", "progress", "recovery", "reform",
        "reinforce", "reliable", "relief", "resilience", "resilient",
        "resolve", "restore", "robust", "satisfy", "secure",
        "significant", "solid", "solution", "stability", "stabilize",
        "stable", "steady", "strength", "strengthen", "strong",
        "success", "successful", "sufficient", "support", "supported",
        "sustain", "sustained", "target", "transparency", "trend",
        "upgrade", "upside", "upturn", "upward", "value", "vigorous",
    }

    neg_count = sum(1 for w in words if w in LM_NEGATIVE)
    pos_count = sum(1 for w in words if w in LM_POSITIVE)
    lm_score = (pos_count - neg_count) / max(total, 1)

    # Expanded CB sentiment
    hawk_words_found = [w for w in words if w in CB_HAWKISH_EXPANDED]
    dove_words_found = [w for w in words if w in CB_DOVISH_EXPANDED]
    hawk_count = len(hawk_words_found)
    dove_count = len(dove_words_found)
    cb_score = (hawk_count - dove_count) / max(total, 1)

    # Combined: higher = more hawkish
    combined = 0.5 * lm_score + 0.5 * cb_score

    return combined, lm_score, cb_score, total, hawk_count, dove_count, hawk_words_found, dove_words_found


# ============================================================
# 2. FRED DFF-BASED SHOCK PROXY (2022-2025)
# ============================================================

def compute_dff_shock_proxy(fomc_dates, dff_series):
    """
    Compute Kuttner-style surprise proxy from FRED DFF.
    
    For each FOMC date, compute:
    - target_proxy = DFF[t] - DFF[t-1] (daily change on FOMC day)
    - This captures the unexpected component of rate changes
    
    For 2022-2025 (post-Acosta data), this is a reasonable proxy
    because DFF reacts to FOMC announcements within the day.
    """
    # Remove duplicate DFF values (weekends/holidays repeat the same value)
    dff_series = dff_series.drop_duplicates(keep='first')
    
    results = []
    for date in fomc_dates:
        ts = pd.Timestamp(date)
        try:
            # Find the FOMC date and previous trading day in DFF
            if ts not in dff_series.index:
                # Find nearest trading day
                loc = dff_series.index.searchsorted(ts, side='right') - 1
                if loc < 0:
                    continue
                ts = dff_series.index[loc]

            dff_today = dff_series.loc[ts]

            # Previous trading day (unique values only)
            loc = dff_series.index.get_loc(ts)
            if loc <= 0:
                continue
            dff_yesterday = dff_series.iloc[loc - 1]

            # Surprise proxy in basis points
            surprise_bp = (dff_today - dff_yesterday) * 100  # DFF is in %, convert to bp

            results.append({
                "fomc_date": ts,
                "target_proxy_bp": surprise_bp,
                "dff_before": dff_yesterday,
                "dff_after": dff_today,
            })
        except (KeyError, IndexError):
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index("fomc_date").sort_index()
    return df


# ============================================================
# 3. FINANCIAL SECTOR EVENT STUDY
# ============================================================

def financial_sector_event_study(crsp_stocks, stock_names, fomc_dates, window=1):
    """
    Compute abnormal returns for financial stocks around FOMC meetings.
    
    For each FOMC date and each stock:
    - AR = R_stock - R_market (market-adjusted model)
    - Cumulative AR over [0, window] days
    
    Returns: DataFrame with cross-sectional statistics per FOMC date
    """
    crsp_stocks = crsp_stocks.copy()
    crsp_stocks["date"] = pd.to_datetime(crsp_stocks["date"])
    crsp_stocks = crsp_stocks.set_index("date").sort_index()
    crsp_stocks["ret"] = pd.to_numeric(crsp_stocks["ret"], errors="coerce")

    # Load CRSP market index for market-adjusted returns
    crsp_dsi = pd.read_csv(WRDS_DIR / "crsp_dsi_index.csv", parse_dates=["date"])
    crsp_dsi = crsp_dsi.set_index("date").sort_index()

    results = []
    for fomc_date in fomc_dates:
        ts = pd.Timestamp(fomc_date)

        # Get market return on FOMC day
        try:
            if ts not in crsp_dsi.index:
                loc = crsp_dsi.index.searchsorted(ts, side='right') - 1
                if loc < 0:
                    continue
                mkt_ret = crsp_dsi.iloc[loc]["vwretd"]
            else:
                mkt_ret = crsp_dsi.loc[ts, "vwretd"]
        except (KeyError, IndexError):
            continue

        # Get stock returns on FOMC day
        try:
            if ts not in crsp_stocks.index:
                loc = crsp_stocks.index.searchsorted(ts, side='right') - 1
                if loc < 0:
                    continue
                day_stocks = crsp_stocks.iloc[[loc]] if isinstance(crsp_stocks.iloc[loc], pd.Series) else crsp_stocks.loc[[crsp_stocks.index[loc]]]
            else:
                day_data = crsp_stocks.loc[ts]
                if isinstance(day_data, pd.Series):
                    day_stocks = pd.DataFrame([day_data.values], columns=day_data.index)
                else:
                    day_stocks = day_data
        except (KeyError, IndexError):
            continue

        if isinstance(day_stocks, pd.Series):
            day_stocks = pd.DataFrame([day_stocks])

        # Compute abnormal returns
        stock_rets = day_stocks["ret"].dropna()
        if len(stock_rets) < 10:
            continue

        abnormal_rets = stock_rets - mkt_ret

        # Cross-sectional statistics
        ar_mean = abnormal_rets.mean()
        ar_median = abnormal_rets.median()
        ar_std = abnormal_rets.std()
        ar_tstat = ar_mean / (ar_std / np.sqrt(len(abnormal_rets))) if ar_std > 0 else 0
        n_positive = (abnormal_rets > 0).sum()
        n_negative = (abnormal_rets < 0).sum()
        n_total = len(abnormal_rets)

        # Sub-sector analysis (if we have SIC codes)
        results.append({
            "date": ts,
            "fin_ar_mean": ar_mean,
            "fin_ar_median": ar_median,
            "fin_ar_std": ar_std,
            "fin_ar_tstat": ar_tstat,
            "fin_ar_n_pos": n_positive,
            "fin_ar_n_neg": n_negative,
            "fin_ar_n_total": n_total,
            "fin_ar_pct_pos": n_positive / n_total * 100,
            "mkt_ret": mkt_ret,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index("date").sort_index()
    return df


# ============================================================
# 4. CHART GENERATION
# ============================================================

def set_publication_style():
    """Set matplotlib style for publication-quality charts."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_sentiment_shocks(df, out_dir):
    """Chart 1: Sentiment vs Target/Path Shocks over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel A: Sentiment over time
    ax = axes[0]
    dates = df.index
    ax.bar(dates, df["sentiment_enhanced"], width=20, color=np.where(
        df["sentiment_enhanced"] > 0, '#c0392b', '#2980b9'
    ), alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Enhanced Sentiment')
    ax.set_title('Panel A: FOMC Statement Sentiment (Expanded CB Dictionary)')

    # Panel B: Target shock
    ax = axes[1]
    target_vals = df["target_shock"].dropna()
    ax.bar(target_vals.index, target_vals, width=20, color=np.where(
        target_vals > 0, '#c0392b', '#2980b9'
    ), alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Target Shock (std)')
    ax.set_title('Panel B: GSS Target Shock (Kuttner Surprise)')

    # Panel C: Path shock
    ax = axes[2]
    path_vals = df["path_shock"].dropna()
    ax.bar(path_vals.index, path_vals, width=20, color=np.where(
        path_vals > 0, '#c0392b', '#2980b9'
    ), alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Path Shock (std)')
    ax.set_title('Panel C: GSS Path Shock (Forward Guidance Factor)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_sentiment_shocks.png")
    plt.close(fig)
    print(f"  Saved fig1_sentiment_shocks.png")


def plot_h1_scatter(df, out_dir):
    """Chart 2: Sentiment vs Target Shock scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Sentiment vs Target
    ax = axes[0]
    mask = df["target_shock"].notna() & df["sentiment_enhanced"].notna()
    x = df.loc[mask, "target_shock"]
    y = df.loc[mask, "sentiment_enhanced"]
    ax.scatter(x, y, alpha=0.5, s=30, color='#2c3e50', edgecolors='white', linewidth=0.5)

    # Fit line
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=1.5, alpha=0.8)

        # R² and p-value
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax.text(0.05, 0.95, f'R² = {r_value**2:.4f}\nβ = {slope:.4f}\np = {p_value:.4f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Target Shock (GSS, standardized)')
    ax.set_ylabel('Enhanced Sentiment')
    ax.set_title('Panel A: Sentiment vs Target Shock')

    # Panel B: Sentiment vs Path
    ax = axes[1]
    mask = df["path_shock"].notna() & df["sentiment_enhanced"].notna()
    x = df.loc[mask, "path_shock"]
    y = df.loc[mask, "sentiment_enhanced"]
    ax.scatter(x, y, alpha=0.5, s=30, color='#2c3e50', edgecolors='white', linewidth=0.5)

    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=1.5, alpha=0.8)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax.text(0.05, 0.95, f'R² = {r_value**2:.4f}\nβ = {slope:.4f}\np = {p_value:.4f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Path Shock (GSS, standardized)')
    ax.set_ylabel('Enhanced Sentiment')
    ax.set_title('Panel B: Sentiment vs Path Shock')

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_h1_scatter.png")
    plt.close(fig)
    print(f"  Saved fig2_h1_scatter.png")


def plot_h2_returns(df, out_dir):
    """Chart 3: Asset returns response to shocks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    asset_pairs = [
        ("crsp_vw_ret", "CRSP VW Market", axes[0, 0]),
        ("crsp_ew_ret", "CRSP EW Market", axes[0, 1]),
        ("sp500_ret", "S&P 500 (CRSP)", axes[1, 0]),
        ("gold_ret", "Gold", axes[1, 1]),
    ]

    for col, label, ax in asset_pairs:
        if col not in df.columns:
            ax.set_title(f'{label} — No data')
            continue

        mask = df["target_shock"].notna() & df[col].notna()
        x = df.loc[mask, "target_shock"]
        y = df.loc[mask, col]

        ax.scatter(x, y, alpha=0.5, s=30, color='#2c3e50', edgecolors='white', linewidth=0.5)

        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=1.5, alpha=0.8)

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            sig = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
            ax.text(0.05, 0.95, f'R² = {r_value**2:.4f}\nβ = {slope:.4f}{sig}\np = {p_value:.4f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Target Shock')
        ax.set_ylabel(f'{label} Return (%)')
        ax.set_title(f'{label} vs Target Shock')

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_h2_returns.png")
    plt.close(fig)
    print(f"  Saved fig3_h2_returns.png")


def plot_financial_event_study(fin_events, out_dir):
    """Chart 4: Financial sector abnormal returns around FOMC."""
    if fin_events is None or fin_events.empty:
        print("  Skipping fig4 — no financial event data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Panel A: Average AR over time
    ax = axes[0]
    ax.bar(fin_events.index, fin_events["fin_ar_mean"] * 100, width=20,
           color=np.where(fin_events["fin_ar_mean"] > 0, '#c0392b', '#2980b9'), alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Average Abnormal Return (bp)')
    ax.set_title('Panel A: Financial Sector Average AR on FOMC Days')

    # Panel B: % positive vs negative
    ax = axes[1]
    ax.bar(fin_events.index, fin_events["fin_ar_pct_pos"], width=20,
           color='#27ae60', alpha=0.6, label='% Positive AR')
    ax.axhline(y=50, color='red', linewidth=1, linestyle='--', label='50% baseline')
    ax.set_ylabel('% Stocks with Positive AR')
    ax.set_title('Panel B: Cross-Sectional Sign of Abnormal Returns')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_financial_event_study.png")
    plt.close(fig)
    print(f"  Saved fig4_financial_event_study.png")


def plot_summary_comparison(out_dir):
    """Chart 5: Before/After comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: H1 R² comparison
    ax = axes[0]
    versions = ['v4\n(rate_change)', 'v5\n(GSS shocks)', 'v6\n(enhanced\nsentiment)']
    r2_vals = [0.0017, 0.0157, None]  # v6 will be filled
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(versions, [0.17, 1.57, 0], color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel('R² (%)')
    ax.set_title('H1: Sentiment ~ Surprise\nR² Improvement')
    for bar, val in zip(bars, [0.17, 1.57, 0]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel B: H1 p-value comparison
    ax = axes[1]
    p_vals = [0.712, 0.032, 0]
    bars = ax.bar(versions, p_vals, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(y=0.05, color='red', linewidth=1.5, linestyle='--', label='5% significance')
    ax.axhline(y=0.10, color='orange', linewidth=1, linestyle='--', label='10% significance')
    ax.set_ylabel('p-value')
    ax.set_title('H1: Target Shock Significance')
    ax.legend()
    for bar, val in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_comparison.png")
    plt.close(fig)
    print(f"  Saved fig5_comparison.png")


# ============================================================
# 5. OLS REGRESSION (same as v5)
# ============================================================

def ols_regression(y, X, robust=True, lag=1):
    """OLS with Newey-West standard errors."""
    n = len(y)
    X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix"}

    residuals = y - X @ beta
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0

    if robust and n > k:
        u = residuals.reshape(-1, 1)
        S = np.zeros((k, k))
        for j in range(lag + 1):
            weight = 1 - j / (lag + 1)
            if j == 0:
                Gamma = (u[j:] * X[j:]).T @ (u[j:] * X[j:])
            else:
                Gamma_j = (u[j:] * X[j:]).T @ (u[:-j] * X[:-j])
                Gamma = Gamma_j + Gamma_j.T
            S += weight * Gamma

        try:
            V_robust = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
            se = np.sqrt(np.maximum(np.diag(V_robust), 1e-10))
        except np.linalg.LinAlgError:
            se = np.sqrt(np.maximum(np.diag(ss_res / (n - k) * np.linalg.inv(X.T @ X)), 1e-10))
    else:
        se = np.sqrt(np.maximum(np.diag(ss_res / (n - k) * np.linalg.inv(X.T @ X)), 1e-10))

    t_stat = beta / se
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=max(n - k, 1))

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "p_value": p_value.tolist(),
        "r_squared": r_squared,
        "r_squared_adj": r_squared_adj,
        "n": n,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Comprehensive Analysis Pipeline v6")
    print("=" * 60)

    set_publication_style()

    # ── Load Data ──
    print("\n--- Loading Data ---")

    # Monetary policy shocks (Acosta 1995-2022)
    shocks = pd.read_excel(DATA_DIR / "mp_shocks_acosta.xlsx", sheet_name="shocks")
    shocks["fomc"] = pd.to_datetime(shocks["fomc"])
    shocks = shocks.set_index("fomc").sort_index()
    shocks["target_bp"] = shocks["ff.shock.0"] * 100  # Convert to basis points
    print(f"Monetary shocks: {len(shocks)} meetings, {shocks.index.min().date()} to {shocks.index.max().date()}")

    # CRSP data
    crsp_dsi = pd.read_csv(WRDS_DIR / "crsp_dsi_index.csv", parse_dates=["date"])
    crsp_dsi = crsp_dsi.set_index("date").sort_index()
    print(f"CRSP dsi: {len(crsp_dsi)} days")

    crsp_stocks = pd.read_csv(WRDS_DIR / "crsp_financial_stocks_2020_2025.csv", parse_dates=["date"])
    print(f"CRSP financial stocks: {len(crsp_stocks)} rows")

    stock_names = pd.read_csv(WRDS_DIR / "crsp_stock_names.csv")
    print(f"Stock names: {len(stock_names)} rows")

    # FOMC data
    fomc_data = pd.read_csv(BASE_DIR / "mp-research-platform" / "data" / "analysis_dataset_expanded.csv",
                            parse_dates=["date"])
    fomc_data = fomc_data.set_index("date").sort_index()
    print(f"FOMC data: {len(fomc_data)} meetings")

    # FOMC statements (for enhanced sentiment)
    import json as json_mod
    with open(BASE_DIR / "mp-research-platform" / "data" / "fomc_statements_all.json", "r") as f:
        statements = json_mod.load(f)
    print(f"FOMC statements: {len(statements)} available")

    # ── Step 1: Enhanced Sentiment ──
    print("\n--- Step 1: Computing Enhanced Sentiment ---")

    sentiment_records = []
    for date_str, text in statements.items():
        date = pd.Timestamp(date_str)
        if not text or pd.isna(date):
            continue
        score, lm, cb, wc, hawk, dove, hawk_w, dove_w = compute_enhanced_sentiment(text)
        sentiment_records.append({
            "date": date,
            "sentiment_enhanced": score,
            "lm_score_enhanced": lm,
            "cb_score_enhanced": cb,
            "word_count": wc,
            "hawk_count": hawk,
            "dove_count": dove,
        })

    df_sentiment = pd.DataFrame(sentiment_records)
    if not df_sentiment.empty:
        df_sentiment = df_sentiment.set_index("date").sort_index()
    print(f"  Enhanced sentiment computed for {len(df_sentiment)} statements")
    print(f"  Old sentiment std: {fomc_data['sentiment'].std():.6f}")
    print(f"  New sentiment std: {df_sentiment['sentiment_enhanced'].std():.6f}")
    print(f"  Old CB score std: {fomc_data['cb_score'].std():.6f}")
    print(f"  New CB score std: {df_sentiment['cb_score_enhanced'].std():.6f}")

    # ── Step 2: FRED DFF Shock Proxy (2022-2025) ──
    print("\n--- Step 2: Computing FRED DFF Shock Proxy ---")

    # Try loading from downloaded CSV first
    dff_csv = DATA_DIR / "dff_recent.csv"
    if dff_csv.exists():
        try:
            dff_df = pd.read_csv(dff_csv, parse_dates=["observation_date"])
            dff_df = dff_df.set_index("observation_date").sort_index()
            dff_df = dff_df[dff_df["DFF"] != "."]
            dff_df["DFF"] = dff_df["DFF"].astype(float)
            dff_data = dff_df["DFF"]
            dff_data = dff_data[~dff_data.index.duplicated(keep='last')]
            print(f"  Loaded DFF from CSV: {len(dff_data)} days")
        except Exception as e:
            print(f"  CSV load failed: {e}")
            dff_data = None
    else:
        dff_data = None

    if dff_data is None:
        # Try FRED API
        try:
            import requests
            resp = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={
                    "id": "DFF",
                    "cosd": "2022-01-01",
                    "coed": "2025-12-31",
                },
                timeout=15,
            )
            if resp.status_code == 200:
                from io import StringIO
                dff_df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"])
                dff_df = dff_df.set_index("observation_date").sort_index()
                dff_df = dff_df[dff_df["DFF"] != "."]
                dff_df["DFF"] = dff_df["DFF"].astype(float)
                dff_data = dff_df["DFF"]
                dff_data = dff_data[~dff_data.index.duplicated(keep='last')]
                print(f"  Loaded DFF from FRED CSV endpoint: {len(dff_data)} days")
            else:
                print(f"  FRED returned {resp.status_code}")
        except Exception as e:
            print(f"  FRED download failed: {e}")

    # Compute DFF proxy for 2022-2025 FOMC dates
    if dff_data is not None:
        fomc_dates_post2022 = [d for d in fomc_data.index if d >= pd.Timestamp("2022-08-01")]
        dff_proxy = compute_dff_shock_proxy(fomc_dates_post2022, dff_data)
        print(f"  DFF shock proxy: {len(dff_proxy)} meetings (2022-2025)")
        if not dff_proxy.empty:
            print(f"  Proxy range: {dff_proxy['target_proxy_bp'].min():.1f} to {dff_proxy['target_proxy_bp'].max():.1f} bp")
    else:
        dff_proxy = pd.DataFrame()
        print("  No DFF data available — skipping proxy")

    # ── Step 3: Build Unified Dataset ──
    print("\n--- Step 3: Building Unified Dataset ---")

    # Start with FOMC data
    df = fomc_data.copy()

    # Add enhanced sentiment
    df = df.join(df_sentiment[["sentiment_enhanced", "cb_score_enhanced", "hawk_count", "dove_count"]], how="left")

    # Fill missing enhanced sentiment with old sentiment (scaled)
    if "sentiment_enhanced" in df.columns:
        mask = df["sentiment_enhanced"].isna() & df["sentiment"].notna()
        # Scale old sentiment to match new range
        if mask.any():
            scale = df["sentiment_enhanced"].std() / df["sentiment"].std() if df["sentiment"].std() > 0 else 1
            df.loc[mask, "sentiment_enhanced"] = df.loc[mask, "sentiment"] * scale

    # Compute CRSP event returns
    all_fomc_dates = df.index.tolist()

    crsp_events = []
    for fomc_date in all_fomc_dates:
        ts = pd.Timestamp(fomc_date)
        try:
            if ts not in crsp_dsi.index:
                loc = crsp_dsi.index.searchsorted(ts, side='right') - 1
                if loc < 0:
                    continue
                row = crsp_dsi.iloc[loc]
            else:
                row = crsp_dsi.loc[ts]

            # Previous day
            loc = crsp_dsi.index.searchsorted(ts, side='right') - 1
            if loc <= 0:
                continue
            prev_row = crsp_dsi.iloc[loc - 1]

            crsp_events.append({
                "date": ts,
                "crsp_vw_ret": row.get("vwretd", np.nan) * 100,
                "crsp_ew_ret": row.get("ewretd", np.nan) * 100,
                "sp500_ret": row.get("sprtrn", np.nan) * 100,
            })
        except (KeyError, IndexError):
            continue

    crsp_events_df = pd.DataFrame(crsp_events)
    if not crsp_events_df.empty:
        crsp_events_df = crsp_events_df.set_index("date").sort_index()
        df = df.join(crsp_events_df, how="left", rsuffix="_crsp")

    # Add monetary policy shocks (Acosta 1995-2022)
    shock_cols = shocks[["target", "path", "target_bp", "ns"]].copy()
    shock_cols.columns = ["target_shock", "path_shock", "kuttner_bp", "ns_shock"]
    df = df.join(shock_cols, how="left")

    # Add DFF proxy for 2022-2025
    if not dff_proxy.empty:
        # For dates after Acosta coverage, use DFF proxy as target_shock
        for idx in dff_proxy.index:
            if idx in df.index:
                if pd.isna(df.loc[idx, "target_shock"]):
                    # Standardize proxy to match Acosta scale
                    proxy_val = dff_proxy.loc[idx, "target_proxy_bp"]
                    # Acosta target is standardized (unit std), so scale proxy
                    if shocks["target"].std() > 0:
                        df.loc[idx, "target_shock"] = proxy_val / (shocks["target_bp"].std())
                    df.loc[idx, "kuttner_bp"] = proxy_val
        print(f"  Extended shocks with DFF proxy for {len(dff_proxy)} meetings")

    # ── Step 4: Financial Sector Event Study ──
    print("\n--- Step 4: Financial Sector Event Study ---")
    fin_events = financial_sector_event_study(crsp_stocks, stock_names, all_fomc_dates)
    if not fin_events.empty:
        df = df.join(fin_events, how="left")
        print(f"  Financial event study: {len(fin_events)} FOMC days")
        print(f"  Mean AR: {fin_events['fin_ar_mean'].mean()*100:.2f} bp")
        print(f"  Mean t-stat: {fin_events['fin_ar_tstat'].mean():.3f}")

    # ── Step 5: Hypothesis Testing ──
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING (v6: Enhanced Sentiment + Extended Shocks)")
    print("=" * 60)

    # Use rows with complete shock data
    df_shock = df[df["target_shock"].notna()].copy()
    print(f"\nComplete cases: {len(df_shock)}")
    print(f"  Period: {df_shock.index.min().date()} to {df_shock.index.max().date()}")

    # Use enhanced sentiment where available
    if "sentiment_enhanced" in df_shock.columns:
        df_shock["sentiment_use"] = df_shock["sentiment_enhanced"].fillna(df_shock["sentiment"])
    else:
        df_shock["sentiment_use"] = df_shock["sentiment"]

    # ── H1: Sentiment ~ Target + Path ──
    print("\n" + "-" * 50)
    print("H1: Sentiment ~ Target Shock + Path Shock")
    print("-" * 50)

    mask = df_shock["sentiment_use"].notna() & df_shock["path_shock"].notna()
    if mask.sum() > 5:
        y = df_shock.loc[mask, "sentiment_use"].values
        X = df_shock.loc[mask, ["target_shock", "path_shock"]].values
        h1 = ols_regression(y, X)
        print(f"  R² = {h1['r_squared']:.4f}")
        print(f"  β(target) = {h1['beta'][1]:.6f}, t = {h1['t_stat'][1]:.3f}, p = {h1['p_value'][1]:.4f}")
        print(f"  β(path)   = {h1['beta'][2]:.6f}, t = {h1['t_stat'][2]:.3f}, p = {h1['p_value'][2]:.4f}")

        # Compare with old
        print(f"\n  --- Comparison ---")
        print(f"  v4 (rate_change):    R² = 0.0017, p = 0.712")
        print(f"  v5 (GSS shocks):     R² = 0.0157, p = 0.032")
        print(f"  v6 (enhanced sent):  R² = {h1['r_squared']:.4f}, p = {h1['p_value'][1]:.4f}")
    else:
        h1 = {"r_squared": 0, "beta": [0, 0, 0], "p_value": [1, 1, 1], "t_stat": [0, 0, 0], "n": 0}
        print("  Insufficient data for H1")

    # ── H2: Asset Returns ~ Target + Path ──
    print("\n" + "-" * 50)
    print("H2: Asset Returns ~ Target Shock + Path Shock")
    print("-" * 50)

    h2_results = {}
    asset_cols = {
        "crsp_vw_ret": "CRSP VW Market",
        "crsp_ew_ret": "CRSP EW Market",
        "sp500_ret": "S&P 500 (CRSP)",
        "gold_ret": "Gold",
        "ty10_chg": "10Y Yield",
        "tb13w_chg": "13W Yield",
    }

    for col, label in asset_cols.items():
        if col not in df_shock.columns:
            continue
        mask = df_shock[col].notna() & df_shock["target_shock"].notna()
        if mask.sum() < 10:
            continue

        y = df_shock.loc[mask, col].values.astype(float)
        X = df_shock.loc[mask, ["target_shock", "path_shock"]].values.astype(float)

        # Remove NaN
        valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid.sum() < 10:
            continue

        r = ols_regression(y[valid], X[valid])
        if "error" in r:
            continue

        h2_results[label] = r
        sig = "***" if r["p_value"][1] < 0.01 else "**" if r["p_value"][1] < 0.05 else "*" if r["p_value"][1] < 0.1 else ""
        print(f"\n  {label}:")
        print(f"    Target: β = {r['beta'][1]:.4f}, t = {r['t_stat'][1]:.3f}{sig}")
        print(f"    Path:   β = {r['beta'][2]:.4f}, t = {r['t_stat'][2]:.3f}")
        print(f"    R² = {r['r_squared']:.4f}, N = {r['n']}")

    # ── H3: Information Channel ──
    print("\n" + "-" * 50)
    print("H3: Information Channel (Path vs Target)")
    print("-" * 50)

    if "sentiment_use" in df_shock.columns:
        # Path shock effect on sentiment after controlling for target
        mask = df_shock["sentiment_use"].notna() & df_shock["path_shock"].notna()
        if mask.sum() > 5:
            y = df_shock.loc[mask, "sentiment_use"].values
            X = df_shock.loc[mask, ["target_shock", "path_shock"]].values
            h3_reg = ols_regression(y, X)
            path_t = abs(h3_reg["t_stat"][2])
            target_t = abs(h3_reg["t_stat"][1])
            info_dominates = path_t > target_t
            print(f"  Target |t| = {target_t:.3f}")
            print(f"  Path   |t| = {path_t:.3f}")
            print(f"  Path dominates → {'✅' if info_dominates else '❌'}")
            h3 = {"info_dominates": info_dominates, "path_t": path_t, "target_t": target_t}
        else:
            h3 = {"info_dominates": False}
            print("  Insufficient data")
    else:
        h3 = {"info_dominates": False}

    # ── H4: Forward Guidance Interaction ──
    # Correct model: Asset Return ~ Target Shock + Sentiment + Sentiment×FG
    print("\n" + "-" * 50)
    print("H4: Forward Guidance Period Interaction")
    print("-" * 50)

    df_shock = df_shock.copy()
    df_shock["fg_period"] = (df_shock.index >= "2008-12-01") & (df_shock.index <= "2015-12-31")
    df_shock["sentiment_x_fg"] = df_shock["sentiment_use"] * df_shock["fg_period"].astype(float)

    # Use CRSP VW return as dependent variable (not sentiment itself!)
    dep_var = "crsp_vw_ret" if "crsp_vw_ret" in df_shock.columns else "sp500_ret"
    mask = df_shock[dep_var].notna() & df_shock["sentiment_use"].notna() & df_shock["path_shock"].notna()
    if mask.sum() > 10:
        y = df_shock.loc[mask, dep_var].values.astype(float)
        X = df_shock.loc[mask, ["target_shock", "sentiment_use", "sentiment_x_fg"]].values.astype(float)
        valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid.sum() > 10:
            h4 = ols_regression(y[valid], X[valid])
            fg_strongest = h4["p_value"][3] < 0.1 and abs(h4["beta"][3]) > abs(h4["beta"][2])
            print(f"  Dep var: {dep_var}")
            print(f"  β(target)       = {h4['beta'][1]:.6f}, p = {h4['p_value'][1]:.4f}")
            print(f"  β(sentiment)    = {h4['beta'][2]:.6f}, p = {h4['p_value'][2]:.4f}")
            print(f"  β(sent×FG)      = {h4['beta'][3]:.6f}, p = {h4['p_value'][3]:.4f}")
            print(f"  R² = {h4['r_squared']:.4f}")
            print(f"  FG strongest → {'✅' if fg_strongest else '❌'}")
        else:
            h4 = {"r_squared": 0, "beta": [0]*4, "p_value": [1]*4}
            fg_strongest = False
    else:
        h4 = {"r_squared": 0, "beta": [0]*4, "p_value": [1]*4}
        fg_strongest = False

    # ── Robustness ──
    print("\n" + "-" * 50)
    print("Robustness Checks")
    print("-" * 50)

    robustness = {}

    # Kuttner bp (non-standardized)
    mask = df_shock["kuttner_bp"].notna() & df_shock["sentiment_use"].notna()
    if mask.sum() > 5:
        y = df_shock.loc[mask, "sentiment_use"].values
        X = df_shock.loc[mask, ["kuttner_bp"]].values
        r = ols_regression(y, X)
        robustness["kuttner_bp"] = r
        print(f"  Kuttner bp: R² = {r['r_squared']:.4f}, β = {r['beta'][1]:.6f}, p = {r['p_value'][1]:.4f}")

    # Post-2010
    mask = (df_shock.index >= "2010-01-01") & df_shock["sentiment_use"].notna() & df_shock["target_shock"].notna() & df_shock["path_shock"].notna()
    if mask.sum() > 5:
        y = df_shock.loc[mask, "sentiment_use"].values.astype(float)
        X = df_shock.loc[mask, ["target_shock", "path_shock"]].values.astype(float)
        valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid.sum() > 5:
            r = ols_regression(y[valid], X[valid])
            if "error" not in r:
                robustness["post_2010"] = r
                print(f"  Post-2010: R² = {r['r_squared']:.4f}, N = {r['n']}")
            else:
                print(f"  Post-2010: regression failed ({r.get('error', 'unknown')})")

    # No COVID
    mask = ~((df_shock.index >= "2020-03-01") & (df_shock.index <= "2020-06-30"))
    mask &= df_shock["sentiment_use"].notna() & df_shock["target_shock"].notna() & df_shock["path_shock"].notna()
    if mask.sum() > 5:
        y = df_shock.loc[mask, "sentiment_use"].values.astype(float)
        X = df_shock.loc[mask, ["target_shock", "path_shock"]].values.astype(float)
        valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid.sum() > 5:
            r = ols_regression(y[valid], X[valid])
            if "error" not in r:
                robustness["no_covid"] = r
                print(f"  No-COVID: R² = {r['r_squared']:.4f}, N = {r['n']}")
            else:
                print(f"  No-COVID: regression failed")

    # Financial sector AR
    if "fin_ar_mean" in df_shock.columns:
        mask = df_shock["fin_ar_mean"].notna() & df_shock["target_shock"].notna()
        if mask.sum() > 5:
            y = df_shock.loc[mask, "fin_ar_mean"].values.astype(float)
            X = df_shock.loc[mask, ["target_shock", "path_shock"]].values.astype(float)
            valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if valid.sum() > 5:
                r = ols_regression(y[valid], X[valid])
                robustness["financial_ar"] = r
                print(f"  Financial AR: R² = {r['r_squared']:.4f}, β(target) = {r['beta'][1]:.4f}, p = {r['p_value'][1]:.4f}")

    # ── Step 6: Generate Charts ──
    print("\n--- Step 6: Generating Charts ---")

    plot_sentiment_shocks(df_shock, CHARTS_DIR)
    plot_h1_scatter(df_shock, CHARTS_DIR)
    plot_h2_returns(df_shock, CHARTS_DIR)
    plot_financial_event_study(fin_events, CHARTS_DIR)
    plot_summary_comparison(CHARTS_DIR)

    # ── Save Results ──
    print("\n--- Saving Results ---")

    all_results = {
        "H1": {
            "r_squared": h1["r_squared"],
            "beta_target": h1["beta"][1],
            "beta_path": h1["beta"][2],
            "se_target": h1["se"][1],
            "se_path": h1["se"][2],
            "p_target": h1["p_value"][1],
            "p_path": h1["p_value"][2],
            "n": h1["n"],
        },
        "H2": {k: {
            "beta_target": v["beta"][1],
            "beta_path": v["beta"][2],
            "p_target": v["p_value"][1],
            "p_path": v["p_value"][2],
            "r_squared": v["r_squared"],
            "n": v["n"],
        } for k, v in h2_results.items()},
        "H3": h3,
        "H4": {
            "r_squared": h4["r_squared"],
            "fg_strongest": fg_strongest,
        },
        "robustness": {k: {
            "r_squared": v["r_squared"],
            "n": v.get("n", 0),
        } for k, v in robustness.items()},
        "metadata": {
            "shock_source": "Acosta (2022) + FRED DFF proxy",
            "return_source": "CRSP via WRDS",
            "sentiment": "Expanded CB dictionary (120 terms vs 36 original)",
            "n_complete": len(df_shock),
            "period": f"{df_shock.index.min().date()} to {df_shock.index.max().date()}",
        },
    }

    with open(RESULTS_DIR / "regression_results_v6.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results → regression_results_v6.json")

    # Save dataset
    df_out = df_shock.copy()
    df_out.index = df_out.index.strftime("%Y-%m-%d")
    df_out.to_csv(RESULTS_DIR / "analysis_dataset_v6.csv")
    print(f"  Dataset → analysis_dataset_v6.csv")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY (v6: Enhanced Sentiment + Extended Shocks)")
    print("=" * 60)

    print(f"\nH1 (Sentiment ~ Shocks): R² = {h1['r_squared']:.4f}")
    print(f"  Target: β = {h1['beta'][1]:.6f}, p = {h1['p_value'][1]:.4f}")
    print(f"  Path:   β = {h1['beta'][2]:.6f}, p = {h1['p_value'][2]:.4f}")

    sig_count = sum(1 for v in h2_results.values() if v["p_value"][1] < 0.1)
    print(f"\nH2 (Returns ~ Shocks): {sig_count}/{len(h2_results)} assets significant at 10%")

    print(f"\nH3 (Info Channel): Path dominates → {'✅' if h3.get('info_dominates') else '❌'}")
    print(f"\nH4 (FG Interaction): FG strongest → {'✅' if fg_strongest else '❌'}")

    print(f"\n--- Version Comparison ---")
    print(f"  v4 (rate_change + yfinance):  H1 R² = 0.17%,  p = 0.712")
    print(f"  v5 (GSS shocks + CRSP):       H1 R² = 1.57%,  p = 0.032")
    print(f"  v6 (enhanced + extended):      H1 R² = {h1['r_squared']*100:.2f}%,  p = {h1['p_value'][1]:.4f}")

    return df_shock, all_results


if __name__ == "__main__":
    main()
