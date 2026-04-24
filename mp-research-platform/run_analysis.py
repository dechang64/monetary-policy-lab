"""
Monetary Policy Research Lab - Real Data Analysis Pipeline v3
Uses yfinance for asset prices, LM dictionary for sentiment, scipy for regressions.
"""
import pandas as pd
import numpy as np
import os, sys, json
import yfinance as yf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.fomc_meetings import get_fomc_data
from data.fomc_statements import get_statements
from data.sentiment import compute_lm_sentiment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def find_loc(idx, date):
    """Find row index for date, fallback to nearest."""
    ts = pd.Timestamp(date)
    try:
        return idx.get_loc(ts)
    except (KeyError, TypeError):
        pos = idx.searchsorted(ts, side='right') - 1
        return max(0, min(pos, len(idx) - 1))


def get_val(df, date, col):
    """Get value from df for date."""
    loc = find_loc(df.index, date)
    v = df.iloc[loc][col]
    return v if not pd.isna(v) else np.nan


def download_prices():
    """Download asset prices from Yahoo Finance."""
    tickers = {
        "^GSPC": "sp500", "^IXIC": "nasdaq", "^VIX": "vix",
        "^TNX": "ty10", "^IRX": "tb13w", "GC=F": "gold",
    }
    data = {}
    for tk, name in tickers.items():
        try:
            df = yf.download(tk, start="1994-01-01", end="2025-12-31", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            print(f"  ✅ {tk} ({name}): {len(df)} obs")
        except Exception as e:
            print(f"  ❌ {tk}: {e}")
    
    pdf = pd.DataFrame(data)
    pdf.index = pd.to_datetime(pdf.index)
    pdf = pdf.sort_index()
    pdf.to_csv(os.path.join(DATA_DIR, "yfinance_prices.csv"))
    return pdf


def compute_returns(pdf, fomc):
    """Compute event-window returns [day0, day+1] for each asset."""
    assets = {"sp500": "sp500_ret", "nasdaq": "nasdaq_ret", "gold": "gold_ret"}
    yields = {"ty10": "ty10_chg", "tb13w": "tb13w_chg"}
    
    for code, col in {**assets, **yields}.items():
        if code not in pdf.columns:
            fomc[col] = np.nan
            continue
        vals = []
        for _, row in fomc.iterrows():
            loc = find_loc(pdf.index, row["date"])
            v0 = pdf.iloc[loc][code]
            v1 = pdf.iloc[min(loc + 1, len(pdf) - 1)][code]
            if pd.isna(v0) or pd.isna(v1):
                vals.append(np.nan)
            elif code in yields:
                vals.append(v1 - v0)
            else:
                vals.append((v1 - v0) / v0 * 100)
        fomc[col] = vals
    
    # Controls
    fomc["vix"] = [get_val(pdf, r["date"], "vix") for _, r in fomc.iterrows()]
    fomc["term_spread"] = [
        get_val(pdf, r["date"], "ty10") - get_val(pdf, r["date"], "tb13w")
        for _, r in fomc.iterrows()
    ]
    return fomc


def run_h1(df):
    """H1: Sentiment vs Surprise correlation."""
    valid = df.dropna(subset=["sentiment", "surprise"])
    if len(valid) < 20:
        return {"r_squared": 0, "beta": 0, "se": 0, "p_value": 1, "n": len(valid)}
    slope, intercept, r, p, se = stats.linregress(valid["surprise"], valid["sentiment"])
    return {"r_squared": r**2, "beta": slope, "se": se, "p_value": p, "n": len(valid)}


def run_h2(df):
    """H2: Incremental predictive power of sentiment."""
    assets = ["sp500_ret", "nasdaq_ret", "gold_ret", "ty10_chg", "tb13w_chg"]
    results = {}
    for asset in assets:
        valid = df.dropna(subset=[asset, "surprise", "sentiment"])
        if len(valid) < 20:
            continue
        
        Y = valid[asset].values
        X1 = valid["surprise"].values
        X2 = valid["sentiment"].values
        
        # Model 1: Y ~ surprise
        X_m1 = np.column_stack([np.ones(len(Y)), X1])
        beta1 = np.linalg.lstsq(X_m1, Y, rcond=None)[0]
        Y_pred1 = X_m1 @ beta1
        ss_res1 = np.sum((Y - Y_pred1)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2_1 = 1 - ss_res1 / ss_tot if ss_tot > 0 else 0
        
        # Model 2: Y ~ surprise + sentiment
        X_m2 = np.column_stack([np.ones(len(Y)), X1, X2])
        beta2 = np.linalg.lstsq(X_m2, Y, rcond=None)[0]
        Y_pred2 = X_m2 @ beta2
        ss_res2 = np.sum((Y - Y_pred2)**2)
        r2_2 = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
        
        # SE for beta2 (sentiment coefficient)
        resid = Y - Y_pred2
        n, k = len(Y), 3
        mse = ss_res2 / (n - k)
        try:
            cov = mse * np.linalg.inv(X_m2.T @ X_m2)
            se_b2 = np.sqrt(cov[2, 2])
        except:
            se_b2 = np.nan
        
        t_stat = beta2[2] / se_b2 if se_b2 > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - k)) if not np.isnan(se_b2) else 1
        
        results[asset] = {
            "beta2": beta2[2], "se": se_b2, "t_stat": t_stat, "p_value": p_val,
            "r2_1": r2_1, "r2_2": r2_2, "inc_r2": r2_2 - r2_1,
            "significant_10": p_val < 0.10, "significant_05": p_val < 0.05,
            "n": n,
        }
    return results


def run_h3(df):
    """H3: Sentiment decomposition (policy vs information proxy)."""
    valid = df.dropna(subset=["sentiment", "surprise"])
    if len(valid) < 20:
        return {"policy_loading": 0, "info_loading": 0, "info_dominates": False}
    
    # Proxy: policy shock = surprise, info shock = sentiment residual from H1
    Y = valid["sentiment"].values
    X = valid["surprise"].values
    slope, intercept, r, p, se = stats.linregress(X, Y)
    residual = Y - (slope * X + intercept)
    
    # Correlation of sentiment with surprise (policy proxy)
    policy_loading = slope
    # Correlation of sentiment residual with surprise (info proxy = orthogonal part)
    info_loading = np.std(residual) / np.std(Y) if np.std(Y) > 0 else 0
    
    return {
        "policy_loading": policy_loading,
        "info_loading": info_loading,
        "info_dominates": info_loading > abs(policy_loading),
        "r_squared": r**2,
    }


def run_h4(df):
    """H4: Regime-dependent effects."""
    results = {}
    for regime in ["conventional", "forward_guidance", "normalization"]:
        sub = df[df["regime"] == regime].dropna(subset=["sp500_ret", "surprise", "sentiment"])
        if len(sub) < 15:
            results[regime] = {"abs_beta2": np.nan, "n": len(sub)}
            continue
        
        Y = sub["sp500_ret"].values
        X = np.column_stack([np.ones(len(Y)), sub["surprise"].values, sub["sentiment"].values])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        results[regime] = {"abs_beta2": abs(beta[2]), "beta2": beta[2], "n": len(sub)}
    
    # Check if FG is strongest
    conv = results.get("conventional", {}).get("abs_beta2", 0)
    fg = results.get("forward_guidance", {}).get("abs_beta2", 0)
    norm = results.get("normalization", {}).get("abs_beta2", 0)
    results["fg_strongest"] = fg > conv and fg > norm
    
    return results


def main():
    print("=" * 60)
    print("Monetary Policy Research Lab - Real Data Analysis v3")
    print("=" * 60)
    
    # 1. FOMC meetings
    print("\n[1/5] Loading FOMC meetings...")
    fomc = get_fomc_data()
    print(f"  {len(fomc)} meetings loaded")
    
    # 2. Sentiment
    print("\n[2/5] Computing sentiment scores...")
    stmts = get_statements()
    fomc["date_str"] = fomc["date"].dt.strftime("%Y-%m-%d")
    stmts["date_str"] = stmts["date"].dt.strftime("%Y-%m-%d")
    merged = fomc.merge(stmts[["date_str", "statement"]], on="date_str", how="inner")
    merged["date"] = pd.to_datetime(merged["date_str"])
    
    scores = []
    for _, row in merged.iterrows():
        combined, lm, cb, n = compute_lm_sentiment(row["statement"])
        scores.append(combined)
    merged["sentiment"] = scores
    print(f"  Matched: {len(merged)} meetings")
    print(f"  Sentiment: mean={merged['sentiment'].mean():.4f}, std={merged['sentiment'].std():.4f}")
    
    # 3. Asset prices
    print("\n[3/5] Downloading asset prices...")
    csv_path = os.path.join(DATA_DIR, "yfinance_prices.csv")
    if os.path.exists(csv_path):
        pdf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"  Loaded from cache: {len(pdf)} rows")
    else:
        pdf = download_prices()
    
    # 4. Event returns
    print("\n[4/5] Computing event returns...")
    merged["surprise"] = merged["rate_change"]
    merged = compute_returns(pdf, merged)
    
    # Clean
    ret_cols = ["sp500_ret", "nasdaq_ret", "gold_ret", "ty10_chg", "tb13w_chg"]
    clean = merged.dropna(subset=["sentiment", "surprise"] + ret_cols[:1])
    print(f"  Clean observations: {len(clean)}")
    print(f"  Surprise: mean={clean['surprise'].mean():.4f}, std={clean['surprise'].std():.4f}")
    
    clean.to_csv(os.path.join(DATA_DIR, "analysis_dataset.csv"), index=False)
    print(f"  Saved analysis dataset")
    
    # 5. Regressions
    print("\n[5/5] Running regressions...")
    
    h1 = run_h1(clean)
    print(f"\n=== H1: Sentiment vs Surprise ===")
    print(f"  β = {h1['beta']:.4f} (SE = {h1['se']:.4f}, p = {h1['p_value']:.4f})")
    print(f"  R² = {h1['r_squared']:.4f}")
    print(f"  N = {h1['n']}")
    
    h2 = run_h2(clean)
    print(f"\n=== H2: Incremental Predictive Power ===")
    for name, r in h2.items():
        marker = "✅" if r["significant_10"] else "  "
        print(f"  {marker} {name}: β₂={r['beta2']:.4f}, p={r['p_value']:.4f}, ΔR²={r['inc_r2']:.4f}")
    
    h3 = run_h3(clean)
    print(f"\n=== H3: Sentiment Decomposition ===")
    print(f"  Policy loading: {h3['policy_loading']:.4f}")
    print(f"  Info loading:   {h3['info_loading']:.4f}")
    print(f"  Info > Policy:  {h3['info_dominates']}")
    
    h4 = run_h4(clean)
    print(f"\n=== H4: Regime-Dependent Effects ===")
    for regime in ["conventional", "forward_guidance", "normalization"]:
        r = h4.get(regime, {})
        print(f"  {regime}: |β₂| = {r.get('abs_beta2', np.nan):.4f} (N={r.get('n', 0)})")
    print(f"  FG strongest: {h4.get('fg_strongest', False)}")
    
    # Save
    results = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    with open(os.path.join(RESULTS_DIR, "regression_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    
    h1_ok = 0 < h1.get("r_squared", 0) < 1
    print(f"\nH1 (Sentiment ≠ Surprise): R² = {h1['r_squared']:.4f} → {'✅ SUPPORTED' if h1_ok else '❌'}")
    
    sig = sum(1 for v in h2.values() if v.get("significant_10", False))
    tot = len(h2)
    print(f"\nH2 (Incremental Power): {sig}/{tot} significant → {'✅' if sig > 0 else '❌'}")
    print(f"\nH3 (Info > Policy): {'✅' if h3.get('info_dominates') else '❌'}")
    print(f"\nH4 (FG Strongest): {'✅' if h4.get('fg_strongest') else '❌'}")
    
    return clean, results


if __name__ == "__main__":
    main()
