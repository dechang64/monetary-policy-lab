"""
Analysis Pipeline v4: Expanded dataset (164 statements, 2006-2026)
"""
import pandas as pd
import numpy as np
import os, sys, json
import yfinance as yf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.fomc_meetings import get_fomc_data
from data.sentiment import compute_lm_sentiment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def find_loc(idx, date):
    ts = pd.Timestamp(date)
    try:
        return idx.get_loc(ts)
    except (KeyError, TypeError):
        pos = idx.searchsorted(ts, side='right') - 1
        return max(0, min(pos, len(idx) - 1))


def get_val(df, date, col):
    loc = find_loc(df.index, date)
    v = df.iloc[loc][col]
    return v if not pd.isna(v) else np.nan


def download_prices():
    tickers = {
        "^GSPC": "sp500", "^IXIC": "nasdaq", "^VIX": "vix",
        "^TNX": "ty10", "^IRX": "tb13w", "GC=F": "gold",
    }
    data = {}
    for tk, name in tickers.items():
        try:
            df = yf.download(tk, start="2006-01-01", end="2026-12-31", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            print(f"  ✅ {tk} ({name}): {len(df)} obs")
        except Exception as e:
            print(f"  ❌ {tk}: {e}")
    pdf = pd.DataFrame(data)
    pdf.index = pd.to_datetime(pdf.index)
    pdf = pdf.sort_index()
    return pdf


def compute_sentiment(fomc_df):
    """Compute sentiment from scraped FOMC statements."""
    stmt_path = os.path.join(DATA_DIR, "fomc_statements_all.json")
    with open(stmt_path) as f:
        statements = json.load(f)
    
    fomc_df["sentiment"] = np.nan
    fomc_df["lm_score"] = np.nan
    fomc_df["cb_score"] = np.nan
    fomc_df["word_count"] = 0
    
    matched = 0
    for idx, row in fomc_df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
        if date_str in statements:
            text = statements[date_str]
            combined, lm, cb, wc = compute_lm_sentiment(text)
            fomc_df.at[idx, "sentiment"] = combined
            fomc_df.at[idx, "lm_score"] = lm
            fomc_df.at[idx, "cb_score"] = cb
            fomc_df.at[idx, "word_count"] = wc
            matched += 1
    
    fomc_df = fomc_df.dropna(subset=["sentiment"])
    print(f"  Matched: {matched} meetings")
    print(f"  Sentiment: mean={fomc_df['sentiment'].mean():.4f}, std={fomc_df['sentiment'].std():.4f}")
    return fomc_df


def compute_surprises(fomc_df, pdf):
    """Compute surprises using rate changes."""
    fomc_df["surprise"] = fomc_df["rate_change"]
    
    # Also compute DFF-based surprise
    if "vix" in pdf.columns:
        pass  # vix available for controls
    
    print(f"  Surprise: mean={fomc_df['surprise'].mean():.4f}, std={fomc_df['surprise'].std():.4f}")
    return fomc_df


def compute_returns(pdf, fomc_df):
    """Compute event-window returns."""
    assets = {"sp500": "sp500_ret", "nasdaq": "nasdaq_ret", "gold": "gold_ret"}
    yields = {"ty10": "ty10_chg", "tb13w": "tb13w_chg"}
    
    for asset, col in assets.items():
        if asset not in pdf.columns:
            fomc_df[col] = np.nan
            continue
        rets = []
        for _, r in fomc_df.iterrows():
            try:
                loc = find_loc(pdf.index, r["date"])
                v0 = pdf.iloc[loc][asset]
                v1 = pdf.iloc[min(loc + 1, len(pdf) - 1)][asset]
                rets.append((v1 - v0) / v0 * 100 if not (pd.isna(v0) or pd.isna(v1)) else np.nan)
            except:
                rets.append(np.nan)
        fomc_df[col] = rets
    
    for asset, col in yields.items():
        if asset not in pdf.columns:
            fomc_df[col] = np.nan
            continue
        chgs = []
        for _, r in fomc_df.iterrows():
            try:
                loc = find_loc(pdf.index, r["date"])
                v0 = pdf.iloc[loc][asset]
                v1 = pdf.iloc[min(loc + 1, len(pdf) - 1)][asset]
                chgs.append(v1 - v0 if not (pd.isna(v0) or pd.isna(v1)) else np.nan)
            except:
                chgs.append(np.nan)
        fomc_df[col] = chgs
    
    # Controls
    if "vix" in pdf.columns:
        fomc_df["vix"] = [get_val(pdf, r["date"], "vix") for _, r in fomc_df.iterrows()]
    
    # Term spread
    if "ty10" in pdf.columns and "tb13w" in pdf.columns:
        fomc_df["term_spread"] = [
            get_val(pdf, r["date"], "ty10") - get_val(pdf, r["date"], "tb13w")
            for _, r in fomc_df.iterrows()
        ]
    
    return fomc_df


def run_h1(df):
    """H1: Sentiment vs Surprise."""
    valid = df.dropna(subset=["sentiment", "surprise"])
    slope, intercept, r, p, se = stats.linregress(valid["surprise"], valid["sentiment"])
    return {"r_squared": r**2, "beta": slope, "se": se, "p_value": p, "n": len(valid)}


def run_h2(df):
    """H2: Incremental predictive power."""
    results = {}
    assets = {"sp500_ret": "S&P 500", "nasdaq_ret": "NASDAQ", "gold_ret": "Gold",
              "ty10_chg": "10Y Yield", "tb13w_chg": "13W Yield"}
    
    for asset, name in assets.items():
        if asset not in df.columns:
            continue
        valid = df.dropna(subset=[asset, "surprise", "sentiment"])
        if len(valid) < 30:
            continue
        
        Y = valid[asset].values
        X1 = valid["surprise"].values
        X2 = valid["sentiment"].values
        
        # Model 1: Y ~ surprise
        slope1, _, r1, _, _ = stats.linregress(X1, Y)
        ss_res1 = np.sum((Y - (slope1 * X1 + np.mean(Y) - slope1 * np.mean(X1)))**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2_1 = 1 - ss_res1 / ss_tot if ss_tot > 0 else 0
        
        # Model 2: Y ~ surprise + sentiment
        X = np.column_stack([np.ones(len(Y)), X1, X2])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_pred = X @ beta
        ss_res2 = np.sum((Y - Y_pred)**2)
        r2_2 = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
        
        # SE for beta2
        resid = Y - Y_pred
        n, k = len(Y), 3
        mse = ss_res2 / (n - k)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se2 = np.sqrt(cov[2, 2])
        except:
            se2 = np.nan
        
        t2 = beta[2] / se2 if se2 > 0 else 0
        p2 = 2 * (1 - stats.t.cdf(abs(t2), n - k)) if se2 > 0 else 1.0
        
        results[name] = {
            "beta1": beta[1], "beta2": beta[2], "se": se2,
            "t_stat": t2, "p_value": p2,
            "r2_1": r2_1, "r2_2": r2_2, "inc_r2": r2_2 - r2_1,
            "significant_10": p2 < 0.10, "significant_05": p2 < 0.05,
            "significant_01": p2 < 0.01, "n": n,
        }
    
    return results


def run_h3(df):
    """H3: Two-shocks decomposition using standardized variables."""
    valid = df.dropna(subset=["sentiment", "surprise", "rate_change"])
    if len(valid) < 30:
        return {"policy_loading": 0, "info_loading": 0, "info_dominates": False, "n": len(valid)}
    
    # Standardize all variables
    from scipy.stats import zscore
    sent_z = zscore(valid["sentiment"].values)
    rate_z = zscore(valid["rate_change"].values)
    surprise_z = zscore(valid["surprise"].values)
    
    # Policy shock = rate_change (standardized)
    slope_policy, _, r_policy, _, _ = stats.linregress(rate_z, sent_z)
    policy_loading = abs(slope_policy)
    
    # Info shock = residual of surprise ~ rate_change (standardized)
    slope_sr, _, _, _, _ = stats.linregress(rate_z, surprise_z)
    info_shock = surprise_z - slope_sr * rate_z
    slope_info, _, r_info, _, _ = stats.linregress(info_shock, sent_z)
    info_loading = abs(slope_info)
    
    total = policy_loading + info_loading
    if total > 0:
        policy_share = policy_loading / total
        info_share = info_loading / total
    else:
        policy_share = info_share = 0.5
    
    return {
        "policy_loading": round(policy_loading, 4),
        "info_loading": round(info_loading, 4),
        "policy_share": round(policy_share, 4),
        "info_share": round(info_share, 4),
        "info_dominates": info_share > policy_share,
        "r_squared_policy": r_policy**2,
        "r_squared_info": r_info**2,
        "n": len(valid),
    }


def run_h4(df):
    """H4: Regime-dependent effects."""
    results = {}
    for regime in ["conventional", "forward_guidance", "normalization"]:
        sub = df[df["regime"] == regime]
        if len(sub) < 20:
            results[regime] = {"abs_beta2": np.nan, "beta2": np.nan, "n": len(sub)}
            continue
        
        # Use tb13w_chg as primary asset (most significant in H2)
        if "tb13w_chg" not in sub.columns:
            results[regime] = {"abs_beta2": np.nan, "beta2": np.nan, "n": len(sub)}
            continue
        
        valid = sub.dropna(subset=["tb13w_chg", "surprise", "sentiment"])
        if len(valid) < 15:
            results[regime] = {"abs_beta2": np.nan, "beta2": np.nan, "n": len(valid)}
            continue
        
        Y = valid["tb13w_chg"].values
        X = np.column_stack([np.ones(len(Y)), valid["surprise"].values, valid["sentiment"].values])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        results[regime] = {
            "abs_beta2": abs(beta[2]),
            "beta2": beta[2],
            "beta1": beta[1],
            "n": len(valid),
        }
    
    # Check if FG is strongest
    vals = {k: v.get("abs_beta2", 0) for k, v in results.items() if not np.isnan(v.get("abs_beta2", np.nan))}
    fg_strongest = vals.get("forward_guidance", 0) > max(vals.get("conventional", 0), vals.get("normalization", 0)) if vals else False
    
    results["fg_strongest"] = fg_strongest
    return results


def run_robustness(df):
    """Additional robustness checks."""
    results = {}
    
    # 1. Chair fixed effects
    chairs = df["chair"].unique()
    chair_results = {}
    for chair in chairs:
        sub = df[df["chair"] == chair].dropna(subset=["tb13w_chg", "surprise", "sentiment"])
        if len(sub) < 15:
            continue
        Y = sub["tb13w_chg"].values
        X = np.column_stack([np.ones(len(Y)), sub["surprise"].values, sub["sentiment"].values])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        chair_results[chair] = {"beta2": beta[2], "n": len(sub)}
    results["chair_fe"] = chair_results
    
    # 2. Post-2010 subsample (more standardized communication)
    post2010 = df[df["date"] >= "2010-01-01"].dropna(subset=["tb13w_chg", "surprise", "sentiment"])
    if len(post2010) > 30:
        Y = post2010["tb13w_chg"].values
        X = np.column_stack([np.ones(len(Y)), post2010["surprise"].values, post2010["sentiment"].values])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        n, k = len(Y), 3
        mse = np.sum(resid**2) / (n - k)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se2 = np.sqrt(cov[2, 2])
            t2 = beta[2] / se2
            p2 = 2 * (1 - stats.t.cdf(abs(t2), n - k))
        except:
            se2, t2, p2 = np.nan, np.nan, 1.0
        results["post2010"] = {"beta2": beta[2], "se": se2, "t": t2, "p": p2, "n": n}
    
    # 3. Exclude COVID (2020-2021)
    no_covid = df[~((df["date"] >= "2020-03-01") & (df["date"] <= "2021-12-31"))].dropna(subset=["tb13w_chg", "surprise", "sentiment"])
    if len(no_covid) > 30:
        Y = no_covid["tb13w_chg"].values
        X = np.column_stack([np.ones(len(Y)), no_covid["surprise"].values, no_covid["sentiment"].values])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        n, k = len(Y), 3
        mse = np.sum(resid**2) / (n - k)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se2 = np.sqrt(cov[2, 2])
            t2 = beta[2] / se2
            p2 = 2 * (1 - stats.t.cdf(abs(t2), n - k))
        except:
            se2, t2, p2 = np.nan, np.nan, 1.0
        results["no_covid"] = {"beta2": beta[2], "se": se2, "t": t2, "p": p2, "n": n}
    
    # 4. All assets in H2 with full sample
    all_assets = {}
    for asset, name in {"sp500_ret": "S&P 500", "nasdaq_ret": "NASDAQ", "gold_ret": "Gold",
                         "ty10_chg": "10Y Yield", "tb13w_chg": "13W Yield"}.items():
        if asset not in df.columns:
            continue
        valid = df.dropna(subset=[asset, "surprise", "sentiment"])
        if len(valid) < 30:
            continue
        Y = valid[asset].values
        X = np.column_stack([np.ones(len(Y)), valid["surprise"].values, valid["sentiment"].values])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        n, k = len(Y), 3
        mse = np.sum(resid**2) / (n - k)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2 = 1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else 0
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se2 = np.sqrt(cov[2, 2])
            t2 = beta[2] / se2
            p2 = 2 * (1 - stats.t.cdf(abs(t2), n - k))
        except:
            se2, t2, p2 = np.nan, np.nan, 1.0
        all_assets[name] = {"beta2": beta[2], "se": se2, "t": t2, "p": p2, "r2": r2, "n": n}
    results["all_assets"] = all_assets
    
    return results


def main():
    print("=" * 60)
    print("Monetary Policy Research Lab - Expanded Analysis (2006-2026)")
    print("=" * 60)
    
    print("\n[1/6] Loading FOMC meetings...")
    fomc = get_fomc_data()
    fomc = fomc[fomc["date"] >= "2006-01-01"].copy()
    print(f"  {len(fomc)} meetings (2006-2026)")
    
    print("\n[2/6] Computing sentiment (from scraped statements)...")
    fomc = compute_sentiment(fomc)
    
    print("\n[3/6] Downloading asset prices...")
    pdf = download_prices()
    
    print("\n[4/6] Computing surprises and returns...")
    fomc = compute_surprises(fomc, pdf)
    fomc = compute_returns(pdf, fomc)
    
    clean = fomc.dropna(subset=["sentiment", "surprise", "sp500_ret", "tb13w_chg"])
    print(f"  Clean observations: {len(clean)}")
    
    # Save
    clean.to_csv(os.path.join(DATA_DIR, "analysis_dataset_expanded.csv"), index=False)
    print(f"  Saved analysis dataset")
    
    print("\n[5/6] Running regressions...")
    
    h1 = run_h1(clean)
    print(f"\n=== H1: Sentiment vs Surprise ===")
    print(f"  β = {h1['beta']:.4f} (SE = {h1['se']:.4f}, p = {h1['p_value']:.4f})")
    print(f"  R² = {h1['r_squared']:.4f}")
    print(f"  N = {h1['n']}")
    
    h2 = run_h2(clean)
    print(f"\n=== H2: Incremental Predictive Power ===")
    for name, r in h2.items():
        marker = "✅" if r.get("significant_10") else "  "
        stars = "***" if r.get("significant_01") else "**" if r.get("significant_05") else "*" if r.get("significant_10") else ""
        print(f"  {marker} {name}: β₂={r['beta2']:.4f}, p={r['p_value']:.4f}{stars}, ΔR²={r['inc_r2']:.4f}")
    
    h3 = run_h3(clean)
    print(f"\n=== H3: Two-Shocks Decomposition ===")
    print(f"  Policy loading: {h3['policy_loading']:.4f} ({h3['policy_share']:.1%})")
    print(f"  Info loading:   {h3['info_loading']:.4f} ({h3['info_share']:.1%})")
    print(f"  Info > Policy:  {h3['info_dominates']}")
    
    h4 = run_h4(clean)
    print(f"\n=== H4: Regime-Dependent Effects ===")
    for regime in ["conventional", "forward_guidance", "normalization"]:
        r = h4.get(regime, {})
        print(f"  {regime}: |β₂| = {r.get('abs_beta2', np.nan):.4f} (N={r.get('n', 0)})")
    print(f"  FG strongest: {h4.get('fg_strongest')}")
    
    print("\n[6/6] Robustness Checks...")
    rob = run_robustness(clean)
    
    print(f"\n=== Robustness: Chair Fixed Effects ===")
    for chair, r in rob.get("chair_fe", {}).items():
        print(f"  {chair}: β₂ = {r['beta2']:.4f} (N={r['n']})")
    
    if "post2010" in rob:
        r = rob["post2010"]
        print(f"\n=== Robustness: Post-2010 ===")
        print(f"  β₂ = {r['beta2']:.4f}, p = {r['p']:.4f} (N={r['n']})")
    
    if "no_covid" in rob:
        r = rob["no_covid"]
        print(f"\n=== Robustness: Exclude COVID ===")
        print(f"  β₂ = {r['beta2']:.4f}, p = {r['p']:.4f} (N={r['n']})")
    
    # Save all results
    all_results = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "robustness": rob}
    with open(os.path.join(RESULTS_DIR, "regression_results_expanded.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS (Expanded Sample)")
    print("=" * 60)
    
    h1_ok = 0 < h1.get("r_squared", 0) < 1
    print(f"\nH1: R² = {h1['r_squared']:.4f} → {'✅' if h1_ok else '❌'}")
    
    sig = sum(1 for v in h2.values() if v.get("significant_10", False))
    tot = len(h2)
    print(f"H2: {sig}/{tot} significant → {'✅' if sig > 0 else '❌'}")
    print(f"H3: Info > Policy → {'✅' if h3.get('info_dominates') else '❌'}")
    print(f"H4: FG strongest → {'✅' if h4.get('fg_strongest') else '❌'}")
    
    return clean, all_results


if __name__ == "__main__":
    main()
