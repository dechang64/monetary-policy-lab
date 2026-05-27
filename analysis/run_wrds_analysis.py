"""
WRDS-Enhanced Analysis Pipeline v5
====================================
Upgrade from v4: Replace yfinance with WRDS CRSP data,
replace rate_change-based surprise with Kuttner/GSS target+path shocks.

Data sources:
- Monetary policy shocks: Acosta (2022) replication of GSS + NS (1995-2022)
- Market returns: CRSP dsi/dsf via WRDS (1990-2024)
- FOMC statements: existing scraper (2006-2025)
- FRED rates: DFF, DGS10, DGS3MO (existing)

Key improvements over v4:
1. Target shock (Kuttner surprise) instead of rate_change
2. Path shock (Gürkaynak forward guidance factor) 
3. CRSP value-weighted returns instead of yfinance
4. Financial sector stock-level analysis
"""

import pandas as pd
import numpy as np
import os, sys, json
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WRDS_DIR = DATA_DIR / "wrds"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_monetary_shocks():
    """Load Acosta (2022) monetary policy shocks (GSS + NS replication)."""
    shock_file = DATA_DIR / "mp_shocks_acosta.xlsx"
    if not shock_file.exists():
        raise FileNotFoundError(
            f"Monetary policy shocks file not found: {shock_file}\n"
            "Download from https://www.acostamiguel.com/replication/MPshocksAcosta.xlsx"
        )
    df = pd.read_excel(shock_file, sheet_name="shocks")
    df["fomc"] = pd.to_datetime(df["fomc"])
    df = df.set_index("fomc").sort_index()
    
    # ff.shock.0 is in percentage points (0.01 = 1bp)
    # Convert to basis points for consistency with literature
    df["target_bp"] = df["ff.shock.0"] * 100  # Kuttner target surprise in bp
    # target and path are standardized (unit std), keep as-is
    
    print(f"Monetary shocks: {len(df)} meetings, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_crsp_index():
    """Load CRSP daily market index from WRDS."""
    f = WRDS_DIR / "crsp_dsi_index.csv"
    if not f.exists():
        raise FileNotFoundError(f"CRSP dsi not found: {f}")
    df = pd.read_csv(f, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    print(f"CRSP dsi: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_crsp_financial_stocks():
    """Load CRSP financial sector daily stocks from WRDS."""
    f = WRDS_DIR / "crsp_financial_stocks_2020_2025.csv"
    if not f.exists():
        print(f"Warning: CRSP financial stocks not found, skipping stock-level analysis")
        return None
    df = pd.read_csv(f, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    print(f"CRSP financial stocks: {len(df)} rows, {df.permno.nunique()} stocks")
    return df


def load_crsp_stock_names():
    """Load CRSP stock name mapping."""
    f = WRDS_DIR / "crsp_stock_names.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df


def load_fomc_data():
    """Load existing FOMC meeting data with sentiment scores."""
    # Use the expanded analysis dataset
    f = BASE_DIR / "mp-research-platform" / "data" / "analysis_dataset_expanded.csv"
    if not f.exists():
        raise FileNotFoundError(f"FOMC data not found: {f}")
    df = pd.read_csv(f, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    print(f"FOMC data: {len(df)} meetings, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_fomc_statements():
    """Load FOMC statements for sentiment analysis."""
    f = BASE_DIR / "mp-research-platform" / "data" / "fomc_statements_all.json"
    if not f.exists():
        return None
    with open(f) as fh:
        data = json.load(fh)
    return data


def compute_event_returns(crsp_dsi, fomc_dates, window=1):
    """
    Compute CRSP market returns around FOMC meetings.
    
    Args:
        crsp_dsi: CRSP daily index DataFrame
        fomc_dates: list of FOMC meeting dates
        window: days after meeting for return calculation
    
    Returns:
        DataFrame with event returns
    """
    results = []
    for date in fomc_dates:
        ts = pd.Timestamp(date)
        if ts not in crsp_dsi.index:
            # Find nearest trading day
            loc = crsp_dsi.index.searchsorted(ts, side="right") - 1
            if loc < 0 or loc >= len(crsp_dsi):
                continue
            ts = crsp_dsi.index[loc]
        
        loc = crsp_dsi.index.get_loc(ts)
        
        # Day-of return (close-to-close)
        ret_day = crsp_dsi.iloc[loc]["vwretd"]
        
        # Pre-meeting return (day before)
        ret_pre = crsp_dsi.iloc[loc - 1]["vwretd"] if loc > 0 else np.nan
        
        # Post-meeting return (day after)
        ret_post = crsp_dsi.iloc[loc + 1]["vwretd"] if loc < len(crsp_dsi) - 1 else np.nan
        
        # 2-day window [0, +1]
        ret_2d = crsp_dsi.iloc[loc:loc + 2]["vwretd"].sum() if loc < len(crsp_dsi) - 1 else np.nan
        
        # S&P 500 index level
        sp_level = crsp_dsi.iloc[loc]["spindx"]
        
        results.append({
            "date": ts,
            "vwretd_day": ret_day,
            "vwretd_pre": ret_pre,
            "vwretd_post": ret_post,
            "vwretd_2d": ret_2d,
            "spindx": sp_level,
            "ewretd_day": crsp_dsi.iloc[loc]["ewretd"],
            "sprtrn_day": crsp_dsi.iloc[loc]["sprtrn"],
        })
    
    df = pd.DataFrame(results).set_index("date").sort_index()
    return df


def compute_financial_stock_event_returns(crsp_stocks, stock_names, fomc_dates):
    """
    Compute financial sector stock returns around FOMC meetings.
    Aggregate to equal-weighted and value-weighted sector returns.
    """
    if crsp_stocks is None:
        return None
    
    # Map permno to ticker if names available
    permno_ticker = {}
    if stock_names is not None:
        # Get most recent ticker for each permno
        names = stock_names.sort_values(["permno", "namedt"]).groupby("permno").last()
        permno_ticker = names["ticker"].to_dict() if "ticker" in names.columns else {}
    
    results = []
    for date in fomc_dates:
        ts = pd.Timestamp(date)
        if ts not in crsp_stocks.index:
            loc = crsp_stocks.index.searchsorted(ts, side="right") - 1
            if loc < 0:
                continue
            ts = crsp_stocks.index[loc]
        
        day_data = crsp_stocks.loc[[ts]] if isinstance(crsp_stocks.loc[ts], pd.DataFrame) else crsp_stocks.loc[[ts]]
        
        # Equal-weighted financial sector return
        rets = pd.to_numeric(day_data["ret"], errors="coerce")
        ew_ret = rets.mean()
        
        # Value-weighted (by shrout * |prc|)
        if "shrout" in day_data.columns and "prc" in day_data.columns:
            prc = pd.to_numeric(day_data["prc"], errors="coerce").abs()
            shrout = pd.to_numeric(day_data["shrout"], errors="coerce")
            mktcap = prc * shrout
            valid = rets.notna() & mktcap.notna() & (mktcap > 0)
            if valid.sum() > 0:
                vw_ret = (rets[valid] * mktcap[valid]).sum() / mktcap[valid].sum()
            else:
                vw_ret = np.nan
        else:
            vw_ret = np.nan
        
        # Bank subset (permnos with SIC 6000-6199)
        n_stocks = len(day_data)
        
        results.append({
            "date": ts,
            "fin_ew_ret": ew_ret,
            "fin_vw_ret": vw_ret,
            "fin_n_stocks": n_stocks,
        })
    
    df = pd.DataFrame(results).set_index("date").sort_index()
    return df


def ols_regression(y, X, robust=True, lag=1):
    """
    OLS regression with optional Newey-West standard errors.
    
    Args:
        y: dependent variable (n,)
        X: independent variables (n, k) — NO constant column needed
        robust: use Newey-West HAC standard errors
        lag: lag length for Newey-West
    
    Returns:
        dict with coefficients, SE, t-stats, p-values, R²
    """
    n = len(y)
    X = np.column_stack([np.ones(n), X])
    k = X.shape[1]
    
    # OLS estimation
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix"}
    
    residuals = y - X @ beta
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0
    
    # Standard errors
    if robust and n > k:
        # Newey-West HAC
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
            XtX_inv = np.linalg.inv(X.T @ X)
            V = XtX_inv @ S @ XtX_inv
            se = np.sqrt(np.maximum(np.diag(V), 1e-10))
        except np.linalg.LinAlgError:
            se = np.sqrt(np.maximum(np.diag(ss_res / (n - k) * np.linalg.inv(X.T @ X)), 1e-10))
    else:
        sigma2 = ss_res / (n - k) if n > k else ss_res / n
        try:
            se = np.sqrt(np.maximum(np.diag(sigma2 * np.linalg.inv(X.T @ X)), 1e-10))
        except np.linalg.LinAlgError:
            return {"error": "Singular XtX"}
    
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=max(n - k, 1)))
    
    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stats.tolist(),
        "p_value": p_values.tolist(),
        "r_squared": float(r_squared),
        "r_squared_adj": float(r_squared_adj),
        "n": int(n),
        "residuals": residuals.tolist(),
    }


def main():
    print("=" * 60)
    print("WRDS-Enhanced Analysis Pipeline v5")
    print("=" * 60)
    
    # ── Load data ──
    print("\n--- Loading Data ---")
    shocks = load_monetary_shocks()
    crsp_dsi = load_crsp_index()
    crsp_stocks = load_crsp_financial_stocks()
    stock_names = load_crsp_stock_names()
    fomc_data = load_fomc_data()
    
    # ── Build unified dataset ──
    print("\n--- Building Unified Dataset ---")
    
    # Get FOMC dates that overlap with shocks data
    common_dates = fomc_data.index.intersection(shocks.index)
    print(f"Overlap between FOMC data and shocks: {len(common_dates)} meetings")
    
    # Compute CRSP event returns for all FOMC dates
    all_fomc_dates = fomc_data.index.tolist()
    crsp_events = compute_event_returns(crsp_dsi, all_fomc_dates)
    
    # Compute financial sector event returns
    fin_events = compute_financial_stock_event_returns(crsp_stocks, stock_names, all_fomc_dates)
    
    # Merge everything using join (avoids reindex issues)
    df = fomc_data.copy()
    
    # Add CRSP returns (replace yfinance) — use join to align by date index
    df = df.join(crsp_events, how="left")
    
    # Add financial sector returns
    if fin_events is not None:
        df = df.join(fin_events, how="left")
    
    # Add monetary policy shocks — use join
    shock_cols = shocks[["target", "path", "target_bp", "ns"]].copy()
    shock_cols.columns = ["target_shock", "path_shock", "kuttner_bp", "ns_shock"]
    df = df.join(shock_cols, how="left")
    
    # ── Analysis ──
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)
    
    # Use only rows with complete data
    df_shock = df.dropna(subset=["target_shock", "path_shock", "sentiment", "vwretd_day"])
    print(f"\nComplete cases for shock analysis: {len(df_shock)}")
    print(f"  Period: {df_shock.index.min().date()} to {df_shock.index.max().date()}")
    
    # ── H1: Sentiment ~ Target Shock + Path Shock ──
    print("\n" + "-" * 40)
    print("H1: Sentiment ~ Target Shock + Path Shock")
    print("-" * 40)
    
    y = df_shock["sentiment"].values
    X = df_shock[["target_shock", "path_shock"]].values
    
    h1 = ols_regression(y, X, robust=True)
    print(f"  R² = {h1['r_squared']:.4f}")
    print(f"  β(target) = {h1['beta'][1]:.4f}, t = {h1['t_stat'][1]:.3f}, p = {h1['p_value'][1]:.4f}")
    print(f"  β(path)   = {h1['beta'][2]:.4f}, t = {h1['t_stat'][2]:.3f}, p = {h1['p_value'][2]:.4f}")
    
    # Compare with old surprise measure
    print("\n  --- Comparison: old rate_change-based surprise ---")
    df_old = df.dropna(subset=["surprise", "sentiment"])
    if len(df_old) > 10:
        h1_old = ols_regression(df_old["sentiment"].values, 
                                df_old[["surprise"]].values, robust=True)
        print(f"  Old: R² = {h1_old['r_squared']:.4f}, β = {h1_old['beta'][1]:.4f}, p = {h1_old['p_value'][1]:.4f}")
    
    # ── H2: Asset Returns ~ Target Shock + Path Shock ──
    print("\n" + "-" * 40)
    print("H2: Asset Returns ~ Target Shock + Path Shock")
    print("-" * 40)
    
    asset_cols = {
        "vwretd_day": "CRSP VW Market",
        "sprtrn_day": "S&P 500 (CRSP)",
        "ewretd_day": "CRSP EW Market",
        "ty10_chg": "10Y Yield Chg",
        "tb13w_chg": "13W T-Bill Chg",
        "gold_ret": "Gold Return",
    }
    
    if fin_events is not None:
        asset_cols["fin_vw_ret"] = "Financial Sector VW"
        asset_cols["fin_ew_ret"] = "Financial Sector EW"
    
    h2_results = {}
    for col, label in asset_cols.items():
        if col not in df_shock.columns:
            continue
        sub = df_shock.dropna(subset=[col])
        if len(sub) < 20:
            continue
        
        y = sub[col].values
        X = sub[["target_shock", "path_shock"]].values
        
        r = ols_regression(y, X, robust=True)
        if "error" in r:
            continue
        
        h2_results[col] = {
            "label": label,
            "beta_target": r["beta"][1],
            "beta_path": r["beta"][2],
            "se_target": r["se"][1],
            "se_path": r["se"][2],
            "t_target": r["t_stat"][1],
            "t_path": r["t_stat"][2],
            "p_target": r["p_value"][1],
            "p_path": r["p_value"][2],
            "r_squared": r["r_squared"],
            "n": r["n"],
            "sig_target_10": r["p_value"][1] < 0.10,
            "sig_target_05": r["p_value"][1] < 0.05,
            "sig_path_10": r["p_value"][2] < 0.10,
            "sig_path_05": r["p_value"][2] < 0.05,
        }
        
        sig_t = "***" if r["p_value"][1] < 0.01 else "**" if r["p_value"][1] < 0.05 else "*" if r["p_value"][1] < 0.10 else ""
        sig_p = "***" if r["p_value"][2] < 0.01 else "**" if r["p_value"][2] < 0.05 else "*" if r["p_value"][2] < 0.10 else ""
        
        print(f"\n  {label}:")
        print(f"    Target: β = {r['beta'][1]:.4f}, t = {r['t_stat'][1]:.3f}{sig_t}")
        print(f"    Path:   β = {r['beta'][2]:.4f}, t = {r['t_stat'][2]:.3f}{sig_p}")
        print(f"    R² = {r['r_squared']:.4f}, N = {r['n']}")
    
    # ── H3: Sentiment ~ Target + Path (information channel) ──
    print("\n" + "-" * 40)
    print("H3: Which shock drives sentiment? (Information Channel)")
    print("-" * 40)
    
    # Already computed in H1, just interpret
    target_dominant = abs(h1["beta"][1]) > abs(h1["beta"][2])
    path_dominant = abs(h1["beta"][2]) > abs(h1["beta"][1])
    target_sig = h1["p_value"][1] < 0.10
    path_sig = h1["p_value"][2] < 0.10
    
    print(f"  Target shock: β = {h1['beta'][1]:.4f}, {'significant' if target_sig else 'not significant'}")
    print(f"  Path shock:   β = {h1['beta'][2]:.4f}, {'significant' if path_sig else 'not significant'}")
    print(f"  Dominant: {'Path (information)' if path_dominant else 'Target (policy)'}")
    
    h3 = {
        "target_beta": h1["beta"][1],
        "path_beta": h1["beta"][2],
        "target_p": h1["p_value"][1],
        "path_p": h1["p_value"][2],
        "path_dominates": path_dominant,
        "info_channel_significant": path_sig,
    }
    
    # ── H4: Forward Guidance Period Interaction ──
    print("\n" + "-" * 40)
    print("H4: Forward Guidance Period Interaction")
    print("-" * 40)
    
    df_shock["fg_period"] = (df_shock.index >= "2008-12-01") & (df_shock.index <= "2015-12-31")
    df_shock["sentiment_x_fg"] = df_shock["sentiment"] * df_shock["fg_period"].astype(float)
    
    y = df_shock["vwretd_day"].values
    X = df_shock[["target_shock", "sentiment", "sentiment_x_fg"]].values
    
    h4 = ols_regression(y, X, robust=True)
    print(f"  R² = {h4['r_squared']:.4f}")
    print(f"  β(target)      = {h4['beta'][1]:.4f}, p = {h4['p_value'][1]:.4f}")
    print(f"  β(sentiment)   = {h4['beta'][2]:.4f}, p = {h4['p_value'][2]:.4f}")
    print(f"  β(sent×FG)     = {h4['beta'][3]:.4f}, p = {h4['p_value'][3]:.4f}")
    fg_strongest = h4["p_value"][3] < h4["p_value"][2]
    print(f"  FG interaction {'stronger' if fg_strongest else 'not stronger'} than base sentiment")
    
    # ── Robustness ──
    print("\n" + "-" * 40)
    print("Robustness Checks")
    print("-" * 40)
    
    robustness = {}
    
    # 1. Post-2010 only
    sub_post2010 = df_shock[df_shock.index >= "2010-01-01"]
    if len(sub_post2010) > 20:
        y = sub_post2010["sentiment"].values
        X = sub_post2010[["target_shock", "path_shock"]].values
        r = ols_regression(y, X, robust=True)
        robustness["post_2010"] = {
            "r_squared": r["r_squared"],
            "beta_target": r["beta"][1],
            "beta_path": r["beta"][2],
            "p_target": r["p_value"][1],
            "p_path": r["p_value"][2],
            "n": r["n"],
        }
        print(f"  Post-2010: R² = {r['r_squared']:.4f}, N = {r['n']}")
    
    # 2. Exclude COVID (Mar 2020)
    sub_no_covid = df_shock[(df_shock.index < "2020-03-01") | (df_shock.index > "2020-06-30")]
    if len(sub_no_covid) > 20:
        y = sub_no_covid["sentiment"].values
        X = sub_no_covid[["target_shock", "path_shock"]].values
        r = ols_regression(y, X, robust=True)
        robustness["no_covid"] = {
            "r_squared": r["r_squared"],
            "beta_target": r["beta"][1],
            "beta_path": r["beta"][2],
            "p_target": r["p_value"][1],
            "p_path": r["p_value"][2],
            "n": r["n"],
        }
        print(f"  No-COVID: R² = {r['r_squared']:.4f}, N = {r['n']}")
    
    # 3. Kuttner bp instead of standardized target
    sub_kuttner = df_shock.dropna(subset=["kuttner_bp"])
    if len(sub_kuttner) > 20:
        y = sub_kuttner["sentiment"].values
        X = sub_kuttner[["kuttner_bp", "path_shock"]].values
        r = ols_regression(y, X, robust=True)
        robustness["kuttner_bp"] = {
            "r_squared": r["r_squared"],
            "beta_kuttner": r["beta"][1],
            "beta_path": r["beta"][2],
            "p_kuttner": r["p_value"][1],
            "p_path": r["p_value"][2],
            "n": r["n"],
        }
        print(f"  Kuttner bp: R² = {r['r_squared']:.4f}, β = {r['beta'][1]:.4f}, N = {r['n']}")
    
    # ── Save results ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nH1 (Sentiment ~ Shocks): R² = {h1['r_squared']:.4f}")
    print(f"  Target: β = {h1['beta'][1]:.4f}, p = {h1['p_value'][1]:.4f}")
    print(f"  Path:   β = {h1['beta'][2]:.4f}, p = {h1['p_value'][2]:.4f}")
    
    sig_assets = sum(1 for v in h2_results.values() if v["sig_target_10"] or v["sig_path_10"])
    print(f"\nH2 (Returns ~ Shocks): {sig_assets}/{len(h2_results)} assets significant at 10%")
    
    print(f"\nH3 (Info Channel): {'Path dominates ✅' if h3['path_dominates'] else 'Target dominates'}")
    print(f"\nH4 (FG Interaction): {'FG stronger ✅' if fg_strongest else 'Base sentiment stronger'}")
    
    # Save
    all_results = {
        "H1": {
            "r_squared": h1["r_squared"],
            "r_squared_adj": h1["r_squared_adj"],
            "beta_target": h1["beta"][1],
            "beta_path": h1["beta"][2],
            "se_target": h1["se"][1],
            "se_path": h1["se"][2],
            "p_target": h1["p_value"][1],
            "p_path": h1["p_value"][2],
            "n": h1["n"],
        },
        "H2": h2_results,
        "H3": h3,
        "H4": {
            "r_squared": h4["r_squared"],
            "beta_target": h4["beta"][1],
            "beta_sentiment": h4["beta"][2],
            "beta_sentiment_x_fg": h4["beta"][3],
            "p_target": h4["p_value"][1],
            "p_sentiment": h4["p_value"][2],
            "p_sentiment_x_fg": h4["p_value"][3],
            "fg_strongest": fg_strongest,
            "n": h4["n"],
        },
        "robustness": robustness,
        "metadata": {
            "shock_source": "Acosta (2022) replication of GSS + NS",
            "return_source": "CRSP via WRDS",
            "n_complete": len(df_shock),
            "period": f"{df_shock.index.min().date()} to {df_shock.index.max().date()}",
        },
    }
    
    out_file = RESULTS_DIR / "regression_results_wrds_v5.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
    
    # Also save the unified dataset
    df_out = df_shock.copy()
    df_out.index = df_out.index.strftime("%Y-%m-%d")
    df_out.to_csv(RESULTS_DIR / "analysis_dataset_wrds_v5.csv")
    print(f"Dataset saved to {RESULTS_DIR / 'analysis_dataset_wrds_v5.csv'}")
    
    return df_shock, all_results


if __name__ == "__main__":
    main()
