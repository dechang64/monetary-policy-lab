"""
Strict Audit Pipeline — re-run ALL regressions from raw data
Outputs: audit_results.json with every number the paper should use
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
WRDS = DATA / "wrds"

# ============================================================
# 1. Newey-West OLS (correct implementation)
# ============================================================
def nw_ols(y, X_cols, data, lag=4):
    """
    OLS with Newey-West HAC standard errors.
    y: column name (str)
    X_cols: list of column names
    data: DataFrame
    lag: NW lag (default 4, as in paper)
    Returns dict with all statistics
    """
    df = data[[y] + X_cols].dropna()
    n = len(df)
    if n < len(X_cols) + 2:
        return {"error": f"n={n} too small"}
    
    y_arr = df[y].values
    X_arr = np.column_stack([np.ones(n)] + [df[c].values for c in X_cols])
    k = X_arr.shape[1]
    
    # OLS
    beta = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    resid = y_arr - X_arr @ beta
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_arr - y_arr.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Newey-West covariance
    XtX_inv = np.linalg.inv(X_arr.T @ X_arr)
    S = np.zeros((k, k))
    # j=0
    for t in range(n):
        S += (X_arr[t:t+1].T * resid[t]) @ (X_arr[t:t+1] * resid[t])
    # j>0
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)
        Gamma = np.zeros((k, k))
        for t in range(j, n):
            Gamma += (X_arr[t:t+1].T * resid[t]) @ (X_arr[t-j:t-j+1] * resid[t-j])
        S += w * (Gamma + Gamma.T)
    
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 1e-15))
    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=max(n - k, 1))
    
    names = ['const'] + X_cols
    result = {}
    for i, name in enumerate(names):
        result[name] = {
            'beta': float(beta[i]),
            'se': float(se[i]),
            't': float(t_stats[i]),
            'p': float(p_values[i])
        }
    result['r2'] = float(r2)
    result['n'] = int(n)
    return result

# ============================================================
# 2. Load data
# ============================================================
print("Loading data...")

# Acosta shocks
shocks = pd.read_excel(DATA / "mp_shocks_acosta.xlsx", sheet_name="shocks")
shocks["fomc"] = pd.to_datetime(shocks["fomc"])
shocks = shocks.set_index("fomc").sort_index()
print(f"  Acosta shocks: {len(shocks)} meetings, {shocks.index.min().date()} to {shocks.index.max().date()}")

# FOMC analysis dataset (has sentiment, returns, etc.)
fomc = pd.read_csv(BASE / "mp-research-platform" / "data" / "analysis_dataset_expanded.csv", parse_dates=["date"])
fomc = fomc.set_index("date").sort_index()
print(f"  FOMC data: {len(fomc)} meetings")

# CRSP market index
crsp_dsi = pd.read_csv(WRDS / "crsp_dsi_index.csv", parse_dates=["date"])
crsp_dsi = crsp_dsi.set_index("date").sort_index()
print(f"  CRSP DSI: {len(crsp_dsi)} days")

# ============================================================
# 3. Build analysis dataset
# ============================================================
print("\nBuilding analysis dataset...")

df = fomc.copy()

# Add Acosta shocks (target, path)
shock_map = shocks[["target", "path"]].copy()
shock_map.columns = ["target_shock", "path_shock"]
df = df.join(shock_map, how="left")

# Kuttner surprise in bp
df["kuttner_bp"] = shocks["ff.shock.0"] * 100
df.loc[df["kuttner_bp"].isna(), "kuttner_bp"] = df.loc[df["kuttner_bp"].isna(), "surprise"]

# CRSP returns
crsp_events = []
for fomc_date in df.index:
    ts = pd.Timestamp(fomc_date)
    try:
        if ts in crsp_dsi.index:
            row = crsp_dsi.loc[ts]
        else:
            loc = crsp_dsi.index.searchsorted(ts, side='right') - 1
            if loc < 0: continue
            row = crsp_dsi.iloc[loc]
        crsp_events.append({
            "date": ts,
            "crsp_vw_ret": float(row.get("vwretd", np.nan)) * 100,
            "crsp_ew_ret": float(row.get("ewretd", np.nan)) * 100,
            "sp500_ret_crsp": float(row.get("sprtrn", np.nan)) * 100,
        })
    except (KeyError, IndexError):
        continue

crsp_df = pd.DataFrame(crsp_events).set_index("date").sort_index()
df = df.join(crsp_df, how="left")

# Sentiment: use sentiment_enhanced if available, else sentiment
if "sentiment_enhanced" in df.columns:
    df["sentiment_use"] = df["sentiment_enhanced"].fillna(df["sentiment"])
else:
    df["sentiment_use"] = df["sentiment"]

# FG period indicator
df["fg_period"] = ((df.index >= "2008-12-01") & (df.index <= "2015-12-31")).astype(int)
df["sent_x_fg"] = df["sentiment_use"] * df["fg_period"]

# Restrict to rows with shock data
df_reg = df[df["target_shock"].notna() & df["path_shock"].notna()].copy()
print(f"  Regression sample: {len(df_reg)} meetings, {df_reg.index.min().date()} to {df_reg.index.max().date()}")

# ============================================================
# 4. Summary Statistics (Table 1)
# ============================================================
print("\n=== TABLE 1: Summary Statistics ===")
table1 = {}
for col, label in [
    ("target_shock", "Target shock"),
    ("path_shock", "Path shock"),
    ("kuttner_bp", "Kuttner surprise (bp)"),
    ("sentiment_use", "Sentiment (combined)"),
    ("lm_score", "Sentiment (LM)"),
    ("cb_score_enhanced" if "cb_score_enhanced" in df_reg.columns else "cb_score", "Sentiment (CB)"),
]:
    if col in df_reg.columns:
        s = df_reg[col].dropna()
        table1[label] = {"mean": float(s.mean()), "std": float(s.std()), 
                         "min": float(s.min()), "max": float(s.max()), "n": int(len(s))}
        print(f"  {label}: mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")

# Correlations
corr_tp = df_reg["target_shock"].corr(df_reg["path_shock"])
corr_st = df_reg["sentiment_use"].corr(df_reg["target_shock"])
corr_sp = df_reg["sentiment_use"].corr(df_reg["path_shock"])
print(f"\n  Correlations: T-P={corr_tp:.2f}, S-T={corr_st:.2f}, S-P={corr_sp:.2f}")

# ============================================================
# 5. H1: Sentiment ~ Target + Path (Table 2)
# ============================================================
print("\n=== TABLE 2: H1 Sentiment ~ Shocks ===")
h1 = nw_ols("sentiment_use", ["target_shock", "path_shock"], df_reg, lag=4)
print(f"  Target: β={h1['target_shock']['beta']:.6f}, t={h1['target_shock']['t']:.3f}, p={h1['target_shock']['p']:.4f}")
print(f"  Path:   β={h1['path_shock']['beta']:.6f}, t={h1['path_shock']['t']:.3f}, p={h1['path_shock']['p']:.4f}")
print(f"  R²={h1['r2']:.4f}, N={h1['n']}")

# H1 with rate_change only
h1_rate = nw_ols("sentiment_use", ["rate_change"], df_reg, lag=4)
print(f"\n  Rate change only: R²={h1_rate['r2']:.4f}, p={h1_rate['rate_change']['p']:.4f}")

# H1 with Kuttner only
h1_kuttner = nw_ols("sentiment_use", ["kuttner_bp"], df_reg, lag=4)
print(f"  Kuttner only: R²={h1_kuttner['r2']:.4f}, p={h1_kuttner['kuttner_bp']['p']:.4f}")

# ============================================================
# 6. H2: Asset Returns ~ Target + Path (Table 4)
# ============================================================
print("\n=== TABLE 4: H2 Asset Returns ===")
h2 = {}
for col, label in [
    ("sp500_ret_crsp", "S&P 500"),
    ("crsp_vw_ret", "CRSP VW"),
    ("crsp_ew_ret", "CRSP EW"),
    ("nasdaq_ret", "NASDAQ"),
    ("gold_ret", "Gold"),
    ("ty10_chg", "10Y Treasury"),
    ("tb13w_chg", "13W T-bill"),
    ("vix", "VIX"),
]:
    if col in df_reg.columns:
        r = nw_ols(col, ["target_shock", "path_shock"], df_reg, lag=4)
        if "error" not in r:
            h2[label] = r
            sig_t = "**" if r["target_shock"]["p"] < 0.05 else "*" if r["target_shock"]["p"] < 0.10 else ""
            sig_p = "**" if r["path_shock"]["p"] < 0.05 else "*" if r["path_shock"]["p"] < 0.10 else ""
            print(f"  {label}: β_t={r['target_shock']['beta']:.3f}{sig_t}(t={r['target_shock']['t']:.2f}), β_p={r['path_shock']['beta']:.3f}{sig_p}(t={r['path_shock']['t']:.2f}), R²={r['r2']:.4f}")

# ============================================================
# 7. H3: Wald Test (Information Channel)
# ============================================================
print("\n=== H3: Wald Test ===")
# Need covariance matrix from H1
y = df_reg["sentiment_use"].dropna()
mask = df_reg["sentiment_use"].notna() & df_reg["target_shock"].notna() & df_reg["path_shock"].notna()
y_arr = df_reg.loc[mask, "sentiment_use"].values
X_arr = np.column_stack([np.ones(mask.sum()), df_reg.loc[mask, "target_shock"].values, df_reg.loc[mask, "path_shock"].values])
n = len(y_arr)
beta = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
resid = y_arr - X_arr @ beta
XtX_inv = np.linalg.inv(X_arr.T @ X_arr)
S = np.zeros((3, 3))
for t in range(n):
    S += (X_arr[t:t+1].T * resid[t]) @ (X_arr[t:t+1] * resid[t])
for j in range(1, 5):
    w = 1 - j/5
    G = np.zeros((3, 3))
    for t in range(j, n):
        G += (X_arr[t:t+1].T * resid[t]) @ (X_arr[t-j:t-j+1] * resid[t-j])
    S += w * (G + G.T)
V = XtX_inv @ S @ XtX_inv

diff = beta[1] - beta[2]
var_diff = V[1,1] + V[2,2] - 2*V[1,2]
wald_stat = diff**2 / var_diff
wald_p = 1 - stats.chi2.cdf(wald_stat, 1)
print(f"  H0: β_target = β_path")
print(f"  Wald χ²={wald_stat:.4f}, p={wald_p:.4f}")

# ============================================================
# 8. H4: FG Interaction (Table 5)
# ============================================================
print("\n=== TABLE 5: H4 FG Interaction ===")
h4 = {}
for col, label in [("sp500_ret_crsp", "S&P 500"), ("nasdaq_ret", "NASDAQ")]:
    r = nw_ols(col, ["target_shock", "path_shock", "sentiment_use", "sent_x_fg"], df_reg, lag=4)
    if "error" not in r:
        h4[label] = r
        print(f"  {label}:")
        for var in ["target_shock", "path_shock", "sentiment_use", "sent_x_fg"]:
            sig = "**" if r[var]["p"] < 0.05 else "*" if r[var]["p"] < 0.10 else ""
            print(f"    β_{var}={r[var]['beta']:.2f}{sig}, t={r[var]['t']:.2f}, p={r[var]['p']:.3f}")
        print(f"    R²={r['r2']:.4f}")

# ============================================================
# 9. Robustness
# ============================================================
print("\n=== Robustness ===")

# Post-2010
df_post2010 = df_reg[df_reg.index >= "2010-01-01"]
r_post2010 = nw_ols("sentiment_use", ["target_shock", "path_shock"], df_post2010, lag=4)
if "error" not in r_post2010:
    print(f"  Post-2010: R²={r_post2010['r2']:.4f}, β_path={r_post2010['path_shock']['beta']:.6f}, p_path={r_post2010['path_shock']['p']:.4f}, N={r_post2010['n']}")

# No COVID
df_nocovid = df_reg[~((df_reg.index >= "2020-03-01") & (df_reg.index <= "2020-06-30"))]
r_nocovid = nw_ols("sentiment_use", ["target_shock", "path_shock"], df_nocovid, lag=4)
if "error" not in r_nocovid:
    print(f"  No-COVID: R²={r_nocovid['r2']:.4f}, β_path={r_nocovid['path_shock']['beta']:.6f}, p_path={r_nocovid['path_shock']['p']:.4f}, N={r_nocovid['n']}")

# NW lag sensitivity
for lag in [1, 2, 3, 4, 5, 6]:
    r = nw_ols("sentiment_use", ["target_shock", "path_shock"], df_reg, lag=lag)
    if "error" not in r:
        print(f"  NW lag={lag}: β_path={r['path_shock']['beta']:.6f}, t={r['path_shock']['t']:.3f}, p={r['path_shock']['p']:.4f}")

# ============================================================
# 10. Save all results
# ============================================================
results = {
    "version": "audit_v1",
    "date": "2026-05-29",
    "sample": {"n": int(len(df_reg)), "period": f"{df_reg.index.min().date()} to {df_reg.index.max().date()}"},
    "table1": table1,
    "correlations": {"target_path": float(corr_tp), "sentiment_target": float(corr_st), "sentiment_path": float(corr_sp)},
    "H1": h1,
    "H1_rate_change": h1_rate,
    "H1_kuttner": h1_kuttner,
    "H2": h2,
    "H3_wald": {"chi2": float(wald_stat), "p": float(wald_p)},
    "H4": h4,
    "robustness": {
        "post_2010": r_post2010 if "error" not in r_post2010 else None,
        "no_covid": r_nocovid if "error" not in r_nocovid else None,
    }
}

with open(BASE / "results" / "audit_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to results/audit_results.json")
