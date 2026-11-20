# -*- coding: utf-8 -*-
"""robustness_tests.py - additional robustness tests (November 2026).

Run from the repo root:
    python3 direction2/robustness_tests.py              # all tests whose data are present
    python3 direction2/robustness_tests.py --wrds       # also download the retail/institutional split (Test 4)
    python3 direction2/robustness_tests.py --only 1 3 6 # run selected tests

Tests
  1  Robust standard errors (HC1, Newey-West, clustered by month, two-way, wild bootstrap)
  2  Official Jarocinski-Karadi shocks in place of the daily sign rule
  3  Beta-based risk ranking, continuous beta, downside beta, sign asymmetry
  4  Retail vs institutional share classes (needs --wrds once)
  5  Credit spread x inflation (downloads BAA, AAA, CPI from FRED)
  6  Power: simple-first regressions, minimum detectable effects, placebo FG regimes,
     multiple-testing corrections over the 42 H4 tests
  7  Fed information: do CBI shocks / tone predict SPF forecast revisions?

Dependencies: numpy, pandas, scipy (statsmodels is NOT needed; all estimators are below).
Outputs: direction2/flow_rerun_output_v8/robustness_tests/ (results.json + report.md)
"""
import argparse
import json
import os
import sys
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats

D2 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(D2)
FLOWS = os.path.join(D2, "flow_rerun_output_v8", "flows_by_class_month_v8.csv")
FOMC = os.path.join(ROOT, "results", "fomc_dataset_2006_latest.csv")
MKT_M = os.path.join(ROOT, "data", "wrds", "crsp_msi_index.csv")
OUT = os.path.join(D2, "flow_rerun_output_v8", "robustness_tests")

# Test 2: the published Jarocinski-Karadi meeting-level file. Download it from
# Marek Jarocinski's website / GitHub, save it here, and set the column names.
JK_FILE = os.path.join(ROOT, "data", "jk_shocks_meeting.csv")
JK_DATE_COL, JK_MP_COL, JK_CBI_COL = "date", "mp_jk", "cbi_jk"

# Test 4: written by --wrds
INVESTOR_FILE = os.path.join(D2, "flow_rerun_output_v8", "flows_by_class_investor_month.csv")

# Test 5: FRED series (monthly). Cached after the first download.
FRED_CACHE = os.path.join(ROOT, "data", "fred_cache")
FRED_SERIES = {"BAA": "BAA", "AAA": "AAA", "CPI": "CPIAUCSL"}

# Test 7: Philadelphia Fed Survey of Professional Forecasters, median forecasts.
# Download from philadelphiafed.org (SPF > Data files > Median forecasts) into data/spf/.
SPF_DIR = os.path.join(ROOT, "data", "spf")
SPF_FILES = {"rgdp": ("Median_RGDP_Growth.xlsx", "DRGDP"),
             "unemp": ("Median_UNEMP_Level.xlsx", "UNEMP"),
             "cpi": ("Median_CPI_Level.xlsx", "CPI")}

RISK = {"government_bonds": 1, "corporate_bonds": 2, "real_assets": 3, "large_cap_equity": 4,
        "developed_market_equity": 5, "emerging_market_equity": 6, "small_cap_equity": 7}
CONTROLS = ["log_tna", "flow_vol_12m"]


# ════════════════════════════════════════════════════════════════════
# Estimation: OLS with classical / HC1 / Newey-West / clustered / two-way SEs
# ════════════════════════════════════════════════════════════════════
class Fit:
    def __init__(self, names, b, V, n, k, r2, df):
        self.names, self.b, self.V, self.n, self.k, self.r2, self.df = names, b, V, n, k, r2, df
        self.se = np.sqrt(np.diag(V))
        self.t = b / self.se
        self.p = 2 * stats.t.sf(np.abs(self.t), df)

    def __getitem__(self, name):
        i = self.names.index(name)
        return {"beta": float(self.b[i]), "se": float(self.se[i]), "t": float(self.t[i]),
                "p": float(self.p[i])}

    def wald_equal(self, a, c):
        i, j = self.names.index(a), self.names.index(c)
        d = self.b[i] - self.b[j]
        v = self.V[i, i] + self.V[j, j] - 2 * self.V[i, j]
        return float(2 * stats.t.sf(abs(d / np.sqrt(v)), self.df))

    def row(self, keep=None):
        keep = keep or [n for n in self.names if n != "const"]
        out = {"n": int(self.n), "r2": round(float(self.r2), 4)}
        for k in keep:
            out[k] = {kk: round(vv, 4) for kk, vv in self[k].items()}
        return out


def _meat_cluster(X, e, g):
    codes = pd.factorize(pd.Series(g).astype(str))[0]
    G = codes.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, X * e[:, None])
    return S.T @ S, G


def ols(df, y, xs, cov="classical", groups=None, lags=4):
    """OLS of df[y] on a constant and df[xs].
    cov: classical | hc1 | hac | cluster (groups=col) | twoway (groups=[col1, col2])."""
    cols = [y] + list(xs) + ([groups] if isinstance(groups, str) else list(groups or []))
    d = df.dropna(subset=[c for c in cols if c in df.columns])
    Y = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    names = ["const"] + list(xs)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ Y
    e = Y - X @ b
    r2 = 1 - (e @ e) / ((Y - Y.mean()) @ (Y - Y.mean()))
    df_t = n - k
    if cov == "classical":
        V = XtXi * (e @ e) / (n - k)
    elif cov == "hc1":
        V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - k)
    elif cov == "hac":  # Newey-West, Bartlett kernel; rows must be in time order
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for L in range(1, lags + 1):
            w = 1 - L / (lags + 1)
            G = Xe[L:].T @ Xe[:-L]
            S += w * (G + G.T)
        V = XtXi @ S @ XtXi * n / (n - k)
    elif cov == "cluster":
        S, G = _meat_cluster(X, e, d[groups])
        V = XtXi @ S @ XtXi * G / (G - 1) * (n - 1) / (n - k)
        df_t = G - 1
    elif cov == "twoway":  # Cameron-Gelbach-Miller
        g1, g2 = d[groups[0]].astype(str), d[groups[1]].astype(str)
        S1, G1 = _meat_cluster(X, e, g1)
        S2, G2 = _meat_cluster(X, e, g2)
        S12, G12 = _meat_cluster(X, e, g1 + "|" + g2)
        c = lambda G: G / (G - 1) * (n - 1) / (n - k)
        V = XtXi @ (c(G1) * S1 + c(G2) * S2 - c(G12) * S12) @ XtXi
        if np.any(np.diag(V) <= 0):  # fall back to the larger one-way matrix
            V = XtXi @ (c(G1) * S1) @ XtXi
        df_t = min(G1, G2) - 1
    else:
        raise ValueError(cov)
    return Fit(names, b, V, n, k, r2, df_t)


def wild_bootstrap_p(df, y, xs, test, reps=9999, seed=7):
    """Wild bootstrap-t (Rademacher weights, null imposed) for H0: beta_test = 0, HC1 t-stat."""
    d = df.dropna(subset=[y] + list(xs)).reset_index(drop=True)
    t_obs = ols(d, y, xs, "hc1")[test]["t"]
    rest = [x for x in xs if x != test]
    fr = ols(d, y, rest, "classical")
    Xr = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in rest])
    fitted = Xr @ fr.b
    u = d[y].to_numpy(float) - fitted
    rng = np.random.default_rng(seed)
    hits = 0
    dd = d.copy()
    for _ in range(reps):
        dd[y] = fitted + u * rng.choice([-1.0, 1.0], len(d))
        if abs(ols(dd, y, xs, "hc1")[test]["t"]) >= abs(t_obs):
            hits += 1
    return (hits + 1) / (reps + 1)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)[::-1]
    adj = np.empty(m); run = 1
    for r, i in enumerate(o):
        run = min(run, p[i] * m / (m - r)); adj[i] = run
    return adj


# ════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════
def load_fomc(drop_no_return=True):
    """Meeting file (161 announcements). MP/CBI by the daily sign rule.

    drop_no_return=True fixes a bug in run_flow_chapter_local.load_fomc():
    np.sign(NaN) != sign(target) is True, so the 2025-26 meetings, which have no
    CRSP return yet, were all coded as MP shocks (8 of the 91 'MP' meetings).
    Here they get MP = CBI = NaN and drop out of the flow regressions."""
    f = pd.read_csv(FOMC, parse_dates=["date"]).set_index("date")
    f["tone100"] = f["sentiment"] * 100
    same = np.sign(f["target_shock"]) == np.sign(f["crsp_vw"])
    f["mp_shock"] = np.where(~same, f["target_shock"], 0.0)
    f["cbi_shock"] = np.where(same, f["target_shock"], 0.0)
    if drop_no_return:
        f.loc[f["crsp_vw"].isna(), ["mp_shock", "cbi_shock"]] = np.nan
    f["fg"] = f["fg_2008_2015"]
    f["tone_x_fg"] = f["tone100"] * f["fg"]
    # tone orthogonalized to Target and Path (residual, full sample)
    fo = ols(f.reset_index(), "tone100", ["target_shock", "path_shock"])
    f["tone_orth"] = f["tone100"] - (fo.b[0] + fo.b[1] * f["target_shock"] + fo.b[2] * f["path_shock"])
    return f


def load_flows(path=FLOWS, by=("asset_class",)):
    a = pd.read_csv(path, parse_dates=["month"])
    a["flow_pct"] = a["dollar_flow"] / a["tna_lag"]
    a["log_tna"] = np.log(a["tna_lag"].where(a["tna_lag"] > 0))
    a = a.sort_values(list(by) + ["month"])
    a["flow_vol_12m"] = a.groupby(list(by))["flow_pct"].transform(
        lambda s: s.rolling(12, min_periods=6).std())
    return a


def panel(agg, fomc, window="same"):
    """Class-month flows matched to meetings. window: same (meeting month),
    post (month after), diff (post minus same). Flows in % of lagged TNA."""
    a = agg.copy()
    keys = [c for c in a.columns if c in ("asset_class", "investor")]
    if window == "diff":
        nxt = a[keys + ["month", "flow_pct"]].copy()
        nxt["month"] = nxt["month"] - pd.DateOffset(months=1)
        a = a.merge(nxt.rename(columns={"flow_pct": "f_next"}), on=keys + ["month"], how="inner")
        a["flow_pct"] = a["f_next"] - a["flow_pct"]
        a["key"] = a["month"].dt.to_period("M")
    else:
        shift = 1 if window == "post" else 0
        a["key"] = (a["month"] - pd.DateOffset(months=shift)).dt.to_period("M")
    fm = fomc.reset_index()
    fm["key"] = fm["date"].dt.to_period("M")
    p = a.merge(fm, on="key", how="inner")
    p["flow"] = p["flow_pct"] * 100
    p["risk"] = p["asset_class"].map(RISK)
    p["month_id"] = p["key"].astype(str)
    for s in ("mp", "cbi"):
        p[f"risk_x_{s}"] = p["risk"] * p[f"{s}_shock"]
        p[f"{s}_pos"] = p[f"{s}_shock"].clip(lower=0)
        p[f"{s}_neg"] = p[f"{s}_shock"].clip(upper=0)
        p[f"risk_x_{s}_pos"] = p["risk"] * p[f"{s}_pos"]
        p[f"risk_x_{s}_neg"] = p["risk"] * p[f"{s}_neg"]
    return p.sort_values(["asset_class", "date"])


H5_X = ["mp_shock", "cbi_shock", "risk_x_mp", "risk_x_cbi"] + CONTROLS
H4_X = ["mp_shock", "cbi_shock"] + CONTROLS


def h4_grid(agg, fomc, cov="classical"):
    out = {}
    for w in ("same", "post", "diff"):
        p = panel(agg, fomc, w)
        for c in RISK:
            f = ols(p[p["asset_class"] == c], "flow", H4_X, cov)
            out[(c, w)] = {"MP": f["mp_shock"], "CBI": f["cbi_shock"], "n": f.n,
                           "wald_p": f.wald_equal("mp_shock", "cbi_shock")}
    return out


# ════════════════════════════════════════════════════════════════════
# Test 1 - robust standard errors
# ════════════════════════════════════════════════════════════════════
def test1(fomc, agg):
    res = {}
    pr = fomc.dropna(subset=["crsp_vw"]).sort_index().reset_index()
    fg = pr[pr["fg"] == 1]
    specs = {
        "M2 full: R ~ Target + Path + Tone": (pr, ["target_shock", "path_shock", "tone100"], "tone100"),
        "M3 full: + FG + Tone x FG": (pr, ["target_shock", "path_shock", "tone100", "fg", "tone_x_fg"], "tone_x_fg"),
        "FG only: R ~ Target + Path + Tone": (fg, ["target_shock", "path_shock", "tone100"], "tone100"),
    }
    for name, (d, xs, key) in specs.items():
        r = {cov: ols(d, "crsp_vw", xs, cov)[key] for cov in ("classical", "hc1", "hac")}
        r["wild_bootstrap_p"] = wild_bootstrap_p(d, "crsp_vw", xs, key, reps=4999)
        r["n"] = int(ols(d, "crsp_vw", xs).n)
        res[name] = {"coef": key, **r}

    p = panel(agg, fomc, "same")
    gov = p[p["asset_class"] == "government_bonds"]
    res["H4 gov bonds, same month: CBI"] = {
        "coef": "cbi_shock", "n": int(ols(gov, "flow", H4_X).n),
        **{cov: ols(gov, "flow", H4_X, cov)["cbi_shock"] for cov in ("classical", "hc1", "hac")}}
    for w in ("same", "post"):
        p = panel(agg, fomc, w)
        res[f"H5 pooled ({w}): Risk x CBI"] = {
            "coef": "risk_x_cbi", "n": int(ols(p, "flow", H5_X).n),
            "classical": ols(p, "flow", H5_X)["risk_x_cbi"],
            "hc1": ols(p, "flow", H5_X, "hc1")["risk_x_cbi"],
            "cluster_month": ols(p, "flow", H5_X, "cluster", "month_id")["risk_x_cbi"],
            "twoway_month_class": ols(p, "flow", H5_X, "twoway", ["month_id", "asset_class"])["risk_x_cbi"]}

    # Influence: leave one meeting out, and drop the autumn-2008 crisis months
    p = panel(agg, fomc, "same")
    loo = []
    for d in sorted(p["date"].unique()):
        q = p[p["date"] != d]
        loo.append({"dropped": str(pd.Timestamp(d).date()),
                    "gov_cbi": ols(q[q["asset_class"] == "government_bonds"], "flow", H4_X)["cbi_shock"]["beta"],
                    "risk_x_cbi": ols(q, "flow", H5_X)["risk_x_cbi"]["beta"]})
    loo = pd.DataFrame(loo)
    res["influence"] = {
        "most_influential_for_gov_cbi": loo.loc[(loo["gov_cbi"] - 0.763).abs().idxmax()].to_dict(),
        "most_influential_for_risk_x_cbi": loo.loc[(loo["risk_x_cbi"] + 0.066).abs().idxmax()].to_dict()}
    for label, mask in (("excluding 2008-09-16", p["date"] != "2008-09-16"),
                        ("excluding Sep-Dec 2008", ~p["date"].between("2008-09-01", "2008-12-31"))):
        q = p[mask]
        gov = q[q["asset_class"] == "government_bonds"]
        res["influence"][label] = {"gov_cbi_hc1": ols(gov, "flow", H4_X, "hc1")["cbi_shock"],
                                   "risk_x_cbi_cluster": ols(q, "flow", H5_X, "cluster", "month_id")["risk_x_cbi"]}
    return res


# ════════════════════════════════════════════════════════════════════
# Test 2 - official Jarocinski-Karadi shocks
# ════════════════════════════════════════════════════════════════════
def test2(fomc, agg):
    if not os.path.exists(JK_FILE):
        return {"skipped": f"Save the JK meeting-level file as {JK_FILE} and set JK_*_COL at the top."}
    jk = pd.read_csv(JK_FILE)
    jk["date"] = pd.to_datetime(jk[JK_DATE_COL]).dt.normalize()
    jk = jk.set_index("date")[[JK_MP_COL, JK_CBI_COL]].rename(columns={JK_MP_COL: "mp_jk", JK_CBI_COL: "cbi_jk"})
    f = fomc.join(jk, how="left")
    out = {"matched_meetings": int(f["mp_jk"].notna().sum()),
           "corr_mp_jk_vs_sign_rule": float(f[["mp_jk", "mp_shock"]].corr().iloc[0, 1]),
           "corr_cbi_jk_vs_sign_rule": float(f[["cbi_jk", "cbi_shock"]].corr().iloc[0, 1]),
           "corr_mp_jk_cbi_jk": float(f[["mp_jk", "cbi_jk"]].corr().iloc[0, 1])}
    g = f.copy()
    g["mp_shock"], g["cbi_shock"] = g["mp_jk"], g["cbi_jk"]
    for w in ("same", "post", "diff"):
        p = panel(agg, g, w)
        gov = p[p["asset_class"] == "government_bonds"]
        fg_ = ols(gov, "flow", H4_X, "hc1")
        out[f"H4 gov bonds ({w})"] = {"MP": fg_["mp_shock"], "CBI": fg_["cbi_shock"], "n": fg_.n,
                                      "wald_p": fg_.wald_equal("mp_shock", "cbi_shock")}
        f5 = ols(p, "flow", H5_X, "cluster", "month_id")
        out[f"H5 pooled ({w})"] = f5.row(["risk_x_mp", "risk_x_cbi"])
    return out


# ════════════════════════════════════════════════════════════════════
# Test 3 - beta-based risk
# ════════════════════════════════════════════════════════════════════
def class_betas(agg):
    """Class value-weighted return backed out of the flow identity:
    Flow = TNA_t - TNA_{t-1}(1+R)  =>  R = (TNA_t - Flow)/TNA_{t-1} - 1.
    Market = CRSP value-weighted index (monthly, through Dec 2024)."""
    a = agg.copy()
    a["ret"] = (a["tna"] - a["dollar_flow"]) / a["tna_lag"] - 1
    m = pd.read_csv(MKT_M, parse_dates=["date"])
    m["p"] = m["date"].dt.to_period("M")
    a["p"] = a["month"].dt.to_period("M")
    a = a.merge(m[["p", "vwretd"]], on="p", how="left")
    static, rolling = {}, []
    for c, g in a.groupby("asset_class"):
        g = g.dropna(subset=["ret", "vwretd"]).sort_values("p")
        x, y = g["vwretd"].to_numpy(), g["ret"].to_numpy()
        down = x < 0
        static[c] = {"beta": float(np.cov(x, y)[0, 1] / np.var(x, ddof=1)),
                     "downside_beta": float(np.cov(x[down], y[down])[0, 1] / np.var(x[down], ddof=1)),
                     "upside_beta": float(np.cov(x[~down], y[~down])[0, 1] / np.var(x[~down], ddof=1)),
                     "mean_ret_pct": float(y.mean() * 100), "months": int(len(g))}
        # rolling 36-month beta using months t-36..t-1 only (no look-ahead)
        s = g.set_index("p")[["ret", "vwretd"]]
        cov = s["ret"].rolling(36, min_periods=24).cov(s["vwretd"])
        var = s["vwretd"].rolling(36, min_periods=24).var()
        rb = (cov / var).shift(1).rename("beta_roll").reset_index()
        rb["asset_class"] = c
        rolling.append(rb)
    roll = pd.concat(rolling)
    roll["rank_roll"] = roll.groupby("p")["beta_roll"].rank()
    return static, roll


def test3(fomc, agg):
    static, roll = class_betas(agg)
    order = sorted(static, key=lambda c: static[c]["beta"])
    out = {"class_betas": static,
           "beta_order": order,
           "asset_class_order": list(RISK)}
    beta = {c: static[c]["beta"] for c in static}
    dbeta = {c: static[c]["downside_beta"] for c in static}
    brank = {c: i + 1 for i, c in enumerate(order)}
    for w in ("same", "post"):
        p = panel(agg, fomc, w)
        p["p"] = p["month"].dt.to_period("M")
        p = p.merge(roll[["p", "asset_class", "beta_roll", "rank_roll"]], on=["p", "asset_class"], how="left")
        variants = {"A. asset-class rank (current)": p["risk"],
                    "B. full-sample beta rank": p["asset_class"].map(brank),
                    "C. continuous beta": p["asset_class"].map(beta),
                    "D. downside beta": p["asset_class"].map(dbeta),
                    "E. rolling 36m beta (lagged; from 2008)": p["beta_roll"],
                    "F. rolling 36m beta rank (lagged; from 2008)": p["rank_roll"]}
        block = {}
        for name, r in variants.items():
            q = p.copy()
            q["rk"] = r
            for s in ("mp", "cbi"):
                q[f"rk_x_{s}"] = q["rk"] * q[f"{s}_shock"]
                q[f"rk_x_{s}_pos"] = q["rk"] * q[f"{s}_pos"]
                q[f"rk_x_{s}_neg"] = q["rk"] * q[f"{s}_neg"]
            xs = ["mp_shock", "cbi_shock", "rk_x_mp", "rk_x_cbi"] + CONTROLS
            f = ols(q, "flow", xs, "cluster", "month_id")
            xs2 = ["mp_shock", "cbi_shock", "rk_x_mp", "rk_x_cbi_pos", "rk_x_cbi_neg"] + CONTROLS
            f2 = ols(q, "flow", xs2, "cluster", "month_id")
            block[name] = {"n": f.n, "risk_x_MP": f["rk_x_mp"], "risk_x_CBI": f["rk_x_cbi"],
                           "risk_x_CBI_positive": f2["rk_x_cbi_pos"], "risk_x_CBI_negative": f2["rk_x_cbi_neg"],
                           "wald_pos_eq_neg_p": f2.wald_equal("rk_x_cbi_pos", "rk_x_cbi_neg")}
            if name.startswith("A"):  # drop government bonds: is it a ladder or a gov-bond effect?
                fd = ols(q[q["asset_class"] != "government_bonds"], "flow", xs, "cluster", "month_id")
                block["A'. asset-class rank, excluding gov bonds"] = {"n": fd.n, "risk_x_CBI": fd["rk_x_cbi"]}
        out[w] = block
    return out


# ════════════════════════════════════════════════════════════════════
# Test 4 - retail vs institutional
# ════════════════════════════════════════════════════════════════════
INVESTOR_SQL_NOTE = """
Same SQL as flow_server_agg.py, with the CRSP header history joined at each fund-month
so that the retail/institutional flag is point-in-time:

  hdr AS (
    SELECT crsp_fundno, chgdt, COALESCE(chgenddt, '9999-12-31') AS chgenddt,
           CASE WHEN inst_fund = 'Y' THEN 'institutional'
                WHEN retail_fund = 'Y' THEN 'retail' ELSE 'unknown' END AS investor
    FROM crsp_q_mutualfunds.fund_hdr_hist)
  ... JOIN hdr h ON m.crsp_fundno = h.crsp_fundno AND m.caldt BETWEEN h.chgdt AND h.chgenddt
  ... GROUP BY asset_class, investor, month
"""


def fetch_investor_split():
    sys.path.insert(0, D2)
    import wrds  # noqa
    from flow_server_agg import SQL
    sql = SQL.replace(
        "j AS (",
        """hdr AS (
  SELECT crsp_fundno, chgdt, COALESCE(chgenddt, DATE '9999-12-31') AS chgenddt,
         CASE WHEN inst_fund = 'Y' THEN 'institutional'
              WHEN retail_fund = 'Y' THEN 'retail' ELSE 'unknown' END AS investor
  FROM crsp_q_mutualfunds.fund_hdr_hist WHERE crsp_fundno IS NOT NULL
),
j AS (""", 1)
    sql = sql.replace("SELECT m.crsp_fundno, m.caldt, m.mret, m.mtna, m.mtna_prev, c.asset_class",
                      "SELECT m.crsp_fundno, m.caldt, m.mret, m.mtna, m.mtna_prev, c.asset_class, h.investor")
    sql = sql.replace("ON m.crsp_fundno = c.crsp_fundno AND m.caldt BETWEEN c.begdt AND c.enddt",
                      "ON m.crsp_fundno = c.crsp_fundno AND m.caldt BETWEEN c.begdt AND c.enddt\n"
                      "  JOIN hdr h ON m.crsp_fundno = h.crsp_fundno AND m.caldt BETWEEN h.chgdt AND h.chgenddt")
    sql = sql.replace("SELECT asset_class, caldt, crsp_fundno, mtna_prev, mtna,",
                      "SELECT asset_class, investor, caldt, crsp_fundno, mtna_prev, mtna,")
    sql = sql.replace("SELECT asset_class, date_trunc('month', caldt)::date AS month,",
                      "SELECT asset_class, investor, date_trunc('month', caldt)::date AS month,")
    sql = sql.replace("FROM f GROUP BY 1, 2 ORDER BY 1, 2", "FROM f GROUP BY 1, 2, 3 ORDER BY 1, 2, 3")
    conn = wrds.Connection()
    df = conn.raw_sql(sql)
    conn.close()
    df.to_csv(INVESTOR_FILE, index=False)
    print(f"saved {INVESTOR_FILE}: {len(df)} rows")


def test4(fomc):
    if not os.path.exists(INVESTOR_FILE):
        return {"skipped": "Run once with --wrds to build " + INVESTOR_FILE, "sql": INVESTOR_SQL_NOTE}
    a = load_flows(INVESTOR_FILE, by=("asset_class", "investor"))
    a = a[a["investor"].isin(["retail", "institutional"])]
    out = {"share_of_tna_institutional": float(
        a[a.investor == "institutional"]["tna"].sum() / a["tna"].sum())}
    for w in ("same", "post", "diff"):
        p = panel(a, fomc, w)
        p["inst"] = (p["investor"] == "institutional").astype(float)
        p["cbi_x_inst"] = p["cbi_shock"] * p["inst"]
        p["mp_x_inst"] = p["mp_shock"] * p["inst"]
        p["risk_x_cbi_x_inst"] = p["risk_x_cbi"] * p["inst"]
        p["risk_x_mp_x_inst"] = p["risk_x_mp"] * p["inst"]
        p["risk_x_inst"] = p["risk"] * p["inst"]
        blk = {}
        for t in ("retail", "institutional"):
            gov = p[(p["asset_class"] == "government_bonds") & (p["investor"] == t)]
            fg_ = ols(gov, "flow", H4_X, "hc1")
            blk[f"gov bonds, {t}"] = {"MP": fg_["mp_shock"], "CBI": fg_["cbi_shock"], "n": fg_.n}
            f5 = ols(p[p["investor"] == t], "flow", H5_X, "cluster", "month_id")
            blk[f"H5 pooled, {t}"] = f5.row(["risk_x_mp", "risk_x_cbi"])
        xs = H5_X + ["inst", "risk_x_inst", "mp_x_inst", "cbi_x_inst", "risk_x_mp_x_inst", "risk_x_cbi_x_inst"]
        f = ols(p, "flow", xs, "cluster", "month_id")
        blk["pooled with institutional interactions"] = f.row(
            ["cbi_shock", "cbi_x_inst", "risk_x_cbi", "risk_x_cbi_x_inst"])
        out[w] = blk
    return out


# ════════════════════════════════════════════════════════════════════
# Test 5 - credit spread x inflation
# ════════════════════════════════════════════════════════════════════
def fred(series):
    os.makedirs(FRED_CACHE, exist_ok=True)
    path = os.path.join(FRED_CACHE, f"{series}.csv")
    if not os.path.exists(path):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        with urllib.request.urlopen(url, timeout=30) as r:
            open(path, "wb").write(r.read())
    s = pd.read_csv(path)
    s.columns = ["date", series]
    s["date"] = pd.to_datetime(s["date"])
    s[series] = pd.to_numeric(s[series], errors="coerce")
    return s.set_index("date")[series]


def macro_state():
    """Monthly BAA-AAA spread (pp) and CPI inflation (% y/y), lagged one month
    so that they are known before the meeting."""
    baa, aaa, cpi = (fred(FRED_SERIES[k]) for k in ("BAA", "AAA", "CPI"))
    m = pd.DataFrame({"spread": baa - aaa, "infl": cpi.pct_change(12) * 100})
    m.index = m.index.to_period("M")
    return m.shift(1)


def test5(fomc, agg):
    try:
        ms = macro_state()
    except Exception as e:  # noqa
        return {"skipped": f"FRED download failed ({e}). Save BAA.csv, AAA.csv, CPIAUCSL.csv "
                           f"from fred.stlouisfed.org into {FRED_CACHE}."}
    f = fomc.copy()
    f["p"] = f.index.to_period("M")
    f = f.join(ms, on="p")
    out = {"spread_median": float(f["spread"].median()), "infl_median": float(f["infl"].median()),
           "spread_range": [float(f["spread"].min()), float(f["spread"].max())]}
    for state in ("spread", "infl"):
        f[f"high_{state}"] = (f[state] > f[state].median()).astype(float)
        f[f"z_{state}"] = (f[state] - f[state].mean()) / f[state].std()
    for w in ("same", "post"):
        p = panel(agg, f, w)
        blk = {}
        for state in ("spread", "infl"):
            for kind in ("high", "z"):
                s = f"{kind}_{state}"
                q = p.copy()
                q["cbi_x_s"] = q["cbi_shock"] * q[s]
                q["mp_x_s"] = q["mp_shock"] * q[s]
                q["risk_x_s"] = q["risk"] * q[s]
                q["risk_x_cbi_x_s"] = q["risk_x_cbi"] * q[s]
                q["risk_x_mp_x_s"] = q["risk_x_mp"] * q[s]
                xs = H5_X + [s, "risk_x_s", "mp_x_s", "cbi_x_s", "risk_x_mp_x_s", "risk_x_cbi_x_s"]
                fit = ols(q, "flow", xs, "cluster", "month_id")
                gov = q[q["asset_class"] == "government_bonds"]
                fg_ = ols(gov, "flow", H4_X + [s, "mp_x_s", "cbi_x_s"], "hc1")
                blk[s] = {"H5 risk_x_CBI": fit["risk_x_cbi"], "H5 risk_x_CBI_x_state": fit["risk_x_cbi_x_s"],
                          "gov CBI": fg_["cbi_shock"], "gov CBI_x_state": fg_["cbi_x_s"], "n": fit.n,
                          "cbi_meetings_in_high_state": int(((f["cbi_shock"].fillna(0) != 0) & (f.get(f"high_{state}", 0) == 1)).sum())}
        out[w] = blk
    return out


# ════════════════════════════════════════════════════════════════════
# Test 6 - power, placebo regimes, multiple testing
# ════════════════════════════════════════════════════════════════════
def test6(fomc, agg, perms=5000, seed=11):
    out = {}
    pr = fomc.dropna(subset=["crsp_vw"]).sort_index().reset_index()
    seq = {"1. Tone only": ["tone100"],
           "2. + Target, Path": ["target_shock", "path_shock", "tone100"],
           "3. + FG, Tone x FG": ["target_shock", "path_shock", "tone100", "fg", "tone_x_fg"]}
    out["simple_first_full_sample"] = {k: ols(pr, "crsp_vw", v, "hc1").row() for k, v in seq.items()}
    fgd = pr[pr["fg"] == 1]
    out["simple_first_FG_only"] = {k: ols(fgd, "crsp_vw", v, "hc1").row() for k, v in list(seq.items())[:2]}

    # minimum detectable effect, 80% power, 5% two-sided: (1.96 + 0.84) x SE
    mde = {}
    f = ols(pr, "crsp_vw", seq["2. + Target, Path"], "hc1")["tone100"]
    mde["Tone on returns, full sample"] = {"beta": f["beta"], "se": f["se"], "mde": 2.8 * f["se"]}
    f = ols(fgd, "crsp_vw", seq["2. + Target, Path"], "hc1")["tone100"]
    mde["Tone on returns, FG only"] = {"beta": f["beta"], "se": f["se"], "mde": 2.8 * f["se"]}
    p = panel(agg, fomc, "same")
    for c in ("government_bonds", "corporate_bonds", "large_cap_equity", "small_cap_equity"):
        f = ols(p[p["asset_class"] == c], "flow", H4_X, "hc1")["cbi_shock"]
        mde[f"CBI on {c} flow (same)"] = {"beta": f["beta"], "se": f["se"], "mde": 2.8 * f["se"]}
    out["minimum_detectable_effect"] = mde

    # placebo regimes for Tone x FG
    xs = seq["3. + FG, Tone x FG"]
    actual = ols(pr, "crsp_vw", xs)["tone_x_fg"]["beta"]
    rng = np.random.default_rng(seed)
    n, nfg = len(pr), int(pr["fg"].sum())
    draws = []
    q = pr.copy()
    for _ in range(perms):
        q["fg"] = 0.0
        q.loc[rng.choice(n, nfg, replace=False), "fg"] = 1.0
        q["tone_x_fg"] = q["tone100"] * q["fg"]
        draws.append(ols(q, "crsp_vw", xs)["tone_x_fg"]["beta"])
    draws = np.array(draws)
    blocks = []
    for s in range(0, n - nfg + 1):
        q["fg"] = 0.0
        q.loc[s:s + nfg - 1, "fg"] = 1.0
        q["tone_x_fg"] = q["tone100"] * q["fg"]
        blocks.append({"start": str(q.loc[s, "date"].date()), "beta": ols(q, "crsp_vw", xs)["tone_x_fg"]["beta"]})
    bb = np.array([b["beta"] for b in blocks])
    out["placebo_FG"] = {
        "actual_tone_x_fg": actual,
        "random_56_meetings_p": float((np.abs(draws) >= abs(actual)).mean()),
        "consecutive_56_blocks": len(blocks),
        "share_of_blocks_at_least_as_negative": float((bb <= actual).mean()),
        "most_negative_block": min(blocks, key=lambda b: b["beta"])}

    # multiple testing across the 42 H4 tests (7 classes x 3 windows x MP/CBI)
    for cov in ("classical", "hc1"):
        g = h4_grid(agg, fomc, cov)
        rows = [(c, w, s, v[s]["beta"], v[s]["p"]) for (c, w), v in g.items() for s in ("MP", "CBI")]
        pv = np.array([r[4] for r in rows])
        hp, bp = holm(pv), bh(pv)
        out[f"H4_42_tests_{cov}"] = {
            "raw_p_below_05": int((pv < .05).sum()),
            "holm_below_05": int((hp < .05).sum()), "bh_below_05": int((bp < .05).sum()),
            "smallest": [{"class": r[0], "window": r[1], "shock": r[2], "beta": round(r[3], 3),
                          "p": round(r[4], 4), "holm": round(h, 3), "bh": round(b, 3)}
                         for r, h, b in sorted(zip(rows, hp, bp), key=lambda z: z[0][4])[:8]]}
    return out


# ════════════════════════════════════════════════════════════════════
# Test 7 - do CBI shocks / tone predict forecast revisions?
# ════════════════════════════════════════════════════════════════════
def spf_revisions():
    """Revision of the forecast for next quarter: x3 (survey q) - x4 (survey q-1).
    SPF deadline approximated as the 15th of the middle month of each quarter."""
    out = None
    for key, (fname, pre) in SPF_FILES.items():
        path = os.path.join(SPF_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        s = pd.read_excel(path)
        s.columns = [str(c).upper() for c in s.columns]
        s = s.sort_values(["YEAR", "QUARTER"]).reset_index(drop=True)
        s[f"rev_{key}"] = pd.to_numeric(s[f"{pre}3"], errors="coerce") - \
            pd.to_numeric(s[f"{pre}4"], errors="coerce").shift(1)
        s["deadline"] = pd.to_datetime(dict(year=s["YEAR"], month=3 * s["QUARTER"] - 1, day=15))
        s = s[["deadline", f"rev_{key}"]]
        out = s if out is None else out.merge(s, on="deadline")
    return out


def test7(fomc):
    try:
        rv = spf_revisions()
    except FileNotFoundError as e:
        return {"skipped": f"Missing {e}. Download the SPF median-forecast files into {SPF_DIR}."}
    rv = rv[(rv["deadline"] > "2006-02-15") & (rv["deadline"] <= "2025-02-15")].reset_index(drop=True)
    f = fomc.dropna(subset=["mp_shock"]).copy()
    agg_rows = []
    for i in range(1, len(rv)):
        lo, hi = rv.loc[i - 1, "deadline"], rv.loc[i, "deadline"]
        w = f[(f.index > lo) & (f.index <= hi)]
        agg_rows.append({"deadline": hi, "mp": w["mp_shock"].sum(), "cbi": w["cbi_shock"].sum(),
                         "tone": w["tone_orth"].mean(), "n_meet": len(w)})
    q = rv.merge(pd.DataFrame(agg_rows), on="deadline")
    out = {"quarters": int(len(q))}
    for y in ("rev_rgdp", "rev_unemp", "rev_cpi"):
        f1 = ols(q, y, ["mp", "cbi"], "hc1")
        f2 = ols(q, y, ["mp", "cbi", "tone"], "hc1")
        out[y] = {"MP": f1["mp"], "CBI": f1["cbi"], "wald_MP_eq_CBI_p": f1.wald_equal("mp", "cbi"),
                  "with_tone": f2.row(["mp", "cbi", "tone"])}
    return out


# ════════════════════════════════════════════════════════════════════
def fmt(d):
    return f"{d['beta']:+.3f} (se {d['se']:.3f}, p {d['p']:.3f})"


def report(R):
    L = ["# Robustness tests: results", ""]
    if "baseline_check" in R:
        b = R["baseline_check"]
        L += ["## Baseline check (classical SEs)", "",
              "| Version | MP meetings | CBI meetings | Gov CBI (same) | Risk x CBI (same) |", "|---|---|---|---|---|"]
        for k, v in b.items():
            L.append(f"| {k} | {v['n_mp']} | {v['n_cbi']} | {fmt(v['gov_cbi'])} | {fmt(v['risk_x_cbi'])} |")
        L.append("")
    if "test1" in R:
        L += ["## Test 1. Robust standard errors", "", "| Result | N | Classical | HC1 | Other |", "|---|---|---|---|---|"]
        for k, v in R["test1"].items():
            if k == "influence":
                continue
            other = []
            for kk in ("hac", "cluster_month", "twoway_month_class"):
                if kk in v:
                    other.append(f"{kk}: p {v[kk]['p']:.3f}")
            if "wild_bootstrap_p" in v:
                other.append(f"wild bootstrap p {v['wild_bootstrap_p']:.3f}")
            L.append(f"| {k} | {v['n']} | {fmt(v['classical'])} | p {v['hc1']['p']:.3f} | {'; '.join(other)} |")
        inf = R["test1"]["influence"]
        g, r = inf["most_influential_for_gov_cbi"], inf["most_influential_for_risk_x_cbi"]
        L += ["", "**Influence (leave one meeting out).** "
              f"Most influential meeting for gov-bond CBI: {g['dropped']} (beta without it {g['gov_cbi']:+.3f}). "
              f"For Risk x CBI: {r['dropped']} (beta without it {r['risk_x_cbi']:+.3f}).", ""]
        for k in ("excluding 2008-09-16", "excluding Sep-Dec 2008"):
            L.append(f"- {k}: gov CBI {fmt(inf[k]['gov_cbi_hc1'])} (HC1); "
                     f"Risk x CBI {fmt(inf[k]['risk_x_cbi_cluster'])} (clustered)")
        L.append("")
    for t, title in (("test2", "Official JK shocks"), ("test4", "Retail vs institutional"),
                     ("test5", "Credit spread x inflation"), ("test7", "SPF forecast revisions")):
        if t in R and "skipped" in R[t]:
            L += [f"## {t[-1]}. {title}", "", f"Skipped: {R[t]['skipped']}", ""]
    if "test3" in R:
        T = R["test3"]
        L += ["## Test 3. Beta-based risk", "", "| Class | Beta | Downside beta | Upside beta |", "|---|---|---|---|"]
        for c in T["asset_class_order"]:
            v = T["class_betas"][c]
            L.append(f"| {c} | {v['beta']:.2f} | {v['downside_beta']:.2f} | {v['upside_beta']:.2f} |")
        L += ["", "Order by beta: " + " < ".join(T["beta_order"]), ""]
        for w in ("same", "post"):
            L += [f"**{w} window** (SE clustered by meeting month)", "",
                  "| Risk measure | N | Risk x CBI | x CBI positive | x CBI negative | pos = neg p |", "|---|---|---|---|---|---|"]
            for k, v in T[w].items():
                if "risk_x_CBI_positive" in v:
                    L.append(f"| {k} | {v['n']} | {fmt(v['risk_x_CBI'])} | {fmt(v['risk_x_CBI_positive'])} | "
                             f"{fmt(v['risk_x_CBI_negative'])} | {v['wald_pos_eq_neg_p']:.3f} |")
                else:
                    L.append(f"| {k} | {v['n']} | {fmt(v['risk_x_CBI'])} | | | |")
            L.append("")
    if "test6" in R:
        T = R["test6"]
        L += ["## Test 6. Power and multiple testing", "", "Simple regressions first (HC1), full sample:", ""]
        for k, v in T["simple_first_full_sample"].items():
            L.append(f"- {k}: Tone {fmt(v['tone100'])}, R2 {v['r2']:.3f}, N {v['n']}"
                     + (f"; Tone x FG {fmt(v['tone_x_fg'])}" if "tone_x_fg" in v else ""))
        L += ["", "FG period only:", ""]
        for k, v in T["simple_first_FG_only"].items():
            L.append(f"- {k}: Tone {fmt(v['tone100'])}, R2 {v['r2']:.3f}, N {v['n']}")
        L += ["", "| Minimum detectable effect (80% power) | Estimate | SE | MDE |", "|---|---|---|---|"]
        for k, v in T["minimum_detectable_effect"].items():
            L.append(f"| {k} | {v['beta']:+.3f} | {v['se']:.3f} | {v['mde']:.3f} |")
        pf = T["placebo_FG"]
        L += ["", f"Placebo FG regimes: actual Tone x FG = {pf['actual_tone_x_fg']:+.3f}. "
              f"Random 56-meeting sets: p = {pf['random_56_meetings_p']:.3f}. "
              f"Consecutive 56-meeting blocks: {pf['consecutive_56_blocks']}, of which "
              f"{pf['share_of_blocks_at_least_as_negative']:.1%} are at least as negative "
              f"(most negative starts {pf['most_negative_block']['start']}, {pf['most_negative_block']['beta']:+.3f}).", ""]
        for cov in ("classical", "hc1"):
            v = T[f"H4_42_tests_{cov}"]
            L.append(f"- 42 H4 tests ({cov} SE): {v['raw_p_below_05']} with raw p < .05; "
                     f"{v['holm_below_05']} survive Holm; {v['bh_below_05']} survive Benjamini-Hochberg.")
        L += ["", "| Class | Window | Shock | Beta | p | Holm | BH |", "|---|---|---|---|---|---|---|"]
        for r in T["H4_42_tests_hc1"]["smallest"]:
            L.append(f"| {r['class']} | {r['window']} | {r['shock']} | {r['beta']:+.3f} | {r['p']:.4f} | {r['holm']:.3f} | {r['bh']:.3f} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--wrds", action="store_true", help="download the retail/institutional split first")
    ap.add_argument("--keep-unreturned-meetings", action="store_true",
                    help="reproduce the published numbers (2025-26 meetings coded MP)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.wrds:
        fetch_investor_split()

    agg = load_flows()
    fomc = load_fomc(drop_no_return=not a.keep_unreturned_meetings)
    R = {"meta": {"run_at": pd.Timestamp.now().isoformat(),
                  "meetings": int(len(fomc)), "with_return": int(fomc["crsp_vw"].notna().sum()),
                  "flow_months": int(agg["month"].nunique())}}

    # baseline check: published coding vs fixed coding
    base = {}
    for label, drop in (("As published (2025-26 coded MP)", False), ("Fixed (2025-26 dropped)", True)):
        f = load_fomc(drop_no_return=drop)
        p = panel(agg, f, "same")
        base[label] = {"n_mp": int((f["mp_shock"].fillna(0) != 0).sum()),
                       "n_cbi": int((f["cbi_shock"].fillna(0) != 0).sum()),
                       "gov_cbi": ols(p[p["asset_class"] == "government_bonds"], "flow", H4_X)["cbi_shock"],
                       "risk_x_cbi": ols(p, "flow", H5_X)["risk_x_cbi"]}
    R["baseline_check"] = base

    tests = {1: lambda: test1(fomc, agg), 2: lambda: test2(fomc, agg), 3: lambda: test3(fomc, agg),
             4: lambda: test4(fomc), 5: lambda: test5(fomc, agg), 6: lambda: test6(fomc, agg),
             7: lambda: test7(fomc)}
    for t in a.only:
        print(f"running test {t} ...", flush=True)
        R[f"test{t}"] = tests[t]()

    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    md = report(R)
    open(os.path.join(OUT, "report.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
