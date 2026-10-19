#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all_results.py
Re-run the Part-1 regressions on the corrected dataset
(results/fomc_dataset_2006_latest.csv, built by build_dataset_2026.py):

  H1  sentiment ~ target + path (+ Wald test of equal effects)
  H2  same-day CRSP return ~ target + path + sentiment  (incremental R2)
  H3  return ~ target + path + sentiment + sentiment*FG + FG  (state dependence)

Reported side by side for:
  * 2006-01 .. 2022-07  (comparable with the old v10.3 results)
  * full sample         (2006-01 .. latest, currently limited by CRSP to 2024-12)

Outputs results/results_v2.json, which the website pages read.
Plain OLS with classical SEs (matches the paper's method).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "results", "fomc_dataset_2006_latest.csv")
OUT = os.path.join(REPO, "results", "results_v2.json")

V1_END = "2022-07-27"          # last meeting in the old sample
FG_START, FG_END = "2008-12-16", "2015-10-28"


def ols(y, X, names):
    """Plain OLS. Returns dict per coefficient plus R2 and n."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    ssr = resid @ resid
    s2 = ssr / (n - k)
    V = s2 * XtX_inv
    se = np.sqrt(np.diag(V))
    tstat = beta / se
    pval = 2 * stats.t.sf(np.abs(tstat), df=n - k)
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ssr / tss if tss > 0 else np.nan
    return {
        "coef": {nm: float(b) for nm, b in zip(names, beta)},
        "se": {nm: float(s) for nm, s in zip(names, se)},
        "p": {nm: float(p) for nm, p in zip(names, pval)},
        "r2": float(r2), "n": int(n),
    }


def wald_equal(res, name_a, name_b):
    """Chi2(1) Wald test of beta_a = beta_b."""
    # recover V implicitly: use coef/se relation is not enough -> recompute not
    # needed here because we pass SEs only for display; approximate via z-stat
    pass


def ols_with_cov(y, X, names):
    """OLS that also returns (XtX_inv, s2) so Wald tests can be exact."""
    res = ols(y, X, names)
    Xa = np.asarray(X, float)
    ya = np.asarray(y, float)
    XtX_inv = np.linalg.inv(Xa.T @ Xa)
    resid = ya - Xa @ np.array([res["coef"][nm] for nm in names])
    s2 = (resid @ resid) / (len(ya) - Xa.shape[1])
    res["_cov"] = s2 * XtX_inv
    return res


def wald_chi2(res, names, idx_a, idx_b):
    R = np.zeros((1, len(names)))
    R[0, idx_a] = 1.0
    R[0, idx_b] = -1.0
    b = np.array([res["coef"][nm] for nm in names])
    V = res["_cov"]
    stat = float((R @ b)[0]) ** 2 / float((R @ V @ R.T)[0, 0])
    return stat, float(stats.chi2.sf(stat, df=1))


def run_h1(df):
    out = {}
    for label, sub in (
        ("2006_2022", df.loc[:V1_END]),
        ("full", df),
    ):
        s = sub.dropna(subset=["sentiment", "target_shock", "path_shock"]).copy()
        s["sent100"] = s["sentiment"] * 100  # net (hawkish - dovish) words per 100 words
        y = s["sent100"].values
        X = np.column_stack([
            np.ones(len(s)), s["target_shock"].values, s["path_shock"].values
        ])
        names = ["const", "target", "path"]
        r = ols_with_cov(y, X, names)
        stat, p = wald_chi2(r, names, 1, 2)
        r.pop("_cov")
        r["wald_equal_p"] = p
        out[label] = r
    return out


def run_h2(df):
    """Return ~ target + path + sentiment, same-day CRSP VW (percent)."""
    out = {}
    for label, sub in (
        ("2006_2022", df.loc[:V1_END]),
        ("full", df),
    ):
        s = sub.dropna(subset=["crsp_vw", "target_shock", "path_shock", "sentiment"]).copy()
        s["sent100"] = s["sentiment"] * 100  # net (hawkish - dovish) words per 100 words
        y = s["crsp_vw"].values
        base = np.column_stack([np.ones(len(s)), s["target_shock"].values,
                                s["path_shock"].values])
        full = np.column_stack([base, s["sent100"].values])
        names = ["const", "target", "path", "sentiment"]
        rb = ols(y, base, ["const", "target", "path"])
        rf = ols_with_cov(y, full, names)
        rf["delta_r2_pp"] = (rf["r2"] - rb["r2"]) * 100
        rf.pop("_cov")
        out[label] = rf
        out[label + "_baseline_r2"] = rb["r2"]
    return out


def run_h3(df):
    """State dependence: sentiment incremental effect under FG (Dec 2008 - Oct 2015)."""
    out = {}
    for label, sub in (
        ("2006_2022", df.loc[:V1_END]),
        ("full", df),
    ):
        s = sub.dropna(subset=["crsp_vw", "target_shock", "path_shock", "sentiment"]).copy()
        s["sent100"] = s["sentiment"] * 100  # net (hawkish - dovish) words per 100 words.copy()
        s["fg"] = ((s.index >= FG_START) & (s.index <= FG_END)).astype(int)
        s["sent100"] = s["sentiment"] * 100
        s["sent_x_fg"] = s["sent100"] * s["fg"]
        y = s["crsp_vw"].values
        X = np.column_stack([
            np.ones(len(s)), s["target_shock"].values, s["path_shock"].values,
            s["sentiment"].values, s["sent_x_fg"].values, s["fg"].values,
        ])
        names = ["const", "target", "path", "sentiment", "sent_x_fg", "fg"]
        r = ols_with_cov(y, X, names)
        r.pop("_cov")
        # FG-period-only regression (the card on the website)
        fg = s[s["fg"] == 1]
        nfg = s[s["fg"] == 0]
        for tag, ss in (("fg_period", fg), ("nonfg_period", nfg)):
            ss = ss.copy(); ss["sent100"] = ss["sentiment"] * 100
            Xs = np.column_stack([np.ones(len(ss)), ss["target_shock"].values,
                                  ss["path_shock"].values, ss["sent100"].values])
            rr = ols(ss["crsp_vw"].values, Xs, ["const", "target", "path", "sentiment"])
            r[tag] = rr
        out[label] = r
    return out


def main():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    print(f"dataset: {len(df)} meetings, {df.index.min().date()} -> {df.index.max().date()}")

    results = {
        "built_on": "results/fomc_dataset_2006_latest.csv (build_dataset_2026.py: body-only statements, same-day CRSP, surprises-driven sample)",
        "unit": "returns in percent, same-day (meeting-day close vs previous close)",
        "H1_tone": run_h1(df),
        "H2_returns": run_h2(df),
        "H3_state_dependence": run_h3(df),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)

    # ---- console summary ----
    for lab in ("2006_2022", "full"):
        h1 = results["H1_tone"][lab]
        print(f"\n[{lab}] H1: sentiment ~ target + path   (N={h1['n']}, R2={h1['r2']:.3f})")
        for nm in ("target", "path"):
            print(f"    beta_{nm} = {h1['coef'][nm]:+.4f} (p={h1['p'][nm]:.3f})")
        print(f"    Wald(target=path) p = {h1['wald_equal_p']:.3f}")
        h2 = results["H2_returns"][lab]
        print(f"[{lab}] H2: crsp_vw ~ shocks + sentiment (N={h2['n']})")
        print(f"    beta_sent = {h2['coef']['sentiment']:+.3f} (p={h2['p']['sentiment']:.3f}); "
              f"delta R2 = {h2['delta_r2_pp']:+.2f} pp")
        h3 = results["H3_state_dependence"][lab]
        fg = h3["fg_period"]
        nfg = h3["nonfg_period"]
        print(f"[{lab}] H3: FG period sentiment beta = {fg['coef']['sentiment']:+.3f} "
              f"(p={fg['p']['sentiment']:.3f}, R2={fg['r2']:.3f}, N={fg['n']}) | "
              f"non-FG beta = {nfg['coef']['sentiment']:+.3f} (p={nfg['p']['sentiment']:.3f}, N={nfg['n']})")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
