# -*- coding: utf-8 -*-
"""models_v2.py — Flows regressions with tone, per Oct-20 expert feedback.

New specifications on top of the v8 audit pipeline:
  A. H4 with tone: Flow(c,w) = a + b1*MP + b2*CBI + b3*Tone + controls
     - tone raw and tone_orth (orthogonalized to Target+Path, per feedback)
  B. Forward-guidance subsample ("at least within the FG period")
  C. Sign asymmetry: Risk x MP_plus / Risk x MP_minus (Ben's "does sign matter")
All regressions report N, coefficients, SEs, p-values (classical OLS, matching
the paper's method).

Run from repo root:  python3 direction2/models_v2.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from run_flow_chapter_local import load_fomc  # noqa: E402

OUT = os.path.join(BASE, "flow_rerun_output_v8")


def load():
    agg = pd.read_csv(os.path.join(OUT, "flows_by_class_month_v8.csv"), parse_dates=["month"])
    agg["flow_pct"] = agg["dollar_flow"] / agg["tna_lag"]
    agg["log_tna"] = np.log(agg["tna_lag"].replace(0, np.nan))
    agg = agg.sort_values(["asset_class", "month"])
    agg["flow_vol_12m"] = agg.groupby("asset_class")["flow_pct"].transform(
        lambda s: s.rolling(12, min_periods=6).std())
    fomc = load_fomc()  # 161 meetings, has target/path/sentiment/crsp_vw/mp_shock/cbi_shock/zlb/fg
    return agg, fomc


def ols_row(y, X):
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, Xc, missing="drop").fit()
    out = {"n": int(res.nobs), "r2": float(res.rsquared)}
    for k in res.params.index:
        out[k] = {"beta": float(res.params[k]), "se": float(res.bse[k]), "p": float(res.pvalues[k])}
    return out, res


def wald_equal(res, a, b):
    try:
        return float(res.t_test(f"{a} = {b}").pvalue)
    except Exception:
        return None


def build_panel(agg, fomc, shift=0):
    a = agg.copy()
    a["key"] = (a["month"] - pd.DateOffset(months=shift)).dt.to_period("M")
    fm = fomc.reset_index().rename(columns={"date": "key"})
    fm["key"] = fm["key"].dt.to_period("M")
    pan = a.merge(fm, on="key", how="inner")
    pan["risk"] = pan["asset_class"].map(
        {"government_bonds": 1, "corporate_bonds": 2, "real_assets": 3, "large_cap_equity": 4,
         "developed_market_equity": 5, "emerging_market_equity": 6, "small_cap_equity": 7})
    pan["tone100"] = pan["sentiment"] * 100
    pan["risk_x_mp"] = pan["risk"] * pan["mp_shock"]
    pan["risk_x_cbi"] = pan["risk"] * pan["cbi_shock"]
    pan["risk_x_tone"] = pan["risk"] * pan["tone100"]
    return pan


def main():
    agg, fomc = load()
    fomc["fg_2008_2015"] = ((fomc.index >= "2008-12-16") & (fomc.index <= "2015-10-28")).astype(int)
    fomc_orth = fomc.copy()
    # tone_orth: residual of tone on target+path (full sample OLS)
    Xo = sm.add_constant(fomc_orth[["target_shock", "path_shock"]], has_constant="add")
    ols_orth = sm.OLS(fomc_orth["sentiment"] * 100, Xo, missing="drop").fit()
    fomc_orth["tone_orth"] = ols_orth.resid
    print(f"tone_orth: R2 of tone~target+path = {ols_orth.rsquared:.4f} "
          f"(target p={ols_orth.pvalues['target_shock']:.3f}, path p={ols_orth.pvalues['path_shock']:.3f})")

    results = {"meta": {
        "run_at": pd.Timestamp.now().isoformat(),
        "note": "Flows regressions with tone (Oct-20 expert feedback). Gap<=3mo spec, point-in-time classes.",
    }}

    # ── A: H4 with tone (raw + orthogonal), three windows ──
    for tone_col, tname in (("tone100", "tone_raw"), ("tone_orth", "tone_orth")):
        block = {}
        for wname, shift in (("same", 0), ("post", 1)):
            pan = build_panel(agg, fomc_orth if tone_col == "tone_orth" else fomc, shift)
            cols = ["mp_shock", "cbi_shock", tone_col, "log_tna", "flow_vol_12m"]
            pan = pan.dropna(subset=["flow_pct"] + cols)
            row, _ = ols_row(pan["flow_pct"] * 100, pan[cols])
            row["wald_mp_eq_cbi_p"] = wald_equal(_, "mp_shock", "cbi_shock")
            block[wname] = row
        # diff
        a0 = agg.copy(); a0["w"] = a0["month"]
        a1 = agg.copy(); a1["w"] = a1["month"] - pd.DateOffset(months=1)
        dm = a1.merge(a0[["asset_class", "w", "flow_pct"]].rename(columns={"flow_pct": "f0"}),
                      on=["asset_class", "w"], how="inner")
        dm["flow_pct"] = dm["flow_pct"] - dm["f0"]
        fm = (fomc_orth if tone_col == "tone_orth" else fomc).reset_index().rename(columns={"date": "key"})
        fm["key"] = fm["key"].dt.to_period("M")
        dm["key"] = dm["w"].dt.to_period("M")
        pan = dm.merge(fm, on="key", how="inner")
        pan["tone100"] = pan["sentiment"] * 100
        pan["risk"] = pan["asset_class"].map({"government_bonds": 1, "corporate_bonds": 2, "real_assets": 3,
                                              "large_cap_equity": 4, "developed_market_equity": 5,
                                              "emerging_market_equity": 6, "small_cap_equity": 7})
        pan["risk_x_tone"] = pan["risk"] * pan[tone_col]
        cols = ["mp_shock", "cbi_shock", tone_col, "log_tna", "flow_vol_12m"]
        pan = pan.dropna(subset=["flow_pct"] + cols)
        block["diff"], _ = ols_row(pan["flow_pct"] * 100, pan[cols])
        results[f"H4_with_{tname}"] = block

    # ── B: FG-period subsample (same window, tone raw) ──
    pan = build_panel(agg, fomc, 0)
    cols = ["mp_shock", "cbi_shock", "tone100", "log_tna", "flow_vol_12m"]
    fg = pan[(pan["fg_2008_2015"] == 1)].dropna(subset=["flow_pct"] + cols)
    nfg = pan[(pan["fg_2008_2015"] == 0)].dropna(subset=["flow_pct"] + cols)
    fg_row, _ = ols_row(fg["flow_pct"] * 100, fg[cols])
    nfg_row, _ = ols_row(nfg["flow_pct"] * 100, nfg[cols])
    results["H4_tone_by_regime"] = {"fg_period": fg_row, "nonfg_period": nfg_row}

    # ── C: sign asymmetry on the ladder (same window) ──
    pan["risk_x_mp_pos"] = pan["risk"] * pan["mp_shock"].clip(lower=0)
    pan["risk_x_mp_neg"] = pan["risk"] * pan["mp_shock"].clip(upper=0)
    pan["risk_x_cbi_pos"] = pan["risk"] * pan["cbi_shock"].clip(lower=0)
    pan["risk_x_cbi_neg"] = pan["risk"] * pan["cbi_shock"].clip(upper=0)
    cols = ["risk_x_mp_pos", "risk_x_mp_neg", "risk_x_cbi_pos", "risk_x_cbi_neg",
            "mp_shock", "cbi_shock", "log_tna", "flow_vol_12m"]
    pan2 = pan.dropna(subset=["flow_pct"] + cols)
    row, _ = ols_row(pan2["flow_pct"] * 100, pan2[cols])
    results["H5_sign_asymmetry"] = row

    with open(os.path.join(OUT, "models_v2_results.json"), "w") as f:
        json.dump(results, f, indent=1)

    # ── Readout ──
    print("\n=== A. H4 with tone (same window) ===")
    for tn in ("H4_with_tone_raw", "H4_with_tone_orth"):
        r = results[tn]["same"]
        print(f"  [{tn}] N={r['n']} R2={r['r2']:.3f}")
        for k in ("mp_shock", "cbi_shock", "tone_raw" if 'raw' in tn else "tone_orth", "tone100"):
            if k in r:
                print(f"    {k:12s} {r[k]['beta']:+.3f} (p={r[k]['p']:.3f})")
    print("\n=== B. tone by regime (same window) ===")
    fg_r = results["H4_tone_by_regime"]["fg_period"]; nfg_r = results["H4_tone_by_regime"]["nonfg_period"]
    print(f"  FG  (N={fg_r['n']}): tone {fg_r['tone100']['beta']:+.3f} (p={fg_r['tone100']['p']:.3f}), "
          f"CBI {fg_r['cbi_shock']['beta']:+.3f} (p={fg_r['cbi_shock']['p']:.3f})")
    print(f"  nFG (N={nfg_r['n']}): tone {nfg_r['tone100']['beta']:+.3f} (p={nfg_r['tone100']['p']:.3f}), "
          f"CBI {nfg_r['cbi_shock']['beta']:+.3f} (p={nfg_r['cbi_shock']['p']:.3f})")
    print("\n=== C. sign asymmetry (same window) ===")
    r = results["H5_sign_asymmetry"]
    for k in ("risk_x_mp_pos", "risk_x_mp_neg", "risk_x_cbi_pos", "risk_x_cbi_neg"):
        print(f"  {k:16s} {r[k]['beta']:+.3f} (p={r[k]['p']:.3f})")
    print(f"  N={r['n']}")
    print("\nsaved", os.path.join(OUT, "models_v2_results.json"))


if __name__ == "__main__":
    main()
