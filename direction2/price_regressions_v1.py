# -*- coding: utf-8 -*-
"""Price-side regressions (Section 4 of the paper) — formal archival run.

Models:
  A. headline reproduction: ty10_chg ~ tone100 within FG subsample
  B. multi-asset matrix: {ty10, ty2, crsp_vw, sp500} x {all, FG, non-FG}
  C. horse race: asset ~ tone100 + MP + CBI (does tone survive controls?)
  D. interaction: asset ~ tone100*FG + tone100 + FG
Output: direction2/flow_rerun_output_v8/price_results_v1.json
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "price_results_v1.json")

df = pd.read_csv(os.path.join(BASE, "results", "fomc_dataset_2006_latest.csv"),
                 index_col=0, parse_dates=True)
df["tone100"] = df["sentiment"] * 100
# same-day JK split shocks (consistent with the flow chapter's fix-C note)
mp = np.where(np.sign(df["target_shock"]) != np.sign(df["crsp_vw"]), df["target_shock"], 0.0)
cbi = np.where(np.sign(df["target_shock"]) == np.sign(df["crsp_vw"]), df["target_shock"], 0.0)
df["mp_shock"], df["cbi_shock"] = mp, cbi

def fit(formula, sub):
    d = sub.dropna(subset=[c for c in ("tone100", "mp_shock", "cbi_shock")
                           if c in formula])
    m = smf.ols(formula, data=d).fit()
    out = {"n": int(m.nobs), "r2": round(float(m.rsquared), 4)}
    for k in m.params.index:
        out[k] = {"beta": round(float(m.params[k]), 4),
                  "se": round(float(m.bse[k]), 4), "p": round(float(m.pvalues[k]), 4)}
    return out

ASSETS = ["ty10_chg", "ty2_chg", "crsp_vw", "sp500"]
fg = df[df["fg_2008_2015"] == 1]
nonfg = df[df["fg_2008_2015"] == 0]

meta = {"run": "2026-10-21", "n_full": int(len(df)), "n_fg": int(len(fg)),
        "note": "Section 4 price-side archival regressions; same-day changes; JK split shocks"}

body = {}
# A. headline
body["A_headline_ty10_FG"] = fit("ty10_chg ~ tone100", fg)
# B. matrix
for a in ASSETS:
    body[f"B_{a}_all"] = fit(f"{a} ~ tone100", df)
    body[f"B_{a}_FG"] = fit(f"{a} ~ tone100", fg)
    body[f"B_{a}_nonFG"] = fit(f"{a} ~ tone100", nonfg)
# C. horse race
for a in ASSETS:
    body[f"C_{a}_FG"] = fit(f"{a} ~ tone100 + mp_shock + cbi_shock", fg)
    body[f"C_{a}_all"] = fit(f"{a} ~ tone100 + mp_shock + cbi_shock", df)
# D. interaction (full sample)
for a in ASSETS:
    d2 = df.dropna(subset=[a, "tone100"]).copy()
    d2["tone_x_fg"] = d2["tone100"] * d2["fg_2008_2015"]
    m = smf.ols(f"{a} ~ tone100 + fg_2008_2015 + tone_x_fg", data=d2).fit()
    body[f"D_{a}_interact"] = {"n": int(m.nobs), "r2": round(float(m.rsquared), 4),
                               **{k: {"beta": round(float(m.params[k]), 4),
                                      "se": round(float(m.bse[k]), 4),
                                      "p": round(float(m.pvalues[k]), 4)}
                                  for k in m.params.index}}

out = {"meta": meta, "results": body}
json.dump(out, open(OUT, "w"), indent=1)
print("saved", OUT)
print("\n=== A. headline (ty10 ~ tone100, FG subsample) ===")
print(json.dumps(body["A_headline_ty10_FG"], indent=1))
print("\n=== D. interaction (ty10) ===")
print(json.dumps(body["D_ty10_chg_interact"], indent=1))
