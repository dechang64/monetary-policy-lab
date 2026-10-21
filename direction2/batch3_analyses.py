# -*- coding: utf-8 -*-
"""Batch-3 analyses: (1) post-2022 tone price regression — the theory
predicts tone re-acquires price relevance in the tightening/SEP era;
(2) placebo: flows on NON-meeting months should show no shock response.
Output: direction2/flow_rerun_output_v8/batch3_results.json
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "batch3_results.json")

res = {}

# ---------- (1) post-2022 tone on prices ----------
df = pd.read_csv(os.path.join(BASE, "results", "fomc_dataset_2006_latest.csv"),
                 index_col=0, parse_dates=True)
df["tone100"] = df["sentiment"] * 100
mp = np.where(np.sign(df["target_shock"]) != np.sign(df["crsp_vw"]), df["target_shock"], 0.0)
cbi = np.where(np.sign(df["target_shock"]) == np.sign(df["crsp_vw"]), df["target_shock"], 0.0)
df["mp_shock"], df["cbi_shock"] = mp, cbi
d = df.dropna(subset=["crsp_vw", "tone100"])

def fit(formula, data):
    m = smf.ols(formula, data=data).fit()
    return {"n": int(m.nobs), "r2": round(float(m.rsquared), 4),
            **{k: {"beta": round(float(m.params[k]), 4), "se": round(float(m.bse[k]), 4),
                   "p": round(float(m.pvalues[k]), 4)} for k in m.params.index}}

post = d[d["post_2022"] == 1]
res["post2022"] = {
    "tone_crsp_FG_like": fit("crsp_vw ~ tone100", post),
    "tone_crsp_all_with_post_int": fit(
        "crsp_vw ~ tone100 + tone_x_post", d.assign(tone_x_post=d["tone100"] * d["post_2022"])),
    "n_post": int(len(post)),
}
# ZLB window for comparison (the dead zone)
zlb = d[d["zlb_2020_2022"] == 1]
res["zlb2020_tone_crsp"] = fit("crsp_vw ~ tone100", zlb)

# ---------- (2) placebo: non-meeting months ----------
fl = pd.read_csv(os.path.join(BASE, "direction2", "flow_rerun_output_v8",
                              "flows_by_class_month_v8.csv"))
fl["month"] = pd.to_datetime(fl["month"])
meetings = set(df.index.strftime("%Y-%m"))
fl["is_meeting"] = [m.strftime("%Y-%m") in meetings for m in fl["month"]]
fl["risk_rank"] = fl["asset_class"].map({
    "government_bonds": 1, "corporate_bonds": 2, "real_assets": 3,
    "large_cap_equity": 4, "developed_market_equity": 5,
    "emerging_market_equity": 6, "small_cap_equity": 7})
nonmeet = fl[~fl["is_meeting"]].copy()
meet = fl[fl["is_meeting"]].copy()

# placebo: random pseudo-shocks (same distribution as real, wrong months)
rng = np.random.default_rng(42)
nonmeet["pseudo_cbi"] = rng.choice(df["cbi_shock"].dropna().values, size=len(nonmeet))
nonmeet["pseudo_risk_x_cbi"] = nonmeet["pseudo_cbi"] * nonmeet["risk_rank"]
mp_ = smf.ols("flow_pct ~ pseudo_cbi + pseudo_risk_x_cbi + log_tna + C(asset_class)",
              data=nonmeet).fit()
res["placebo_nonmeeting"] = {
    "n": int(mp_.nobs), "n_months": int(nonmeet["month"].nunique()),
    "pseudo_cbi": {"beta": round(float(mp_.params["pseudo_cbi"]), 4),
                   "p": round(float(mp_.pvalues["pseudo_cbi"]), 4)},
    "pseudo_risk_x_cbi": {"beta": round(float(mp_.params["pseudo_risk_x_cbi"]), 4),
                          "p": round(float(mp_.pvalues["pseudo_risk_x_cbi"]), 4)},
}
# real-meeting reference (should reproduce CBI risk-ordering)
meet["risk_x_cbi"] = meet["risk_rank"] * 0  # CBI shock unknown at class-month level here;
# the reference is the archived H5 (Risk x CBI = -0.066, p=0.036) - noted in json.
res["note"] = ("Placebo uses only non-meeting months (n months=%d of %d total). "
               "Real-meeting reference: Risk x CBI = -0.066 (p=0.036) from "
               "flow_results_rerun_v8.json. Fund-level heterogeneity NOT runnable "
               "locally (server-side aggregation); queued." % (
                   res["placebo_nonmeeting"]["n_months"], fl["month"].nunique()))

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
