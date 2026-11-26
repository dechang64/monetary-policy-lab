# -*- coding: utf-8 -*-
"""Test 4, official-JK variant: retail vs institutional flows.

The sign-rule variant lives in robustness_tests.py::test4 (results.json).
This script recomputes the same panel under the OFFICIAL Jarociński-Karadi
series, which is the professor's headline specification.

Critical detail (audit finding 2026-11-26): load_flows MUST be called with
by=("asset_class", "investor"). The default groups the 12-month flow-vol
rolling window by asset class only; because the investor CSV interleaves
retail/institutional/unknown rows within a class-month, the default ordering
makes flow_vol_12m roll ACROSS investor types - silent control corruption
that shifts every coefficient (~30% here). The sign-rule test4 in
robustness_tests.py already passes `by` correctly; an earlier ad-hoc run of
this script did not, which is why the first published numbers were wrong.

Output: professor_revisions.json["test4_official_jk"].
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

import robustness_tests as RT


def main():
    fomc = RT.load_fomc()
    jk = pd.read_csv(RT.JK_FILE)
    jk["date"] = pd.to_datetime(jk[RT.JK_DATE_COL]).dt.normalize()
    jk = jk.set_index("date")[["mp_jk", "cbi_jk"]]
    fomc = fomc.join(jk, how="left")
    fomc["mp_shock"], fomc["cbi_shock"] = fomc["mp_jk"], fomc["cbi_jk"]

    a = RT.load_flows(RT.INVESTOR_FILE, by=("asset_class", "investor"))  # REQUIRED
    a = a[a.investor.isin(["retail", "institutional"])]
    out = {
        "share_of_tna_institutional": float(
            a[a.investor == "institutional"]["tna"].sum() / a["tna"].sum()),
        "note": "by=(asset_class, investor): flow_vol_12m rolls within investor groups",
    }
    for w in ("same", "post", "diff"):
        p = RT.panel(a, fomc, w)
        blk = {}
        for t in ("retail", "institutional"):
            gov = p[(p.asset_class == "government_bonds") & (p.investor == t)]
            fg_ = RT.ols(gov, "flow", RT.H4_X, "hc1")
            blk[f"gov bonds, {t}"] = {
                "MP": fg_["mp_shock"], "CBI": fg_["cbi_shock"], "n": int(fg_.n)}
            f5 = RT.ols(p[p.investor == t], "flow", RT.H5_X, "cluster", "month_id")
            blk[f"H5 pooled, {t}"] = f5.row(["risk_x_mp", "risk_x_cbi"])
        p["investor_dummy"] = (p["investor"] == "institutional").astype(float)
        p["risk_x_inst"] = p["risk"] * p["investor_dummy"]
        p["mp_x_inst"] = p["mp_shock"] * p["investor_dummy"]
        p["cbi_x_inst"] = p["cbi_shock"] * p["investor_dummy"]
        p["risk_x_mp_x_inst"] = p["risk_x_mp"] * p["investor_dummy"]
        p["risk_x_cbi_x_inst"] = p["risk_x_cbi"] * p["investor_dummy"]
        xs = RT.H5_X + ["investor_dummy", "risk_x_inst", "mp_x_inst", "cbi_x_inst",
                        "risk_x_mp_x_inst", "risk_x_cbi_x_inst"]
        f = RT.ols(p, "flow", xs, "cluster", "month_id")
        blk["pooled with institutional interactions"] = f.row(
            ["cbi_shock", "cbi_x_inst", "risk_x_cbi", "risk_x_cbi_x_inst"])
        out[f"{w} (official JK)"] = blk

    pr_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "direction2/flow_rerun_output_v8/professor_revisions.json")
    pr = json.load(open(pr_path))
    pr["test4_official_jk"] = out
    json.dump(pr, open(pr_path, "w"), indent=1)
    print("written test4_official_jk ->", pr_path)
    for w in ("same", "post", "diff"):
        b = out[f"{w} (official JK)"]
        hi = b["H5 pooled, institutional"]["risk_x_cbi"]
        hr = b["H5 pooled, retail"]["risk_x_cbi"]
        print(f"{w}: RiskxCBI inst {hi['beta']:+.2f} (p {hi['p']:.3f}) "
              f"retail {hr['beta']:+.2f} (p {hr['p']:.3f})")


if __name__ == "__main__":
    main()
