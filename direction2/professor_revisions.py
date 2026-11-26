# -*- coding: utf-8 -*-
"""Professor-revision regressions (Nov 2026 memo, items A4/A5/E11).

A4  Official Jarociński-Karadi MP/CBI series as the main shock measure:
    robustness_tests.py test2 already covers flows; here we redo the
    price-side specs (M2/M3 tone regressions and H1 sentiment) with
    mp_jk/cbi_jk added as controls, so the deck can report
    "official series, not just our sign rule".
A5  Cluster-by-meeting-month SEs on every pooled spec (H5 panel).
E11 Tone split: statement sentiment (0.5 LM + 0.5 CB, tone100) vs the
    LLM hawkishness score (-100..+100). Which one drives the FG-period
    return response?  E11 wants R ~ Target + Path + each measure
    separately and together, full sample and FG-only.

Writes direction2/flow_rerun_output_v8/professor_revisions.json and prints
a markdown summary. Uses only the helpers in robustness_tests.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_tests as RT  # noqa: E402

OUT = os.path.join(RT.D2, "flow_rerun_output_v8", "professor_revisions.json")
LLM_FILE = os.path.join(RT.D2, "flow_rerun_output_v8", "llm_statement_scores.jsonl")


def load():
    fomc = RT.load_fomc()
    agg = RT.load_flows()
    return fomc, agg


def prices(fomc):
    """Price-side regressions: R_vw ~ Target + Path (+ Tone variants).
    Returns dict of spec -> {coef: {beta, se, p}, n, r2} with HC1 SEs."""
    df = fomc.dropna(subset=["crsp_vw", "target_shock", "path_shock"]).copy()
    df["tone100"] = df["sentiment"] * 100.0
    # FIX (audit 2026-11-26): use the dataset's own fg_2008_2015 flag.
    # A hand-rolled window ..2015-12-31 counts 57 meetings but the dataset
    # marks 2015-12-16 fg=0 (no FG language) -> 56, matching slide 23 and
    # robustness test1. All deck numbers use the flag.
    df["fg"] = df["fg_2008_2015"]
    df["tone_x_fg"] = df["tone100"] * df["fg"]

    # merge official JK shocks (A4)
    jk = pd.read_csv(RT.JK_FILE)
    jk["date"] = pd.to_datetime(jk[RT.JK_DATE_COL]).dt.normalize()
    jk = jk.set_index("date")[["mp_jk", "cbi_jk"]]
    df = df.join(jk, how="left")

    # merge LLM hawkishness (E11)
    llm = {}
    with open(LLM_FILE) as fh:
        for line in fh:
            d = json.loads(line)
            llm[pd.Timestamp(d["date"]).normalize()] = d["score"]
    s = pd.Series(llm, name="hawk")
    df = df.join(s, how="left")
    df["hawk100"] = df["hawk"] * 1.0          # already -100..100
    df["hawk_x_fg"] = df["hawk100"] * df["fg"]

    out = {}
    specs = [
        ("M2 full: R ~ T + P + sentiment", ["target_shock", "path_shock", "tone100"]),
        ("M3 full: + FG interaction (sentiment)", ["target_shock", "path_shock", "tone100",
                                                   "fg", "tone_x_fg"]),
        ("E11 full: R ~ T + P + hawkishness", ["target_shock", "path_shock", "hawk100"]),
        ("E11 full: + FG interaction (hawkishness)", ["target_shock", "path_shock", "hawk100",
                                                      "fg", "hawk_x_fg"]),
        ("E11 full: sentiment + hawkishness together", ["target_shock", "path_shock", "tone100",
                                                        "hawk100"]),
        ("FG only: T + P + sentiment", ["target_shock", "path_shock", "tone100"]),
        ("FG only: T + P + hawkishness", ["target_shock", "path_shock", "hawk100"]),
        ("E11 FG only: sentiment + hawkishness", ["target_shock", "path_shock", "tone100",
                                                  "hawk100"]),
        # E10 (professor): slide-23 horse race with MP/CBI controls.
        # sign-rule version = the published 48%-R2 spec; official JK = new.
        ("E10 FG: T + P + tone + MP(sign-rule) + CBI(sign-rule)",
         ["target_shock", "path_shock", "tone100", "mp_shock", "cbi_shock"], "fg"),
        ("E10 FG: T + P + tone + MP(official JK) + CBI(official JK)",
         ["target_shock", "path_shock", "tone100", "mp_jk", "cbi_jk"], "fg"),
        ("E10 FG: tone + MP(official JK) + CBI(official JK)",
         ["tone100", "mp_jk", "cbi_jk"], "fg"),
        ("E10 full: tone + MP(official JK) + CBI(official JK)",
         ["tone100", "mp_jk", "cbi_jk"], None),
    ]
    for spec in specs:
        name, xs = spec[0], spec[1]
        restrict_fg = len(spec) > 2 and spec[2] == "fg"
        sub = df.dropna(subset=xs + ["crsp_vw"])
        if restrict_fg or name.startswith("FG only") or name.startswith("E11 FG"):
            sub = sub[(sub["fg"] == 1)]
        if len(sub) < 10:
            out[name] = {"skipped": f"n={len(sub)}"}
            continue
        r = RT.ols(sub, "crsp_vw", xs, "hc1")
        out[name] = {"n": int(r.n), "r2": float(r.r2),
                     "coefs": {c: {"beta": r[c]["beta"], "se": r[c]["se"],
                                   "p": r[c]["p"]} for c in xs}}
    return out


def flows_cluster(fomc, agg):
    """A5: the H5 pooled spec with cluster-by-month SEs under BOTH shock
    measures (sign rule and official JK), all three windows."""
    out = {}
    for shock_name, repl in [("sign rule", None), ("official JK", ("mp_jk", "cbi_jk"))]:
        g = fomc.copy()
        if repl:
            jk = pd.read_csv(RT.JK_FILE)
            jk["date"] = pd.to_datetime(jk[RT.JK_DATE_COL]).dt.normalize()
            jk = jk.set_index("date")[["mp_jk", "cbi_jk"]]
            g = g.join(jk, how="left")
            g["mp_shock"], g["cbi_shock"] = g["mp_jk"], g["cbi_jk"]
        for w in ("same", "post", "diff"):
            p = RT.panel(agg, g, w)
            f5 = RT.ols(p, "flow", RT.H5_X, "cluster", "month_id")
            out[f"{shock_name} · {w}"] = {
                "n": int(f5.n), "r2": float(f5.r2),
                "risk_x_cbi": f5["risk_x_cbi"],
                "risk_x_mp": f5["risk_x_mp"],
            }
    return out


def main():
    fomc, agg = load()
    res = {"meta": {"run_at": pd.Timestamp.now().isoformat(),
                    "jk_source": "github.com/marekjarocinski/jkshocks_update_fed (MP_pm/CBI_pm)",
                    "llm_scores": os.path.basename(LLM_FILE)}}
    res["prices_tone_split_and_jk"] = prices(fomc)
    res["flows_clustered_both_shocks"] = flows_cluster(fomc, agg)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved:", OUT)

    # markdown summary
    print("\n## E11 tone split (price side)\n")
    for name, spec in res["prices_tone_split_and_jk"].items():
        if "skipped" in spec:
            print(f"- {name}: skipped ({spec['skipped']})")
            continue
        bits = []
        for c in ["tone100", "hawk100", "tone_x_fg", "hawk_x_fg", "target_shock", "path_shock"]:
            if c in spec["coefs"]:
                v = spec["coefs"][c]
                bits.append(f"{c} {v['beta']:+.3f} (p {v['p']:.3f})")
        print(f"- {name} [n={spec['n']}, R2={spec['r2']*100:.1f}%]: " + "; ".join(bits))
    print("\n## A5 clustered flows (both shock measures)\n")
    for k, v in res["flows_clustered_both_shocks"].items():
        rc, rm = v["risk_x_cbi"], v["risk_x_mp"]
        print(f"- {k} [n={v['n']}]: Risk×CBI {rc['beta']:+.3f} (p {rc['p']:.3f}) | "
              f"Risk×MP {rm['beta']:+.3f} (p {rm['p']:.3f})")


if __name__ == "__main__":
    main()
