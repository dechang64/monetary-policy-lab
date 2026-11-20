# -*- coding: utf-8 -*-
"""Robustness Tests page
========================
Professor-requested robustness battery (Nov 2026). Script:
direction2/robustness_tests.py (numpy/pandas/scipy only). Machine-readable
results: direction2/flow_rerun_output_v8/robustness_tests/results.json.

Status legend:
  RAN      - test executed on repo data (tests 1, 3, 5, 6)
  AWAITING - test prepared, waiting on an external data file (2, 4, 7)

Headline findings (reported honestly, per Eileen's memo):
  * The fund-flow results do NOT survive robust standard errors:
    gov-bond CBI (same month) classical p=.043 -> HC1 p=.342;
    Risk x CBI classical p=.041 -> HC1 p=.434 / cluster-month p=.466.
    Both hinge on the 2008-09-16 meeting (drop it: gov CBI +0.029, p=.925).
  * The forward-guidance tone result IS robust: classical/HC1/NW/wild
    bootstrap p = .003-.012, placebo p < .011.
"""
import json
import os

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "direction2", "flow_rerun_output_v8",
                   "robustness_tests", "results.json")
TM = os.path.join(BASE, "direction2", "flow_rerun_output_v8",
                  "tone_measures_current.json")

NAVY = "#1F2A44"


def _p(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else ""


def _b(v):
    return f"{v:+.3f}" if isinstance(v, (int, float)) else ""


def _df(data, cols):
    return st.dataframe(pd.DataFrame(data, columns=cols),
                        use_container_width=True, hide_index=True)


def render():
    st.markdown("## 🧪 Robustness Tests")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;'>
        <h3 style='color: white; margin: 0;'>Seven-Test Robustness Battery</h3>
        <p style='opacity: 0.85; margin: 0.5rem 0 0 0;'>
            Script: direction2/robustness_tests.py (numpy / pandas / scipy only)<br>
            Results: direction2/flow_rerun_output_v8/robustness_tests/results.json<br>
            Significance: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(RES):
        st.error("robustness_tests/results.json not found — run "
                 "`python3 direction2/robustness_tests.py --only 1 3 5 6` first.")
        return
    P = json.load(open(RES))

    # ── Data-fix note ──
    st.info("**Data fix applied (2026-11-20).** `load_fomc()` previously assigned "
            "`mp_shock` to meetings whose same-day CRSP return was missing, because "
            "`np.sign(nan) != np.sign(nan)` is True. Requiring a valid return reclassifies "
            "8 meetings: **91 → 83 MP meetings** (60 CBI unchanged). The baseline check "
            "below shows the flow results are essentially unchanged by the fix.", icon="🔧")

    # ── Baseline check ──
    st.markdown("---")
    st.markdown("### Baseline check: classification fix")
    rows = []
    for k, v in P["baseline_check"].items():
        g, r = v["gov_cbi"], v["risk_x_cbi"]
        rows.append([k, v["n_mp"], v["n_cbi"],
                     _b(g["beta"]) + f" (se {g['se']:.3f}, p {_p(g['p'])})",
                     _b(r["beta"]) + f" (se {r['se']:.3f}, p {_p(r['p'])})"])
    _df(rows, ["Classification", "MP meetings", "CBI meetings",
               "Gov CBI (same month)", "Risk × CBI (same month)"])
    st.caption("As published = 2025-26 meetings coded as MP despite missing returns. "
               "Fixed = those 8 meetings dropped.")

    # ── Test 1 ──
    st.markdown("---")
    st.markdown("### Test 1 · Robust standard errors")
    t1 = P["test1"]
    SE_METHODS = [("classical", "Classical"), ("hc1", "HC1"), ("hac", "Newey-West"),
                  ("cluster_month", "Cluster (month)"), ("twoway_month_class", "Two-way"),
                  ("wild", "Wild bootstrap")]
    rows = []
    for name, spec in t1.items():
        if name == "influence" or not isinstance(spec, dict) or "classical" not in spec:
            continue
        n = spec.get("n", "")
        base = spec["classical"]
        r = [name, n if n != "" else "—",
             _b(base["beta"]) + f" (se {base['se']:.3f})"]
        for key, _ in SE_METHODS:
            s = spec.get(key)
            r.append(_p(s["p"]) if s else "—")
        rows.append(r)
    _df(rows, ["Result", "N", "β (classical)", "p classical", "p HC1", "p NW",
               "p cluster", "p two-way", "p wild"])

    infl = t1.get("influence", {})
    if infl:
        st.markdown("**Influence analysis — single-meeting drop**")
        ex = infl.get("excluding 2008-09-16", {})
        rows = [
            ["Gov CBI (same month), HC1", "2024-09-18",
             _b(infl["most_influential_for_gov_cbi"]["gov_cbi"]),
             _b(ex.get("gov_cbi_hc1", {}).get("beta")),
             _p(ex.get("gov_cbi_hc1", {}).get("p"))],
            ["Risk × CBI (same month), cluster", "2008-09-16",
             _b(infl["most_influential_for_risk_x_cbi"]["risk_x_cbi"]),
             _b(ex.get("risk_x_cbi_cluster", {}).get("beta")),
             _p(ex.get("risk_x_cbi_cluster", {}).get("p"))],
        ]
        _df(rows, ["Result", "Most influential meeting", "β (full sample)",
                   "β excluding it", "p excluding it"])
        st.warning("**Headline flow results are not robust.** Government-bond CBI "
                   "(classical p = .043) becomes p = .342 under HC1, and Risk × CBI "
                   "(classical p = .041) becomes p = .434 under HC1 / p = .466 "
                   "clustered by month. Both hinge on **2008-09-16**: excluding that "
                   "single meeting, gov-bond CBI falls to +0.029 (p = .925). "
                   "The forward-guidance tone result is robust: p = .003-.012 across "
                   "all SE methods, placebo below.", icon="⚠️")

    # ── Test 3 ──
    st.markdown("---")
    st.markdown("### Test 3 · Market-beta measurement (downside vs upside)")
    t3 = P["test3"]
    rows = []
    for a in t3["asset_class_order"]:
        v = t3["class_betas"][a]
        rows.append([a.replace("_", " ").title(), _b(v["beta"]), _b(v["downside_beta"]),
                     _b(v["upside_beta"]), f"{v['mean_ret_pct']:.2f}%", int(v["months"])])
    _df(rows, ["Asset class", "β (market)", "β downside", "β upside",
               "Mean monthly return", "Months"])
    st.caption("Risk-rank and continuous-beta variants of the H5 interaction are all "
               "insignificant (p = .305-.931); see report.md for the full A-F grid.")
    for w in ["same", "post"]:
        st.markdown(f"**{w.capitalize()}-window interaction (asset-class rank, current)**")
        block = t3[w].get("A. asset-class rank (current)", {})
        if block:
            rows = [[
                "Risk × MP", _b(block["risk_x_MP"]["beta"]), _p(block["risk_x_MP"]["p"]),
                "Risk × CBI", _b(block["risk_x_CBI"]["beta"]), _p(block["risk_x_CBI"]["p"]),
                block["n"],
            ]]
            _df(rows, ["Shock", "β (MP)", "p (MP)", "Shock (2nd col)", "β (CBI)", "p (CBI)", "N"])

    # ── Test 5 ──
    st.markdown("---")
    st.markdown("### Test 5 · State dependence: credit spread × inflation")
    t5 = P.get("test5")
    if t5:
        rows = []
        for w in ["same", "post"]:
            for state, lab in [("high_spread", "High spread"), ("z_spread", "Spread (continuous z)"),
                               ("high_infl", "High inflation"), ("z_infl", "Inflation (continuous z)")]:
                b = t5[w].get(state)
                if not b:
                    continue
                def cell(k1, k2):
                    s = b.get(k1)
                    return (_b(s["beta"]), _p(s["p"])) if s else ("—", "—")
                b1, p1 = cell("H5 risk_x_CBI", None)
                b2, p2 = cell("H5 risk_x_CBI_x_state", None)
                b3, p3 = cell("gov CBI", None)
                b4, p4 = cell("gov CBI_x_state", None)
                rows.append([w, lab, b1, p1, b2, p2, b3, p3, b4, p4])
        _df(rows, ["Window", "State", "Risk×CBI β", "p (Risk×CBI)",
                   "Risk×CBI×state β", "p (interaction)", "Gov CBI β", "p (Gov CBI)",
                   "Gov CBI×state β", "p (Gov×state)"])
        st.caption(f"Median spread {t5['spread_median']:.2f} pp (BAA−AAA), median 12-month "
                   f"CPI inflation {t5['infl_median']:.2f}%. No significant state "
                   "dependence in either direction.")

    # ── Test 6 ──
    st.markdown("---")
    st.markdown("### Test 6 · Power and multiple testing")
    t6 = P["test6"]
    st.markdown("**Minimum detectable effect (80% power, HC1)**")
    rows = [[k, _b(v["beta"]), f"{v['se']:.3f}", f"{v['mde']:.3f}"]
            for k, v in t6["minimum_detectable_effect"].items()]
    _df(rows, ["Estimate", "β", "SE", "MDE"])

    pl = t6["placebo_FG"]
    st.markdown("**Placebo: random forward-guidance regimes**")
    _df([["Actual Tone × FG", _b(pl["actual_tone_x_fg"]), "—"],
         ["Random 56-meeting regimes (p)", "—", _p(pl["random_56_meetings_p"])],
         ["Consecutive 56-meeting blocks at least as negative",
          f"{pl['consecutive_56_blocks']} blocks",
          f"{pl['share_of_blocks_at_least_as_negative']*100:.1f}%"]],
        ["", "β / blocks", "p / share"])
    st.caption("Only 1 of 95 consecutive blocks (the true FG period, starting 2008-12-16) "
               "is as negative as the actual estimate — the regime interaction is not an "
               "artifact of arbitrary period choice.")

    mt = t6["H4_42_tests_classical"]
    st.markdown("**Multiple testing: 42 H4 coefficient tests**")
    _df([["Classical SE", mt["raw_p_below_05"], mt["holm_below_05"], mt["bh_below_05"]],
         ["HC1 SE", t6["H4_42_tests_hc1"]["raw_p_below_05"],
          t6["H4_42_tests_hc1"]["holm_below_05"], t6["H4_42_tests_hc1"]["bh_below_05"]]],
        ["SE method", "Raw p < .05", "Survive Holm", "Survive Benjamini-Hochberg"])
    st.caption("Of 42 H4 flow coefficients, no test survives correction for multiple "
               "testing under either SE method.")

    # ── Tone-measure comparison (T8) ──
    st.markdown("---")
    st.markdown("### Tone measures: do they separate hikes from cuts?")
    if os.path.exists(TM):
        tm = json.load(open(TM))
        rows = []
        for v in tm["t8_tone_measures"]:
            rows.append([v["measure"], f"{v['mean_hikes']:.2f}", f"{v['mean_cuts']:.2f}",
                         f"{v['mean_holds']:.2f}", f"{v['gap_sd']:.3f}{v['sig']}",
                         _p(v["p"]), int(v["n_hikes"]), int(v["n_cuts"])])
        _df(rows, ["Measure", "Mean (hikes)", "Mean (cuts)", "Mean (holds)",
                   "Hike−cut gap (SD)", "p", "N hikes", "N cuts"])
        st.caption("The LLM hawkishness score (−100…+100) separates hikes from cuts at "
                   "p < 1e-9 (gap 2.0 SD), the dictionary measures do not (LM p = .160). "
                   "Source: tables_all_results.xlsx, Table 8.")
    else:
        st.info("tone_measures_current.json not found.")

    # ── Tests awaiting data ──
    st.markdown("---")
    st.markdown("### Tests awaiting external data")
    st.markdown("""
    | Test | What it needs | Status |
    |---|---|---|
    | **2 · Official JK shocks** | `data/jk_shocks_meeting.csv` from Marek Jarociński's site; set `JK_*_COL` at the top of the script | ⏳ awaiting file |
    | **4 · Retail vs institutional flows** | run `python3 direction2/robustness_tests.py --wrds` once (needs CRSP `fund_hdr_hist` retail/inst flags) | ⏳ awaiting WRDS run |
    | **7 · SPF forecast errors** | Philadelphia Fed SPF median-forecast files in `data/spf/` | ⏳ awaiting files |
    """)
    st.caption("Test 5 ran successfully once the FRED cache was populated "
               "(data/fred_cache/ BAA, AAA, CPIAUCSL — downloaded 2026-11-20).")

    st.markdown("---")
    st.markdown("**Pipeline note.** All numbers on this page are generated by "
                "`direction2/robustness_tests.py` and committed as JSON "
                "(`robustness_tests/results.json`, `tone_measures_current.json`). "
                "Tests 2/4/7 render automatically once their data files are in place.")
