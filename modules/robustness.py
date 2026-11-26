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

    # ── Test 2: official JK series (NOW RAN) ──
    st.markdown("---")
    st.markdown("### Test 2 · Official Jarociński–Karadi MP/CBI series (RAN 2026-11-26)")
    t2 = P.get("test2", {})
    if t2 and "matched_meetings" in t2:
        st.success("**The flow results get STRONGER under the official series.** "
                   "Risk × CBI (same month) is +5.28 with **p = .005 under month-clustered SE** "
                   "(sign-rule series: p = .466). H4 gov-bond CBI: −43.1 (p = .037), "
                   "Wald MP=CBI p = .025. Source: Marek Jarociński's updated replication files "
                   "(github.com/marekjarocinski/jkshocks_update_fed, MP_pm/CBI_pm), matched "
                   "161/161 of our meeting dates.", icon="✅")
        rows = [[
            str(t2.get("matched_meetings")),
            f"{t2.get('corr_mp_jk_vs_sign_rule', 0):.3f}",
            f"{t2.get('corr_cbi_jk_vs_sign_rule', 0):.3f}",
            f"{t2.get('corr_mp_jk_cbi_jk', 0):.3f}",
        ]]
        _df(rows, ["Matched meetings", "corr MP: official vs sign-rule",
                   "corr CBI: official vs sign-rule", "corr MP–CBI within official"])
        st.caption("The sign-rule CBI is nearly uncorrelated with the official CBI (ρ = .04): "
                   "they capture different information components, so the official-series "
                   "results are new evidence, not a re-run.")
        hrows = []
        for k, v in t2.items():
            if k.startswith("H4"):
                hrows.append([k, v["n"], _b(v["CBI"]["beta"]) + f" (p {_p(v['CBI']['p'])})",
                              _b(v["MP"]["beta"]) + f" (p {_p(v['MP']['p'])})",
                              f"{v['wald_p']:.3f}"])
        if hrows:
            _df(hrows, ["H4 gov bonds (HC1)", "N", "β CBI (official)", "β MP (official)", "Wald p"])
        frows = []
        for k, v in t2.items():
            if k.startswith("H5"):
                r, m = v["risk_x_cbi"], v["risk_x_mp"]
                frows.append([k, v.get("n", ""), _b(r["beta"]) + f" (p {_p(r['p'])})",
                              _b(m["beta"]) + f" (p {_p(m['p'])})"])
        if frows:
            _df(frows, ["H5 pooled (month-clustered)", "N", "Risk × CBI", "Risk × MP"])
    else:
        st.info("Test 2 has not been run yet.")

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

    # ── E11: tone split (sentiment vs hawkishness) ──
    st.markdown("---")
    st.markdown("### Tone split: sentiment vs hawkishness (professor E11)")
    PR = os.path.join(BASE, "direction2/flow_rerun_output_v8/professor_revisions.json")
    if os.path.exists(PR):
        pr = json.load(open(PR))
        split = pr.get("prices_tone_split_and_jk", {})
        rows = []
        for name, spec in split.items():
            if "skipped" in spec:
                continue
            c = spec["coefs"]
            rows.append([name, spec["n"], f"{spec['r2']*100:.1f}%",
                         (_b(c["tone100"]["beta"]) + f" (p {_p(c['tone100']['p'])})") if "tone100" in c else "—",
                         (_b(c["hawk100"]["beta"]) + f" (p {_p(c['hawk100']['p'])})") if "hawk100" in c else "—",
                         (_b(c["tone_x_fg"]["beta"]) + f" (p {_p(c['tone_x_fg']['p'])})") if "tone_x_fg" in c else "—",
                         (_b(c["hawk_x_fg"]["beta"]) + f" (p {_p(c['hawk_x_fg']['p'])})") if "hawk_x_fg" in c else "—"])
        _df(rows, ["Specification", "N", "R²", "Sentiment β (p)", "Hawkishness β (p)",
                   "Sentiment × FG β (p)", "Hawkishness × FG β (p)"])
        st.success("**Answer to E11: it is sentiment, not hawkishness.** In the FG period "
                   "(N = 56, dataset flag), the return response loads on statement sentiment "
                   "(tone100 −1.11, p = .005 even when both measures are included), while the "
                   "LLM hawkishness score is at most marginal (p = .08–.92; only the "
                   "both-measures spec reaches 10%). Sentiment breadth, not directional "
                   "hawkishness, is what prices react to.", icon="✅")
        st.caption("Source: direction2/professor_revisions.py (LLM hawkishness from "
                   "llm_statement_scores.jsonl, 5-model consensus).")

        # ── E10: slide-23 horse race, official vs sign-rule controls ──
        st.markdown("**E10 · Slide-23 horse race with MP/CBI controls, official vs sign-rule**")
        e10 = {k: v for k, v in split.items() if k.startswith("E10")}
        if e10:
            rows = []
            for name, spec in e10.items():
                if "skipped" in spec:
                    continue
                c = spec["coefs"]
                tone = (_b(c["tone100"]["beta"]) + f" (p {_p(c['tone100']['p'])})") if "tone100" in c else "—"
                mp = (_b(c["mp_jk"]["beta"]) + f" (p {_p(c['mp_jk']['p'])})") if "mp_jk" in c else (
                     (_b(c["mp_shock"]["beta"]) + f" (p {_p(c['mp_shock']['p'])})") if "mp_shock" in c else "—")
                cbi = (_b(c["cbi_jk"]["beta"]) + f" (p {_p(c['cbi_jk']['p'])})") if "cbi_jk" in c else (
                      (_b(c["cbi_shock"]["beta"]) + f" (p {_p(c['cbi_shock']['p'])})") if "cbi_shock" in c else "—")
                rows.append([name, spec["n"], f"{spec['r2']*100:.1f}%", tone, mp, cbi])
            _df(rows, ["Specification", "N", "R²", "Tone β (p)", "MP β (p)", "CBI β (p)"])
            st.caption("**Tone survives the official-series controls**: −0.96 (p = .009) with "
                       "official JK MP/CBI in the FG subsample (published sign-rule version: "
                       "−1.22, p = .001, R² 50.5% — matches slide 23's 48% spec). The "
                       "full-sample spec stays insignificant (p = .29). "
                       "Source: direction2/professor_revisions.py.")

        # A5 clustered flows, both shock measures
        st.markdown("**A5 · Clustered flows, both shock measures side by side**")
        cl = pr.get("flows_clustered_both_shocks", {})
        rows = []
        for k, v in cl.items():
            r, m = v["risk_x_cbi"], v["risk_x_mp"]
            rows.append([k, v["n"], _b(r["beta"]) + f" (p {_p(r['p'])})",
                         _b(m["beta"]) + f" (p {_p(m['p'])})"])
        _df(rows, ["Shock measure · window", "N", "Risk × CBI (clustered)", "Risk × MP (clustered)"])
        st.caption("Month-clustered SEs on the class-month panel. Official JK series: "
                   "same +5.28 (p = .005), post +3.45 (p = .006), diff −1.79 (p = .37) — "
                   "the same-window effect partially unwinds two months out, consistent "
                   "with H6. Sign-rule series remains insignificant.")

    # ── Professor Part I: papers to add ──
    st.markdown("---")
    st.markdown("### Related literature added per the professor's memo (Part I)")
    st.markdown("""
    | Paper | Relevance |
    |---|---|
    | **Bauer & Swanson (2023)**, JME — *An Alternative Explanation for the 'Fed Information Effect'* | Strong pre-FOMC evidence ⇒ the Fed-information-effect literature must control for it. Our statement-based CBI is designed to avoid the pre-FOMC confound (C4) |
    | **Kroencke, Schmeling & Schrimpf**, CEPR DP14037 — *The FOMC Risk Shift* | Fund flows around FOMC, risk-on/off shifts — **closest paper to ours**; our CBI/risk-ladder channel is distinct |
    | **Banegas, Montes-Rojas & Siga (2022)**, JIMF | MP shocks and mutual-fund flows by fund strategy |
    | **Brooks, Katz & Lustig (2018)**, NBER 25127 | Bond-fund investors pull money out after hikes with a lag — directly relevant to H6 |
    | **Bernanke & Kuttner (2005)**, JoF 60(3) | Why prices move at all: the canonical mechanism |
    """)

    # ── Test 4: retail vs institutional (RAN 2026-11-26) ──
    st.markdown("---")
    st.markdown("### Test 4 · Retail vs institutional flows (RAN 2026-11-26)")
    PR4 = os.path.join(BASE, "direction2/flow_rerun_output_v8/professor_revisions.json")
    t4s = P.get("test4", {})
    t4o = json.load(open(PR4)).get("test4_official_jk", {}) if os.path.exists(PR4) else {}
    if t4o:
        inst_share = t4o.get("share_of_tna_institutional", 0)
        st.success("**Institutions are the ladder movers.** Institutional share of TNA: "
                   f"{inst_share*100:.1f}%. Under the OFFICIAL JK series, the Risk x CBI "
                   "ladder response is significant for institutional share classes "
                   "(same +5.46, p = .012; post +4.27, p = .014) and insignificant for "
                   "retail (p = .13 / .84). Retail funds only show a delayed bounce two "
                   "months out (diff +30.2, p = .054 on gov-bond CBI) - slow money "
                   "chasing what fast money already traded. Sign-rule series: no "
                   "significant split either way.", icon="✅")
        rows = []
        for w in ["same", "post", "diff"]:
            b = t4o.get(f"{w} (official JK)", {})
            if not b:
                continue
            gr, gi = b["gov bonds, retail"]["CBI"], b["gov bonds, institutional"]["CBI"]
            hr, hi = b["H5 pooled, retail"]["risk_x_cbi"], b["H5 pooled, institutional"]["risk_x_cbi"]
            rows.append([w, _b(gr["beta"]) + f" (p {_p(gr['p'])})",
                         _b(gi["beta"]) + f" (p {_p(gi['p'])})",
                         _b(hr["beta"]) + f" (p {_p(hr['p'])})",
                         _b(hi["beta"]) + f" (p {_p(hi['p'])})"])
        _df(rows, ["Window", "Gov CBI — retail", "Gov CBI — institutional",
                   "Risk × CBI — retail", "Risk × CBI — institutional"])
        st.caption("Official JK shocks, gov-bond cross-section HC1 / ladder panel "
                   "month-clustered. Sign-rule variant (results.json test4) shows no "
                   "significant split either way - consistent with the sign-rule series "
                   "being a different information component. Source: professor_revisions.json (test4_official_jk); WRDS data pulled 2026-11-26 via CRSP fund_hdr_hist point-in-time join.")

        # ── Test 7: SPF forecast revisions (RAN 2026-11-26) ──
    st.markdown("---")
    st.markdown("### Test 7 · SPF forecast revisions (RAN 2026-11-26)")
    t7 = P.get("test7", {})
    if t7 and t7.get("quarters"):
        st.success("Philadelphia Fed SPF median-forecast files downloaded and the test ran: "
                   "**75 survey quarters** (2006Q2–2025Q1). The official shock components "
                   "show weak, statistically indistinguishable links to forecast revisions "
                   "(all Wald MP = CBI p > .08); the orthogonalized tone measure is a "
                   "borderline predictor of GDP and CPI revisions.", icon="✅")
        rows = []
        for y, lab in [("rev_rgdp", "Real GDP growth revision"), ("rev_unemp", "Unemployment revision"),
                       ("rev_cpi", "CPI inflation revision")]:
            v = t7.get(y)
            if not v:
                continue
            mp, cbi, wt = v["MP"], v["CBI"], v["with_tone"]
            rows.append([lab, _b(mp["beta"]) + f" (p {_p(mp['p'])})",
                         _b(cbi["beta"]) + f" (p {_p(cbi['p'])})",
                         f"{v['wald_MP_eq_CBI_p']:.3f}",
                         _b(wt["tone"]["beta"]) + f" (p {_p(wt['tone']['p'])})"])
        _df(rows, ["SPF outcome (per quarter)", "MP β (p)", "CBI β (p)", "Wald p (MP=CBI)",
                   "Tone β (p), both in"])
        st.caption("Revision = quarter-t median forecast minus quarter-t−1 (Eileen's spec). "
                   "Honest reading: the macro-revision benchmark is uninformative between MP "
                   "and CBI at this sample size; the tone signal (higher tone → upward GDP "
                   "and CPI revisions) is consistent with the breadth interpretation but is "
                   "only marginal. Source: robustness_tests/results.json, test7; files in "
                   "data/spf/ rebuilt from the Fed's combined medianLevel/medianGrowth workbooks.")
    else:
        st.info("Test 7 has not been run yet.")

    # Test 4 status update
    # ── Tests awaiting data ──
    st.markdown("---")
    st.markdown("### Tests awaiting external data")
    st.markdown("""
    | Test | What it needs | Status |
    |---|---|---|
    """)
    st.caption("Test 5 ran successfully once the FRED cache was populated "
               "(data/fred_cache/ BAA, AAA, CPIAUCSL — downloaded 2026-11-20).")

    st.markdown("---")
    st.markdown("**Pipeline note.** All numbers on this page are generated by "
                "`direction2/robustness_tests.py` and committed as JSON "
                "(`robustness_tests/results.json`, `tone_measures_current.json`). "
                "All seven tests have now run (tests 1-7 complete as of 2026-11-26).")
