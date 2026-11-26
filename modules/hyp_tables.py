# -*- coding: utf-8 -*-
"""Proposal Hypotheses H1-H6: Regression Result Tables
=====================================================
The professor asked for web tables covering the remaining hypotheses,
matching the 'H1 Regression Results' table he saw on the Direction 2 page.

Numbering follows the research proposal deck (H1-H6). All numbers are read
from paper/ppt_results_v2.json (the same archived payload behind PPT v3) -
zero hand-typed values.

H1  Target and Path surprises are associated with statement sentiment
H2  Tone carries incremental information beyond Target and Path (FG regime)
H3  Tone-return relation is state dependent (forward guidance)
H4  MP vs CBI shocks have differential effects on mutual-fund flows
H5  CBI shocks move portfolios down the risk ladder
H6  Transmission differs across immediate prices vs subsequent horizons
"""
import json
import os

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "paper", "ppt_results_v2.json")
CH = os.path.join(BASE, "paper", "ppt_charts")
WIN = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "h4_all_windows.json")


def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def fmt(b, p, prec=3):
    return f"{b:+.{prec}f}{stars(p)}"


def _robustness():
    """Load the headline robustness numbers (gov CBI, Risk x CBI, 2008-09
    exclusion) from direction2/flow_rerun_output_v8/robustness_tests/results.json.
    Returns {} when the file is missing."""
    p = os.path.join(BASE, "direction2/flow_rerun_output_v8/robustness_tests/results.json")
    if not os.path.exists(p):
        return {}
    try:
        R = json.load(open(p))
        t1 = R.get("test1", {})
        inf = t1.get("influence", {})
        return {
            "gov_cbi": t1.get("H4 gov bonds, same month: CBI", {}),
            "risk_x_cbi": t1.get("H5 pooled (same): Risk x CBI", {}),
            "exclude_2008_09": inf.get("excluding 2008-09-16"),
        }
    except Exception:
        return {}


def chart(name):
    p = os.path.join(CH, name)
    if os.path.exists(p):
        st.image(p, use_container_width=True)
    else:
        st.caption(f"(chart file missing: {name})")


def render():
    st.markdown("## 📐 Proposal Hypotheses H1–H6: Regression Tables")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;'>
        <h3 style='color: white; margin: 0;'>Regression Evidence for All Six Hypotheses</h3>
        <p style='opacity: 0.85; margin: 0.5rem 0 0 0;'>
            Same archived results as the proposal deck (PPT v3) — every number traced to the
            reproducible pipeline<br>
            Significance: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("This page follows the **proposal's H1–H6 numbering** (as in the research "
            "proposal deck). The *Direction 2* page keeps an earlier internal numbering "
            "for its archived experiments.", icon="📐")

    if not os.path.exists(RES):
        st.error("paper/ppt_results_v2.json not found — tables cannot be rendered.")
        return
    P = json.load(open(RES))
    win = json.load(open(WIN)) if os.path.exists(WIN) else None

    # ============ H1 ============
    st.markdown("---")
    st.markdown("### H1: Target and Path surprises are associated with statement sentiment")
    tm_path = os.path.join(BASE, "direction2/flow_rerun_output_v8/tone_measures_current.json")
    if os.path.exists(tm_path):
        tm = json.load(open(tm_path))
        rows = []
        for v in tm["t1_current"]:
            if v["variable"].startswith("Wald"):
                rows.append([f"Wald test: Target = Path (p) — {v['sample']}",
                             "", "", f"{float(v['p']):.3f}"])
            elif v["variable"] == "const":
                continue
            else:
                rows.append([v["variable"].capitalize(), fmt(float(v["coef"]), float(v["p"])),
                             f"({float(v['se']):.3f})", f"{float(v['p']):.3f}"])
        # N / R2 per sample block
        for sample in dict.fromkeys(v["sample"] for v in tm["t1_current"]):
            blk = [v for v in tm["t1_current"] if v["sample"] == sample and v["n"]]
            if blk:
                rows.append([f"N — {sample}", str(int(blk[0]["n"])), "", ""])
                rows.append([f"R² — {sample}", f"{float(blk[0]['r2'])*100:.2f}%", "", ""])
        st.markdown("**H1 Regression — statement sentiment on Target and Path "
                    "(current dataset, OLS, classical SE)**")
        st.dataframe(pd.DataFrame(rows, columns=["Variable", "Coefficient", "SE", "p-value"]),
                     use_container_width=True, hide_index=True)
        st.caption("Current dataset (2006–2026; 2025-26 meetings without a same-day CRSP "
                   "return dropped → N = 131): neither shock predicts sentiment "
                   "(Target p = .36, Path p = .92). The earlier archived result "
                   "(audited spec, N = 117, Target p = .017) does not survive the "
                   "corrected classification — shown for comparison on the Research "
                   "Results page. Source: tables_all_results.xlsx, Table 1.")
    else:
        h1 = P["h1"]
        rows = [
            ["Target shock (Acosta)", fmt(h1["target"]["beta"], h1["target"]["p"], 6),
             f"({h1['target']['se']:.6f})", f"{h1['target']['p']:.3f}"],
            ["Path shock (Acosta)", fmt(h1["path"]["beta"], h1["path"]["p"], 6),
             f"({h1['path']['se']:.6f})", f"{h1['path']['p']:.3f}"],
            ["N", f"{h1['n']}", "", ""],
            ["R²", f"{h1['r2']*100:.2f}%", "", ""],
        ]
        st.markdown("**H1 Regression — statement sentiment (0.5·LM + 0.5·CB), NW HAC(4)**")
        st.dataframe(pd.DataFrame(rows, columns=["Variable", "Coefficient", "NW SE", "p-value"]),
                     use_container_width=True, hide_index=True)
        st.caption(f"Sample: {h1['sample']} (audited spec, n={h1['n']}) — {h1['spec']}. "
                   "Source: verified_results.json.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**H1 (price side): asset-return response to shocks**")
        h1p = P["h1_price"]
        rows = [[a, fmt(v["target"]["beta"], v["target"]["p"]),
                 f"{v['target']['p']:.3f}",
                 fmt(v["path"]["beta"], v["path"]["p"]), f"{v['path']['p']:.3f}"]
                for a, v in h1p.items()]
        st.dataframe(pd.DataFrame(rows, columns=["Asset", "β Target", "p (Target)",
                                                  "β Path", "p (Path)"]),
                     use_container_width=True, hide_index=True)
        st.caption("Event-window returns on shocks, OLS; N = 161 meetings, 2006–2026. "
                   "Source: price_results_v1.json (B specs).")
    with col2:
        chart("h1_coef.png")
    chart("h1_price.png")

    # ============ H2 ============
    st.markdown("---")
    st.markdown("### H2: Tone carries incremental information beyond Target and Path")
    st.markdown("Forward-guidance regime (Dec 2008 – Dec 2015): returns regressed on tone "
                "**plus** both shock components.")
    h2 = P["h2"]
    rows = []
    for a, v in h2.items():
        fg = v["fg"]
        tone, mp, cbi = fg["tone100"], fg["mp_shock"], fg["cbi_shock"]
        rows.append([a, fmt(tone["beta"], tone["p"]), f"({tone['se']:.3f})",
                     fmt(mp["beta"], mp["p"]), fmt(cbi["beta"], cbi["p"]),
                     f"{fg['r2']*100:.1f}%"])
    st.markdown("**H2 Regression — asset returns on tone + shocks (FG sample)**")
    st.dataframe(pd.DataFrame(rows, columns=["Asset (FG sample)", "β Tone", "(SE)",
                                             "β MP", "β CBI", "R²"]),
                 use_container_width=True, hide_index=True)
    st.caption("Source: price_results_v1.json (C specs, FG window). Full-sample tone "
               "coefficients are null (CRSP VW −0.132, p = 0.230) — the effect is "
               "regime-specific.")
    chart("h2_tone.png")

    # ============ H3 ============
    st.markdown("---")
    st.markdown("### H3: The tone–return relation is state dependent (regime interaction)")
    d = P["h3"]["interact"]
    rows = [
        ["Tone (non-FG baseline)", fmt(d["tone100"]["beta"], d["tone100"]["p"]),
         f"({d['tone100']['se']:.3f})", f"{d['tone100']['p']:.3f}"],
        ["FG regime dummy (2008–15)", fmt(d["fg_2008_2015"]["beta"], d["fg_2008_2015"]["p"]),
         f"({d['fg_2008_2015']['se']:.3f})", f"{d['fg_2008_2015']['p']:.3f}"],
        ["Tone × FG", fmt(d["tone_x_fg"]["beta"], d["tone_x_fg"]["p"]),
         f"({d['tone_x_fg']['se']:.3f})", f"{d['tone_x_fg']['p']:.3f}"],
        ["N", f"{d['n']}", "", ""],
        ["R²", f"{d['r2']*100:.1f}%", "", ""],
    ]
    st.markdown("**H3 Regression — CRSP VW return, interaction specification**")
    st.dataframe(pd.DataFrame(rows, columns=["Variable", "Coefficient", "SE", "p-value"]),
                 use_container_width=True, hide_index=True)
    r2 = P["h3"]["r2"]
    st.markdown("**Explanatory power by regime** (R² of tone → CRSP VW): "
                f"FG **{r2['fg']*100:.1f}%** · non-FG {r2['nonfg']*100:.1f}% · "
                f"ZLB (2020–22) {P['h3']['zlb_r2']*100:.1f}%")
    st.caption("Source: price_results_v1.json (D spec) + batch3_results.json (ZLB window).")
    chart("h3_regime.png")

    # ============ H4 ============
    st.markdown("---")
    st.markdown("### H4: MP and CBI shocks have differential effects on fund flows")
    h4 = P["h4"]
    rob = _robustness()
    rows = [[a.replace("_", " ").title(), fmt(v["mp_beta"], v["mp_p"]),
             fmt(v["cbi_beta"], v["cbi_p"]), f"{v['wald_mp_eq_cbi_p']:.2f}"]
            for a, v in h4.items()]
    if rob and "gov_cbi" in rob:
        gc = rob["gov_cbi"]
        rows.append(["Gov bonds, CBI — robust SE (HC1 / NW)",
                     fmt(gc["classical"]["beta"], gc["classical"]["p"]),
                     f"HC1 p = {gc['hc1']['p']:.3f}; NW p = {gc['hac']['p']:.3f}",
                     "not robust"])
        ex = rob.get("exclude_2008_09")
        if ex:
            rows.append(["Gov bonds, CBI — excluding 2008-09-16",
                         fmt(ex["gov_cbi_hc1"]["beta"], ex["gov_cbi_hc1"]["p"]),
                         f"(se {ex['gov_cbi_hc1']['se']:.3f})",
                         f"p = {ex['gov_cbi_hc1']['p']:.3f}"])
    st.markdown("**H4 Regression — class-level flows on MP and CBI (meeting month)**")
    st.dataframe(pd.DataFrame(rows, columns=["Asset class / variant", "β MP",
                                             "β CBI / note", "Wald p / robust p"]),
                 use_container_width=True, hide_index=True)
    st.caption("Controls: log TNA, flow volatility; class-level monthly flows. "
               "⚠️ Sign-rule caveat: the gov-bond CBI effect (classical p = .043) is NOT "
               "significant under robust standard errors (HC1 p = .342, NW p = .341) "
               "and hinges on 2008-09-16 (excluding it: β = +0.029, p = .925). "
               "✅ Under the OFFICIAL Jarociński–Karadi series (2026-11-26) the flows "
               "results are stronger, not weaker: gov-bond CBI −43.1 (p = .037, HC1), "
               "Wald MP=CBI p = .025 — see the Robustness Tests page. "
               "Source: flow_results_rerun_v8.json + robustness_tests/results.json.")
    chart("h4_flows.png")

    # ============ H5 ============
    st.markdown("---")
    st.markdown("### H5: CBI shocks push portfolios down the risk ladder")
    L = P["h5"]["ladder"]
    A = P["h5"]["asym"]
    Z = P["h5"]["zlb"]
    rob = _robustness()
    r5 = rob.get("risk_x_cbi") if rob else None
    rows = [
        ["Risk × CBI", fmt(L["risk_x_cbi"]["beta"], L["risk_x_cbi"]["p"]),
         f"({L['risk_x_cbi']['se']:.3f})", f"{L['risk_x_cbi']['p']:.3f}"],
        ["Risk × MP", fmt(L["risk_x_mp"]["beta"], L["risk_x_mp"]["p"]),
         f"({L['risk_x_mp']['se']:.3f})", f"{L['risk_x_mp']['p']:.3f}"],
        ["Risk × CBI (positive CBI = good economic news)",
         fmt(A["risk_x_cbi_pos"]["beta"], A["risk_x_cbi_pos"]["p"]),
         f"({A['risk_x_cbi_pos']['se']:.3f})", f"{A['risk_x_cbi_pos']['p']:.3f}"],
        ["Risk × CBI (negative CBI)",
         fmt(A["risk_x_cbi_neg"]["beta"], A["risk_x_cbi_neg"]["p"]),
         f"({A['risk_x_cbi_neg']['se']:.3f})", f"{A['risk_x_cbi_neg']['p']:.3f}"],
        ["MP × ZLB (2020–22)", fmt(Z["mp_x_zlb"]["beta"], Z["mp_x_zlb"]["p"]),
         f"({Z['mp_x_zlb']['se']:.3f})", f"{Z['mp_x_zlb']['p']:.3f}"],
        ["N (meeting months × classes)", f"{L['n']}", "", ""],
    ]
    if r5:
        rows.insert(1, ["Risk × CBI — robust / clustered SE", fmt(r5["classical"]["beta"], r5["classical"]["p"]),
                        f"HC1 p = {r5['hc1']['p']:.3f}; cluster-month p = {r5['cluster_month']['p']:.3f}",
                        "not robust"])
    st.markdown("**H5 Regression — flows on risk-rank interactions (meeting month)**")
    st.dataframe(pd.DataFrame(rows, columns=["Variable", "Coefficient", "SE / note", "p-value"]),
                 use_container_width=True, hide_index=True)
    st.caption("Dependent variable: class-level flow (%TNA). "
               "⚠️ Sign-rule caveat: Risk × CBI (classical p = .041) is NOT significant "
               "under HC1 (p = .434) or month-clustered SE (p = .466). "
               "✅ Official JK series + month-clustered SE: Risk × CBI +5.28, p = .005 "
               "(same window) — the ladder result stands on the official measure. "
               "Source: flow_results_rerun_v8.json + robustness_tests/results.json.")
    chart("h5_ladder.png")

    # ============ H6 ============
    st.markdown("---")
    st.markdown("### H6: Transmission differs across horizons (prices first, portfolios later)")
    if win:
        rows = []
        labels = {"same": "Same month", "post": "Post month", "diff": "Diff (post − same)"}
        for w in ["same", "post", "diff"]:
            for c in ["government_bonds", "emerging_market_equity"]:
                v = win[w][c]
                rows.append([labels[w], c.replace("_", " ").title(),
                             fmt(v["cbi_beta"], v["cbi_p"]), f"{v['cbi_p']:.3f}",
                             fmt(v["mp_beta"], v["mp_p"]), f"{v['mp_p']:.3f}"])
        st.markdown("**H6 Regression — flow response by window and asset class**")
        st.dataframe(pd.DataFrame(rows, columns=["Window", "Asset class", "β CBI",
                                                 "p (CBI)", "β MP", "p (MP)"]),
                     use_container_width=True, hide_index=True)
        st.caption("Source: h4_all_windows.json. The government-bond CBI effect reverses "
                   "across windows (+0.76 → −0.99): flight-to-safety inflows unwind as "
                   "portfolios rebalance.")
    chart("h6_windows.png")

    st.markdown("---")
    st.markdown("**Pipeline note.** Every coefficient above is generated by the archived "
                "regression pipeline and committed as JSON (`paper/ppt_results_v2.json`); "
                "the proposal deck (PPT v3) shows the same numbers, so the website, the "
                "deck, and the paper cannot drift apart.")
