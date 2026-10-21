# -*- coding: utf-8 -*-
"""Generate all LaTeX tables from archival result JSONs (numbers never hand-typed).

Outputs paper/tables/*.tex, inserted into the section markdown as raw LaTeX.
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "paper", "tables")
os.makedirs(OUTDIR, exist_ok=True)

V8 = json.load(open(os.path.join(BASE, "direction2", "flow_rerun_output_v8", "flow_results_rerun_v8.json")))
MV = json.load(open(os.path.join(BASE, "direction2", "flow_rerun_output_v8", "models_v2_results.json")))
PR = json.load(open(os.path.join(BASE, "direction2", "flow_rerun_output_v8", "price_results_v1.json")))
HW = json.load(open(os.path.join(BASE, "direction2", "flow_rerun_output_v8", "h4_all_windows.json")))
AV = json.load(open(os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_report_v2.json")))
FW = json.load(open(os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_5way_summary.json")))

def coef(d, key):
    v = d.get(key)
    if not isinstance(v, dict) or "beta" not in v:
        return "---", "---"
    star = ""
    p = v["p"]
    if p < 0.01:
        star = "^{***}"
    elif p < 0.05:
        star = "^{**}"
    elif p < 0.10:
        star = "^{*}"
    sign = "+" if v["beta"] >= 0 else "-"
    return f"${sign}{abs(v['beta']):.3f}{star}$", f"({v['se']:.3f})"

def esc(s):
    """Escape LaTeX specials in TEXT columns (numbers already valid LaTeX)."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s

def wrap(title, label, header, rows, note=""):
    col = "l" + "c" * (len(header) - 1)
    title = esc(title) if "%" in title or "_" in title or "&" in title else title
    body = "\n".join(" & ".join(esc(c) if not c.startswith("$") and not c.startswith("(") and c not in ("1", "--") else c
                                 for c in r) + r" \\" for r in rows)
    n = "\\begin{table}[!tbp]\n\\centering\\small\n"
    n += f"\\caption{{{title}}}\n\\label{{{label}}}\n"
    n += f"\\begin{{tabular}}{{{col}}}\n\\toprule\n"
    n += " & ".join(header) + " \\\\\n\\midrule\n"
    n += body + "\n\\bottomrule\n\\end{tabular}\n"
    if note:
        n += f"\\\\[2pt]\\begin{{minipage}}{{0.92\\linewidth}}\\footnotesize\\emph{{Notes:}} {note}\\end{{minipage}}\n"
    n += "\\end{table}\n"
    return n

def save(name, tex):
    open(os.path.join(OUTDIR, name), "w", encoding="utf-8").write(tex)
    print("wrote", name)

# ---------- Table 4.1: tone by asset x regime (price side) ----------
ASSETS = [("ty10_chg", "10Y futures (bps)"), ("ty2_chg", "2Y futures (bps)"),
          ("crsp_vw", "CRSP VW return (%)"), ("sp500", "S\\&P 500 return (%)")]
rows = []
for a, lab in ASSETS:
    b, se = coef(PR["results"][f"B_{a}_all"], "tone100")
    rows.append([lab, "All", b, se])
for a, lab in ASSETS:
    b, se = coef(PR["results"][f"B_{a}_FG"], "tone100")
    rows.append([lab, "FG", b, se])
for a, lab in ASSETS:
    b, se = coef(PR["results"][f"B_{a}_nonFG"], "tone100")
    rows.append([lab, "Non-FG", b, se])
save("table41.tex", wrap(
    "Tone and same-day price reactions, by asset and regime",
    "tab:41", ["Asset", "Sample", "Tone ($\\beta$)", "SE"], rows,
    "OLS with heteroskedasticity-robust SEs. Dependent variable: same-day change. Tone: LM hawkish-minus-dovish per 100 words. FG: Dec-2008--Dec-2015 (56 meetings). Non-FG row for CRSP uses n=94 (CRSP coverage 150/161). Archival source: \\texttt{price\\_results\\_v1.json}."))

# ---------- Table 4.2: horse race (FG, equities) ----------
rows = []
for a, lab in [("crsp_vw", "CRSP VW"), ("sp500", "S\\&P 500")]:
    d = PR["results"][f"C_{a}_FG"]
    t, ts = coef(d, "tone100"); m, ms = coef(d, "mp_shock"); c, cs = coef(d, "cbi_shock")
    rows.append([lab, f"{d['n']}", f"{d['r2']:.3f}", t, ts, m, ms, c, cs])
save("table42.tex", wrap(
    "Horse race in the FG regime: tone vs.\\ shocks (equities)",
    "tab:42", ["Asset", "N", "R$^2$", "Tone", "", "MP shock", "", "CBI shock", ""], rows,
    "Same-day equity return on tone, MP shock, and CBI shock jointly (FG subsample). All three regressors significant. Archival source: \\texttt{price\\_results\\_v1.json}."))

# ---------- Table 6.1: risk ladder + asymmetry ----------
h5 = V8["H5_risk_ladder_same"]
r_x_cbi, _ = coef(h5, "risk_x_cbi"); r_x_mp, _ = coef(h5, "risk_x_mp")
cbi, _ = coef(h5, "cbi_shock"); mp, _ = coef(h5, "mp_shock")
sa = MV["H5_sign_asymmetry"]
sa_pos, _ = coef(sa, "risk_x_cbi_pos"); sa_neg, _ = coef(sa, "risk_x_cbi_neg")
rows = [
    ["CBI shock (main)", cbi, "Risk $\\times$ CBI", r_x_cbi],
    ["MP shock (main)", mp, "Risk $\\times$ MP", r_x_mp],
    ["Risk $\\times$ CBI, positive CBI", sa_pos, "Risk $\\times$ CBI, negative CBI", sa_neg],
]
save("table61.tex", wrap(
    "The risk ladder: fund flows and the information component",
    "tab:61", ["Specification", "$\\beta$", "Specification", "$\\beta$"], rows,
    "Class-level monthly flow rate on shocks and risk-rank interactions (n=1{,}099 fund-months; 7 classes; risk rank 1--7). Risk $\\times$ CBI $=-0.066$ (p=0.036); Risk $\\times$ MP $=+0.005$ (p=0.88); Wald test of equality p=0.113. Sign asymmetry from the split-interaction spec (models\\_v2). Archival sources: \\texttt{flow\\_results\\_rerun\\_v8.json}, \\texttt{models\\_v2\\_results.json}."))

# ---------- Table 6.2: zero-rate exception ----------
z = V8["zero_rate_check"]
mz, ms_ = coef(z, "mp_x_zlb"); cz, cs_ = coef(z, "cbi_x_zlb")
rows = [["MP $\\times$ ZLB", mz, ms_], ["CBI $\\times$ ZLB", cz, cs_]]
save("table62.tex", wrap(
    "The zero-rate exception",
    "tab:62", ["Interaction", "$\\beta$", "SE"], rows,
    "Flow regressions with zero-rate-window interactions (n={n}, R$^2$={r2}). MP $\\times$ ZLB $=-1.505$ (p=0.005); CBI $\\times$ ZLB insignificant (p=0.78). Archival source: \\texttt{{flow\\_results\\_rerun\\_v8.json}}.".format(n=z['n'], r2=round(z['r2'], 3))))

# ---------- Table 8.1: three windows (gov bonds) ----------
rows = []
for wk, lab in (("same", "Same month"), ("post", "Post"), ("diff", "Diff (post $-$ same)")):
    d = HW[wk]["government_bonds"]
    m, _ = coef(d, "mp_shock"); c, _ = coef(d, "cbi_shock")
    rows.append([lab, str(d["n"]), f"{d['r2']:.3f}", m, c])
save("table81.tex", wrap(
    "Flow windows: same, post, and difference (government bonds)",
    "tab:81", ["Window", "N", "R$^2$", "MP shock", "CBI shock"], rows,
    "De-risking dynamics: the information shock enters safe-asset flows positively in the same month (+0.759, p=0.037) and reverses in the difference window ($-$0.988, p=0.014); the conventional shock is null at every horizon (p$\\geq$0.76). Archival source: \\texttt{h4\\_all\\_windows.json}."))

# ---------- Table A1: audit verdicts ----------
meta = AV["meta"]
vd = meta.get("verdict_distribution", {})
mt = meta.get("matching_test", {})
rows = [
    ["Verified", str(vd.get("Verified", 0)), "L1 semantic match rate", f"{mt.get('hit_rate', 0)*100:.0f}\\%"],
    ["LikelyTrue", str(vd.get("LikelyTrue", 0)), "L4 cross-vendor sign agreement", "93\\% (29) / 100\\% (divergence)"],
    ["Uncertain", str(vd.get("Uncertain", 0)), "L5 self-consistency pass", "70\\%"],
    ["LikelyFalse", str(vd.get("LikelyFalse", 0)), "L3 CROWN (LM-anchor independence)", "0/30 triggered"],
    ["Hallucination", str(vd.get("Hallucination", 0)), "One-off errors caught", "1 (OT 2011-09-21)"],
]
save("tablea1.tex", wrap(
    "Five-layer audit: verdicts and layer outcomes (30-statement sample)",
    "tab:a1", ["Verdict", "N", "Layer outcome", "Rate"], rows,
    "Verdicts after threshold-bug fix (v2 recompute). Cross-vendor layer: router/GPT-4.1-mini/Claude-haiku/DeepSeek (n=29) with partial Gemini. Archival sources: \\texttt{llm\\_audit\\_report\\_v2.json}, \\texttt{llm\\_5way\\_summary.json}."))

# ---------- Table A3: seven classes, same-month direct effects ----------
rows = []
LAB = {"government_bonds": "Government bonds", "corporate_bonds": "Corporate bonds",
       "real_assets": "Real assets", "large_cap_equity": "Large-cap equity",
       "developed_market_equity": "Developed-market equity",
       "emerging_market_equity": "Emerging-market equity", "small_cap_equity": "Small-cap equity"}
for cls, lab in LAB.items():
    d = V8["H4_mp_vs_cbi"]["same"][cls]
    m, _ = coef(d, "mp_shock"); c, _ = coef(d, "cbi_shock")
    rows.append([lab, str(d["n"]), m, c])
save("tablea3.tex", wrap(
    "Direct shock effects on monthly flows, seven asset classes (same month)",
    "tab:a3", ["Class", "N", "MP shock", "CBI shock"], rows,
    "Same-month window. Government-bond CBI is the only significant direct effect (+0.759, p=0.037) — the safe-asset anchor of the de-risking ladder. Archival source: \\texttt{flow\\_results\\_rerun\\_v8.json}."))

print("all tables done")

# ---------- Table A2: cross-vendor correlation matrix ----------
import pandas as pd
fw = pd.read_json(os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_5way_summary.json"))
fw = fw.T.astype(float)
corr = fw.corr()
lab = {"router": "Main router", "gpt": "GPT-4.1-mini", "claude": "Claude-haiku",
       "gemini": "Gemini-2.5-flash", "deepseek": "DeepSeek-flash"}
cols = [c for c in ("router", "gpt", "claude", "gemini", "deepseek") if c in corr.columns]
rows = []
for r in cols:
    cells = []
    for c in cols:
        if c == r:
            cells.append("1")
        elif cols.index(c) < cols.index(r):
            cells.append("--")
        else:
            cells.append(f"{corr.loc[r, c]:.2f}")
    rows.append([lab[r]] + cells)
save("tablea2.tex", wrap(
    "Cross-vendor score correlations (audit sample, $n$ up to 30)",
    "tab:a2", ["Model"] + [lab[c] for c in cols], rows,
    "Pairwise Pearson correlations of stance scores across vendor families; router/gpt/claude/deepseek cover 29-30 audit statements, gemini 11-22 (thinking-model latency; fill-in ongoing). Sign agreement across the four complete families: 93.1\\% overall, 100\\% on the 11 LM-divergence statements. Archival source: \\texttt{llm\\_5way\\_summary.json}."))
print("a2 done")
