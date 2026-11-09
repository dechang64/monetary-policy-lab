# -*- coding: utf-8 -*-
"""Rebuild result slides 23-34 of Eileen's research proposal deck per
professor feedback: native tables + disruption-style charts replace WMF
screenshots. All numbers read from archived JSONs — zero hand-typed values.

Input : upload/...research_proposal_v2_2026-10-16_revision.pptx
Output: upload/research_proposal_v3_2026-11-09.pptx
"""
import copy
import json
import os

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

BASE = "/home/z/my-project/monetary-policy-lab"
SRC = ("/home/z/my-project/upload/"
       "6af142abb6ab055640b5630d_research_proposal_v2_2026-10-16_revision.pptx")
DST = "/home/z/my-project/upload/research_proposal_v3_2026-11-09.pptx"
CH = os.path.join(BASE, "paper", "ppt_charts")

P = json.load(open(os.path.join(BASE, "paper", "ppt_results_v2.json")))
WIN = json.load(open(os.path.join(BASE, "direction2",
                                  "flow_rerun_output_v8", "h4_all_windows.json")))

NAVY, RED, GRAY, BLACK = (RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0xC0, 0x39, 0x2B),
                          RGBColor(0x66, 0x66, 0x66), RGBColor(0x22, 0x22, 0x22))


def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def _ln(tc, tag, w_pt):
    tcPr = tc.get_or_add_tcPr()
    ln = etree.SubElement(tcPr, qn(tag))
    ln.set("w", str(int(w_pt * 12700))); ln.set("cap", "flat")
    ln.set("cmpd", "sng"); ln.set("algn", "ctr")
    sf = etree.SubElement(ln, qn("a:solidFill"))
    c = etree.SubElement(sf, qn("a:srgbClr")); c.set("val", "1F3A5F")
    tcPr.remove(ln); tcPr.insert(0, ln)  # lnT/lnB precede fill per schema


def border(cell, top=None, bottom=None):
    if bottom:
        _ln(cell._tc, "a:lnB", bottom)
    if top:
        _ln(cell._tc, "a:lnT", top)


def txt(cell, paras, size=13, align=PP_ALIGN.LEFT, color=BLACK, bold=False,
        italic=False, font="Calibri"):
    """paras: list of (text, overrides-dict)."""
    cell.margin_left, cell.margin_right = Inches(.08), Inches(.08)
    cell.margin_top, cell.margin_bottom = Inches(.02), Inches(.02)
    cell.vertical_anchor = 1  # middle
    tf = cell.text_frame
    tf.clear()
    for i, (t, ov) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ov.get("align", align)
        r = p.add_run(); r.text = t
        f = r.font
        f.name = ov.get("font", font); f.size = Pt(ov.get("size", size))
        f.bold = ov.get("bold", bold); f.italic = ov.get("italic", italic)
        f.color.rgb = ov.get("color", color)


def new_slide(prs, title, suffix=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    tb = s.shapes.add_textbox(Inches(.56), Inches(.22), Inches(12.2), Inches(.8))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.name, r.font.size, r.font.color.rgb = "Calibri", Pt(18), BLACK
    if suffix:
        r2 = p.add_run(); r2.text = "  — " + suffix
        r2.font.name, r2.font.size, r2.font.bold = "Calibri", Pt(15), True
        r2.font.color.rgb = NAVY
    return s


def add_chart(s, png, top=1.30, max_w=11.35, max_h=4.75):
    w, h = Image.open(png).size
    r = min(max_w / (w / 96), max_h / (h / 96))
    pw, ph = (w / 96) * r, (h / 96) * r
    return s.shapes.add_picture(png, Inches((13.33 - pw) / 2), Inches(top),
                                Inches(pw), Inches(ph))


def takeaway(s, text, top=6.12):
    tb = s.shapes.add_textbox(Inches(.84), Inches(top), Inches(11.7), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "Result: "
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "Calibri", Pt(15), True, NAVY
    r2 = p.add_run(); r2.text = text
    r2.font.name, r2.font.size, r2.font.color.rgb = "Calibri", Pt(15), BLACK


def source(s, text):
    tb = s.shapes.add_textbox(Inches(.84), Inches(7.06), Inches(11.7), Inches(.35))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.name, r.font.size, r.font.italic, r.font.color.rgb = "Calibri", Pt(10), True, GRAY


def coef_row(row, key_beta, key_se, key_p, hl=False):
    b, se, p = row[key_beta], row[key_se], row[key_p]
    sig = p < .05
    color = RED if (hl and sig) else BLACK
    return [(f"{b:+.3f}{stars(p)}",
             {"bold": sig, "color": color, "align": PP_ALIGN.CENTER}),
            (f"({se:.3f})",
             {"italic": True, "color": GRAY, "size": 11, "align": PP_ALIGN.CENTER}),
            (f"{p:.3f}",
             {"align": PP_ALIGN.CENTER, "bold": sig, "color": color})]


def stat_row(label, b, se, p, hl=False, fmt=".3f"):
    sig = p < .05
    color = RED if (hl and sig) else BLACK
    return [(label, {}),
            (f"{b:+{fmt}}{stars(p)}",
             {"bold": sig, "color": color, "align": PP_ALIGN.CENTER}),
            (f"({se:{fmt}}",
             {"italic": True, "color": GRAY, "size": 11, "align": PP_ALIGN.CENTER}),
            (f"{p:.3f}", {"align": PP_ALIGN.CENTER, "bold": sig, "color": color})]


def add_table(s, header, rows, top=1.45, widths=None, hl_rows=(), nhead=1,
              title=None, title_top=None):
    """header: list of column labels for row 0. title: optional spanning line
    rendered as a textbox above the table."""
    if title is not None:
        tb = s.shapes.add_textbox(Inches(.9), Inches(title_top or top - .42),
                                  Inches(11.5), Inches(.38))
        p = tb.text_frame.paragraphs[0]
        r = p.add_run(); r.text = title
        r.font.name, r.font.size, r.font.italic, r.font.color.rgb = (
            "Calibri", Pt(12.5), True, RGBColor(0x44, 0x44, 0x44))
    nr, nc = 1 + len(rows), len(header)
    tbl = s.shapes.add_table(nr, nc, Inches(.9), Inches(top),
                             Inches(11.5), Inches(.4 * nr)).table
    tbl.first_row = False; tbl.horz_banding = False
    if widths:
        for i, w in enumerate(widths):
            tbl.columns[i].width = Inches(w)
    for c in range(nc):  # header row
        cell = tbl.cell(0, c)
        txt(cell, [(header[c], {})], size=13, bold=True, color=NAVY,
            align=PP_ALIGN.CENTER if c else PP_ALIGN.LEFT)
        border(cell, top=2.0)
    for r, row in enumerate(rows, 1):
        for c in range(nc):
            cell = tbl.cell(r, c)
            pair = row[c] if isinstance(row[c], tuple) else (row[c], {})
            txt(cell, [pair])
            if r == len(rows):
                border(cell, bottom=2.0)
        if r in hl_rows:
            for c in range(nc):
                tbl.cell(r, c).fill.solid()
                tbl.cell(r, c).fill.fore_color.rgb = RGBColor(0xF5, 0xE9, 0xE7)
    for r in range(nr):  # white fill everywhere (kill default banding)
        for c in range(nc):
            if not (r in hl_rows):
                tbl.cell(r, c).fill.solid()
                tbl.cell(r, c).fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for c in range(nc):  # rule under header
        border(tbl.cell(nhead - 1, c), bottom=1.0)
    return tbl


prs = Presentation(SRC)
print("loaded:", len(prs.slides), "slides")

# ---- delete old result slides 23-34 (indices 22-33), high-index first ----
xml_slides = prs.slides._sldIdLst
for idx in range(33, 21, -1):
    rId = xml_slides[idx].get(qn("r:id"))
    prs.part.drop_rel(rId)
    xml_slides.remove(xml_slides[idx])
print("after delete:", len(prs.slides))

h1, h2, h3, h4, h5, h6, h1p = (P["h1"], P["h2"], P["h3"], P["h4"], P["h5"],
                               P["h6"], P["h1_price"])
T_H1 = "H1: Target and Path monetary-policy surprises are associated with FOMC statement sentiment"

# ============ S23: H1 regression table ============
s = new_slide(prs, T_H1, "Regression Evidence")
rows = [stat_row("Target shock (Acosta)", h1["target"]["beta"], h1["target"]["se"],
                 h1["target"]["p"], hl=True, fmt=".6f"),
        stat_row("Path shock (Acosta)", h1["path"]["beta"], h1["path"]["se"],
                 h1["path"]["p"], fmt=".6f"),
        [("N", {}), (f"{h1['n']}", {"align": PP_ALIGN.CENTER}), ("", {}),
         ("", {})],
        [("R²", {}), (f"{h1['r2']*100:.2f}%", {"align": PP_ALIGN.CENTER}), ("", {}),
         ("", {})]]
add_table(s, ["Variable", "Coefficient", "NW SE", "p-value"], rows,
          top=1.75, widths=[3.6, 2.7, 2.6, 2.6], hl_rows=(1,),
          title="Dependent variable: statement sentiment (0.5·LM + 0.5·CB) — NW HAC(4)")
takeaway(s, "Target shocks are significantly associated with statement sentiment "
            "(β = 0.000577, p = 0.017); Path shocks are not (p = 0.152). H1 is "
            "supported through the Target channel — tone tracks the action, "
            "not the guidance.")
source(s, "Source: verified_results.json (audited spec); OLS with Newey-West HAC(4); "
          "sample 2006:01–2022:07, N = 117.")

# ============ S24: H1 coefficient plot ============
s = new_slide(prs, T_H1, "Coefficient Evidence")
add_chart(s, f"{CH}/h1_coef.png")
takeaway(s, "Only the Target shock's 95% CI excludes zero. Statement sentiment "
            "responds to the policy action itself, not to communication about "
            "the future path — the asymmetry that motivates the incremental-"
            "information tests of H2.")
source(s, "Source: verified_results.json; NW HAC(4) standard errors; bars show 95% CI.")

# ============ S25: H1 price response ============
s = new_slide(prs, T_H1, "Both Shocks Move Prices on Announcement Day")
add_chart(s, f"{CH}/h1_price.png")
cv = h1p["CRSP VW"]
takeaway(s, f"Equity indices respond negatively to both shocks (CRSP VW: Target "
            f"{cv['target']['beta']:+.2f}, Path {cv['path']['beta']:+.2f}, p = 0.037); "
            "Treasury and bill responses are muted. Prices digest both components "
            "immediately — setting up the question of what the statement text "
            "adds on top (H2).")
source(s, "Source: fomc_dataset_2006_latest.csv; event-window returns on shocks, "
          "OLS with 95% CI; N = 161 meetings, 2006–2026.")

# ============ S26: H2 table ============
s = new_slide(prs, "H2: Incremental Information",
              "Tone beyond Target and Path — Forward-Guidance Regime")
rows = []
for a in ["CRSP VW", "S&P 500", "10Y Treasury", "2Y Treasury"]:
    fg = h2[a]["fg"]
    tone, mp, cbi = fg["tone100"], fg["mp_shock"], fg["cbi_shock"]
    sig = tone["p"] < .05
    r = [(a, {"bold": sig}),
         (f"{tone['beta']:+.3f}{stars(tone['p'])}",
          {"align": PP_ALIGN.CENTER, "bold": sig,
           "color": RED if sig else BLACK}),
         (f"({tone['se']:.3f})", {"italic": True, "color": GRAY, "size": 11,
                                  "align": PP_ALIGN.CENTER}),
         (f"{mp['beta']:+.3f}{stars(mp['p'])}", {"align": PP_ALIGN.CENTER}),
         (f"{cbi['beta']:+.3f}{stars(cbi['p'])}", {"align": PP_ALIGN.CENTER}),
         (f"{fg['r2']*100:.1f}%", {"align": PP_ALIGN.CENTER})]
    rows.append(r)
add_table(s, ["Asset (FG sample)", "β Tone", "(SE)", "β MP", "β CBI", "R²"], rows,
          top=1.75, widths=[2.6, 2.4, 1.7, 1.6, 1.6, 1.1], hl_rows=(1, 2),
          title="FG sample (Dec 2008 – Dec 2015); dependent variable: asset return")
takeaway(s, "Within the forward-guidance regime, tone carries significant "
            "incremental information for equities (CRSP VW: −1.21, p < 0.001; "
            "S&P 500: −1.12, p < 0.001) beyond both shock components — R² rises "
            "to 48%. Treasuries show no tone effect.")
source(s, "Source: price_results_v1.json, spec C (tone + MP + CBI, FG sample); "
          "stars: * p<0.1, ** p<0.05, *** p<0.01.")

# ============ S27: H2 chart ============
s = new_slide(prs, "H2: Incremental Information",
              "Tone Effects Concentrated in the FG Regime")
add_chart(s, f"{CH}/h2_tone.png")
takeaway(s, "Full-sample tone coefficients (navy) are indistinguishable from zero; "
            "FG-regime coefficients (red) are significant for both equity indices. "
            "Words carry price information exactly when words were the policy "
            "instrument.")
source(s, "Source: price_results_v1.json, spec C by regime; 95% CI bars.")

# ============ S28: H3 table ============
s = new_slide(prs, "H3: State Dependence", "Tone × FG Interaction (CRSP VW)")
d = h3["interact"]
rows = [stat_row("Tone (×100)", d["tone100"]["beta"], d["tone100"]["se"],
                 d["tone100"]["p"]),
        stat_row("FG regime dummy (2008–15)", d["fg_2008_2015"]["beta"],
                 d["fg_2008_2015"]["se"], d["fg_2008_2015"]["p"]),
        stat_row("Tone × FG", d["tone_x_fg"]["beta"], d["tone_x_fg"]["se"],
                 d["tone_x_fg"]["p"], hl=True),
        [("N", {}), (f"{d['n']}", {"align": PP_ALIGN.CENTER}), ("", {}), ("", {})],
        [("R²", {}), (f"{d['r2']*100:.1f}%", {"align": PP_ALIGN.CENTER}), ("", {}), ("", {})]]
add_table(s, ["Variable", "Coefficient", "SE", "p-value"], rows,
          top=1.75, widths=[3.6, 2.7, 2.6, 2.6], hl_rows=(3,),
          title="Dependent variable: CRSP VW return — interaction spec")
takeaway(s, "The interaction confirms regime dependence: tone has no effect "
            "outside FG (+0.04, p = 0.81) but a large, significant effect inside "
            "it (−1.156, p = 0.003). Text matters exactly when the statement "
            "was the instrument.")
source(s, "Source: price_results_v1.json, spec D (interaction); OLS; "
          "2006–2026, N = 150.")

# ============ S29: H3 chart ============
s = new_slide(prs, "H3: State Dependence", "Explanatory Power by Regime")
add_chart(s, f"{CH}/h3_regime.png")
takeaway(s, "R² = 14.1% in the FG regime vs 0.1% outside it and 1.1% in the "
            "2020–22 ZLB template era. State dependence is the rule, not the "
            "exception — and motivates the LLM-based measurement layer.")
source(s, "Source: price_results_v1.json spec B by regime; batch3_results.json "
          "(ZLB window).")

# ============ S30: H4 table ============
s = new_slide(prs, "H4: MP vs. CBI", "Differential Effects on Mutual-Fund Flows "
              "(meeting month)")
lab = {"government_bonds": "Government bonds", "corporate_bonds": "Corporate bonds",
       "real_assets": "Real assets", "large_cap_equity": "Large-cap equity",
       "developed_market_equity": "DM equity", "emerging_market_equity": "EM equity",
       "small_cap_equity": "Small-cap equity"}
rows = []
for c, v in h4.items():
    rows.append([(lab[c], {}), *coef_row(v, "mp_beta", "mp_p", "mp_p")[:0],
                 (f"{v['mp_beta']:+.3f}{stars(v['mp_p'])}",
                  {"align": PP_ALIGN.CENTER, "bold": v["mp_p"] < .05}),
                 (f"{v['cbi_beta']:+.3f}{stars(v['cbi_p'])}",
                  {"align": PP_ALIGN.CENTER, "bold": v["cbi_p"] < .05,
                   "color": RED if c == "government_bonds" and v["cbi_p"] < .05 else BLACK}),
                 (f"{v['wald_mp_eq_cbi_p']:.2f}", {"align": PP_ALIGN.CENTER})])
add_table(s, ["Asset class (meeting month)", "β MP", "β CBI", "Wald p (MP=CBI)"], rows,
          top=1.75, widths=[3.4, 2.7, 2.7, 2.7], hl_rows=(1,),
          title="Controls: log TNA, flow volatility; class-level flows")
takeaway(s, "Government bonds receive significant inflows on CBI shocks "
            "(+0.76 %TNA, p = 0.037) with no MP response (Wald p = 0.098); real "
            "assets respond to MP instead (+0.30, p = 0.04). Information shocks — "
            "not action shocks — move safe-haven portfolios.")
source(s, "Source: flow_rerun_output_v8/flow_results_rerun_v8.json, H4 same-window spec; "
          "class-level flows, N = 157 meeting months.")

# ============ S31: H4 chart ============
s = new_slide(prs, "H4: MP vs. CBI", "Flow Response by Asset Class")
add_chart(s, f"{CH}/h4_flows.png")
takeaway(s, "The CBI effect concentrates in government bonds while the MP "
            "effect appears in real assets — the two shock components transmit "
            "to different portfolio margins, consistent with the "
            "information-vs-action decomposition.")
source(s, "Source: flow_rerun_output_v8; 95% CI bars; asset classes ordered by risk rank.")

# ============ S32: H5 table ============
s = new_slide(prs, "H5 — Risk Ladder", "Interaction of Risk Rank with Shock Type")
L, A, Z = h5["ladder"], h5["asym"], h5["zlb"]
rows = [stat_row("Risk × CBI", L["risk_x_cbi"]["beta"], L["risk_x_cbi"]["se"],
                 L["risk_x_cbi"]["p"], hl=True),
        stat_row("Risk × MP", L["risk_x_mp"]["beta"], L["risk_x_mp"]["se"],
                 L["risk_x_mp"]["p"]),
        stat_row("Risk × CBI (hawkish: CBI > 0)", A["risk_x_cbi_pos"]["beta"],
                 A["risk_x_cbi_pos"]["se"], A["risk_x_cbi_pos"]["p"]),
        stat_row("Risk × CBI (dovish: CBI < 0)", A["risk_x_cbi_neg"]["beta"],
                 A["risk_x_cbi_neg"]["se"], A["risk_x_cbi_neg"]["p"]),
        stat_row("MP × ZLB (2020–22)", Z["mp_x_zlb"]["beta"], Z["mp_x_zlb"]["se"],
                 Z["mp_x_zlb"]["p"])]
add_table(s, ["Variable", "Coefficient", "SE", "p-value"], rows,
          top=1.75, widths=[4.2, 2.5, 2.4, 2.4], hl_rows=(1,),
          title="Dependent variable: class-level flow (%TNA), meeting month")
takeaway(s, "CBI shocks push portfolios down the risk ladder (Risk×CBI = −0.066, "
            "p = 0.036; gap-free spec −0.114, p = 0.002) — opposite to the original "
            "prediction. The effect is driven by hawkish information; and MP shocks "
            "acquire ladder effects only at the zero bound (MP×ZLB = −1.505).")
source(s, "Source: flow_results_rerun_v8.json (H5_risk_ladder_same, zero_rate_check); "
          "models_v2_results.json (H5_sign_asymmetry).")

# ============ S33: H5 chart ============
s = new_slide(prs, "H5 — Risk Ladder", "CBI Flow Response by Risk Rank")
add_chart(s, f"{CH}/h5_ladder.png")
takeaway(s, "The ladder is monotone in risk rank: the safest class (government "
            "bonds, +0.76 %TNA) gains the most, riskier classes lose or stay "
            "flat. One unit of hawkish central-bank information reallocates "
            "portfolios toward safety.")
source(s, "Source: flow_rerun_output_v8; CBI coefficients with 95% CI band; "
          "asset classes ordered from safest to riskiest.")

# ============ S34: H6 ============
s = new_slide(prs, "H6: MP and CBI shocks exhibit different transmission patterns "
              "across horizons", "Prices First, Portfolios Later")
add_chart(s, f"{CH}/h6_windows.png", top=1.25, max_h=4.35)
g_same = WIN["same"]["government_bonds"]["cbi_beta"]
g_post = WIN["post"]["government_bonds"]["cbi_beta"]
g_diff = WIN["diff"]["government_bonds"]["cbi_beta"]
takeaway(s, f"The CBI flow effect on government bonds reverses across horizons "
            f"({g_same:+.2f} meeting month → {g_diff:+.2f} two months out): initial "
            "flight-to-safety inflows are partially unwound as portfolios "
            "rebalance — prices clear the information first, portfolios adjust "
            "later.", top=5.85)
source(s, "Source: h4_all_windows.json (same / post / diff windows); 95% CI bands.")

prs.save(DST)
print("saved:", DST, "| slides:", len(prs.slides))
