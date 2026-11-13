# -*- coding: utf-8 -*-
"""V7 -> V8: complete the appendix rebuild.

Problems fixed:
1. The AI-generated appendix dropped rows (T4 78 data rows -> ~50 shown,
   T6 31 -> 26, T7 44 -> 33). Now every row from the source xlsx renders,
   auto-paginated at 13 data rows per slide with "(x of y)" titles.
2. Slides 72-90 ("ARCHIVED" v10.3-era outputs) removed per user request -
   they mixed pre-audit numbers with current results and confused readers.
3. Body slides 1-43 (user-edited in V7) and figures F00-F07 untouched.

Source: upload/...tables_all_results.xlsx (compiled Nov 12, 2026, 9 CURRENT
tables). Output: upload/research_proposal_v8_2026-11-13.pptx
"""
import copy
import os

import openpyxl
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

SRC = ("/home/z/my-project/upload/"
       "6af68dd62bb7711382e48efd_research_proposal_v7_2026-11-12.pptx")
XLSX = "/home/z/my-project/upload/6af68dd99f9cb7db6025521c_tables_all_results.xlsx"
DST = "/home/z/my-project/upload/research_proposal_v8_2026-11-13.pptx"

NAVY = RGBColor(0x1F, 0x2A, 0x44)
BAR = RGBColor(0xCC, 0x00, 0x33)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x88, 0x88, 0x88)
LIGHT = RGBColor(0xF2, 0xF4, 0xF8)

SHEETS = [
    ("T1_H1_Tone", None, None),
    ("T2_H2_Returns", None, None),
    ("T3_H3_ForwardGuidance", None, None),
    ("T4_Price_Specs_A-D", None, None),
    ("T5_H4_Flows_MP_vs_CBI", "Same (meeting month)", "Same window"),
    ("T5_H4_Flows_MP_vs_CBI", "Post (month after)", "Post window"),
    ("T5_H4_Flows_MP_vs_CBI", "Diff (Post − Same)", "Diff window"),
    ("T6_H5_RiskLadder", None, None),
    ("T7_Tone_in_Flows", None, None),
    ("T8_LLM_vs_Dictionary", None, None),
    ("T9_Flow_Data_Summary", None, None),
]
NUMCOLS = {"Coefficient", "SE", "p-value", "p", "R²", "Mean flow (% TNA)",
           "SD flow (% TNA)", "β MP", "β CBI", "Wald MP = CBI (p)",
           "Mean (hikes)", "Mean (cuts)", "Mean (holds)", "Hike–cut gap (SD)"}


def fmt_cell(v, col):
    if v is None or str(v).strip() == "":
        return ""
    s = str(v).strip()
    if col in NUMCOLS:
        try:
            x = float(s)
            return f"{x:.3f}" if abs(x) >= 0.001 or x == 0 else f"{x:.4f}"
        except ValueError:
            return s
    if col in ("N", "N hikes", "N cuts", "Months", "Max funds",
               "Avg. funds per month"):
        try:
            return str(int(float(s)))
        except ValueError:
            return s
    return s


def add_title(s, text, sub=None):
    tb = s.shapes.add_textbox(Inches(.55), Inches(.28), Inches(12.2), Inches(.8))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = (
        "Georgia", Pt(20), False, NAVY)
    bar = s.shapes.add_shape(1, Inches(.58), Inches(1.02), Inches(2.6), Inches(.045))
    bar.fill.solid(); bar.fill.fore_color.rgb = BAR; bar.line.fill.background()
    if sub:
        tb2 = s.shapes.add_textbox(Inches(.55), Inches(1.12), Inches(12.2), Inches(.4))
        p2 = tb2.text_frame.paragraphs[0]
        r2 = p2.add_run(); r2.text = sub
        r2.font.name, r2.font.size, r2.font.italic = "Calibri", Pt(12), True
        r2.font.color.rgb = GRAY
    return 1.55 if sub else 1.30  # table top


def add_grid(s, header, rows, top, widths=None):
    nr, nc = 1 + len(rows), len(header)
    tbl = (s.shapes.add_table(nr, nc, Inches(.55), Inches(top),
           Inches(12.24), Inches(.34 * nr)).table)
    tbl.first_row = False; tbl.horz_banding = False
    if widths:
        tot = sum(widths)
        for i, w in enumerate(widths):
            tbl.columns[i].width = Emu(int(12.24 / tot * w * 914400))
    for c in range(nc):
        cell = tbl.cell(0, c)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_top = cell.margin_bottom = Emu(18000)
        p = cell.text_frame.paragraphs[0]
        if c:
            p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(header[c])
        r.font.name, r.font.size, r.font.bold = "Calibri", Pt(10.5), True
        r.font.color.rgb = WHITE
    for ri, row in enumerate(rows, 1):
        for c in range(nc):
            cell = tbl.cell(ri, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if ri % 2 == 0 else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_top = cell.margin_bottom = Emu(15000)
            p = cell.text_frame.paragraphs[0]
            if c:
                p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(row[c])
            r.font.name, r.font.size = "Calibri", Pt(10.5)
            r.font.color.rgb = BLACK
    return tbl


prs = Presentation(SRC)
print("loaded:", len(prs.slides))
xml_slides = prs.slides._sldIdLst

# ---- record slide ids to delete LATER (old tables 45-63, ARCHIVED 72-90) ----
drop_ids = {int(xml_slides[i].get("id")) for i in range(44, 63)}
drop_ids |= {int(xml_slides[i].get("id")) for i in range(71, 90)}
print("will drop", len(drop_ids), "slides after adding new ones")

# ---- rebuild tables from xlsx (ADD FIRST: new parts get partnames beyond the
# existing 90, so no name collision with the soon-to-be-orphaned old parts;
# python-pptx leaves dropped parts in the package, and identical partnames
# corrupt the deck via zip-entry clobbering) ----
wb = openpyxl.load_workbook(XLSX, read_only=True)
blank = prs.slides[63].slide_layout          # figure-slide layout (blank-ish)
new_slides = []

for sheet, window_key, window_lab in SHEETS:
    ws = wb[sheet]
    grid = [list(r) for r in ws.iter_rows(values_only=True)]
    title = str(grid[0][0]).strip()
    sub = str(grid[1][0]).strip() if grid[1][0] else None
    header = [str(c).strip() if c is not None else "" for c in grid[3]]
    data = [r for r in grid[4:] if any(c is not None and str(c).strip() for c in r)]
    if window_key:
        data = [r for r in data if str(r[0]).strip() == window_key]
        sub = f"{sub} — {window_lab}" if sub else window_lab
    # widths: first two cols wide, rest even
    widths = [2.6, 1.6] + [1.0] * (len(header) - 2) if len(header) > 2 else [2.6, 1.0]
    per = 13
    chunks = [data[i:i + per] for i in range(0, len(data), per)] or [[]]
    n_pag = len(chunks)
    for pi, chunk in enumerate(chunks, 1):
        t = title if n_pag == 1 else f"{title} ({pi} of {n_pag})"
        s = prs.slides.add_slide(blank)
        top = add_title(s, t, sub if pi == 1 else None)
        rows = [[fmt_cell(v, header[c]) for c, v in enumerate(r)] for r in chunk]
        add_grid(s, header, rows, top, widths)
        new_slides.append(s)
        print(f"  slide_id={s.slide_id} rows={len(rows)} :: {t[:50]}")
        sub = None                              # subtitle only on page 1

print("new table slides:", len(new_slides))

# ---- move new slides right after the divider (index 43), in order ----
id_map = {}
for sldId in list(xml_slides):
    id_map[int(sldId.get("id"))] = sldId
els = [id_map[s.slide_id] for s in new_slides]
for el in els:
    xml_slides.remove(el)
anchor = id_map[int(xml_slides[43].get("id"))]   # divider (V7 slide 44)
for el in els:
    anchor.addnext(el)                            # chain: divider -> T1 -> ...
    anchor = el

# ---- NOW delete the old table slides + ARCHIVED slides (by recorded id) ----
for el in [id_map[i] for i in drop_ids]:
    rId = el.get(qn("r:id"))
    prs.part.drop_rel(rId)
    xml_slides.remove(el)
print("after delete:", len(prs.slides))

prs.save(DST)
print("saved:", DST, "| slides:", len(prs.slides))
