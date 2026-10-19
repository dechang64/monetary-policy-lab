#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-page glossary for the Priyanka meeting (Eileen's pre-meeting checklist, item 2)."""
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
sec = doc.sections[0]
sec.left_margin = Inches(0.7); sec.right_margin = Inches(0.7)
sec.top_margin = Inches(0.6); sec.bottom_margin = Inches(0.6)
style = doc.styles['Normal']
style.font.name = 'Georgia'
style.font.size = Pt(9.5)

h = doc.add_heading('Glossary — What Moves Money: Rate News vs. Economy News', level=1)
h.add_run('\nOne page. Plain-words definitions; source paper in parentheses. Keep beside you during the meeting.').italic = True
h.add_run(' ').font.size = Pt(2)

rows = [
    ("Announcement surprise",
     "The part of the FOMC decision the market did NOT price in beforehand. Ex ante = what was expected; ex post = what actually happened. The difference is the surprise."),
    ("Target shock",
     "Unexpected change in the current policy rate, measured in fed funds futures in a narrow window around the 2:00pm release. (Kuttner 2001; GSS 2005.)"),
    ("Path shock",
     "Unexpected change in expected FUTURE rates, measured in Eurodollar futures. A 'same decision, different future' surprise. (GSS 2005.)"),
    ("MP shock (rate news)",
     "A target-rate surprise on a day when stock prices move AGAINST the rate move. Reading: the Fed acted, markets repriced discount rates. Pure policy action. (Jarociński & Karadi 2020.)"),
    ("CBI shock (economy news)",
     "A target-rate surprise on a day when stock prices move WITH the rate move. Reading: the Fed revealed what it knows about the economy; markets updated beliefs. Central-bank information. (Jarociński & Karadi 2020.)"),
    ("Tone",
     "Hawkish words minus dovish words per 100 words in the statement text (Loughran-McDonald dictionaries). Descriptive language measure — NOT a shock, NOT a price. We use 'tone' throughout; never 'sentiment'."),
    ("Statement / body-only text",
     "The FOMC statement itself with all website navigation removed. 161 meetings, Jan 2006 - Apr 2026."),
    ("Fund flow",
     "Change in a fund's total net assets (TNA) minus the assets that came from returns: Flow = TNA(t) - TNA(t-1) x (1 + return). Positive = investors put money in. Measured per asset class, per month, as % of lagged TNA."),
    ("Risk ladder",
     "An ORDINAL ranking 1-7 of the seven CRSP asset classes from safest to riskiest: 1 gov bonds, 2 corporate bonds, 3 real assets, 4 large-cap equity, 5 developed-market equity, 6 emerging-market equity, 7 small-cap equity. It is a ranking, not a beta."),
    ("Gap rule (flow measurement)",
     "Most CRSP funds report TNA quarterly, not monthly. We only compute flows when two consecutive reports are at most 3 months apart, and drop |flow| > 50% of lagged TNA. Without this, multi-month market drift is misread as investor money moving."),
    ("Forward guidance (FG) regime",
     "Dec 16, 2008 - Dec 15, 2015: the period when the fed funds rate was at zero and the Fed steered expectations through words. Example sentence (Aug 9, 2011): 'likely to warrant exceptionally low levels for the federal funds rate at least through mid-2013.'"),
    ("Zero-rate (ZLB) regime",
     "Mar 15, 2020 - Jan 26, 2022: the COVID zero-rate period, kept separate from FG in the regressions."),
    ("Same / Post / Diff windows",
     "Same = the calendar month containing the meeting. Post = the following month. Diff = Post minus Same. Separates immediate from delayed money movement."),
    ("MP x ZLB",
     "Interaction test: does the flow response to rate news change during the zero-rate period? Found: yes, -1.51 (p=0.005) — rate news carried more (negative) flow response when rates could not move."),
]

table = doc.add_table(rows=0, cols=2)
table.style = 'Light Grid Accent 1'
for term, definition in rows:
    r = table.add_row()
    c0, c1 = r.cells
    p0 = c0.paragraphs[0]; run = p0.add_run(term); run.bold = True; run.font.size = Pt(9)
    p1 = c1.paragraphs[0]; run = p1.add_run(definition); run.font.size = Pt(9)
    c0.width = Inches(1.55); c1.width = Inches(5.75)

doc.add_paragraph()
p = doc.add_paragraph()
run = p.add_run('Four headline numbers to remember:  161 meetings (2006-2026) · 91 MP / 60 CBI · gov bonds CBI flow +0.76% (p=0.037) · Risk×CBI = −0.066 (p=0.036, de-risking).')
run.bold = True; run.font.size = Pt(9.5)

doc.save('/home/z/my-project/monetary-policy-lab/proposal/talk15/glossary_one_page.docx')
print('glossary saved')
