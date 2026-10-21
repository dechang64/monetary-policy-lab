# -*- coding: utf-8 -*-
"""Generate the paper's figures (1-4) from archival files. Publication style.

F1 preview  : tone -> same-day CRSP VW return, FG vs non-FG (bin scatter)
F2 timeline : tone series with regime shading
F3 dynamics : flow response by window (same/post/diff), safe vs risky classes
F4 compress : LM tone vs LLM score scatter, divergence statements flagged
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 10.5,
    "axes.labelsize": 10, "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GREY = "#1f4e79", "#b23a3a", "#8a8a8a"

# ---------- shared data ----------
df = pd.read_csv(os.path.join(BASE, "results", "fomc_dataset_2006_latest.csv"),
                 index_col=0, parse_dates=True)
df["tone100"] = df["sentiment"] * 100

# ---------- Figure 1: preview (tone -> equity, FG vs non-FG) ----------
d1 = df.dropna(subset=["crsp_vw", "tone100"])
fg, nf = d1[d1["fg_2008_2015"] == 1], d1[d1["fg_2008_2015"] == 0]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for dat, c, lab in ((fg, RED, "Forward guidance era (2008-15)"),
                    (nf, BLUE, "Outside forward guidance")):
    bins = pd.qcut(dat["tone100"], 8, duplicates="drop")
    gm = dat.groupby(bins, observed=True).agg(x=("tone100", "mean"), y=("crsp_vw", "mean"))
    ax.scatter(gm.x, gm.y, s=34, color=c, zorder=3, label=f"{lab} (bins)")
    b, a = np.polyfit(dat["tone100"], dat["crsp_vw"], 1)
    xs = np.linspace(dat["tone100"].min(), dat["tone100"].max(), 50)
    ax.plot(xs, a + b * xs, color=c, lw=1.6, alpha=0.85,
            label=f"{lab} (slope {b:+.2f})")
ax.axhline(0, color=GREY, lw=0.7, ls=":")
ax.set_xlabel("Statement tone (LM per-100-words, hawkish +)")
ax.set_ylabel("Same-day CRSP VW return (%)")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_preview.png"), bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ---------- Figure 2: tone timeline with regime shading ----------
fig, ax = plt.subplots(figsize=(7.2, 3.4))
d2 = df.dropna(subset=["tone100"])
spans = [("2008-12-16", "2015-12-16", "#dfe8f2", "Forward guidance"),
         ("2020-03-15", "2022-01-26", "#f7e3d7", "Zero-rate window"),
         ("2022-03-16", "2026-04-29", "#e8e8e8", "Tightening cycle")]
for s, e, c, lab in spans:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.65,
               label=lab, zorder=1)
ax.plot(d2.index, d2["tone100"], color=BLUE, lw=1.1, marker="o", ms=2.6,
        zorder=3, label="Statement tone")
ax.axhline(0, color=GREY, lw=0.7, ls=":")
for dt, lab in (("2011-09-21", "OT"), ("2012-09-13", "QE3"), ("2022-06-15", "+75bp")):
    ax.annotate(lab, (pd.Timestamp(dt), df.loc[dt, "tone100"]),
                textcoords="offset points", xytext=(0, 9), ha="center",
                fontsize=8, color=RED)
ax.set_ylabel("Tone (per-100-words)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_timeline.png"), bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ---------- Figure 3: flow dynamics by window ----------
hw = json.load(open(os.path.join(BASE, "direction2", "flow_rerun_output_v8",
                                 "h4_all_windows.json")))
CLASSES = [("government_bonds", "Government bonds"),
           ("corporate_bonds", "Corporate bonds"),
           ("large_cap_equity", "Large-cap equity"),
           ("small_cap_equity", "Small-cap equity")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True)
for ax, comp, ttl in ((axes[0], "cbi_shock", "CBI (information) shock"),
                      (axes[1], "mp_shock", "MP (conventional) shock")):
    xs = np.arange(3)
    for i, (cls, lab) in enumerate(CLASSES):
        ys = [hw[w][cls][comp]["beta"] for w in ("same", "post", "diff")]
        shade = "#1f4e79" if "bond" in cls else ("#b23a3a" if "equity" in cls else GREY)
        ax.plot(xs, ys, marker="o", ms=5, lw=1.4, color=shade, label=lab)
    ax.axhline(0, color=GREY, lw=0.7, ls=":")
    ax.set_xticks(xs, ["Same month", "Post", "Diff"])
    ax.set_title(ttl)
    ax.set_ylabel("Flow response (pp per unit shock)" if comp == "cbi_shock" else "")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_dynamics.png"), bbox_inches="tight")
plt.close(fig)
print("fig3 done")

# ---------- Figure 4: score compression ----------
recs = [json.loads(l) for l in open(os.path.join(
    BASE, "direction2", "flow_rerun_output_v8", "llm_statement_scores.jsonl"),
    encoding="utf-8") if '"error"' not in l]
ll = pd.DataFrame(recs)
ll["date"] = pd.to_datetime(ll["date"])
df4 = pd.read_csv(os.path.join(BASE, "results", "fomc_dataset_2006_latest.csv"),
                  index_col=0, parse_dates=True)
df4["tone100"] = df4["sentiment"] * 100
mrg = ll.set_index("date").join(df4[["tone100"]]).dropna(subset=["tone100", "score"])
lm_dates = {"2007-08-07", "2012-09-13", "2015-10-28", "2017-06-14", "2017-09-20",
            "2017-11-01", "2020-06-10", "2020-09-16", "2020-11-05", "2020-12-16",
            "2022-06-15"}
mrg["div"] = [d.strftime("%Y-%m-%d") in lm_dates for d in mrg.index]
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.scatter(mrg.loc[~mrg["div"], "tone100"], mrg.loc[~mrg["div"], "score"],
           s=22, color=BLUE, alpha=0.65, label=f"Other meetings (n={int((~mrg['div']).sum())})")
dv = mrg[mrg["div"]]
ax.scatter(dv["tone100"], dv["score"], s=44, color=RED, marker="D",
           label=f"Paradigm-shift statements (n={len(dv)})")
for d, r in dv.iterrows():
    ax.annotate(d.strftime("%Y-%m"), (r["tone100"], r["score"]),
                textcoords="offset points", xytext=(6, 3), fontsize=7, color=RED)
ax.axhline(0, color=GREY, lw=0.7, ls=":")
ax.axvline(0, color=GREY, lw=0.7, ls=":")
ax.set_xlabel("LM dictionary tone (per-100-words)")
ax.set_ylabel("LLM statement score (−100..+100)")
fig.text(0.99, 0.01, "Flagged set: LM std 1.10 vs LLM std 71", ha="right",
         fontsize=7.5, color=GREY)
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_compression.png"), bbox_inches="tight")
plt.close(fig)
print("fig4 done")
print("ALL FIGURES DONE ->", OUT)
