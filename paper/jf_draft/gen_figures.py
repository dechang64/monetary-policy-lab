# -*- coding: utf-8 -*-
"""Generate the four JF-style figures from the audited JSON files.

JF house style: black-and-white friendly, serif labels, no chartjunk,
95% confidence intervals, vectors (PDF).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/z/my-project/monetary-policy-lab"
OUT = os.path.join(ROOT, "paper/jf_draft/figures")
os.makedirs(OUT, exist_ok=True)

RES = json.load(open(f"{ROOT}/direction2/flow_rerun_output_v8/robustness_tests/results.json"))
PR = json.load(open(f"{ROOT}/direction2/flow_rerun_output_v8/professor_revisions.json"))
TM = json.load(open(f"{ROOT}/direction2/flow_rerun_output_v8/tone_measures_current.json"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

WINDOWS = ["same", "post", "diff"]
WLAB = ["Same month", "Post month", "Diff (post $-$ same)"]


# ── Figure 1: the ladder response across windows, both shock measures ──
def fig1():
    t2 = RES["test2"]
    cl = PR["flows_clustered_both_shocks"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)

    # Panel A: Risk x CBI, official vs sign-rule (clustered SEs)
    ax = axes[0]
    off = [cl[f"official JK · {w}"]["risk_x_cbi"] for w in WINDOWS]
    sgn = [cl[f"sign rule · {w}"]["risk_x_cbi"] for w in WINDOWS]
    ax.errorbar(x - 0.12, [v["beta"] for v in off],
                yerr=[1.96 * v["se"] for v in off], fmt="o", color="black",
                capsize=3, lw=1.2, label="Official JK series")
    ax.errorbar(x + 0.12, [v["beta"] for v in sgn],
                yerr=[1.96 * v["se"] for v in sgn], fmt="s", color="0.55",
                capsize=3, lw=1.2, label="Sign rule")
    ax.axhline(0, color="0.3", lw=0.7, ls=":")
    ax.set_xticks(x, WLAB, fontsize=8.5)
    ax.set_ylabel(r"$\beta$ on Risk$\times$CBI")
    ax.set_title("A. Risk ladder (month-clustered SEs)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)

    # Panel B: gov-bond flow CBI coefficient, official series (HC1)
    ax = axes[1]
    h4 = [t2[f"H4 gov bonds ({w})"]["CBI"] for w in WINDOWS]
    mp = [t2[f"H4 gov bonds ({w})"]["MP"] for w in WINDOWS]
    ax.errorbar(x - 0.12, [v["beta"] for v in h4],
                yerr=[1.96 * v["se"] for v in h4], fmt="o", color="black",
                capsize=3, lw=1.2, label="CBI (official)")
    ax.errorbar(x + 0.12, [v["beta"] for v in mp],
                yerr=[1.96 * v["se"] for v in mp], fmt="s", color="0.55",
                capsize=3, lw=1.2, label="MP (official)")
    ax.axhline(0, color="0.3", lw=0.7, ls=":")
    ax.set_xticks(x, WLAB, fontsize=8.5)
    ax.set_ylabel(r"$\beta$ on gov-bond flows")
    ax.set_title("B. Government-bond flows (HC1)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_flows_windows.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig1 written")


# ── Figure 2: tone effect, FG period — sentiment vs hawkishness ──
def fig2():
    ps = PR["prices_tone_split_and_jk"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    specs = [("FG only: T + P + sentiment", "Sentiment only"),
             ("FG only: T + P + hawkishness", "Hawkishness only"),
             ("E11 FG only: sentiment + hawkishness", "Both in the race")]
    rows = []
    for k, lab in specs:
        c = ps[k]["coefs"]
        rows.append((lab, c.get("tone100"), c.get("hawk100")))
    n = len(rows)
    for i, (lab, t, h) in enumerate(rows):
        if t is not None:
            ax.errorbar(t["beta"], i - 0.16, xerr=1.96 * t["se"], fmt="o",
                        color="black", capsize=3, lw=1.3, markersize=5,
                        label="Sentiment (tone100)" if i == 0 else None)
        if h is not None:
            h100 = {"beta": h["beta"] * 100, "se": h["se"] * 100}
            ax.errorbar(h100["beta"], i + 0.16, xerr=1.96 * h100["se"], fmt="s",
                        color="0.55", capsize=3, lw=1.3, markersize=5,
                        label="Hawkishness (LLM, rescaled)" if i == 0 else None)
    ax.axvline(0, color="0.3", lw=0.7, ls=":")
    ax.set_yticks(range(n), [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.6)
    ax.set_xlabel(r"$\beta$ on same-day CRSP VW return (per 100 units of measure)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_tone_split.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig2 written (v2)")


# ── Figure 3: score compression — hike/cut gap by measure ──
def fig3():
    rows = TM["t8_tone_measures"]
    labs = {"Loughran–McDonald dictionary": "LM dictionary",
            "Central-bank word list": "Central-bank list",
            "Combined tone (½ LM + ½ CB)": "Combined (LM+CB)/2",
            "LLM hawkishness score (−100 to +100)": "LLM consensus"}
    names = [labs.get(r["measure"], r["measure"]) for r in rows]
    gaps = [r["gap_sd"] for r in rows]
    ps = [r["p"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bars = ax.barh(names, gaps, color=["0.75", "0.75", "0.55", "black"], height=0.55)
    for b, p in zip(bars, ps):
        ax.text(b.get_width() + 0.04, b.get_y() + b.get_height() / 2,
                f"p = {p:.3f}" if p >= .001 else "p < .001", va="center", fontsize=8.5)
    ax.set_xlabel("Hike–cut gap (SD units)")
    ax.set_xlim(0, max(gaps) * 1.28)
    ax.invert_yaxis()
    ax.set_title("Separating hikes from cuts: by tone measure", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_score_compression.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig3 written")


# ── Figure 4: the official-vs-sign-rule CBI identity scatter ──
def fig4():
    # per-meeting series
    import pandas as pd
    jk = pd.read_csv(f"{ROOT}/data/jk_shocks_meeting.csv", parse_dates=["date"])
    f = pd.read_csv(f"{ROOT}/results/fomc_dataset_2006_latest.csv")
    f["date"] = pd.to_datetime(f["date"])
    m = f.merge(jk, on="date", how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.scatter(m["mp_shock_daily"], m["mp_jk"], s=6, color="black", alpha=0.55)
    r = m[["mp_shock_daily", "mp_jk"]].corr().iloc[0, 1]
    ax.set_xlabel("Sign-rule MP shock")
    ax.set_ylabel("Official JK MP shock")
    ax.set_title(f"A. MP: $\\rho = {r:.2f}$", fontsize=10)
    ax = axes[1]
    ax.scatter(m["cbi_shock_daily"], m["cbi_jk"], s=6, color="black", alpha=0.55)
    r = m[["cbi_shock_daily", "cbi_jk"]].corr().iloc[0, 1]
    ax.set_xlabel("Sign-rule CBI shock")
    ax.set_ylabel("Official JK CBI shock")
    ax.set_title(f"B. CBI: $\\rho = {r:.2f}$", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_series_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig4 written")


fig1()
fig2()
fig3()
fig4()
print("all figures ->", OUT)
