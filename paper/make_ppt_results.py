# -*- coding: utf-8 -*-
"""PPT v3 results: compute H1 regressions fresh, load archived JSONs,
emit disruption-style charts (white bg, navy markers, red fit line, gray
95% CI band, clean axes) + pre-formatted table payload for the deck rebuild.

Outputs: paper/ppt_charts/*.png + paper/ppt_results_v2.json
All numbers derive from archived result files or statsmodels runs on the
frozen dataset — zero hand-typed values.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = "/home/z/my-project/monetary-policy-lab"
OUT = os.path.join(BASE, "paper", "ppt_charts")
V8 = os.path.join(BASE, "direction2", "flow_rerun_output_v8")

NAVY, RED, GRAY = "#1F3A5F", "#C0392B", "#C9C9C9"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "axes.edgecolor": "#555555",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def band_ci(x, res):
    """95% mean-prediction CI for OLS y~x."""
    xs = np.linspace(x.min(), x.max(), 100)
    X = sm.add_constant(xs, has_constant="add")
    pr = res.get_prediction(X)
    return xs, pr.conf_int()[:, 0], pr.conf_int()[:, 1]


payload = {}

# ============ dataset ============
df = pd.read_csv(os.path.join(BASE, "results", "fomc_dataset_2006_latest.csv"),
                 index_col=0, parse_dates=True)

# ---------- H1: archived audited spec (verified_results.json, n=117 NW HAC4) ----------
ver = json.load(open(os.path.join(BASE, "results", "verified_results.json")))
h1v = ver["H1"]
payload["h1"] = {
    "sample": ver["sample"]["period"], "n": h1v["n"], "r2": h1v["r2"],
    "spec": "sentiment ~ target + path, NW HAC(4)",
    "target": {"beta": h1v["target_shock"]["beta"], "se": h1v["target_shock"]["se"],
               "p": h1v["target_shock"]["p"]},
    "path": {"beta": h1v["path_shock"]["beta"], "se": h1v["path_shock"]["se"],
             "p": h1v["path_shock"]["p"]},
}

# ================= CHART 1: H1 coefficient plot (slide 24) =================
fig, ax = plt.subplots(figsize=(8.2, 4.3))
for i, (lab, k) in enumerate([("Target shock", "target"), ("Path shock", "path")]):
    b, se, p = h1v[f"{k}_shock"]["beta"], h1v[f"{k}_shock"]["se"], h1v[f"{k}_shock"]["p"]
    ax.errorbar([i], [b], yerr=1.96 * se, fmt="o", ms=10, color=NAVY,
                ecolor=GRAY, elinewidth=6, capsize=0, zorder=3)
    ax.annotate(f"{b:.6f}{stars(p)}\n(p = {p:.3f})", (i + .07, b),
                va="center", fontsize=13.5, color=NAVY, fontweight="bold")
ax.axhline(0, color="#888", lw=.9, ls="--")
ax.set_xticks([0, 1], ["Target shock", "Path shock"], fontsize=13)
ax.set_xlim(-.55, 1.75)
ax.set_ylabel("Coefficient on statement sentiment")
ax.set_title("FOMC Shocks and Statement Sentiment (H1)", fontsize=13.5, pad=8)
ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
ax.text(.02, .97, f"Bars: 95% CI  |  N = {h1v['n']}  |  R² = {h1v['r2']*100:.1f}%  |  NW HAC(4)\n"
                  f"Sample: {ver['sample']['period']}", transform=ax.transAxes,
        ha="left", va="top", fontsize=10, color="#666")
fig.tight_layout()
fig.savefig(f"{OUT}/h1_coef.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- H1-price: Target/Path betas across assets ----------
assets = [("crsp_vw", "CRSP VW"), ("crsp_ew", "CRSP EW"), ("sp500", "S&P 500"),
          ("ty10_chg", "10Y Treasury"), ("ty2_chg", "2Y Treasury"), ("tb3m_chg", "3M T-bill")]
h1p = {}
for col, lab in assets:
    d = df.dropna(subset=[col, "target_shock", "path_shock"])
    m = sm.OLS(d[col], sm.add_constant(d[["target_shock", "path_shock"]])).fit()
    h1p[lab] = {"target": {"beta": float(m.params["target_shock"]),
                           "se": float(m.bse["target_shock"]), "p": float(m.pvalues["target_shock"])},
                "path": {"beta": float(m.params["path_shock"]),
                         "se": float(m.bse["path_shock"]), "p": float(m.pvalues["path_shock"])},
                "n": int(m.nobs)}
payload["h1_price"] = h1p

# ---------- archived loads ----------
price = json.load(open(os.path.join(V8, "price_results_v1.json")))["results"]
flow = json.load(open(os.path.join(V8, "flow_results_rerun_v8.json")))
win = json.load(open(os.path.join(V8, "h4_all_windows.json")))
models2 = json.load(open(os.path.join(V8, "models_v2_results.json")))
payload["h2"], payload["h3"], payload["h4"], payload["h6"] = {}, {}, {}, {}
payload["h5"] = {"ladder": flow["H5_risk_ladder_same"], "asym": models2["H5_sign_asymmetry"],
                 "zlb": flow["zero_rate_check"]}

CLASS_ORDER = ["government_bonds", "corporate_bonds", "real_assets", "large_cap_equity",
               "developed_market_equity", "emerging_market_equity", "small_cap_equity"]
CLASS_LAB = {"government_bonds": "Govt\nBonds", "corporate_bonds": "Corp\nBonds",
             "real_assets": "Real\nAssets", "large_cap_equity": "Large-Cap\nEquity",
             "developed_market_equity": "DM\nEquity", "emerging_market_equity": "EM\nEquity",
             "small_cap_equity": "Small-Cap\nEquity"}

# ================= CHART 2: H1 price coef plot (slide 25) =================
fig, ax = plt.subplots(figsize=(11.4, 4.0))
labs = [l for _, l in assets]
xt = np.arange(len(labs))
for i, (key, col, lab) in enumerate([("target", NAVY, "Target (level) shock"),
                                     ("path", RED, "Path (guidance) shock")]):
    betas = [h1p[l][key]["beta"] for l in labs]
    errs = [1.96 * h1p[l][key]["se"] for l in labs]
    ax.errorbar(xt + (i - .5) * .18, betas, yerr=errs, fmt="o", ms=7, lw=1.6,
                capsize=4, color=col, label=lab, zorder=3)
    for x, b, e, l in zip(xt + (i - .5) * .18, betas, errs, labs):
        if h1p[l][key]["p"] < .05:
            ax.annotate(stars(h1p[l][key]["p"]), (x, b + e + .04), ha="center",
                        fontsize=13, color=col, fontweight="bold")
ax.axhline(0, color="#888888", lw=.9, ls="--", zorder=1)
ax.set_xticks(xt, labs, fontsize=11.5)
ax.set_ylabel("Coefficient on shock (95% CI)")
ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
ax.legend(fontsize=11, loc="lower left")
ax.set_title("Asset-Price Response to Monetary-Policy Shocks", fontsize=12.5, pad=8)
ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo - .12, yhi + .22)  # room for stars
fig.tight_layout()
fig.savefig(f"{OUT}/h1_price.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ================= CHART 3: H2 tone coef by asset, FG vs full (slide 27) =================
h2map = [("crsp_vw", "CRSP VW"), ("sp500", "S&P 500"), ("ty10_chg", "10Y Treasury"), ("ty2_chg", "2Y Treasury")]
fig, ax = plt.subplots(figsize=(11.4, 4.0))
xt = np.arange(len(h2map))
for i, (sample, col, lab) in enumerate([("FG", NAVY, "Forward-guidance regime"),
                                        ("all", "#8FA3BC", "Full sample")]):
    betas, errs, ps = [], [], []
    for k, _ in h2map:
        spec = price[f"C_{k}_{sample}"]["tone100"]
        betas.append(spec["beta"]); errs.append(1.96 * spec["se"]); ps.append(spec["p"])
    ax.errorbar(xt + (i - .5) * .18, betas, yerr=errs, fmt="o", ms=7, lw=1.6, capsize=4,
                color=col, label=lab, zorder=3)
    for x, b, e, p in zip(xt + (i - .5) * .18, betas, errs, ps):
        if p < .05:
            ax.annotate(stars(p), (x, b - e - .05), ha="center", fontsize=13, color=col,
                        fontweight="bold")
ax.axhline(0, color="#888888", lw=.9, ls="--", zorder=1)
ax.set_xticks(xt, [l for _, l in h2map], fontsize=11.5)
ax.set_ylabel("Coefficient on tone (95% CI)")
ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
ax.legend(fontsize=11, loc="upper right", framealpha=.95, edgecolor="#BBBBBB")
ax.set_title("Incremental Tone Effect on Asset Returns, controlling for Target and Path",
             fontsize=12.5, pad=8)
ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo - .35, yhi + .3)  # stars + legend clearance
fig.tight_layout()
fig.savefig(f"{OUT}/h2_tone.png", dpi=200, bbox_inches="tight"); plt.close(fig)
for k, l in h2map:
    payload["h2"][l] = {"fg": price[f"C_{k}_FG"], "all": price[f"C_{k}_all"]}

# ================= CHART 4: H3 regime (slide 29) =================
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
d = price["D_crsp_vw_interact"]
names = [("tone100", "Tone"), ("fg_2008_2015", "FG regime"), ("tone_x_fg", "Tone × FG")]
for i, (k, lab) in enumerate(names):
    s = d[k]
    ax.errorbar(i, s["beta"], yerr=1.96 * s["se"], fmt="o", ms=8, lw=1.8, capsize=5,
                color=RED if k == "tone_x_fg" else NAVY, zorder=3)
    ax.annotate(f"{s['beta']:+.3f}{stars(s['p'])}", (i, s["beta"] + 1.96 * s["se"] + .18),
                ha="center", fontsize=11.5,
                color=RED if k == "tone_x_fg" else NAVY, fontweight="bold")
ax.axhline(0, color="#888888", lw=.9, ls="--")
ax.set_xticks(range(3), [l for _, l in names], fontsize=11.5)
ax.set_ylabel("Coefficient (95% CI)")
ax.set_title("Panel A: Interaction — CRSP VW", fontsize=12.5, pad=8)
ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
ax = axes[1]
regs = [("FG (2008–15)", NAVY), ("Non-FG", "#8FA3BC"), ("ZLB (2020–22)", GRAY)]
r2 = [price["B_crsp_vw_FG"]["r2"] * 100, price["B_crsp_vw_nonFG"]["r2"] * 100,
      price["B_crsp_vw_all"]["r2"] * 100]
# ZLB-window R²: batch3 zlb2020 spec
b3 = json.load(open(os.path.join(V8, "batch3_results.json")))
r2[2] = b3["zlb2020_tone_crsp"]["r2"] * 100
bars = ax.bar(range(3), r2, width=.55, color=[c for _, c in regs], edgecolor="white")
for i, v in enumerate(r2):
    ax.text(i, v + .4, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#333")
ax.set_xticks(range(3), [l for l, _ in regs], fontsize=11.5)
ax.set_ylabel("R² of tone → returns (%)")
ax.set_ylim(0, max(r2) * 1.25)
ax.set_title("Panel B: Explanatory power by regime", fontsize=12.5, pad=8)
ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
fig.tight_layout()
fig.savefig(f"{OUT}/h3_regime.png", dpi=200, bbox_inches="tight"); plt.close(fig)
payload["h3"] = {"interact": d, "r2": {l: price[f"B_crsp_vw_{s}"]["r2"]
                                       for l, s in [("fg", "FG"), ("nonfg", "nonFG"), ("all", "all")]},
                 "zlb_r2": b3["zlb2020_tone_crsp"]["r2"]}

# ================= CHART 5: H4 flows coef by class (slide 31) =================
fig, ax = plt.subplots(figsize=(11.4, 4.15))
xt = np.arange(len(CLASS_ORDER))
for i, (key, col, lab) in enumerate([("cbi", RED, "CBI (information) shock"),
                                     ("mp", NAVY, "MP (action) shock")]):
    betas = [flow["H4_mp_vs_cbi"]["same"][c][f"{key}_shock"]["beta"] for c in CLASS_ORDER]
    errs = [1.96 * flow["H4_mp_vs_cbi"]["same"][c][f"{key}_shock"]["se"] for c in CLASS_ORDER]
    ax.errorbar(xt + (i - .5) * .18, betas, yerr=errs, fmt="o", ms=7, lw=1.6, capsize=4,
                color=col, label=lab, zorder=3)
    for x, b, e, c in zip(xt + (i - .5) * .18, betas, errs, CLASS_ORDER):
        p = flow["H4_mp_vs_cbi"]["same"][c][f"{key}_shock"]["p"]
        if p < .05:
            ax.annotate(stars(p), (x, b + e + .06), ha="center", fontsize=13, color=col,
                        fontweight="bold")
ax.axhline(0, color="#888888", lw=.9, ls="--", zorder=1)
ax.set_xticks(xt, [CLASS_LAB[c] for c in CLASS_ORDER], fontsize=11)
ax.set_ylabel("Flow response, %TNA (95% CI)")
ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
ax.legend(fontsize=11, loc="upper right")
ax.set_title("Mutual-Fund Flow Response to Shocks by Asset Class (meeting month)",
             fontsize=12.5, pad=8)
ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo - .1, yhi + .15)
yt = np.array([t for t in ax.get_yticks() if ylo - .1 <= t <= yhi + .15])
ax.set_yticks(yt)  # label the whisker zone too
fig.tight_layout()
fig.savefig(f"{OUT}/h4_flows.png", dpi=200, bbox_inches="tight"); plt.close(fig)
payload["h4"] = {c: flow["H4_mp_vs_cbi"]["same"][c] for c in CLASS_ORDER}

# ================= CHART 6: H5 risk ladder (slide 33) =================
fig, ax = plt.subplots(figsize=(11.4, 4.15))
xb = np.arange(1, 8)
betas = [flow["H4_mp_vs_cbi"]["same"][c]["cbi_shock"]["beta"] for c in CLASS_ORDER]
errs = [1.96 * flow["H4_mp_vs_cbi"]["same"][c]["cbi_shock"]["se"] for c in CLASS_ORDER]
ax.fill_between(xb, [b - e for b, e in zip(betas, errs)], [b + e for b, e in zip(betas, errs)],
                color=GRAY, alpha=.4, label="95% CI", zorder=1)
ax.plot(xb, betas, "-o", color=RED, lw=2.2, ms=8, zorder=3, label="CBI coefficient")
ax.axhline(0, color="#888888", lw=.9, ls="--", zorder=2)
for x, b, e, c in zip(xb, betas, errs, CLASS_ORDER):
    p = flow["H4_mp_vs_cbi"]["same"][c]["cbi_shock"]["p"]
    ax.annotate(f"{b:+.2f}{stars(p)}", (x, b + e + .09), ha="center", fontsize=11,
                color=RED if p < .05 else "#555555", fontweight="bold" if p < .05 else "normal")
ax.set_xticks(xb, [CLASS_LAB[c] for c in CLASS_ORDER], fontsize=11)
ax.set_xlabel("Asset-class risk rank (safer $\\rightarrow$ riskier)")
ax.set_ylabel("Flow response to CBI shock, %TNA")
ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(.5, -.27), ncol=2)
ax.set_title("The Risk Ladder: Hawkish Information Pushes Portfolios Toward Safer Assets",
             fontsize=12.5, pad=8)
ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo - .12, yhi + .28)  # band top clipping fix
fig.tight_layout()
fig.savefig(f"{OUT}/h5_ladder.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ================= CHART 7: H6 event-study across windows (slide 34) =================
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.15))
for ax, (cls, title) in zip(axes, [("government_bonds", "Government bonds"),
                                  ("emerging_market_equity", "EM equity")]):
    wpos = [1, 2, 3]
    wlab = ["Meeting\nmonth", "Month\nafter", "2 months\nafter"]
    betas = [win[w][cls]["cbi_shock"]["beta"] for w in ["same", "post", "diff"]]
    lo = [win[w][cls]["cbi_shock"]["beta"] - 1.96 * win[w][cls]["cbi_shock"]["se"] for w in ["same", "post", "diff"]]
    hi = [win[w][cls]["cbi_shock"]["beta"] + 1.96 * win[w][cls]["cbi_shock"]["se"] for w in ["same", "post", "diff"]]
    ax.fill_between(wpos, lo, hi, color=GRAY, alpha=.4, label="95% CI", zorder=1)
    ax.plot(wpos, betas, "-o", color=NAVY, lw=2.2, ms=8, zorder=3, label="CBI coefficient")
    ax.axhline(0, color="#888888", lw=.9, ls="--", zorder=2)
    for x, b in zip(wpos, betas):
        ax.annotate(f"{b:+.2f}", (x, b + .12), ha="center", fontsize=11.5,
                    color=NAVY, fontweight="bold")
    ax.set_xticks(wpos, wlab, fontsize=11)
    ax.set_ylabel("Flow response to CBI shock, %TNA")
    ax.grid(axis="y", color="#E6E6E6", lw=.7, zorder=0)
    ax.set_title(f"{title}", fontsize=12.5, pad=8)
axes[0].legend(fontsize=10.5, loc="lower left")
fig.suptitle("Transmission Horizons: CBI Flow Effects Reverse as Portfolios Rebalance",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/h6_windows.png", dpi=200, bbox_inches="tight"); plt.close(fig)
payload["h6"] = {w: {c: win[w][c] for c in ["government_bonds", "emerging_market_equity"]}
                 for w in ["same", "post", "diff"]}

json.dump(payload, open(os.path.join(BASE, "paper", "ppt_results_v2.json"), "w"), indent=1)
print("charts:", sorted(os.listdir(OUT)))
print("h1 target beta:", round(payload["h1"]["target"]["beta"], 6),
      "p:", round(payload["h1"]["target"]["p"], 4))
