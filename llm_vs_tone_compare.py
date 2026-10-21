# -*- coding: utf-8 -*-
"""Compare LLM statement scores (text-based) vs LM tone, after Oct-20 feedback.

Outputs:
  - correlations: LLM vs tone, LLM vs shocks (161 meetings)
  - disagreement samples (|LLM - tone| large)
  - H4 flows model with LLM score instead of tone (does LLM add flow power?)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "direction2"))
from run_flow_chapter_local import load_fomc  # noqa: E402

OUT_V8 = os.path.join(BASE, "direction2", "flow_rerun_output_v8")
SCORES = os.path.join(OUT_V8, "llm_statement_scores.jsonl")

fomc = load_fomc()
fomc["tone100"] = fomc["sentiment"] * 100
fomc["fg_2008_2015"] = ((fomc.index >= "2008-12-16") & (fomc.index <= "2015-10-28")).astype(int)

recs = []
with open(SCORES, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
            if "score" in r and "error" not in r:
                recs.append(r)
        except json.JSONDecodeError:
            continue
llm = pd.DataFrame(recs)
llm["date"] = pd.to_datetime(llm["date"])
llm = llm.set_index("date")["score"].rename("llm_score")
print(f"LLM scores loaded: {len(llm)}/{len(fomc)}")

df = fomc.join(llm, how="inner").dropna(subset=["llm_score", "tone100"])
print(f"matched: {len(df)}")

print("\n=== Correlations (Pearson) ===")
for name, col in (("LM tone", "tone100"), ("Target", "target_shock"),
                  ("Path", "path_shock"), ("CBI", "cbi_shock"), ("MP", "mp_shock")):
    c = df["llm_score"].corr(df[col])
    print(f"  LLM vs {name:10s}: r = {c:+.3f}")

print("\n=== Disagreements: |z(LLM) - z(tone)| > 2 ===")
zs = lambda s: (s - s.mean()) / s.std()
df["z_llm"], df["z_tone"] = zs(df["llm_score"]), zs(df["tone100"])
dis = df[(df["z_llm"] - df["z_tone"]).abs() > 2]
for d, r in dis.iterrows():
    print(f"  {d.date()}  LLM {r['llm_score']:+.0f} vs tone {r['tone100']:+.2f}")

print("\n=== H4 flows: LLM score vs LM tone as the statement variable ===")
agg = pd.read_csv(os.path.join(OUT_V8, "flows_by_class_month_v8.csv"), parse_dates=["month"])
agg["flow_pct"] = agg["dollar_flow"] / agg["tna_lag"]
agg["log_tna"] = np.log(agg["tna_lag"].replace(0, np.nan))
agg = agg.sort_values(["asset_class", "month"])
agg["flow_vol_12m"] = agg.groupby("asset_class")["flow_pct"].transform(
    lambda s: s.rolling(12, min_periods=6).std())
fm = df.reset_index().rename(columns={"date": "key"})
fm["key"] = fm["key"].dt.to_period("M")
a = agg.copy()
a["key"] = a["month"].dt.to_period("M")
pan = a.merge(fm, on="key", how="inner")
pan["risk"] = pan["asset_class"].map({
    "government_bonds": 1, "corporate_bonds": 2, "real_assets": 3,
    "large_cap_equity": 4, "developed_market_equity": 5,
    "emerging_market_equity": 6, "small_cap_equity": 7})
pan = pan.dropna(subset=["flow_pct", "mp_shock", "cbi_shock", "llm_score", "tone100", "log_tna", "flow_vol_12m"])

for var, label in (("tone100", "LM tone"), ("llm_score", "LLM score (scaled/10)")):
    pan["_stmt"] = pan[var] if var == "tone100" else pan[var] / 10.0
    X = sm.add_constant(pan[["mp_shock", "cbi_shock", "_stmt", "log_tna", "flow_vol_12m"]],
                        has_constant="add")
    res = sm.OLS(pan["flow_pct"] * 100, X).fit()
    print(f"  [{label:22s}] beta_stmt = {res.params['_stmt']:+.3f} (p={res.pvalues['_stmt']:.3f})  N={int(res.nobs)}")

print("\nNote: LLM score scale is -100..100, tone is per-100-words; beta magnitudes are not"
      " directly comparable across the two — compare significance and incremental R2, not levels.")
