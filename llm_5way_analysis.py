# -*- coding: utf-8 -*-
"""Five-way cross-model agreement analysis (audit Layer 4, model-level)."""
import json

import pandas as pd

BASE = __file__.rsplit("/", 1)[0]
OUT = BASE + "/direction2/flow_rerun_output_v8/llm_crossmodel_scores.jsonl"
SCORES = BASE + "/direction2/flow_rerun_output_v8/llm_statement_scores.jsonl"
AUDIT = BASE + "/direction2/flow_rerun_output_v8/llm_audit_results.jsonl"

allrecs = []
for line in open(OUT, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if not r.get("error") and "score" in r:  # error: None or "" both pass
        allrecs.append(r)
MODELS = {"amax-router": "router", "gpt-4.1-mini": "gpt",
          "claude-haiku-4-5-20251001": "claude", "gemini-2.5-flash": "gemini",
          "deepseek-flash": "deepseek"}
audit = [json.loads(l)["date"] for l in open(AUDIT, encoding="utf-8")]
rscores = {}
for line in open(SCORES, encoding="utf-8"):
    r = json.loads(line)
    if r.get("score") is not None and r.get("error") is None:
        rscores[r["date"]] = r["score"]

rows = {}
for d in sorted(set(audit)):
    vals = {"router": rscores.get(d)}
    for r in allrecs:
        if r["date"] == d and r["model"] in MODELS:
            vals[MODELS[r["model"]]] = r["score"]
    if all(v is not None for v in vals.values()) and len(vals) == 5:
        rows[d] = vals

df = pd.DataFrame(rows).T
def direction(s):
    return "hawkish" if s >= 25 else ("dovish" if s <= -25 else "neutral")
dirs = df.applymap(direction)
n_agree = dirs.nunique(axis=1) == 1
lm_dates = {"2011-09-21", "2012-09-13", "2015-10-28", "2017-09-20", "2017-11-01",
            "2020-06-10", "2020-09-16", "2020-11-05", "2020-12-16", "2022-06-15",
            "2007-08-07"}
is_lm = pd.Series([d in lm_dates for d in df.index], index=df.index)
print("=== 五方方向一致率（router/gpt/claude/gemini/deepseek, ±25）===")
print(f"全部样本: {n_agree.mean():.1%} (n={len(df)})")
print(f"LM 分歧样本: {n_agree[is_lm].mean():.1%} (n={int(is_lm.sum())})")
print("\n方向不一致的样本：")
for d in df.index[~n_agree]:
    print(f"  {d}: {list(dirs.loc[d])}  scores={[int(x) for x in df.loc[d]]}")
print("\n=== 五方相关矩阵 ===")
print(df.corr().round(2).to_string())
df.to_json(BASE + "/direction2/flow_rerun_output_v8/llm_5way_summary.json", orient="index")
print("\nsaved llm_5way_summary.json")
