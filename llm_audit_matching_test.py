# -*- coding: utf-8 -*-
"""Matching test for L1 semantic layer: does each one_line embed closer to
ITS OWN statement than to random statements? (contrastive baseline)
"""
import json
import os
import numpy as np
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
RESL = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_results.jsonl")
V2 = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_report_v2.json")
STMTS = json.load(open(os.path.join(BASE, "data", "fomc_statements_clean.json"), encoding="utf-8"))

client = OpenAI(api_key=os.environ["AMAX_API_KEY"], base_url="https://ai.amaxsmp.com/v1",
                timeout=60, max_retries=2)

def emb(text):
    r = client.embeddings.create(model="text-embedding-v4", input=text[:6000])
    return np.array(r.data[0].embedding)

recs = [json.loads(l) for l in open(RESL, encoding="utf-8")]
rng = np.random.default_rng(7)
all_dates = sorted(STMTS.keys())
E = {}  # cache statement embeddings

def stmt_emb(d):
    if d not in E:
        E[d] = emb(STMTS[d])
    return E[d]

hits, margins, rows = 0, [], []
for r in recs:
    d, ol = r["date"], r["base"].get("one_line", "")
    if not ol:
        continue
    e1 = emb(ol)
    own = float(np.dot(e1, stmt_emb(d)) / (np.linalg.norm(e1) * np.linalg.norm(stmt_emb(d))))
    decoys = [x for x in rng.choice(all_dates, size=4, replace=False) if x != d][:3]
    decoy_cos = []
    for dd in decoys:
        e2 = stmt_emb(dd)
        decoy_cos.append(float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))))
    m = own - max(decoy_cos)
    margins.append(m)
    rows.append({"date": d, "own": round(own, 3), "best_decoy": round(max(decoy_cos), 3),
                 "margin": round(m, 3), "match": bool(m > 0)})
    hits += int(m > 0)

v2 = json.load(open(V2))
v2["meta"]["matching_test"] = {"hit_rate": round(hits / len(rows), 3), "n": len(rows),
                               "mean_margin": round(float(np.mean(margins)), 3),
                               "min_margin": round(float(np.min(margins)), 3),
                               "failures": [r for r in rows if not r["match"]]}
json.dump(v2, open(V2, "w"), indent=1, default=str)
print(f"matching hit rate: {hits}/{len(rows)} = {hits/len(rows):.1%}")
print(f"mean margin: {np.mean(margins):+.3f} | min: {np.min(margins):+.3f}")
for r in rows:
    if not r["match"]:
        print("  FAIL", r)
