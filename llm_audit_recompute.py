# -*- coding: utf-8 -*-
"""Recompute audit verdicts: fix the 2/3 threshold bug + semantic L1 via
text-embedding-v4 (the real 'vector fact-check' layer from the Rust design).

L4 bug: consensus_threshold=0.667 rejects 2/3=0.6667 (float) — use vote
count >= 2 of 3 instead (the Rust rule's intent).
L1 fix: word-overline coverage penalizes paraphrase; add cosine similarity
between one_line and statement text (text-embedding-v4), supported if >= 0.5.
"""
import json
import os
import numpy as np
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
RESL = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_results.jsonl")
OUT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_report_v2.json")
STMTS = json.load(open(os.path.join(BASE, "data", "fomc_statements_clean.json"), encoding="utf-8"))

client = OpenAI(api_key=os.environ["AMAX_API_KEY"], base_url="https://ai.amaxsmp.com/v1",
                timeout=60, max_retries=2)

def emb(text):
    r = client.embeddings.create(model="text-embedding-v4", input=text[:6000])
    return np.array(r.data[0].embedding)

recs = [json.loads(l) for l in open(RESL, encoding="utf-8")]
out = {"meta": {"fix": "L4 vote>=2; L1 semantic via text-embedding-v4 cos>=0.5",
                "n": len(recs)}, "statements": {}}
for r in recs:
    d = r["date"]
    text = STMTS.get(d, "")
    v = dict(r)
    # semantic L1
    if text and r["base"].get("one_line"):
        e1, e2 = emb(r["base"]["one_line"]), emb(text)
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    else:
        cos = None
    v["L1_semantic"] = {"cosine": round(cos, 3) if cos is not None else None,
                        "supported": bool(cos is not None and cos >= 0.5)}
    # L4 fixed: majority of valid votes (>=2 of 3)
    votes = r["L4_vote"]["votes"]
    top = max(set(votes), key=votes.count) if votes else None
    v["L4_fixed"] = {"votes": votes, "majority": top,
                     "verified": bool(len(votes) >= 2 and votes.count(top) >= 2)}
    # verdict
    l1 = v["L1_semantic"]["supported"] or r["L1_retrieval"]["supported"]
    l4 = v["L4_fixed"]["verified"]
    l5 = r["L5_selfconsistency"]["passes"]
    crown = bool(r.get("L3_crown", {}).get("crown_triggered", False))
    if l1 and l4 and l5 and not crown:
        v["verdict"] = "Verified"
    elif l1 and (l4 or l5):
        v["verdict"] = "LikelyTrue"
    elif not l1 and (l4 or l5):
        v["verdict"] = "Uncertain"
    else:
        v["verdict"] = "LikelyFalse"
    out["statements"][d] = v

from collections import Counter
vd = Counter(v["verdict"] for v in out["statements"].values())
out["meta"]["verdict_distribution"] = dict(vd)
out["meta"]["L1_semantic_support_rate"] = round(float(np.mean(
    [v["L1_semantic"]["supported"] for v in out["statements"].values()])), 3)
out["meta"]["L1_semantic_mean_cos"] = round(float(np.mean(
    [v["L1_semantic"]["cosine"] for v in out["statements"].values() if v["L1_semantic"]["cosine"] is not None])), 3)
out["meta"]["L4_fixed_consensus_rate"] = round(float(np.mean(
    [v["L4_fixed"]["verified"] for v in out["statements"].values()])), 3)
json.dump(out, open(OUT, "w"), indent=1, default=str)
print("verdicts:", dict(vd))
print("L1 semantic support:", out["meta"]["L1_semantic_support_rate"],
      "| mean cos:", out["meta"]["L1_semantic_mean_cos"])
print("L4 fixed consensus:", out["meta"]["L4_fixed_consensus_rate"])
print("saved", OUT)
