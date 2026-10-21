# -*- coding: utf-8 -*-
"""Audit LLM statement scoring with the 5-layer hallucination defense
(ported from federated-ai-platform/src/hallucination/mod.rs).

Layers (Rust -> audit mapping):
  1. RetrievalConsistency : one_line's terms must be retrievable from the
     statement text (term coverage >= retrieval_threshold 0.50).
  2. VectorFactCheck      : strong-declaration keywords in one_line verified
     word-by-word against the statement (verified_ratio reported).
  3. CROWN                : re-ask with the LM dictionary score shown as
     reference; if the LLM's confidence drops > crown_delta (0.10) it was
     anchored by the reference -> keep the independent answer (flagged).
  4. MultiNodeVote        : 3 prompt variants vote on direction; consensus
     >= 2/3 -> Verified.
  5. SelfConsistency      : 3 samples at temperature 0.3; direction agreement
     >= 0.70 and score range <= 30 -> passes.

Verdicts mirror DefenseResult: Verified / LikelyTrue / Uncertain /
LikelyFalse / Hallucination.
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_statement_scores.jsonl")
OUT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_report.json")
RESL = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_results.jsonl")

CFG = dict(retrieval_threshold=0.50, crown_delta=0.10,
           self_consistency_threshold=0.70, consensus_threshold=0.667,
           sample_n=30, repeat_k=3, seed=42)

STOP = set("""a an the of to in on for at by with from as is are was were be been being
that this these those it its and or but not no nor so than then too very can will just
should now has have had do does did would could may might must shall about into over
under again further once here there when where why how all any both each few more most
other some such only own same s t don should ve ll re d m o y ain aren couldn didn
doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn""".split())

client = OpenAI(api_key=os.environ.get("AMAX_API_KEY", ""), base_url="https://ai.amaxsmp.com/v1")

PROMPT = """You are an expert reader of FOMC statements. Read the statement below and rate the overall policy stance signal on a scale from -100 (extremely dovish) to +100 (extremely hawkish). Judge ONLY the words used, not outcomes.
Return strict JSON: {{"score": <int>, "confidence": "<high|medium|low>", "one_line": "<max 15 words>"}}

STATEMENT {variant}:
{text}"""

VARIANTS = {
    "base": "({date}):\n{text}",
    "nodate": "(undated):\n{text}",
    "reorder": "(undated, policy-instruction-first):\n{text}\nRemember: judge ONLY the words used, not outcomes.",
}

def terms(s):
    return [t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(t) >= 2 and t not in STOP]

def ask(text, variant="base", date="", temperature=0.1, show_lm=None):
    body = VARIANTS[variant].format(date=date, text=text[:3800])
    if show_lm is not None:
        body += (f"\n\nREFERENCE: a Loughran-McDonald dictionary scored this statement "
                 f"{show_lm:+.1f} (hawkish-positive). Note dictionaries handle structural "
                 f"operation and forward-guidance language poorly. Give YOUR OWN independent reading.")
    resp = client.chat.completions.create(model="amax-router",
        messages=[{"role": "user", "content": PROMPT.format(variant=body, text="", date="")}],
        temperature=temperature, max_tokens=120)
    m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
        return {"score": float(j["score"]), "confidence": j.get("confidence", "low"),
                "one_line": j.get("one_line", "")}
    except Exception:
        return None

CONF2F = {"high": 0.9, "medium": 0.7, "low": 0.5}

def direction(score):
    return "hawkish" if score >= 20 else ("dovish" if score <= -20 else "neutral")

# ────────── Layers ──────────
def layer1_retrieval(one_line, text):
    ts = terms(one_line)
    if not ts:
        return 0.0, 0
    body = text.lower()
    hit = sum(1 for t in ts if t in body)
    return hit / len(ts), len(ts)

def layer2_factcheck(one_line, text):
    strong = [t for t in terms(one_line) if t in
              {"taper", "tapering", "balance", "sheet", "inflation", "employment", "labor",
               "maximum", "accommodative", "rate", "rates", "hike", "hikes", "cut", "cuts",
               "raise", "raises", "lower", "lowers", "purchase", "purchases", "holdings",
               "patience", "patient", "guidance", "range", "target"}]
    if not strong:
        return None, 0
    body = text.lower()
    hit = sum(1 for t in strong if t in body)
    return hit / len(strong), len(strong)

def main():
    sys.path.insert(0, os.path.join(BASE, "direction2"))
    from run_flow_chapter_local import load_fomc
    fomc = load_fomc()
    fomc["tone100"] = fomc["sentiment"] * 100
    statements = json.load(open(os.path.join(BASE, "data", "fomc_statements_clean.json"), encoding="utf-8"))
    recs = [json.loads(l) for l in open(SCORES, encoding="utf-8")]
    recs = [r for r in recs if "score" in r and "error" not in r]

    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").join(fomc[["tone100"]], how="left")
    zs = lambda s: (s - s.mean()) / s.std()
    df["z_llm"], df["z_tone"] = zs(df["score"]), zs(df["tone100"])
    df["disagree"] = (df["z_llm"] - df["z_tone"]).abs() > 2

    rng = np.random.default_rng(CFG["seed"])
    dispick = df[df["disagree"]].index.tolist()
    others = df[~df["disagree"]].index.tolist()
    samp = dispick + list(rng.choice(others, size=min(CFG["sample_n"] - len(dispick), len(others)),
                                     replace=False))
    samp = sorted(samp)
    print(f"audit sample: {len(samp)} statements ({len(dispick)} disagreements + rest random)")

    report = {"meta": {"n_sample": len(samp), "cfg": CFG,
                       "layers": ["retrieval", "factcheck", "crown", "vote", "selfconsistency"]},
              "statements": {}}

    # resume: load previously persisted statement results
    done = set()
    if os.path.exists(RESL):
        for line in open(RESL, encoding="utf-8"):
            try:
                rec = json.loads(line)
                report["statements"][rec["date"]] = rec
                done.add(rec["date"])
            except Exception:
                continue
        if done:
            print(f"resume: {len(done)} statements already done", flush=True)

    for i, date in enumerate(samp):
        d = date.strftime("%Y-%m-%d")
        if d in done:
            continue
        text = statements.get(d, "")
        if not text:
            continue
        row = df.loc[date]
        base_rec = {"score": row["score"], "confidence": "medium", "one_line": "", "tone": row["tone100"]}
        # fresh base call (for one_line audit + CROWN baseline confidence)
        b = ask(text, date=d)
        if b:
            base_rec = dict(b, tone=row["tone100"])
        r = {"base": base_rec}

        # L1 retrieval + L2 factcheck on the base one_line
        cov, n_terms = layer1_retrieval(base_rec["one_line"], text)
        r["L1_retrieval"] = {"coverage": round(cov, 3), "n_terms": n_terms,
                             "supported": bool(cov >= CFG["retrieval_threshold"])}
        fc, n_strong = layer2_factcheck(base_rec["one_line"], text)
        r["L2_factcheck"] = {"verified_ratio": None if fc is None else round(fc, 3),
                             "n_strong": n_strong}

        # L3 CROWN: does showing the LM score anchor the LLM?
        s = ask(text, date=d, show_lm=row["tone100"])
        if s:
            drop = CONF2F[base_rec["confidence"]] - CONF2F[s["confidence"]]
            changed = direction(s["score"]) != direction(base_rec["score"])
            r["L3_crown"] = {"indep_score": base_rec["score"], "social_score": s["score"],
                             "indep_conf": CONF2F[base_rec["confidence"]],
                             "social_conf": CONF2F[s["confidence"]],
                             "drop": round(drop, 2), "direction_changed": changed,
                             "crown_triggered": drop > CFG["crown_delta"] and changed}
        time.sleep(0.4)

        # L4 vote + L5 self-consistency
        votes, samples = [], []
        for v in VARIANTS:
            sv = ask(text, variant=v, temperature=0.3)
            if sv:
                votes.append(direction(sv["score"]))
                samples.append(sv["score"])
            time.sleep(0.3)
        counts = pd.Series(votes).value_counts()
        strength = counts.iloc[0] / len(votes) if len(votes) else 0
        r["L4_vote"] = {"votes": votes, "consensus": counts.index[0] if len(votes) else None,
                        "strength": round(strength, 2),
                        "verified": bool(strength >= CFG["consensus_threshold"])}
        k = len(samples)
        agree = counts.iloc[0] / k if k else 0
        spread = (max(samples) - min(samples)) if k > 1 else 0
        r["L5_selfconsistency"] = {"samples": samples, "direction_agreement": round(agree, 2),
                                   "score_range": spread,
                                   "passes": bool(agree >= CFG["self_consistency_threshold"] and spread <= 30)}
        report["statements"][d] = r
        # incremental persistence (resume-safe)
        with open(RESL, "a", encoding="utf-8") as rf:
            rf.write(json.dumps({"date": d, **r}, ensure_ascii=False, default=str) + "\n")
        print(f"  [{i+1}/{len(samp)}] {d} L1={r['L1_retrieval']['supported']}", flush=True)

    # aggregate
    S = report["statements"]
    agg = {
        "L1_support_rate": round(float(np.mean([v["L1_retrieval"]["supported"] for v in S.values()])), 3),
        "L1_mean_coverage": round(float(np.mean([v["L1_retrieval"]["coverage"] for v in S.values()])), 3),
        "L2_mean_verified": round(float(np.mean([v["L2_factcheck"]["verified_ratio"] for v in S.values()
                                                 if v["L2_factcheck"]["verified_ratio"] is not None])), 3),
        "L3_crown_triggered_n": int(sum(1 for v in S.values()
                                        if "L3_crown" in v and v["L3_crown"]["crown_triggered"])),
        "L3_direction_changed_n": int(sum(1 for v in S.values()
                                          if "L3_crown" in v and v["L3_crown"]["direction_changed"])),
        "L4_consensus_rate": round(float(np.mean([v["L4_vote"]["verified"] for v in S.values()])), 3),
        "L5_pass_rate": round(float(np.mean([v["L5_selfconsistency"]["passes"] for v in S.values()])), 3),
        "L5_mean_range": round(float(np.mean([v["L5_selfconsistency"]["score_range"] for v in S.values()])), 1),
    }
    report["aggregate"] = agg

    # verdict per statement (mirror DefenseResult logic)
    for d, v in S.items():
        l1 = v["L1_retrieval"]["supported"]
        l4 = v["L4_vote"]["verified"]
        l5 = v["L5_selfconsistency"]["passes"]
        crown = v.get("L3_crown", {})
        if l1 and l4 and l5 and not crown.get("crown_triggered", False):
            v["verdict"] = "Verified"
        elif l1 and (l4 or l5):
            v["verdict"] = "LikelyTrue"
        elif not l1 and (l4 or l5):
            v["verdict"] = "Uncertain"
        elif not (l4 or l5):
            v["verdict"] = "LikelyFalse"
        else:
            v["verdict"] = "Hallucination"
    vd = pd.Series([v["verdict"] for v in S.values()]).value_counts()
    report["verdict_distribution"] = vd.to_dict()

    json.dump(report, open(OUT, "w"), indent=1)
    print("\n=== AGGREGATE ===")
    for k2, v2 in agg.items():
        print(f"  {k2:26s} {v2}")
    print("  verdicts:", dict(vd))
    print("saved", OUT)


if __name__ == "__main__":
    main()
