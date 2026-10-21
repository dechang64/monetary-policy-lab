# -*- coding: utf-8 -*-
"""Cross-model scoring of the audit sample (Layer 4 model-level).

4 models from different families + the main amax-router = 5 voting nodes
(MultiNodeVoter, model-level). Sample = the 30 audit statements.
Output: direction2/flow_rerun_output_v8/llm_crossmodel_scores.jsonl
"""
import json
import os
import re
import time

from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_audit_sample.json")
OUT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_crossmodel_scores.jsonl")

MODELS = ["gpt-4.1-mini", "claude-haiku-4-5-20251001", "gemini-2.5-flash", "deepseek-flash"]

PROMPT = """You are an expert reader of FOMC statements. Read the statement below and rate the overall policy stance signal on a scale from -100 (extremely dovish) to +100 (extremely hawkish). Judge ONLY the words used, not outcomes.
Return strict JSON: {{"score": <int>, "confidence": "<high|medium|low>", "one_line": "<max 15 words>"}}

STATEMENT ({date}):
{text}"""


def main():
    import sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.join(BASE, "direction2"))
    from run_flow_chapter_local import load_fomc
    fomc = load_fomc()
    fomc["tone100"] = fomc["sentiment"] * 100
    recs = [json.loads(l) for l in open(os.path.join(
        BASE, "direction2", "flow_rerun_output_v8", "llm_statement_scores.jsonl"), encoding="utf-8")]
    recs = [r for r in recs if "score" in r and "error" not in r]
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").join(fomc[["tone100"]], how="left")
    zs = lambda s: (s - s.mean()) / s.std()
    df["z_llm"], df["z_tone"] = zs(df["score"]), zs(df["tone100"])
    df["disagree"] = (df["z_llm"] - df["z_tone"]).abs() > 2
    rng = np.random.default_rng(42)  # same seed as the audit script
    dispick = df[df["disagree"]].index.tolist()
    others = df[~df["disagree"]].index.tolist()
    samp = dispick + list(rng.choice(others, size=min(30 - len(dispick), len(others)), replace=False))
    dates = sorted(d.strftime("%Y-%m-%d") for d in samp)
    print(f"cross-model sample: {len(dates)} statements ({len(dispick)} disagreements)", flush=True)
    raw = json.load(open(os.path.join(BASE, "data", "fomc_statements_clean.json"), encoding="utf-8"))
    client = OpenAI(api_key=os.environ["AMAX_API_KEY"], base_url="https://ai.amaxsmp.com/v1",
                    timeout=60, max_retries=2)

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["model"], r["date"]))
            except Exception:
                pass

    todo = [(m, d) for m in MODELS for d in dates if (m, d) not in done]
    print(f"cross-model: {len(MODELS)} models x {len(dates)} dates, {len(todo)} calls", flush=True)
    fout = open(OUT, "a", encoding="utf-8")
    n = 0
    for model in MODELS:
        for date in dates:
            if (model, date) in done:
                continue
            text = raw.get(date, "")
            if not text:
                continue
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT.format(date=date, text=text[:3800])}],
                    temperature=0.1, max_tokens=120)
                content = resp.choices[0].message.content or ""
                m = re.search(r"\{.*\}", content, re.S)
                rec = {"model": model, "date": date, "error": None}
                if not m:
                    rec["error"] = content[:100]
                else:
                    j = json.loads(m.group(0))
                    rec.update({"score": int(j["score"]), "confidence": j.get("confidence", "?"),
                                "one_line": j.get("one_line", "")})
            except Exception as e:
                rec = {"model": model, "date": date, "error": type(e).__name__ + " " + str(e)[:80]}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            if n % 10 == 0:
                print(f"  [{n}/{len(todo)}]", flush=True)
            time.sleep(0.4)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
