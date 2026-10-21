# -*- coding: utf-8 -*-
"""LLM statement scoring — reads the full statement text (audit fix).

Replaces the old meeting-metadata proxy. Output: direction2/flow_rerun_output_v8/
llm_statement_scores.jsonl  (date, score, confidence, one_line)
Resume-safe: existing entries are skipped.
"""
import json
import os
import re
import sys
import time

from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "fomc_statements_clean.json")
OUT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_statement_scores.jsonl")

sys.path.insert(0, os.path.join(BASE, "direction2"))
from run_flow_chapter_local import load_fomc  # noqa: E402

fomc = load_fomc()
dates = [d.strftime("%Y-%m-%d") for d in fomc.index]
statements = json.load(open(RAW, encoding="utf-8"))

done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["date"])
            except Exception:
                pass

client = OpenAI(api_key=os.environ.get("AMAX_API_KEY", ""), base_url="https://ai.amaxsmp.com/v1")

PROMPT = """You are an expert reader of FOMC statements. Read the statement below and rate the overall policy stance signal on a scale from -100 (extremely dovish) to +100 (extremely hawkish). Judge ONLY the words used, not outcomes.
Return strict JSON: {{"score": <int>, "confidence": "<high|medium|low>", "one_line": "<max 15 words>"}}

STATEMENT ({date}):
{text}"""

n_total = len(dates)
n_done = len(done)
print(f"scoring {n_total - n_done}/{n_total} statements", flush=True)

with open(OUT, "a", encoding="utf-8") as fout:
    for i, date in enumerate(dates):
        if date in done:
            continue
        text = statements.get(date, "")
        if not text:
            print(f"  {date}: no text, skip", flush=True)
            continue
        resp = client.chat.completions.create(
            model="amax-router",
            messages=[{"role": "user", "content": PROMPT.format(date=date, text=text[:3800])}],
            temperature=0.1, max_tokens=120,
        )
        content = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            rec = {"date": date, "error": content[:120]}
        else:
            try:
                j = json.loads(m.group(0))
                rec = {"date": date, "score": int(j["score"]),
                       "confidence": j.get("confidence", "?"), "one_line": j.get("one_line", "")}
            except Exception as e:
                rec = {"date": date, "error": str(e)[:120]}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_done += 1
        if (i + 1) % 10 == 0:
            print(f"  [{n_done}/{n_total}]", flush=True)
        time.sleep(0.8)

print("DONE", flush=True)
