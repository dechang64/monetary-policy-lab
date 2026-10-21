# -*- coding: utf-8 -*-
"""Gemini fill-in: thinking model needs max_tokens=2000."""
import json
import os
import re

from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "direction2", "flow_rerun_output_v8", "llm_crossmodel_scores.jsonl")
raw = json.load(open(os.path.join(BASE, "data", "fomc_statements_clean.json"), encoding="utf-8"))

client = OpenAI(api_key=os.environ["AMAX_API_KEY"], base_url="https://ai.amaxsmp.com/v1",
                timeout=120, max_retries=2)
xs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
ok_dates = {(r["model"], r["date"]) for r in xs if not r.get("error") and r["model"] == "gemini-2.5-flash"}
audit = [json.loads(l)["date"] for l in open(os.path.join(
    BASE, "direction2", "flow_rerun_output_v8", "llm_audit_results.jsonl"), encoding="utf-8")]
todo = [d for d in sorted(set(audit)) if ("gemini-2.5-flash", d) not in ok_dates]
print("todo:", len(todo), flush=True)

PROMPT = """You are an expert reader of FOMC statements. Rate the policy stance signal of the statement below on a scale from -100 (extremely dovish) to +100 (extremely hawkish). Judge ONLY the words used, not outcomes.
Reply with ONLY a JSON object: {{"score": <int>, "confidence": "<high|medium|low>", "one_line": "<max 15 words>"}}

STATEMENT ({date}):
{text}"""

fout = open(OUT, "a", encoding="utf-8")
okc = 0
for d in todo:
    try:
        resp = client.chat.completions.create(model="gemini-2.5-flash",
            messages=[{"role": "user", "content": PROMPT.format(date=d, text=raw[d][:3800])}],
            temperature=0.1, max_tokens=2000)
        c = re.sub(r"```(?:json)?", "", resp.choices[0].message.content or "").strip()
        m = re.search(r"\{[^{}]*\}", c, re.S)
        if not m:
            fout.write(json.dumps({"model": "gemini-2.5-flash", "date": d, "error": c[:80]}) + "\n")
            print(d, "FAIL", c[:50], flush=True)
            continue
        j = json.loads(m.group(0))
        fout.write(json.dumps({"model": "gemini-2.5-flash", "date": d, "error": None,
                               "score": int(j["score"]), "confidence": j.get("confidence", "?"),
                               "one_line": j.get("one_line", "")}) + "\n")
        okc += 1
        print(d, j["score"], flush=True)
    except Exception as e:
        print(d, "EXC", type(e).__name__, str(e)[:50], flush=True)
fout.close()
print("fixed:", okc, "/", len(todo), flush=True)
