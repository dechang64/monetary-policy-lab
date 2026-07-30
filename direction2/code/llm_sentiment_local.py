# -*- coding: utf-8 -*-
"""
LLM-based FOMC Sentiment Analysis (Local Version)
Uses amax router with OpenAI-compatible API.

Usage:
    pip install openai
    python llm_sentiment_local.py

Output:
    ../results/llm_sentiment_results.csv
"""

import pandas as pd
import numpy as np
import json
import time
import os
import sys
import re
from scipy import stats
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
API_KEY = "sk-IWzK4nWmbGSVPOoRCOeTFFgQQSqwRSijgnudaKkNz7yqkpKG"
BASE_URL = "https://ai.amaxsmp.com/v1"
MODEL = "amax-router"  # Smart routing, no need to specify model

CSV_PATH = "../results/minutes_sentiment_corrected.csv"
OUTPUT_PATH = "../results/llm_sentiment_results.csv"

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============================================================
# Load FOMC data
# ============================================================
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} FOMC meetings from {CSV_PATH}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Model: {MODEL} (amax smart routing)")
print()

# ============================================================
# Process each FOMC meeting
# ============================================================
results = []

if os.path.exists(OUTPUT_PATH):
    existing = pd.read_csv(OUTPUT_PATH)
    results = existing.to_dict("records")
    print(f"Resuming from {len(results)} existing results")

start_idx = len(results)

for idx in range(start_idx, len(df)):
    row = df.iloc[idx]

    prompt = f"""You are a monetary policy expert analyzing FOMC meeting sentiment.

FOMC Meeting Context:
- Date: {row['date']}
- Decision: {row['decision']} (rate change: {row['rate_change']} bps)
- Chair: {row['chair']}
- S&P 500 return: {row['sp500_ret']:.4f}
- VIX level: {row['vix']:.2f}
- 10Y Treasury change: {row.get('ty10_chg', 0):.4f}
- Target shock (GSS): {row['target_shock']:.4f}
- Path shock (GSS): {row['path_shock']:.4f}
- Equity return (CRSP): {row.get('vwretd_day', 0):.4f}

Based on this context, rate the FOMC meeting on three dimensions:
1. hawkish_dovish: +1.0 = strongly hawkish (tightening), -1.0 = strongly dovish (easing), 0.0 = neutral
2. forward_guidance: +1.0 = tightening forward guidance, -1.0 = easing forward guidance, 0.0 = neutral/none
3. information_revelation: +1.0 = high information revelation, -1.0 = low information, 0.0 = neutral

Respond in JSON format ONLY:
{{"hawkish_dovish": 0.00, "forward_guidance": 0.00, "information_revelation": 0.00}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a monetary policy expert. Respond only in JSON format."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            scores = json.loads(json_match.group())
            h = float(scores.get("hawkish_dovish", np.nan))
            fg = float(scores.get("forward_guidance", np.nan))
            ir = float(scores.get("information_revelation", np.nan))
        else:
            h = fg = ir = np.nan
            print(f"  [{idx+1}] No JSON: {content[:80]}")
    except Exception as e:
        h = fg = ir = np.nan
        print(f"  [{idx+1}] Error: {e}")

    results.append({
        "date": row["date"],
        "decision": row["decision"],
        "rate_change": row["rate_change"],
        "chair": row["chair"],
        "lm_sentiment": row["sentiment"],
        "lm_score": row["lm_score"],
        "lm_cb_score": row["cb_score"],
        "llm_hawkish": h,
        "llm_forward_guidance": fg,
        "llm_information": ir,
    })

    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(df)} meetings")
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    time.sleep(0.3)

# ============================================================
# Save and analyze
# ============================================================
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved {len(result_df)} results to {OUTPUT_PATH}")

valid = result_df.dropna(subset=["llm_hawkish"])
print(f"Valid LLM scores: {len(valid)}/{len(result_df)}")

if len(valid) > 2:
    print(f"\n{'='*60}")
    print("Correlation: LM Dictionary vs LLM")
    print(f"{'='*60}")

    for lm_col, llm_col, label in [
        ("lm_sentiment", "llm_hawkish", "LM Sentiment vs LLM Hawkish"),
        ("lm_score", "llm_hawkish", "LM Score vs LLM Hawkish"),
        ("lm_cb_score", "llm_information", "LM CB Score vs LLM Information"),
    ]:
        if lm_col in valid.columns and llm_col in valid.columns:
            r, p = stats.pearsonr(valid[lm_col], valid[llm_col])
            rho, p_rho = stats.spearmanr(valid[lm_col], valid[llm_col])
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"  {label}:")
            print(f"    Pearson r = {r:.4f} {sig} (p={p:.4f})")
            print(f"    Spearman rho = {rho:.4f} {sig} (p={p_rho:.4f})")

    print(f"\n{'='*60}")
    print("Decision-level Means")
    print(f"{'='*60}")
    print(f"{'Decision':15s} {'N':>4s} {'LM Sentiment':>13s} {'LLM Hawkish':>13s}")
    print("-" * 48)
    for dec in ["rate_hike", "unchanged", "rate_cut"]:
        subset = valid[valid["decision"] == dec]
        if len(subset) > 0:
            print(f"{dec:15s} {len(subset):4d} {subset['lm_sentiment'].mean():13.4f} "
                  f"{subset['llm_hawkish'].mean():13.4f}")

    print(f"\n{'Chair':15s} {'N':>4s} {'LM Sentiment':>13s} {'LLM Hawkish':>13s}")
    print("-" * 48)
    for chair in ["Greenspan", "Bernanke", "Yellen", "Powell"]:
        subset = valid[valid["chair"] == chair]
        if len(subset) > 0:
            print(f"{chair:15s} {len(subset):4d} {subset['lm_sentiment'].mean():13.4f} "
                  f"{subset['llm_hawkish'].mean():13.4f}")

print(f"\n✅ Done. Send llm_sentiment_results.csv back for B-S baseline comparison.")
