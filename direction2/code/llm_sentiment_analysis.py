#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based FOMC Sentiment Analysis
Uses GLM-4.6 to rate 117 FOMC meetings on hawkish/dovish, forward guidance, information
Compares with LM dictionary scores from Phase 1
"""
import pandas as pd, numpy as np, json, subprocess, re, time, os, sys

csv_path = '../results/minutes_sentiment_corrected.csv'
output_path = '../results/llm_sentiment_results.csv'
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} FOMC meetings")

# Check for existing results (resume capability)
if os.path.exists(output_path):
    existing = pd.read_csv(output_path)
    start_idx = len(existing)
    print(f"Resuming from meeting {start_idx}")
else:
    existing = pd.DataFrame()
    start_idx = 0

results = []
for idx in range(start_idx, len(df)):
    row = df.iloc[idx]
    
    prompt = f"""You are a monetary policy expert analyzing FOMC meeting sentiment.

FOMC Meeting Context:
- Date: {row['date']}
- Decision: {row['decision']} (rate change: {row['rate_change']} bps)
- Chair: {row['chair']}
- Regime: {row.get('regime', 'unknown')}
- S&P 500 return: {row['sp500_ret']:.4f}
- VIX level: {row['vix']:.2f}
- 10Y Treasury change: {row.get('ty10_chg', 0):.4f}
- Target shock (GSS): {row['target_shock']:.4f}
- Path shock (GSS): {row['path_shock']:.4f}
- Equity return (CRSP): {row.get('vwretd_day', 0):.4f}

Based on this context, rate the FOMC meeting on three dimensions:
1. hawkish_dovish: +1.0 = strongly hawkish (tightening), -1.0 = strongly dovish (easing), 0.0 = neutral
2. forward_guidance: +1.0 = strongly tightening FG signal, -1.0 = strongly easing FG signal, 0.0 = no clear FG
3. information_revelation: +1.0 = high information revelation about economy, -1.0 = low information, 0.0 = neutral

Respond in JSON ONLY (no markdown, no explanation):
{{"hawkish_dovish": X.XX, "forward_guidance": X.XX, "information_revelation": X.XX}}"""

    try:
        result = subprocess.run(
            ['z-ai', 'chat', '-m', 'glm-4.6', '-p', prompt, '-t', '0.3', '--thinking', 'disabled'],
            capture_output=True, text=True, timeout=30, cwd='/home/z/my-project'
        )
        response = result.stdout.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            scores = json.loads(json_match.group())
            result_row = {
                'date': row['date'],
                'decision': row['decision'],
                'rate_change': row['rate_change'],
                'chair': row['chair'],
                'lm_sentiment': row['sentiment'],
                'lm_score': row['lm_score'],
                'lm_cb_score': row['cb_score'],
                'llm_hawkish': float(scores.get('hawkish_dovish', np.nan)),
                'llm_forward_guidance': float(scores.get('forward_guidance', np.nan)),
                'llm_information': float(scores.get('information_revelation', np.nan)),
            }
            results.append(result_row)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(df)} meetings")
                # Save incrementally
                pd.concat([existing, pd.DataFrame(results)]).to_csv(output_path, index=False)
        else:
            print(f"  Meeting {idx+1} ({row['date']}): FAILED to parse response")
            results.append({
                'date': row['date'], 'decision': row['decision'],
                'rate_change': row['rate_change'], 'chair': row['chair'],
                'lm_sentiment': row['sentiment'], 'lm_score': row['lm_score'],
                'lm_cb_score': row['cb_score'],
                'llm_hawkish': np.nan, 'llm_forward_guidance': np.nan, 'llm_information': np.nan,
            })
    except Exception as e:
        print(f"  Meeting {idx+1} ({row['date']}): ERROR - {e}")
        results.append({
            'date': row['date'], 'decision': row['decision'],
            'rate_change': row['rate_change'], 'chair': row['chair'],
            'lm_sentiment': row['sentiment'], 'lm_score': row['lm_score'],
            'lm_cb_score': row['cb_score'],
            'llm_hawkish': np.nan, 'llm_forward_guidance': np.nan, 'llm_information': np.nan,
        })
    
    time.sleep(0.5)  # Rate limiting

# Save final results
final_df = pd.concat([existing, pd.DataFrame(results)])
final_df.to_csv(output_path, index=False)
print(f"\n✅ Saved {len(final_df)} results to {output_path}")

# Correlation analysis
from scipy import stats
valid = final_df.dropna(subset=['llm_hawkish'])
print(f"\nValid observations: {len(valid)}")
print(f"\n{'='*60}")
print("Correlation: LM Dictionary vs LLM (GLM-4.6)")
print(f"{'='*60}")

comparisons = [
    ('lm_sentiment', 'llm_hawkish', 'LM Sentiment vs LLM Hawkish'),
    ('lm_score', 'llm_hawkish', 'LM Score vs LLM Hawkish'),
    ('lm_cb_score', 'llm_information', 'LM CB Score vs LLM Information'),
]

for lm_col, llm_col, label in comparisons:
    if lm_col in valid.columns and llm_col in valid.columns:
        r, p = stats.pearsonr(valid[lm_col], valid[llm_col])
        rho, p_rho = stats.spearmanr(valid[lm_col], valid[llm_col])
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f"  {label}:")
        print(f"    Pearson r = {r:.4f} {sig} (p={p:.4f})")
        print(f"    Spearman ρ = {rho:.4f} {sig} (p={p_rho:.4f})")

# Decision-level analysis
print(f"\n{'='*60}")
print("Decision-level Means")
print(f"{'='*60}")
print(f"{'Decision':15s} {'N':>4s} {'LM Sentiment':>13s} {'LLM Hawkish':>13s}")
print("-" * 48)
for dec in ['rate_hike', 'unchanged', 'rate_cut']:
    subset = valid[valid['decision'] == dec]
    if len(subset) > 0:
        print(f"{dec:15s} {len(subset):4d} {subset['lm_sentiment'].mean():13.4f} "
              f"{subset['llm_hawkish'].mean():13.4f}")

# Chair-level analysis
print(f"\n{'Chair':15s} {'N':>4s} {'LM Sentiment':>13s} {'LLM Hawkish':>13s}")
print("-" * 48)
for chair in ['Greenspan', 'Bernanke', 'Yellen', 'Powell']:
    subset = valid[valid['chair'] == chair]
    if len(subset) > 0:
        print(f"{chair:15s} {len(subset):4d} {subset['lm_sentiment'].mean():13.4f} "
              f"{subset['llm_hawkish'].mean():13.4f}")

print(f"\n✅ Analysis complete")
