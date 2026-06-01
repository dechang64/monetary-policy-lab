#!/usr/bin/env python3
"""
Delta — Multi-Agent 分歧研究平台 v3
增强版：FinDPO情感分析 + 信息论分歧测量 + 多模型回归选择

运行方式：
  pip install streamlit pandas plotly scikit-learn xgboost tensorflow
  streamlit run app.py

作者：思怡 / 2026-05-31
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statistics, re, time, math
from datetime import datetime, timedelta

# ============================================================
# 信息论工具（v3新增）
# ============================================================

def js_divergence(p: list, q: list) -> float:
    """Jensen-Shannon散度，衡量两个概率分布的分歧"""
    def _kl(a, b):
        a, b = [max(x, 1e-10) for x in a], [max(x, 1e-10) for x in b]
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0)
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * (_kl(p, m) + _kl(q, m))

def shannon_entropy(probs: list) -> float:
    """香农熵 H = -sum(P·logP)"""
    return -sum(p * math.log(max(p, 1e-10)) for p in probs if p > 0)

def sentiment_to_probs(score: float) -> list:
    """将1-10的情绪评分映射为三维概率向量（FinDPO风格输出）"""
    # score 1-10 → [P_negative, P_neutral, P_positive]
    if score <= 3:
        neg = 0.6 + (3 - score) * 0.1
        pos = 0.1 + (score - 1) * 0.05
    elif score <= 6:
        neg = 0.2 + (6 - score) * 0.05
        pos = 0.2 + (score - 4) * 0.05
    else:
        neg = 0.1 + (7 - score) * 0.05
        pos = 0.6 + (score - 7) * 0.1
    neu = max(0.0, 1.0 - neg - pos)
    return [neg, neu, pos]

def compute_info_metrics(r1_scores: list, r2_scores: list) -> dict:
    """
    计算v3新增的信息论分歧指标
    r1_scores / r2_scores: [sentiment, technical, fundamental] 各Agent的评分（1-10）
    """
    # Round 1 三维概率向量
    p1 = [sentiment_to_probs(s) for s in r1_scores]
    p2 = [sentiment_to_probs(s) for s in r2_scores]

    # JS散度（Sent-Agent视角：S vs T vs F）
    js_pre  = js_divergence(p1[0], p1[1]) + js_divergence(p1[1], p1[2]) + js_divergence(p1[0], p1[2])
    js_post = js_divergence(p2[0], p2[1]) + js_divergence(p2[1], p2[2]) + js_divergence(p2[0], p2[2])
    js_pre  /= 3; js_post /= 3

    # 情感熵（Round 2情绪Agent）
    probs_r2_sent = sentiment_to_probs(r2_scores[0])
    H_sentiment = shannon_entropy(probs_r2_sent)
    confidence   = max(probs_r2_sent)
    confidence_low = confidence < 0.4

    # IA（信息不对称度）：评分变化越大 → 信息不对称程度越高
    changes = [abs(r2_scores[i] - r1_scores[i]) for i in range(3)]
    IA = sum(changes) / (len(r1_scores) * 9) * 2  # 归一化到[0,1]
    IA = min(1.0, IA)

    # D_irreducible（不可约分歧）
    d_post_std = statistics.stdev(r2_scores) if len(r2_scores) > 1 else 0.0
    D_irreducible = js_post / max(IA, 1e-10) if IA < 1.0 else d_post_std

    return {
        "JS_pre":  round(js_pre,  4),
        "JS_post": round(js_post, 4),
        "H_sentiment":    round(H_sentiment,    4),
        "confidence":     round(confidence,     3),
        "confidence_low": confidence_low,
        "IA":            round(IA,             4),
        "D_irreducible": round(min(D_irreducible, 1.0), 4),
    }

# ============================================================
# LLM 调用层（OpenClaw 平台内嵌，无需 API Key）
# ============================================================

LLM_TASK_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"]
}
DEBATE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}, "changed": {"type": "boolean"}},
    "required": ["score", "reason", "changed"]
}

SENTIMENT_PROMPT = """你是一位资深市场情绪分析师。请对以下股票给出1-10的情绪评分。

股票：{ticker}
时间窗口：{start_date} 至 {end_date}

评分标准：
  1-3分：整体负面情绪占主导（恐慌、抛售、悲观）
  4-6分：中性或混合情绪
  7-10分：整体正面情绪占主导（乐观、买入、信心）

严格按以下JSON格式输出，不要加任何前缀或说明：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

TECHNICAL_PROMPT = """你是一位技术分析专家。请对以下股票给出1-10的技术评分。

股票：{ticker}

评分标准：
  1-3分：明显下跌趋势（空头排列，RSI超卖但无反弹）
  4-6分：震荡或横盘（无明显趋势）
  7-10分：明显上涨趋势（多头排列，价格创或接近新高）

严格按以下JSON格式输出，不要加任何前缀或说明：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

FUNDAMENTAL_PROMPT = """你是一位基本面分析师。请对以下股票的估值水平给出1-10的评分。

股票：{ticker}

评分标准：
  1-3分：估值过高（PE>30，PB>5，ROE偏低）
  4-6分：估值合理区间
  7-10分：估值偏低或合理偏低（PE<20，PB<2，有一定ROE支撑）

严格按以下JSON格式输出，不要加任何前缀或说明：
{{"score": <数字>, "reason": "<理由，50字以内>"}}"""

DEBATE_PROMPT = """你是{my_role}分析师。其他两位分析师的评分如下：
- 情绪分析师：{sentiment_score}分，理由：{sentiment_reason}
- 技术分析师：{technical_score}分，理由：{technical_reason}
- 基本面分析师：{fundamental_score}分，理由：{fundamental_reason}

你的任务是：参考其他分析师的意见，决定你的最终评分。
- 如果其他分析师的评分提供了重要信息，让你改变看法，输出调整后的评分
- 如果其他分析师的评分没有改变你的判断，输出保持不变的评分

严格按以下JSON格式输出（不要加任何前缀）：
{{"score": <数字>, "reason": "<理由，50字以内>", "changed": true/false}}"""


def call_llm(prompt: str, schema: dict = None) -> dict:
    """通过 OpenClaw 内置 llm-task 工具调用 LLM"""
    from openclaw.tools import llm_task
    try:
        result = llm_task(prompt=prompt, schema=schema or LLM_TASK_SCHEMA, timeoutMs=90000)
        return result if isinstance(result, dict) else {"score": 5.0, "reason": str(result)}
    except Exception as e:
        return {"score": 5.0, "reason": f"[LLM调用失败: {e}]"}


def extract_score(val) -> float:
    if isinstance(val, dict):
        v = float(val.get("score", 5.0))
    else:
        patterns = [r'"score"\s*:\s*(\d+(?:\.\d+)?)', r'评分[：:\s]*(\d+(?:\.\d+)?)\s*分']
        v = 5.0
        for p in patterns:
            m = re.search(p, str(val))
            if m:
                v = float(m.group(1))
                break
    return max(1.0, min(10.0, v))


def extract_changed(val) -> bool:
    if isinstance(val, dict):
        return bool(val.get("changed", False))
    m = re.search(r'"changed"\s*:\s*(true|false)', str(val))
    return m.group(1) == "true" if m else False


def sentiment_agent(ticker: str, start: str, end: str) -> dict:
    return call_llm(SENTIMENT_PROMPT.format(ticker=ticker, start_date=start, end_date=end))

def technical_agent(ticker: str) -> dict:
    return call_llm(TECHNICAL_PROMPT.format(ticker=ticker))

def fundamental_agent(ticker: str) -> dict:
    return call_llm(FUNDAMENTAL_PROMPT.format(ticker=ticker))

def debate_agent(my_role: str, sent: dict, tech: dict, fund: dict) -> dict:
    return call_llm(DEBATE_PROMPT.format(
        my_role=my_role,
        sentiment_score=extract_score(sent), sentiment_reason=sent.get("reason", ""),
        technical_score=extract_score(tech), technical_reason=tech.get("reason", ""),
        fundamental_score=extract_score(fund), fundamental_reason=fund.get("reason", ""),
    ), DEBATE_SCHEMA)


# ============================================================
# 核心分析引擎
# ============================================================

def run_full_analysis(ticker: str) -> dict:
    now = datetime.now()
    start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    ticker = ticker.upper()

    with st.spinner("🔍 Round 1：三Agent独立分析..."):
        sent = sentiment_agent(ticker, start, end)
        tech = technical_agent(ticker)
        fund = fundamental_agent(ticker)

    r1 = {
        "sentiment": extract_score(sent), "technical": extract_score(tech), "fundamental": extract_score(fund),
        "sentiment_reason": sent.get("reason", ""), "technical_reason": tech.get("reason", ""),
        "fundamental_reason": fund.get("reason", ""),
    }

    with st.spinner("🔄 Round 2：辩论中..."):
        sent_r2 = debate_agent("情绪", r1, r1, r1)
        tech_r2 = debate_agent("技术", r1, r1, r1)
        fund_r2 = debate_agent("基本面", r1, r1, r1)

    r2 = {
        "sentiment": extract_score(sent_r2), "technical": extract_score(tech_r2), "fundamental": extract_score(fund_r2),
        "sentiment_changed": extract_changed(sent_r2), "technical_changed": extract_changed(tech_r2),
        "fundamental_changed": extract_changed(fund_r2),
    }

    s1 = [r1["sentiment"], r1["technical"], r1["fundamental"]]
    s2 = [r2["sentiment"], r2["technical"], r2["fundamental"]]
    d_pre = statistics.stdev(s1) if len(s1) > 1 else 0.0
    d_post = statistics.stdev(s2) if len(s2) > 1 else 0.0
    c_shift = (d_pre - d_post) / d_pre if d_pre > 0 else 0.0

    # v3新增：信息论指标
    info = compute_info_metrics(s1, s2)

    return {
        "ticker": ticker, "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "r1": r1, "r2": r2,
        "disagreements": {
            "D_pre": round(d_pre, 3), "D_post": round(d_post, 3), "C_shift": round(c_shift, 3),
            "D(Sent,Tech)": round(abs(r2["sentiment"] - r2["technical"]), 3),
            "D(Sent,Fund)": round(abs(r2["sentiment"] - r2["fundamental"]), 3),
            "D(Tech,Fund)": round(abs(r2["technical"] - r2["fundamental"]), 3),
        },
        "info_theory": info,
    }


page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "⚡ Event Study Engine",
        "🎯 Two-Shocks Decomposition",
        "💬 FOMC Sentiment Analysis",
        "🔄 Capital Flow Analysis",
        "📚 Paper Replication Lab",
        "📊 WRDS Results (v10.2)",
        "⚙️ Data Explorer",
        "🔬 Phase 1 Research",
        "🧠 Federated AI Intelligence",
    ],
    label_visibility="collapsed",
)

def plot_radar(r1, r2):
    cats = ["情绪 Agent", "技术 Agent", "基本面 Agent"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Round 1（辩论前）", "Round 2（辩论后）"],
                        specs=[[{"type": "polar"}, {"type": "polar"}]])
    for col, (scores, color, name) in enumerate([
        ([r1["sentiment"], r1["technical"], r1["fundamental"]], "#636EFA", "Round 1"),
        ([r2["sentiment"], r2["technical"], r2["fundamental"]], "#FF6B84", "Round 2"),
    ], 1):
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], theta=cats + [cats[0]],
            fill='toself', fillcolor=color.replace("#", "rgba(").replace("FA", ",0.25)").replace("84", ",0.25"),
            line=dict(color=color, width=2), name=name
        ), row=1, col=col)
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,10], tickvals=[2,4,6,8,10])), row=1, col=col)
    fig.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Route to Pages ──
if page == "🏠 Dashboard":
    from modules import dashboard
    dashboard.render()
elif page == "⚡ Event Study Engine":
    from modules import event_study
    event_study.render()
elif page == "🎯 Two-Shocks Decomposition":
    from modules import two_shocks
    two_shocks.render()
elif page == "💬 FOMC Sentiment Analysis":
    from modules import sentiment
    sentiment.render()
elif page == "🔄 Capital Flow Analysis":
    from modules import capital_flow
    capital_flow.render()
elif page == "📚 Paper Replication Lab":
    from modules import replication
    replication.render()
elif page == "📊 WRDS Results (v10.2)":
    from modules import wrds_results
    wrds_results.render()
elif page == "⚙️ Data Explorer":
    from modules import data_explorer
    data_explorer.render()
elif page == "🔬 Phase 1 Research":
    from modules import research
    research.render()
elif page == "🧠 Federated AI Intelligence":
    # Use English module
    from modules import fed_intelligence_en
    fed_intelligence_en.render()
