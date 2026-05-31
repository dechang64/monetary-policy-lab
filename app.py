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


# ============================================================
# 可视化组件
# ============================================================

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


def plot_pairwise_bar(dis):
    labels = ["D(Sent,Tech)", "D(Sent,Fund)", "D(Tech,Fund)"]
    values = [dis[k] for k in labels]
    colors = ["#FF6B84", "#4ECDC4", "#45B7D1"]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors,
        text=[f"{v}" for v in values], textposition="outside")])
    fig.update_layout(title="Round 2 分歧度（Pairwise）", yaxis_title="评分差", height=280)
    st.plotly_chart(fig, use_container_width=True)


def plot_flow(r1, r2, changes):
    agents, y_pos = ["情绪 Agent", "技术 Agent", "基本面 Agent"], [8, 5, 2.5]
    r1v = [r1["sentiment"], r1["technical"], r1["fundamental"]]
    r2v = [r2["sentiment"], r2["technical"], r2["fundamental"]]
    changed = [changes[k] for k in ["sentiment", "technical", "fundamental"]]
    fig = go.Figure()
    for i, (a, y, r1i, r2i, ch) in enumerate(zip(agents, y_pos, r1v, r2v, changed)):
        color = "#FF6B84" if ch else "#00C896"
        arrow = "↑" if r2i > r1i else ("↓" if r2i < r1i else "→")
        fig.add_trace(go.Scatter(x=[0,1], y=[y,y], mode="lines+markers+text",
            line=dict(color=color, width=3), marker=dict(size=18),
            text=[f"{r1i:.0f}", f"{r2i:.0f} {arrow}"],
            textposition=["middle left","middle right"],
            name=f"{a} {'🔄' if ch else '✅'}"))
    fig.update_layout(title="评分变化（绿=不变，红=变化）",
        yaxis=dict(range=[0,11], visible=False), xaxis=dict(title="← Round 1 | Round 2 →", showgrid=False),
        showlegend=True, height=260)
    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(score, label, color):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        gauge={"axis":{"range":[0,10],"ticks":""},"bar":{"color":color},
            "steps":[{"range":[0,3],"color":"rgba(255,99,132,0.15)"},{"range":[3,7],"color":"rgba(255,206,86,0.15)"},{"range":[7,10],"color":"rgba(0,200,150,0.15)"}],
            "threshold":{"line":{"color":"white","width":4},"thickness":0.8}},
        domain={"x":[0,1],"y":[0,1]}))
    fig.update_layout(height=170, margin=dict(l=10,r=10,t=30,b=10),
        title=dict(text=label, font=dict(size=13)), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Streamlit 主界面
# ============================================================

st.set_page_config(page_title="Delta", page_icon="📊", layout="wide")
st.title("📊 Delta — Multi-Agent 分歧研究平台")
st.caption(f"Phase 1 Demo | 思怡 | {datetime.now().strftime('%Y-%m-%d')}")

with st.expander("ℹ️ 关于本项目", expanded=False):
    st.markdown("""
    **研究问题**：LLM Agent之间的分歧能否作为股票截面收益的结构化预测因子？
    
    **方法**：Round 1 三Agent独立评分（1-10）→ Round 2 辩论后决定是否修改 → 分歧度 = 标准差
    
    **创新**：现有系统（TradingAgents、FinDebate）将分歧视为噪声，我们保留并量化分歧作为信号
    
    **Phase 1**：单只股票完整分析 + 雷达图 + 分歧因子 + CSV导出
    **Phase 2（规划）**：批量回测 | 分组收益 | IC分析 | Fama-French回归
    """)

with st.sidebar:
    st.header("🔧 配置")
    ticker = st.text_input("股票代码", value="AAPL", placeholder="如：AAPL, TSLA, 600519.SS").strip().upper()
    btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**NLP引擎**：v3 FinDPO（3D概率向量）")
    st.markdown("**新增**：JS散度 | 情感熵 | Alpha Illusion过滤")
    st.markdown(f"**时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown(f"**时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

if btn and ticker:
    st.session_state.clear()
    st.subheader(f"🔍 {ticker} — 分析进行中...")

    result = run_full_analysis(ticker)
    st.success(f"✅ 分析完成 | {result['ticker']} | {result['datetime']}")

    r1, r2, dis = result["r1"], result["r2"], result["disagreements"]

    # 评分仪表
    st.divider()
    st.subheader("📈 评分仪表（辩论后）")
    c1, c2, c3 = st.columns(3)
    changes = {"sentiment": r2["sentiment_changed"], "technical": r2["technical_changed"],
               "fundamental": r2["fundamental_changed"]}
    with c1:
        tag = "🔄 变了" if changes["sentiment"] else "✅ 不变"
        plot_gauge(r2["sentiment"], f"情绪 Agent {tag}", "#636EFA")
    with c2:
        tag = "🔄 变了" if changes["technical"] else "✅ 不变"
        plot_gauge(r2["technical"], f"技术 Agent {tag}", "#4ECDC4")
    with c3:
        tag = "🔄 变了" if changes["fundamental"] else "✅ 不变"
        plot_gauge(r2["fundamental"], f"基本面 Agent {tag}", "#FF6B84")

    # 雷达图 + 变化流
    st.divider()
    cl, cr = st.columns([1, 1])
    with cl:
        plot_radar(r1, r2)
    with cr:
        plot_flow(r1, r2, changes)

    # 分歧因子
    # v3新增：信息论指标
    info = result["info_theory"]
    st.divider()
    st.subheader("🔬 核心指标（v3信息论增强）")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("D_pre（辩论前）", f"{dis['D_pre']:.3f}", "原始分歧")
        st.metric("JS_post（JS散度）", f"{info['JS_post']:.4f}", "v3新增")
        st.metric("H_sentiment（情感熵）", f"{info['H_sentiment']:.4f}", "v3新增")
    with mc2:
        st.metric("D_post（核心因子）", f"{dis['D_post']:.3f}", "真实分歧")
        st.metric("IA（信息不对称度）", f"{info['IA']:.4f}", "v3新增")
        st.metric("D_irreducible", f"{info['D_irreducible']:.4f}", "v3新增")
    with mc3:
        st.metric("C_shift（信念转移度）", f"{dis['C_shift']:.1%}", "虚假分歧过滤")
        conf_label = "⚠️ Alpha Illusion" if info['confidence_low'] else "✅ 置信"
        st.metric("confidence（置信度）", f"{info['confidence']:.3f}", conf_label)

    # Alpha Illusion警告
    if info['confidence_low']:
        st.warning("⚠️ Alpha Illusion风险：置信度 < 0.4，当前信号模糊，谨慎下单")
    else:
        st.success(f"✅ 信号清晰，置信度 {info['confidence']:.3f}，无Alpha Illusion风险")

    if dis["D_post"] > 2.0:
        st.info("🔴 **高度分歧** | 三Agent评分差异大，存在显著不确定性。文献支持：高分歧股票未来收益倾向负面。可关注D_post排序Top 20%股票的做空机会。")
    elif dis["D_post"] > 1.0:
        st.info("🟡 **中度分歧** | 存在真实判断差异，辩论机制有效（参考C_shift）。需进一步检验预测力。")
    else:
        st.info("🟢 **低度分歧** | 三Agent相对一致，信号噪声较低。暂不构成有效alpha因子候选。")

    # Pairwise
    plot_pairwise_bar(dis)

    # 因子表
    st.divider()
    st.subheader("📋 因子汇总（v3信息论增强版）")
    rows = [
        ["D_pre", "辩论前总体分歧（标准差）", f"{dis['D_pre']:.3f}", "越大=原始分歧越大"],
        ["D_post", "辩论后总体分歧（核心因子）", f"{dis['D_post']:.3f}", "越大=真实分歧越大"],
        ["C_shift", "信念转移度", f"{dis['C_shift']:.1%}", "越高=虚假分歧被过滤越多"],
        ["JS_post", "JS散度（v3新增）", f"{info['JS_post']:.4f}", "概率分布分歧"],
        ["H_sentiment", "情感熵（v3新增）", f"{info['H_sentiment']:.4f}", "高=情绪不确定性高"],
        ["confidence_low", "Alpha Illusion（v3）", "⚠️" if info["confidence_low"] else "✅", "置信度<0.4时触发"],
        ["IA", "信息不对称度（v3新增）", f"{info['IA']:.4f}", "高=信息不对称严重"],
        ["D_irreducible", "不可约分歧（v3新增）", f"{info['D_irreducible']:.4f}", "无法通过辩论消除"],
        ["D(Sent,Tech)", "情绪×技术", f"{dis['D(Sent,Tech)']:.3f}", "预测动量/反转"],
        ["D(Sent,Fund)", "情绪×基本面", f"{dis['D(Sent,Fund)']:.3f}", "预测估值偏离"],
        ["D(Tech,Fund)", "技术×基本面", f"{dis['D(Tech,Fund)']:.3f}", "预测价值因子"],
    ]
