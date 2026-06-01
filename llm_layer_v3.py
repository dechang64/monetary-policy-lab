"""
Delta — LLM 调用层 v3 (升级版)
=================================
升级内容：
  - FinDPO (iacornelius/FinDPO-FinGPT3.5) 替代 rule-based 作为情感分析底座
  - 输出: [P_positive, P_negative, P_neutral] → 概率向量熵 + 置信度指标
  - 保留辩论机制（Debate Round 1 → Round 2），Debate层仍用 OpenClaw llm-task
  - 支持 FinDPO / FinBERT / rule-based 三种模式自动切换

作者：思怡 / 2026-05-31
"""

import re
import math
import statistics
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────
# 1. FinDPO / FinBERT 情感分析引擎
# ─────────────────────────────────────────────────────────

class FinDPOEngine:
    """
    调用 FinDPO (iacornelius/FinDPO-FinGPT3.5) 做金融情感分析。
    输出三维概率向量，替代单一评分。
    """

    def __init__(self, model_name: str = "iacornelius/FinDPO-FinGPT3.5"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self._loaded = False

    def load(self):
        """懒加载：首次调用时加载模型"""
        if self._loaded:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._loaded = True
            return True
        except Exception as e:
            print(f"[FinDPO] 加载失败: {e}，回退到 FinBERT")
            return False

    def analyze(self, text: str) -> dict:
        """
        返回三维概率 + 派生指标
        FinDPO标签: 0=positive, 1=negative, 2=neutral
        """
        if not self._loaded:
            ok = self.load()
            if not ok:
                return self._fallback_analysis(text)

        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        p_pos = probs[0].item()
        p_neg = probs[1].item()
        p_neu = probs[2].item()

        # 情感分: -1~1 (positive - negative)
        sentiment_score = p_pos - p_neg

        # 熵（信息不确定性）
        entropy = -sum(p * math.log(max(p, 1e-10)) for p in [p_pos, p_neg, p_neu])

        # 置信度
        confidence = max(p_pos, p_neg, p_neu)

        # 标签
        label = "Dovish" if sentiment_score > 0.15 else ("Hawkish" if sentiment_score < -0.15 else "Neutral")

        return {
            "score": sentiment_score,           # -1 ~ 1
            "label": label,
            "prob_positive": p_pos,
            "prob_negative": p_neg,
            "prob_neutral":  p_neu,
            "entropy":      entropy,            # 0 ~ ln(3) ≈ 1.10
            "confidence":   confidence,        # 0 ~ 1
            "confidence_low": confidence < 0.4, # 低置信度信号
            "mode": "findpo",
        }

    def _fallback_analysis(self, text: str) -> dict:
        """FinDPO不可用时回退到 FinBERT"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = mdl(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            p_pos, p_neg, p_neu = probs[0].item(), probs[1].item(), probs[2].item()
            score = p_pos - p_neg
            entropy = -sum(p * math.log(max(p, 1e-10)) for p in [p_pos, p_neg, p_neu])
            confidence = max(p_pos, p_neg, p_neu)
            label = "Dovish" if score > 0.15 else ("Hawkish" if score < -0.15 else "Neutral")
            return {
                "score": score, "label": label,
                "prob_positive": p_pos, "prob_negative": p_neg, "prob_neutral": p_neu,
                "entropy": entropy, "confidence": confidence, "confidence_low": confidence < 0.4,
                "mode": "finbert",
            }
        except Exception:
            return self._rule_based_analyze(text)

    def _rule_based_analyze(self, text: str) -> dict:
        """最终回退：rule-based 关键词匹配"""
        text_lower = text.lower()
        hawkish_kw = ["inflation","tighten","restrictive","strong growth","above target","higher for longer","no rate cuts"]
        dovish_kw  = ["easing","cut rates","moderate growth","downside risks","patient","balanced","disinflation"]
        h = sum(1 for kw in hawkish_kw if kw in text_lower)
        d = sum(1 for kw in dovish_kw if kw in text_lower)
        total = h + d
        score = (d - h) / max(total, 1)
        entropy = 0.95  # rule-based 高不确定性
        confidence = 0.5
        label = "Hawkish" if score < -0.15 else ("Dovish" if score > 0.15 else "Neutral")
        return {
            "score": score, "label": label,
            "prob_positive": max(0, score), "prob_negative": max(0, -score), "prob_neutral": 1-abs(score),
            "entropy": entropy, "confidence": confidence, "confidence_low": True,
            "mode": "rule-based",
        }


# 全局单例（避免重复加载）
_fin_dpo_engine = None

def get_fin_dpo_engine() -> FinDPOEngine:
    global _fin_dpo_engine
    if _fin_dpo_engine is None:
        _fin_dpo_engine = FinDPOEngine()
    return _fin_dpo_engine


# ─────────────────────────────────────────────────────────
# 2. 专业 Agent Prompt 模板（FinDPO增强版）
# ─────────────────────────────────────────────────────────

def sentiment_prompt(ticker: str, start_date: str, end_date: str) -> str:
    return (
        f"请对以下股票给出1-10的情绪评分（1=极度恐慌，10=极度乐观）。\n"
        f"股票：{ticker}\n"
        f"时间窗口：{start_date} 至 {end_date}\n"
        f"严格按JSON格式输出，不要加任何前缀：\n"
        f'{{"score": <数字 1-10>, "reason": "<理由，50字以内>"}}'
    )

def technical_prompt(ticker: str) -> str:
    return (
        f"请对以下股票给出1-10的技术评分（1=明显空头，10=明显多头）。\n"
        f"股票：{ticker}\n"
        f"严格按以下JSON格式输出：\n"
        f'{{"score": <数字 1-10>, "reason": "<理由，50字以内>"}}'
    )

def fundamental_prompt(ticker: str) -> str:
    return (
        f"请对以下股票给出1-10的基本面评分（1=估值过高，10=估值偏低）。\n"
        f"股票：{ticker}\n"
        f"严格按以下JSON格式输出：\n"
        f'{{"score": <数字 1-10>, "reason": "<理由，50字以内>"}}'
    )

def debate_prompt(my_role: str, sent: dict, tech: dict, fund: dict) -> str:
    return (
        f"你是{my_role}分析师。其他两位分析师的评分如下：\n"
        f"- 情绪分析师：{sent['score']}分，理由：{sent.get('reason','')}\n"
        f"- 技术分析师：{tech['score']}分，理由：{tech.get('reason','')}\n"
        f"- 基本面分析师：{fund['score']}分，理由：{fund.get('reason','')}\n"
        f"参考其他分析师意见，决定你的最终评分。\n"
        f"严格按JSON格式输出：\n"
        f'{{"score": <数字 1-10>, "reason": "<理由>", "changed": true/false}}'
    )


# ─────────────────────────────────────────────────────────
# 3. LLM 调用（通过 OpenClaw llm-task）
# ─────────────────────────────────────────────────────────

def call_llm(prompt: str, schema: dict) -> dict:
    """通过 OpenClaw llm-task 工具调用 LLM"""
    try:
        from openclaw.tools import llm_task
        result = llm_task(prompt=prompt, schema=schema, timeoutMs=60000)
        if isinstance(result, dict):
            return result
        import json
        return json.loads(str(result))
    except Exception as e:
        return {"score": 5.0, "reason": f"[调用失败: {e}]", "error": str(e)}


# ─────────────────────────────────────────────────────────
# 4. JS散度计算（升级：基于概率向量）
# ─────────────────────────────────────────────────────────

def js_divergence(p_pos_s, p_neg_s, p_pos_t, p_neg_t, p_pos_f, p_neg_f) -> float:
    """
    简化JS散度：基于情绪概率向量计算三 Agent 分歧度。
    用 (P_pos - P_neg) 作为每个 Agent 的情绪方向标量。
    D = |s_i - s_j| 的归一化版本。
    """
    s_vals = [p_pos_s - p_neg_s, p_pos_t - p_neg_t, p_pos_f - p_neg_f]
    mean = sum(s_vals) / 3
    # Jensen-Shannon 散度（简化）
    kl = sum(
        (max(v, -1) - max(mean, -1))**2 / 3
        for v in s_vals
    )
    # 归一化到 [0,1]
    return min(kl / 0.5, 1.0)


def std_divergence(scores: list) -> float:
    """标准差分歧度"""
    return statistics.stdev(scores) if len(scores) > 1 else 0.0


def range_divergence(scores: list) -> float:
    """极差分歧度"""
    return max(scores) - min(scores) if scores else 0.0


def entropy_from_probs(p_pos: float, p_neg: float, p_neu: float) -> float:
    """从三维概率计算香农熵"""
    probs = [p_pos, p_neg, p_neu]
    return -sum(p * math.log(max(p, 1e-10)) for p in probs)


# ─────────────────────────────────────────────────────────
# 5. 完整两轮分析（升级版）
# ─────────────────────────────────────────────────────────

def run_full_analysis(ticker: str, llm_call_fn) -> dict:
    """
    升级版两轮分析：
      - Round 1: FinDPO情绪 + llm-task技术 + llm-task基本面
      - Round 2: 三方辩论（llm-task）
      - 输出: 概率向量 + 熵 + 置信度 + JS散度 + 标准差 + 极差
    """
    now = datetime.now()
    start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end   = now.strftime("%Y-%m-%d")

    # ── Round 1 ─────────────────────────────────────────
    # 情绪：用 FinDPO
    fin_dpo = get_fin_dpo_engine()
    fin_sent = fin_dpo.analyze(
        f"股票 {ticker} 在 {start} 至 {end} 的财经新闻"
    )
    sentiment_r1 = {
        "score": fin_sent["score"] * 5 + 5,  # 归一化到 1~10
        "reason": f"FinDPO[{fin_sent['mode']}]: prob_pos={fin_sent['prob_positive']:.2f}, prob_neg={fin_sent['prob_negative']:.2f}",
        "fin_score": fin_sent["score"],        # 原始 -1~1
        "prob_positive": fin_sent["prob_positive"],
        "prob_negative": fin_sent["prob_negative"],
        "prob_neutral":  fin_sent["prob_neutral"],
        "entropy":       fin_sent["entropy"],  # 0~1.1
        "confidence":    fin_sent["confidence"],
        "mode":          fin_sent["mode"],
    }

    # 技术 + 基本面：仍用 llm-task（Debate）
    technical   = llm_call_fn(technical_prompt(ticker))
    fundamental = llm_call_fn(fundamental_prompt(ticker))

    r1 = {
        "sentiment": sentiment_r1["score"],
        "technical":  technical.get("score", 5.0),
        "fundamental": fundamental.get("score", 5.0),
        "sentiment_reason": sentiment_r1["reason"],
        "technical_reason": technical.get("reason", ""),
        "fundamental_reason": fundamental.get("reason", ""),
        # FinDPO 扩展字段
        "fin_score": sentiment_r1["fin_score"],
        "sentiment_entropy": sentiment_r1["entropy"],
        "sentiment_confidence": sentiment_r1["confidence"],
        "prob_pos": sentiment_r1["prob_positive"],
        "prob_neg": sentiment_r1["prob_negative"],
        "prob_neu":  sentiment_r1["prob_neutral"],
        "sentiment_mode": sentiment_r1["mode"],
    }

    # ── Round 2: Debate ──────────────────────────────────
    sent_r2 = llm_call_fn(debate_prompt("情绪", r1, r1, r1))  # 情绪Agent只看其他两个
    tech_r2 = llm_call_fn(debate_prompt("技术", r1, r1, r1))
    fund_r2 = llm_call_fn(debate_prompt("基本面", r1, r1, r1))

    r2 = {
        "sentiment": sent_r2.get("score", r1["sentiment"]),
        "technical":  tech_r2.get("score", r1["technical"]),
        "fundamental": fund_r2.get("score", r1["fundamental"]),
        "sentiment_changed": sent_r2.get("changed", False),
        "technical_changed":  tech_r2.get("changed", False),
        "fundamental_changed": fund_r2.get("changed", False),
    }

    # ── 分歧度计算 ───────────────────────────────────────
    s1 = [r1["sentiment"], r1["technical"], r1["fundamental"]]
    s2 = [r2["sentiment"], r2["technical"], r2["fundamental"]]

    # 三路信号（升级版：同时输出JS散度、标准差、极差）
    d_pre_std  = std_divergence(s1)
    d_pre_rng  = range_divergence(s1)
    d_post_std = std_divergence(s2)
    d_post_rng = range_divergence(s2)

    # JS散度（基于FinDPO概率向量）
    js_pre  = js_divergence(
        sentiment_r1["prob_pos"], sentiment_r1["prob_neg"],  # sent prob
        0.5, 0.5,   # tech prob (近似)
        0.5, 0.5,   # fund prob (近似)
    )
    js_post = d_post_std / max(d_pre_std, 0.01)  # 归一化

    c_shift = (d_pre_std - d_post_std) / max(d_pre_std, 0.001)

    # FinDPO情绪熵（Round 1输出）
    sentiment_entropy_r1 = sentiment_r1["entropy"]
    confidence_low_r1   = sentiment_r1["confidence"] < 0.4

    return {
        "ticker": ticker,
        "r1": r1,
        "r2": r2,
        "fin_dpo": {
            "score": sentiment_r1["fin_score"],          # -1~1
            "prob_positive": sentiment_r1["prob_positive"],
            "prob_negative": sentiment_r1["prob_negative"],
            "prob_neutral":  sentiment_r1["prob_neutral"],
            "entropy":        sentiment_entropy_r1,
            "confidence":    sentiment_r1["confidence"],
            "confidence_low": confidence_low_r1,
            "mode":           sentiment_r1["mode"],
        },
        "disagreements": {
            "D_pre_std":  round(d_pre_std,  4),
            "D_post_std": round(d_post_std, 4),
            "D_pre_range":  round(d_pre_rng,  4),
            "D_post_range": round(d_post_rng, 4),
            "JS_pre": round(js_pre, 4),
            "JS_post": round(js_post, 4),
            "C_shift": round(c_shift, 4),
            "D(Sent,Tech)": round(abs(r2["sentiment"] - r2["technical"]), 3),
            "D(Sent,Fund)": round(abs(r2["sentiment"] - r2["fundamental"]), 3),
            "D(Tech,Fund)": round(abs(r2["technical"] - r2["fundamental"]), 3),
        }
    }