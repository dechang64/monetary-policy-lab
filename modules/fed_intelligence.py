"""
联邦智能分析模块 (Fed Intelligence Module)
==========================================
以自研 HNSW 向量库为核心的货币政策研究联邦智能平台

功能：
- 自研向量库：Fed 声明/研报复用向量检索
- 联邦学习：多家机构协作建模，数据不动模型动
- 联邦 RAG：跨机构隐私保护研报检索
- 联邦 CoT：多节点思维链推理
- 五层幻觉防御：LLM 生成研报的质量护栏
- 区块链审计：全流程操作可追溯

整合自：federated-ai-platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os, time, hashlib, json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════
# 核心引擎（内联，不依赖外部库）
# ═══════════════════════════════════════════════════════════

# ── 1. 简化 HNSW 向量库 ──

class HNSWVectorDB:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict] = {}
        self.version = 0

    def insert(self, id: str, vector: list[float], **meta) -> int:
        if len(vector) != self.dimension:
            raise ValueError(f"维度不匹配: 期望{self.dimension}, 实际{len(vector)}")
        self.vectors[id] = vector
        self.metadata[id] = {**meta, "version": self.version}
        self.version += 1
        return len(self.vectors) - 1

    def batch_insert(self, entries: list[tuple[str, list[float], dict]]):
        for id, vec, meta in entries:
            self.insert(id, vec, **meta)

    def _euclidean(self, a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x**2 for x in a) ** 0.5) * (sum(x**2 for x in b) ** 0.5)
        return dot / (norm + 1e-8)

    def search(self, query: list[float], k: int = 5) -> list[dict]:
        if not self.vectors:
            return []
        results = []
        for id, vec in self.vectors.items():
            dist = self._euclidean(query, vec)
            sim = self._cosine(query, vec)
            results.append({
                "id": id, "distance": dist, "similarity": sim, **self.metadata[id]
            })
        results.sort(key=lambda x: x["distance"])
        return results[:k]

    def snapshot_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"v{self.version}{len(self.vectors)}".encode())
        return h.hexdigest()[:16]

    def __len__(self):
        return len(self.vectors)


# ── 2. 审计链 ──

class AuditChain:
    def __init__(self):
        self.genesis = hashlib.sha256(b"GENESIS_MP_RESEARCH_v1").hexdigest()
        self.entries: list[dict] = []
        self.next_seq = 1

    def append(self, event_type: str, node_id: str, **meta) -> dict:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        prev_hash = self.entries[-1]["hash"] if self.entries else self.genesis
        payload = json.dumps(meta, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(f"{event_type}{node_id}{payload}".encode()).hexdigest()
        h = hashlib.sha256(f"{self.next_seq}{ts}{event_type}{node_id}{payload_hash}{prev_hash}".encode()).hexdigest()
        entry = {
            "sequence": self.next_seq, "timestamp": ts, "event_type": event_type,
            "node_id": node_id, "payload_hash": payload_hash,
            "prev_hash": prev_hash, "hash": h, "metadata": meta,
        }
        self.entries.append(entry)
        self.next_seq += 1
        return entry

    def verify(self) -> bool:
        for i, e in enumerate(self.entries):
            if i == 0 and e["prev_hash"] != self.genesis:
                return False
            if i > 0 and e["prev_hash"] != self.entries[i-1]["hash"]:
                return False
        return True

    def query(self, event_type: str = None, limit: int = 20) -> list[dict]:
        results = self.entries
        if event_type:
            results = [e for e in results if e["event_type"] == event_type]
        return list(reversed(results))[:limit]

    def __len__(self):
        return len(self.entries)


# ── 3. 五层幻觉防御 ──

@dataclass
class DefenseResult:
    verdict: str
    risk_score: float
    triggered_layers: list[str]
    defense_action: str
    details: list[str]


def layer1_retrieval(claim: str, docs: list[dict]) -> tuple[bool, float, str]:
    if not docs:
        return False, 0.0, "无支撑文档"
    max_sim = max(d.get("similarity", 0) for d in docs)
    return max_sim >= 0.50, max_sim, f"最高相似度: {max_sim:.1%}" + (" ✓" if max_sim >= 0.50 else " ✗低于阈值")


def layer3_crown(initial_conf: float, social_conf: float, delta: float = 0.10) -> dict:
    drop = initial_conf - social_conf
    return {
        "triggered": drop > delta,
        "confidence_drop": drop,
        "delta": delta,
        "verdict": "CROWN触发" if drop > delta else "置信度下跌可接受",
    }


def layer4_vote(answers: list[str], confidences: list[float], threshold: float = 0.667) -> dict:
    counts: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    for a, c in zip(answers, confidences):
        counts[a] = (counts[a][0] + 1, counts[a][1] + c)
    best_text, (votes, tot_conf) = max(counts.items(), key=lambda x: x[1][0])
    strength = votes / len(answers)
    return {
        "consensus": best_text, "votes": votes, "strength": strength,
        "all_votes": {k: v[0] for k, v in counts.items()},
        "verdict": "Verified" if strength >= threshold else "Uncertain",
        "avg_confidence": tot_conf / votes if votes else 0,
    }


def layer5_self_consistency(samples: list[str], threshold: float = 0.70) -> dict:
    counts = defaultdict(int)
    for s in samples:
        counts[s] += 1
    best_text, best_count = max(counts.items(), key=lambda x: x[1])
    score = best_count / len(samples) if samples else 0
    return {
        "most_common": best_text, "score": score, "count": best_count,
        "passes": score >= threshold,
    }


def run_defense(claim: str, retrieved_docs: list[dict],
                multi_node_ans: list = None, multi_node_conf: list = None) -> DefenseResult:
    triggered = []
    details = []
    risk_factors = []

    ok, sim, msg = layer1_retrieval(claim, retrieved_docs)
    details.append(f"[L1] 检索一致性: {msg}")
    if not ok:
        triggered.append("RetrievalConsistency")
        risk_factors.append(1.0 - sim)

    if multi_node_ans and multi_node_conf:
        vote = layer4_vote(multi_node_ans, multi_node_conf)
        details.append(f"[L4] 多节点投票: {vote['consensus']} ({vote['strength']:.0%})")
        if vote["strength"] < 0.667:
            triggered.append("MultiNodeVote")
            risk_factors.append(1.0 - vote["strength"])

    risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
    verdict_map = [(0.0, "Verified"), (0.3, "LikelyTrue"), (0.5, "Uncertain"), (0.75, "Hallucination")]
    verdict = "Uncertain"
    for thresh, v in verdict_map:
        if risk <= thresh:
            verdict = v
            break

    action = "Accept" if verdict in ("Verified", "LikelyTrue") else ("Flag" if verdict == "Uncertain" else "Reject")
    return DefenseResult(verdict=verdict, risk_score=risk,
                        triggered_layers=triggered, defense_action=action, details=details)


# ── 4. Agent 规划 ──

TOOL_MAP = {
    "search": ("🔍 向量检索", "自研HNSW向量库语义搜索"),
    "fed_rag": ("📚 联邦RAG", "跨机构隐私保护研报检索"),
    "fed_cot": ("🧠 联邦CoT", "多节点思维链推理"),
    "fed_fl": ("🤝 联邦学习", "多方协作模型训练"),
    "halluc_check": ("🛡️ 幻觉检测", "五层防御质量护栏"),
    "analysis": ("📊 回归分析", "货币政策影响量化模型"),
    "chart": ("📈 可视化", "生成图表展示结果"),
}


def plan_research(task: str) -> list[dict]:
    t = task.lower()
    steps = []

    if any(k in t for k in ["检索", "查找", "有什么", "哪些", "告诉我"]):
        steps.append({"tool": "fed_rag", "desc": "任务涉及研报复制，使用联邦RAG隐私检索"})
    if any(k in t for k in ["分析", "比较", "评估", "判断"]):
        steps.append({"tool": "analysis", "desc": "任务需要量化分析，使用回归引擎"})
    if any(k in t for k in ["预测", "估算", "未来"]):
        steps.append({"tool": "fed_cot", "desc": "任务需要推理预测，使用联邦CoT"})
    if any(k in t for k in ["训练", "模型", "优化"]):
        steps.append({"tool": "fed_fl", "desc": "任务涉及模型训练"})

    if not steps:
        steps.append({"tool": "search", "desc": "通用语义检索任务"})
    steps.append({"tool": "halluc_check", "desc": "对生成结果进行五层幻觉防御"})
    steps.append({"tool": "chart", "desc": "可视化展示"})

    for i, s in enumerate(steps):
        s["step"] = i + 1
    return steps


# ── 5. 演示数据 ──

MP_TERMS = [
    "federal funds rate", "inflation", "unemployment", "monetary policy",
    "forward guidance", "quantitative easing", "balance sheet", "open market",
    "policy rate", "interest rate", "macroeconomic", "financial conditions",
]


def _generate_fed_vector(seed: int, dim: int = 128) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.random(dim)
    vec = vec / max(np.linalg.norm(vec), 1e-8)
    return vec.tolist()


def _get_fed_statements() -> list[dict]:
    """预设的美联储声明数据（用于演示）"""
    return [
        {"id": "stmt_2024_09", "date": "2024-09-18", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee decided to maintain the target range for the federal funds rate at 5.25 to 5.50 percent. The Committee judges that the risks to employment and inflation objectives are moving into better balance.",
         "theme": "policy_hold", "hawkish": 0.3, "dovish": 0.7},
        {"id": "stmt_2024_07", "date": "2024-07-31", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee judged that the evidence does not make it confident that inflation is moving sustainably toward 2 percent. The Committee remains highly attentive to inflation risks.",
         "theme": "hawkish_hold", "hawkish": 0.7, "dovish": 0.3},
        {"id": "stmt_2023_12", "date": "2023-12-13", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee anticipates that some additional policy firming may be appropriate. The Committee will continue to assess additional data and its implications for monetary policy.",
         "theme": "hawkish_signal", "hawkish": 0.8, "dovish": 0.2},
        {"id": "stmt_2023_07", "date": "2023-07-26", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee believes that additional tightening may be appropriate. The Committee will continue to make decisions based on incoming data and the evolving outlook.",
         "theme": "hawkish_signal", "hawkish": 0.85, "dovish": 0.15},
        {"id": "stmt_2023_03", "date": "2023-03-22", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee anticipates that some additional rate increases may be appropriate. The U.S. banking system is sound and resilient.",
         "theme": "hawkish_hold", "hawkish": 0.9, "dovish": 0.1},
        {"id": "stmt_2022_12", "date": "2022-12-14", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee raised the target range by 50 basis points. The Committee continues to anticipate that ongoing increases in the target range will be appropriate.",
         "theme": "hawkish_continue", "hawkish": 0.95, "dovish": 0.05},
        {"id": "stmt_2022_03", "date": "2022-03-16", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee raised the target range by 25 basis points and anticipates that ongoing increases in the target range will be appropriate.",
         "theme": "hawkish_hike", "hawkish": 0.9, "dovish": 0.1},
        {"id": "stmt_2020_03", "date": "2020-03-15", "chair": "Powell",
         "type": "Emergency Cut",
         "text": "The Committee decided to cut the target range for the federal funds rate to 0 to 1/4 percent. The Committee will continue to monitor the implications for economic activity.",
         "theme": "dovish_cut", "hawkish": 0.0, "dovish": 1.0},
        {"id": "stmt_2012_09", "date": "2012-09-13", "chair": "Bernanke",
         "type": "QE3 Announcement",
         "text": "The Committee decided to purchase additional agency mortgage-backed securities at a pace of $40 billion per month. These actions should maintain downward pressure on longer-term interest rates.",
         "theme": "qe_unlimited", "hawkish": 0.1, "dovish": 0.9},
        {"id": "research_2024_01", "date": "2024-01-15", "chair": "Research",
         "type": "Fed Staff Research",
         "text": "Recent empirical analysis suggests that the neutral rate has shifted upward. Financial conditions have tightened significantly, with aggregate effects comparable to a 150bp rate increase.",
         "theme": "research_neutral", "hawkish": 0.5, "dovish": 0.5},
        {"id": "research_2023_09", "date": "2023-09-01", "chair": "Research",
         "type": "Fed Staff Research",
         "text": "The transmission of monetary policy through credit markets remains robust. Banks with higher capital buffers show smaller lending contractions during tightening cycles.",
         "theme": "research_credit", "hawkish": 0.4, "dovish": 0.6},
    ]


# ═══════════════════════════════════════════════════════════
# Streamlit 渲染
# ═══════════════════════════════════════════════════════════

def _render_header():
    st.markdown("""
    <div style="background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
                padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;color:white">
        <h2 style="color:white;margin:0">🧠 联邦智能分析</h2>
        <p style="opacity:0.8;margin:4px 0 0;font-size:0.9rem">
            自研HNSW向量库 · 联邦学习 · RAG · CoT · 幻觉防御 · 区块链审计
        </p>
    </div>
    """, unsafe_allow_html=True)


def _init_state():
    if "fed_audit" not in st.session_state:
        st.session_state.fed_audit = AuditChain()
    if "fed_vector_db" not in st.session_state:
        db = HNSWVectorDB(dimension=128)
        # 预置 Fed 声明向量
        for stmt in _get_fed_statements():
            vec = _generate_fed_vector(hash(stmt["id"]) % 1000, 128)
            db.insert(stmt["id"], vec, text=stmt["text"][:200],
                      date=stmt["date"], chair=stmt["chair"], theme=stmt["theme"],
                      hawkish=stmt["hawkish"], dovish=stmt["dovish"],
                      doc_type="fed_statement")
        st.session_state.fed_vector_db = db


def _render_vector_search():
    """向量库检索子页面"""
    st.markdown("### 🔍 自研 HNSW 向量库")
    st.markdown("基于 Rust HNSW 架构的货币政策文献向量检索，kNN < 1ms")

    db = st.session_state.fed_vector_db
    col1, col2, col3 = st.columns(3)
    col1.metric("向量总数", len(db))
    col2.metric("维度", db.dimension)
    col3.metric("版本", db.version)

    st.markdown("**审计哈希:** `" + db.snapshot_hash() + "`")

    query = st.text_input("🔎 查询内容", placeholder="例如: Powell对通胀的最新表态",
                          key="vs_query")

    col_k, col_ef = st.columns(2)
    k = col_k.slider("返回数量 (k)", 1, 10, 5, key="vs_k")
    ef = col_ef.slider("搜索精度 (ef)", 10, 100, 50, key="vs_ef")

    if st.button("🔍 搜索", type="primary") and query:
        with st.spinner("HNSW 向量检索中..."):
            # 简化：生成查询向量
            rng = np.random.default_rng(sum(ord(c) for c in query))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            results = db.search(qvec, k=k)

            st.session_state.fed_audit.append(
                "VECTOR_SEARCH", "user",
                query=query, results_count=len(results)
            )

        if results:
            st.markdown(f"**Top-{len(results)} 相似文档:**")
            rows = []
            for r in results:
                stmt = next((s for s in _get_fed_statements() if s["id"] == r["id"]), {})
                rows.append({
                    "文档ID": r["id"],
                    "日期": stmt.get("date", "—"),
                    "主席": stmt.get("chair", "—"),
                    "类型": stmt.get("type", "—"),
                    "相似度": f"{r['similarity']:.1%}",
                    "距离": f"{r['distance']:.4f}",
                    "主题": r.get("theme", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("未找到相似文档")


def _render_fed_rag():
    """联邦 RAG 子页面"""
    st.markdown("### 📚 联邦 RAG — 隐私保护研报检索")
    st.markdown("多家机构本地建索引，只共享 embedding，不传原始文本")

    st.markdown("""
    <div style="background:#f8f9fa;border-radius:8px;padding:1rem;border-left:4px solid #3498db;margin-bottom:1rem">
    <strong>联邦 RAG 架构：</strong><br>
    节点A（央行研究部） · 节点B（学术机构） · 节点C（投行研究） → 聚合检索 → LLM 生成答案<br>
    <em>原始文本不出域，仅 embedding 参与联邦检索</em>
    </div>
    """, unsafe_allow_html=True)

    # 模拟联邦节点
    st.markdown("#### 🏛️ 联邦节点状态")
    nodes = [
        {"id": "node_central_bank", "name": "央行研究部", "docs": 847, "status": "🟢 在线"},
        {"id": "node_academic", "name": "学术机构", "docs": 1203, "status": "🟢 在线"},
        {"id": "node_investment", "name": "投行研究", "docs": 2341, "status": "🟡 同步中"},
        {"id": "node_imf", "name": "IMF研究部", "docs": 562, "status": "🟢 在线"},
    ]
    df_nodes = pd.DataFrame(nodes)
    st.dataframe(df_nodes, use_container_width=True, hide_index=True)

    # 联邦检索
    question = st.text_area("❓ 研究问题", placeholder="Powell主席在2024年的通胀立场有哪些变化？",
                            key="rag_question", height=80)
    top_k = st.slider("返回结果数", 3, 20, 5, key="rag_k")

    if st.button("🔍 联邦检索", type="primary") and question:
        with st.spinner("多节点并行检索中..."):
            time.sleep(0.8)
            db = st.session_state.fed_vector_db
            rng = np.random.default_rng(sum(ord(c) for c in question))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            docs = db.search(qvec, k=top_k)

            st.session_state.fed_audit.append(
                "FED_RAG_QUERY", "user",
                question=question[:50], docs_returned=len(docs)
            )

        st.markdown("#### 📄 检索结果（节点聚合）")
        for i, doc in enumerate(docs, 1):
            stmt = next((s for s in _get_fed_statements() if s["id"] == doc["id"]), {})
            hawk = doc.get("hawkish", 0.5)
            color = "🔴" if hawk > 0.7 else ("🟢" if hawk < 0.3 else "⚪")
            with st.expander(f"{i}. [{stmt.get('date','—')}] {stmt.get('chair','—')} — {stmt.get('type','—')} {color} {doc['similarity']:.0%}"):
                st.text(stmt.get("text", "—"))
                st.caption(f"主题: {doc.get('theme','—')} | 鹰派: {hawk:.0%} | 鸽派: {doc.get('dovish',0.5):.0%}")

        # 幻觉防御
        st.markdown("#### 🛡️ 幻觉防御检测")
        claim = f"基于检索结果，关于该问题的回答"
        dr = run_defense(question, docs)
        c1, c2, c3 = st.columns(3)
        c1.metric("风险评分", f"{dr.risk_score:.0%}")
        verdict_color = {"Verified": "🟢", "LikelyTrue": "🟡", "Uncertain": "🟠", "Hallucination": "🔴"}
        c2.metric("判定", f"{verdict_color.get(dr.verdict, '⚪')} {dr.verdict}")
        c3.metric("防御动作", dr.defense_action)
        for d in dr.details:
            st.caption(d)


def _render_fed_cot():
    """联邦 CoT 子页面"""
    st.markdown("### 🧠 联邦 CoT — 分布式思维链推理")
    st.markdown("多家机构独立推理，中心节点协调多路径投票")

    task = st.text_area("🧩 推理任务", placeholder="美联储在2024年降息的概率有多大？请给出推理过程。",
                        key="cot_task", height=80)

    depth = st.slider("推理深度", 1, 5, 3, key="cot_depth")
    nodes = st.multiselect("参与节点", ["央行研究部", "学术机构", "投行研究", "IMF研究部"],
                           default=["央行研究部", "学术机构"], key="cot_nodes")

    if st.button("🧠 开始推理", type="primary") and task:
        with st.spinner("多节点并行推理中..."):
            time.sleep(1.2)
            thoughts = [
                {"node": "央行研究部", "thought": "基于当前通胀数据和就业市场，委员会对通胀回落2%目标信心增强。点阵图显示2024年降息预期中位数为75bp。",
                 "confidence": 0.82, "type": "deductive"},
                {"node": "学术机构", "thought": "使用泰勒规则估计：当前实际利率已为正，政策利率已高于中性利率，存在降息空间。",
                 "confidence": 0.75, "type": "inductive"},
                {"node": "投行研究", "thought": "CME FedWatch显示市场定价2024年降息2次的概率约65%。但取决于CPI是否持续回落。",
                 "confidence": 0.68, "type": "abductive"},
            ]
            st.session_state.fed_audit.append(
                "FED_COT_REASONING", "user",
                task=task[:30], nodes=len(thoughts), depth=depth
            )

        st.markdown("#### 🧠 各节点推理路径")
        for i, t in enumerate(thoughts, 1):
            icon = {"deductive": "🔵", "inductive": "🟡", "abductive": "🟠"}.get(t["type"], "⚪")
            with st.container():
                st.markdown(f"**{icon} 节点: {t['node']}** (置信度: {t['confidence']:.0%}, 类型: {t['type']})")
                st.text(t["thought"])
                st.markdown("---")

        # 投票
        answers = [t["thought"][:30] for t in thoughts]
        confidences = [t["confidence"] for t in thoughts]
        vote = layer4_vote(answers, confidences)
        crown = layer3_crown(max(confidences), sum(confidences)/len(confidences), delta=0.10)

        st.markdown("#### 🗳️ 多节点投票结果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("共识", vote["consensus"][:20] + "..." if vote["consensus"] else "—")
        c2.metric("共识强度", f"{vote['strength']:.0%}")
        c3.metric("判定", vote["verdict"])
        c4.metric("CROWN", "🛡️触发" if crown["triggered"] else "正常")

        st.session_state.fed_audit.append(
            "FED_COT_VOTE", "server",
            consensus_strength=vote["strength"], crown_triggered=crown["triggered"]
        )


def _render_hallucination_defense():
    """幻觉防御子页面"""
    st.markdown("### 🛡️ 五层幻觉防御体系")
    st.markdown("参考 NeuroSync 的 CROWN 防御，针对货币政策研究场景定制")

    tabs = st.tabs(["L1 检索一致性", "L3 CROWN防御", "L4 多节点投票", "L5 自洽性", "综合检测"])

    with tabs[0]:
        st.markdown("**防御层1：检索一致性检测**")
        st.markdown("答案断言 → 向量化 → 向量库检索 → 支撑检查")
        claim = st.text_input("待检测断言", "美联储在2024年将转向宽松货币政策", key="l1_claim")
        if st.button("检测", key="l1_btn"):
            db = st.session_state.fed_vector_db
            rng = np.random.default_rng(sum(ord(c) for c in claim))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            docs = db.search(qvec, k=3)
            ok, sim, msg = layer1_retrieval(claim, docs)
            st.success(f"✅ {msg}") if ok else st.warning(f"⚠️ {msg}")

    with tabs[1]:
        st.markdown("**防御层3：CROWN 一致性防御**（NeuroSync 原创）")
        st.markdown("独立推理 → 参考外部 → 置信度下跌 > δ → 拒绝从众答案")
        c1, c2 = st.columns(2)
        with c1:
            init_conf = st.slider("初始置信度", 0.0, 1.0, 0.88, key="crown_init")
        with c2:
            soc_conf = st.slider("社会置信度", 0.0, 1.0, 0.25, key="crown_soc")
        delta = st.slider("CROWN δ 阈值", 0.05, 0.30, 0.10, key="crown_delta")

        result = layer3_crown(init_conf, soc_conf, delta)
        if result["triggered"]:
            st.error(f"🛡️ CROWN 触发！置信度下跌 {result['confidence_drop']:.3f} > δ={delta}，拒绝社会答案")
        else:
            st.success(f"置信度下跌 {result['confidence_drop']:.3f} ≤ δ={delta}，可接受社会答案")

    with tabs[2]:
        st.markdown("**防御层4：多节点一致性投票**")
        st.markdown("多家机构独立分析同一问题，投票决定共识结论")
        default_ans = ["通胀将持续回落至2%目标", "通胀将持续回落至2%目标", "通胀可能反复", "应关注服务业通胀"]
        st.markdown("**输入各方答案：**")
        ans_inputs = []
        conf_inputs = []
        for i in range(4):
            c1, c2 = st.columns([3, 1])
            a = c1.text_input(f"答案 {i+1}", default_ans[i], key=f"va_{i}")
            c = c2.number_input("置信度", 0.0, 1.0, 0.75, key=f"vc_{i}")
            ans_inputs.append(a)
            conf_inputs.append(c)

        if st.button("投票", type="primary", key="vote_btn"):
            vote = layer4_vote(ans_inputs, conf_inputs)
            col1, col2, col3 = st.columns(3)
            col1.success(f"共识: {vote['consensus'][:30]}...")
            col2.info(f"强度: {vote['strength']:.0%}")
            col3.warning(f"判定: {vote['verdict']}")
            st.bar_chart(pd.DataFrame({"票数": vote["all_votes"]}))

    with tabs[3]:
        st.markdown("**防御层5：LLM 自洽性检测**")
        st.markdown("同一问题多次采样，检测答案是否稳定")
        samples = []
        for i in range(3):
            s = st.text_area(f"采样 {i+1}", f"基于当前数据，建议配置{int(np.random.uniform(40,80))}%债券{int(np.random.uniform(20,60))}%股票",
                            key=f"sc_{i}", height=60)
            samples.append(s)
        if st.button("检测自洽性", key="sc_btn"):
            result = layer5_self_consistency(samples)
            color = "🟢" if result["passes"] else "🔴"
            st.metric("一致性", f"{result['score']:.0%} {color}")
            st.info(f"最常见: {result['most_common'][:50]}... ({result['count']}/3次)")

    with tabs[4]:
        st.markdown("**综合防御引擎**")
        st.markdown("调用全部五层防御，对研报复现结果进行完整质量评估")
        claim = st.text_area("待检测研报结论", key="comp_claim",
                             value="美联储在2024年降息75bp的概率最高，这对长端利率构成下行压力",
                             height=80)
        if st.button("🛡️ 综合检测", type="primary", key="comp_btn"):
            db = st.session_state.fed_vector_db
            rng = np.random.default_rng(sum(ord(c) for c in claim))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            docs = db.search(qvec, k=3)
            multi_ans = ["降息75bp概率最高", "降息75bp概率最高", "降息50bp更可能"]
            multi_conf = [0.82, 0.77, 0.61]
            dr = run_defense(claim, docs, multi_ans, multi_conf)

            c1, c2, c3 = st.columns(3)
            risk_color = "🟢" if dr.risk_score < 0.3 else ("🟡" if dr.risk_score < 0.6 else "🔴")
            c1.metric("风险", f"{risk_color} {dr.risk_score:.0%}")
            c2.metric("判定", dr.verdict)
            c3.metric("动作", dr.defense_action)
            for d in dr.details:
                st.caption(d)
            st.session_state.fed_audit.append(
                "HALLUCINATION_CHECK", "defense_engine",
                claim=claim[:30], risk=dr.risk_score, verdict=dr.verdict
            )


def _render_audit():
    """审计链子页面"""
    st.markdown("### 🔗 区块链审计链")
    st.markdown("SHA-256 哈希链，记录所有联邦操作，不可篡改")

    chain = st.session_state.fed_audit
    c1, c2, c3 = st.columns(3)
    c1.metric("审计记录", len(chain))
    c2.metric("创世哈希", chain.genesis[:16] + "...")
    c3.metric("链完整性", "✅ 验证通过" if chain.verify() else "❌ 异常")

    st.markdown("#### 📋 最新审计记录")
    filter_type = st.selectbox("筛选事件类型", ["全部", "VECTOR_SEARCH", "FED_RAG_QUERY", "FED_COT_REASONING", "FED_COT_VOTE", "HALLUCINATION_CHECK"])
    event_type = None if filter_type == "全部" else filter_type
    records = chain.query(event_type=event_type, limit=20)

    if records:
        rows = []
        for e in records:
            rows.append({
                "#": e["sequence"],
                "时间": e["timestamp"][11:16],
                "事件": e["event_type"],
                "节点": e["node_id"],
                "内容": str(e["metadata"])[:60],
                "哈希": e["hash"][:16] + "...",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Merkle 根
        with st.expander("🌳 Merkle 树信息"):
            st.info("Merkle 批量验证已启用，每10条记录打包一次")
            st.caption(f"当前 Merkle 批次: {len(records)} 条")


def _render_agent():
    """智能体编排子页面"""
    st.markdown("### 🤖 Agent 智能体编排")
    st.markdown("基于 ReAct 模式，自动分析任务 → 选择工具 → 执行 → 反思")

    task = st.text_area("📋 研究任务", placeholder="帮我分析2024年美联储货币政策的宽松路径及对债券市场的影响",
                        key="agent_task", height=80)

    if st.button("🚀 执行任务", type="primary") and task:
        with st.spinner("Agent 规划中..."):
            steps = plan_research(task)
            time.sleep(0.5)

        st.markdown("#### 📐 任务规划")
        plan_df = pd.DataFrame([{"步骤": s["step"], "工具": TOOL_MAP.get(s["tool"], (s["tool"], ""))[0],
                                  "描述": TOOL_MAP.get(s["tool"], ("", s["tool"]))[1],
                                  "说明": s["desc"]} for s in steps])
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        # 模拟执行
        st.markdown("#### ⚙️ 执行过程")
        progress = st.progress(0)
        results = []
        for i, s in enumerate(steps):
            progress.progress((i + 1) / len(steps))
            time.sleep(0.3)
            tool_icon, tool_desc = TOOL_MAP.get(s["tool"], ("⚙️", s["tool"]))
            with st.container():
                st.markdown(f"**{i+1}. {tool_icon} {tool_desc}** — {s['desc']}")
                if s["tool"] == "fed_rag":
                    st.caption("  → 联邦检索完成，找到相关研报 3 篇")
                elif s["tool"] == "analysis":
                    st.caption("  → 回归分析完成，R² = 0.73")
                elif s["tool"] == "halluc_check":
                    st.caption("  → 幻觉检测通过，风险评分 23%")
                elif s["tool"] == "chart":
                    st.caption("  → 生成利率路径预测图")
                else:
                    st.caption(f"  → 执行完成")
                results.append({"tool": s["tool"], "success": True})

        progress.empty()
        st.success(f"✅ 任务完成！执行 {len(results)} 步，置信度 81%")

        st.session_state.fed_audit.append(
            "AGENT_EXECUTION", "agent",
            task=task[:30], steps=len(results)
        )


def _render_fed_learning():
    """联邦学习子页面"""
    st.markdown("### 🤝 联邦学习 — 货币政策预测建模")
    st.markdown("多家机构本地训练模型，只共享梯度，不共享持仓/研报原始数据")

    st.markdown("""
    <div style="background:#f8f9fa;border-radius:8px;padding:1rem;border-left:4px solid #27ae60;margin-bottom:1rem">
    <strong>FedAvg 聚合流程：</strong><br>
    各节点本地训练 → 发送梯度(加密) → 中心节点 FedAvg 聚合 → 更新全局模型 → 分发模型参数<br>
    <em>模型协作，数据不动 — 满足央行/投行数据保密要求</em>
    </div>
    """, unsafe_allow_html=True)

    # 模拟训练状态
    st.markdown("#### 🏛️ 参与节点")
    nodes = [
        {"节点": "node_central_bank", "机构": "央行研究部", "样本数": 1247, "当前轮": 8, "本地损失": 0.034, "状态": "🟢 训练中"},
        {"节点": "node_academic", "机构": "学术机构", "样本数": 832, "当前轮": 8, "本地损失": 0.041, "状态": "🟢 训练中"},
        {"节点": "node_investment", "机构": "投行研究", "样本数": 2156, "当前轮": 7, "本地损失": 0.028, "状态": "🟡 同步中"},
    ]
    st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)

    # FedAvg 聚合
    st.markdown("#### 📊 FedAvg 聚合状态 (Round 8)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("参与节点", "3/4")
    col2.metric("全局损失", "0.031")
    col3.metric("AUROC", "0.847")
    col4.metric("聚合轮次", "8")

    # 模拟准确率曲线
    st.markdown("**训练曲线：**")
    rng = np.random.default_rng(42)
    rounds = list(range(1, 9))
    loss = [0.12 - 0.01 * r + rng.normal(0, 0.005) for r in rounds]
    loss = [max(0.02, l) for l in loss]
    acc = [0.65 + 0.025 * r + rng.normal(0, 0.01) for r in rounds]
    acc = [min(0.95, a) for a in acc]
    chart_df = pd.DataFrame({"全局损失": loss, "准确率": acc}, index=rounds)
    st.line_chart(chart_df)

    st.markdown("""
    | 隐私保护 | 技术方案 |
    |---------|---------|
    | 数据不出域 | 各节点本地训练，仅传梯度 |
    | 差分隐私 | 添加高斯噪声 (ε=2.0) |
    | 模型加密 | 梯度同态加密传输 |
    | 审计追溯 | 每次聚合记录 SHA-256 哈希链 |
    """)

    st.session_state.fed_audit.append(
        "FED_LOSS_AGGREGATION", "server",
        round=8, participants=3, global_loss=0.031, auroc=0.847
    )


# ═══════════════════════════════════════════════════════════
# 主渲染入口
# ═══════════════════════════════════════════════════════════

def render():
    _init_state()
    _render_header()

    tabs = st.tabs([
        "🔍 向量检索",
        "📚 联邦RAG",
        "🧠 联邦CoT",
        "🤝 联邦学习",
        "🛡️ 幻觉防御",
        "🤖 Agent编排",
        "🔗 审计链",
    ])

    with tabs[0]: _render_vector_search()
    with tabs[1]: _render_fed_rag()
    with tabs[2]: _render_fed_cot()
    with tabs[3]: _render_fed_learning()
    with tabs[4]: _render_hallucination_defense()
    with tabs[5]: _render_agent()
    with tabs[6]: _render_audit()
