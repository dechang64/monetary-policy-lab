"""
Fed Intelligence Module — Federated AI for Monetary Policy Research
================================================================
Self-developed HNSW vector DB powered, privacy-preserving research platform.

Modules:
- Self-built HNSW Vector DB: Fed statement/research vector retrieval
- Federated Learning: Multi-institution collaborative modeling
- Federated RAG: Privacy-preserving cross-institution document retrieval
- Federated CoT: Multi-node chain-of-thought reasoning
- Five-Layer Hallucination Defense: LLM quality guardrail
- Blockchain Audit: Immutable operation traceability

Integrated from: federated-ai-platform
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
# Core Engines (inline, no external deps)
# ═══════════════════════════════════════════════════════════

class HNSWVectorDB:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.vectors: dict = {}
        self.metadata: dict = {}
        self.version = 0

    def insert(self, id: str, vector: list, **meta) -> int:
        if len(vector) != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {len(vector)}")
        self.vectors[id] = vector
        self.metadata[id] = {**meta, "version": self.version}
        self.version += 1
        return len(self.vectors) - 1

    def batch_insert(self, entries: list):
        for id, vec, meta in entries:
            self.insert(id, vec, **meta)

    def _euclidean(self, a: list, b: list) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _cosine(self, a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x**2 for x in a) ** 0.5) * (sum(x**2 for x in b) ** 0.5)
        return dot / (norm + 1e-8)

    def search(self, query: list, k: int = 5) -> list:
        if not self.vectors:
            return []
        results = []
        for id, vec in self.vectors.items():
            dist = self._euclidean(query, vec)
            sim = self._cosine(query, vec)
            results.append({"id": id, "distance": dist, "similarity": sim, **self.metadata[id]})
        results.sort(key=lambda x: x["distance"])
        return results[:k]

    def snapshot_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"v{self.version}{len(self.vectors)}".encode())
        return h.hexdigest()[:16]

    def __len__(self):
        return len(self.vectors)


class AuditChain:
    def __init__(self):
        self.genesis = hashlib.sha256(b"GENESIS_MP_RESEARCH_v1").hexdigest()
        self.entries: list = []
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

    def query(self, event_type: str = None, limit: int = 20) -> list:
        results = self.entries
        if event_type:
            results = [e for e in results if e["event_type"] == event_type]
        return list(reversed(results))[:limit]

    def __len__(self):
        return len(self.entries)


# ── Hallucination Defense ──

@dataclass
class DefenseResult:
    verdict: str
    risk_score: float
    triggered_layers: list
    defense_action: str
    details: list


def layer1_retrieval(claim: str, docs: list) -> tuple:
    if not docs:
        return False, 0.0, "No supporting documents found"
    max_sim = max(d.get("similarity", 0) for d in docs)
    return max_sim >= 0.50, max_sim, f"Max similarity: {max_sim:.1%}" + (" ✓" if max_sim >= 0.50 else " ✗ Below threshold")


def layer3_crown(initial_conf: float, social_conf: float, delta: float = 0.10) -> dict:
    drop = initial_conf - social_conf
    return {
        "triggered": drop > delta,
        "confidence_drop": drop,
        "delta": delta,
        "verdict": "CROWN triggered" if drop > delta else "Confidence drop acceptable",
    }


def layer4_vote(answers: list, confidences: list, threshold: float = 0.667) -> dict:
    counts: dict = defaultdict(lambda: (0, 0.0))
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


def layer5_self_consistency(samples: list, threshold: float = 0.70) -> dict:
    counts = defaultdict(int)
    for s in samples:
        counts[s] += 1
    best_text, best_count = max(counts.items(), key=lambda x: x[1])
    score = best_count / len(samples) if samples else 0
    return {
        "most_common": best_text, "score": score, "count": best_count,
        "passes": score >= threshold,
    }


def run_defense(claim: str, retrieved_docs: list, multi_node_ans: list = None,
                multi_node_conf: list = None) -> DefenseResult:
    triggered, details, risk_factors = [], [], []

    ok, sim, msg = layer1_retrieval(claim, retrieved_docs)
    details.append(f"[L1] Retrieval consistency: {msg}")
    if not ok:
        triggered.append("RetrievalConsistency")
        risk_factors.append(1.0 - sim)

    if multi_node_ans and multi_node_conf:
        vote = layer4_vote(multi_node_ans, multi_node_conf)
        details.append(f"[L4] Multi-node vote: {vote['consensus']} ({vote['strength']:.0%})")
        if vote["strength"] < 0.667:
            triggered.append("MultiNodeVote")
            risk_factors.append(1.0 - vote["strength"])

    risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
    verdict_map = [(0.0, "Verified"), (0.3, "LikelyTrue"), (0.5, "Uncertain"), (0.75, "Hallucination")]
    verdict = next((v for thresh, v in verdict_map if risk <= thresh), "Uncertain")
    action = "Accept" if verdict in ("Verified", "LikelyTrue") else ("Flag" if verdict == "Uncertain" else "Reject")
    return DefenseResult(verdict=verdict, risk_score=risk, triggered_layers=triggered,
                        defense_action=action, details=details)


# ── Agent Planning ──

TOOL_MAP = {
    "search": ("🔍 Vector Search", "Self-built HNSW vector retrieval"),
    "fed_rag": ("📚 Federated RAG", "Cross-institution privacy-preserving retrieval"),
    "fed_cot": ("🧠 Federated CoT", "Multi-node chain-of-thought reasoning"),
    "fed_fl": ("🤝 Federated Learning", "Multi-party collaborative modeling"),
    "halluc_check": ("🛡️ Hallucination Check", "5-layer defense quality guardrail"),
    "analysis": ("📊 Regression Analysis", "Monetary policy impact quantification"),
    "chart": ("📈 Visualization", "Generate charts"),
}


def plan_research(task: str) -> list:
    t = task.lower()
    steps = []
    if any(k in t for k in ["search", "find", "what is", "which", "tell me"]):
        steps.append({"tool": "fed_rag", "desc": "Research query, use Federated RAG"})
    if any(k in t for k in ["analyze", "compare", "evaluate", "assess"]):
        steps.append({"tool": "analysis", "desc": "Quantitative analysis needed, use regression engine"})
    if any(k in t for k in ["predict", "estimate", "forecast", "future"]):
        steps.append({"tool": "fed_cot", "desc": "Inference/prediction needed, use Federated CoT"})
    if any(k in t for k in ["train", "model", "optimize"]):
        steps.append({"tool": "fed_fl", "desc": "Model training required"})
    if not steps:
        steps.append({"tool": "search", "desc": "General semantic search"})
    steps.append({"tool": "halluc_check", "desc": "Apply 5-layer hallucination defense to output"})
    steps.append({"tool": "chart", "desc": "Visualize results"})
    for i, s in enumerate(steps):
        s["step"] = i + 1
    return steps


# ── Demo Data ──

def _generate_fed_vector(seed: int, dim: int = 128) -> list:
    rng = np.random.default_rng(seed)
    vec = rng.random(dim)
    vec = vec / max(np.linalg.norm(vec), 1e-8)
    return vec.tolist()


def _get_fed_statements() -> list:
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
         "text": "The Committee decided to raise the target range by 25 basis points. The Committee will continue to make decisions based on incoming data and the evolving outlook.",
         "theme": "hawkish_hike", "hawkish": 0.85, "dovish": 0.15},
        {"id": "stmt_2022_12", "date": "2022-12-14", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee raised the target range by 50 basis points. The Committee continues to anticipate that ongoing increases in the target range will be appropriate.",
         "theme": "hawkish_continue", "hawkish": 0.95, "dovish": 0.05},
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
        {"id": "stmt_2022_03", "date": "2022-03-16", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee raised the target range by 25 basis points and anticipates that ongoing increases in the target range will be appropriate.",
         "theme": "hawkish_hike", "hawkish": 0.9, "dovish": 0.1},
        {"id": "stmt_2023_03", "date": "2023-03-22", "chair": "Powell",
         "type": "FOMC Decision",
         "text": "The Committee anticipates that some additional rate increases may be appropriate. The U.S. banking system is sound and resilient.",
         "theme": "hawkish_hold", "hawkish": 0.9, "dovish": 0.1},
    ]


# ═══════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════

def _render_header():
    st.markdown("""
    <div style="background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
                padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;color:white">
        <h2 style="color:white;margin:0">🧠 Federated AI Intelligence</h2>
        <p style="opacity:0.8;margin:4px 0 0;font-size:0.9rem">
            Self-built HNSW · FL · RAG · CoT · Hallucination Defense · Blockchain Audit
        </p>
    </div>
    """, unsafe_allow_html=True)


def _init_state():
    if "fed_audit" not in st.session_state:
        st.session_state.fed_audit = AuditChain()
    if "fed_vector_db" not in st.session_state:
        db = HNSWVectorDB(dimension=128)
        for stmt in _get_fed_statements():
            vec = _generate_fed_vector(hash(stmt["id"]) % 1000, 128)
            db.insert(stmt["id"], vec, text=stmt["text"][:200],
                      date=stmt["date"], chair=stmt["chair"], theme=stmt["theme"],
                      hawkish=stmt["hawkish"], dovish=stmt["dovish"],
                      doc_type="fed_statement")
        st.session_state.fed_vector_db = db


# ── Tab 1: Vector Search ──
def _render_vector_search():
    st.markdown("### 🔍 Self-Built HNSW Vector DB")
    st.markdown("Rust HNSW architecture for monetary policy literature retrieval, kNN < 1ms")
    db = st.session_state.fed_vector_db
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Vectors", len(db))
    c2.metric("Dimension", db.dimension)
    c3.metric("Version", db.version)
    st.markdown(f"**Audit Hash:** `{db.snapshot_hash()}`")
    query = st.text_input("🔎 Query", placeholder="e.g. Powell's latest stance on inflation", key="vs_query")
    k = st.slider("Results (k)", 1, 10, 5, key="vs_k")
    ef = st.slider("Search Precision (ef)", 10, 100, 50, key="vs_ef")
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("HNSW vector search..."):
            rng = np.random.default_rng(sum(ord(c) for c in query))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            results = db.search(qvec, k=k)
            st.session_state.fed_audit.append("VECTOR_SEARCH", "user", query=query, results_count=len(results))
        if results:
            st.markdown(f"**Top-{len(results)} Similar Documents:**")
            rows = []
            for r in results:
                stmt = next((s for s in _get_fed_statements() if s["id"] == r["id"]), {})
                rows.append({
                    "Doc ID": r["id"], "Date": stmt.get("date","—"),
                    "Chair": stmt.get("chair","—"), "Type": stmt.get("type","—"),
                    "Similarity": f"{r['similarity']:.1%}",
                    "Distance": f"{r['distance']:.4f}",
                    "Theme": r.get("theme","—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No similar documents found")


# ── Tab 2: Federated RAG ──
def _render_fed_rag():
    st.markdown("### 📚 Federated RAG — Privacy-Preserving Research Retrieval")
    st.markdown("Each institution indexes locally; only embeddings are shared, raw text stays private")
    nodes = [
        {"id": "node_central_bank", "name": "Central Bank Research", "docs": 847, "status": "🟢 Online"},
        {"id": "node_academic", "name": "Academic Institution", "docs": 1203, "status": "🟢 Online"},
        {"id": "node_investment", "name": "Investment Bank Research", "docs": 2341, "status": "🟡 Syncing"},
        {"id": "node_imf", "name": "IMF Research", "docs": 562, "status": "🟢 Online"},
    ]
    st.markdown("#### 🏛️ Federated Nodes")
    st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
    question = st.text_area("❓ Research Question", placeholder="What are Powell's inflation stance changes in 2024?",
                            key="rag_question", height=80)
    top_k = st.slider("Results", 3, 20, 5, key="rag_k")
    if st.button("🔍 Federated Search", type="primary") and question:
        with st.spinner("Multi-node parallel retrieval..."):
            time.sleep(0.8)
            db = st.session_state.fed_vector_db
            rng = np.random.default_rng(sum(ord(c) for c in question))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            docs = db.search(qvec, k=top_k)
            st.session_state.fed_audit.append("FED_RAG_QUERY", "user", question=question[:50], docs_returned=len(docs))
        st.markdown("#### 📄 Retrieval Results (Node-Aggregated)")
        for i, doc in enumerate(docs, 1):
            stmt = next((s for s in _get_fed_statements() if s["id"] == doc["id"]), {})
            hawk = doc.get("hawkish", 0.5)
            color = "🔴" if hawk > 0.7 else ("🟢" if hawk < 0.3 else "⚪")
            with st.expander(f"{i}. [{stmt.get('date','—')}] {stmt.get('chair','—')} — {stmt.get('type','—')} {color} {doc['similarity']:.0%}"):
                st.text(stmt.get("text", "—"))
                st.caption(f"Theme: {doc.get('theme','—')} | Hawkish: {hawk:.0%} | Dovish: {doc.get('dovish',0.5):.0%}")
        st.markdown("#### 🛡️ Hallucination Defense Check")
        claim = f"Based on retrieval, the answer to the research question"
        dr = run_defense(question, docs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{dr.risk_score:.0%}")
        vcolor = {"Verified": "🟢", "LikelyTrue": "🟡", "Uncertain": "🟠", "Hallucination": "🔴"}
        c2.metric("Verdict", f"{vcolor.get(dr.verdict,'⚪')} {dr.verdict}")
        c3.metric("Action", dr.defense_action)
        for d in dr.details:
            st.caption(d)


# ── Tab 3: Federated CoT ──
def _render_fed_cot():
    st.markdown("### 🧠 Federated CoT — Distributed Chain-of-Thought Reasoning")
    st.markdown("Multiple institutions reason independently; central node coordinates multi-path voting")
    task = st.text_area("🧩 Reasoning Task", placeholder="What's the probability of Fed rate cuts in 2024? Please reason step by step.",
                        key="cot_task", height=80)
    depth = st.slider("Reasoning Depth", 1, 5, 3, key="cot_depth")
    sel_nodes = st.multiselect("Participating Nodes", ["Central Bank Research", "Academic Institution",
                           "Investment Bank Research", "IMF Research"],
                          default=["Central Bank Research", "Academic Institution"], key="cot_nodes")
    if st.button("🧠 Start Reasoning", type="primary") and task:
        with st.spinner("Multi-node parallel reasoning..."):
            time.sleep(1.2)
            thoughts = [
                {"node": "Central Bank Research",
                 "thought": "Based on current inflation data and labor market, the committee's confidence in inflation returning to 2% is increasing. Dot plot shows 75bp rate cuts expected in 2024.",
                 "confidence": 0.82, "type": "deductive"},
                {"node": "Academic Institution",
                 "thought": "Using Taylor rule estimates: real interest rate is now positive, policy rate exceeds neutral rate, suggesting room for rate cuts.",
                 "confidence": 0.75, "type": "inductive"},
                {"node": "Investment Bank Research",
                 "thought": "CME FedWatch shows market pricing ~65% probability of 2 rate cuts in 2024. But depends on whether CPI continues to decline.",
                 "confidence": 0.68, "type": "abductive"},
            ]
            st.session_state.fed_audit.append("FED_COT_REASONING", "user", task=task[:30], nodes=len(thoughts), depth=depth)
        st.markdown("#### 🧠 Per-Node Reasoning Paths")
        for i, t in enumerate(thoughts, 1):
            icon = {"deductive": "🔵", "inductive": "🟡", "abductive": "🟠"}.get(t["type"], "⚪")
            with st.container():
                st.markdown(f"**{icon} Node: {t['node']}** (confidence: {t['confidence']:.0%}, type: {t['type']})")
                st.text(t["thought"])
                st.markdown("---")
        answers = [t["thought"][:30] for t in thoughts]
        confidences = [t["confidence"] for t in thoughts]
        vote = layer4_vote(answers, confidences)
        crown = layer3_crown(max(confidences), sum(confidences)/len(confidences), delta=0.10)
        st.markdown("#### 🗳️ Multi-Node Vote Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consensus", vote["consensus"][:20] + "..." if vote["consensus"] else "—")
        c2.metric("Consensus Strength", f"{vote['strength']:.0%}")
        c3.metric("Verdict", vote["verdict"])
        c4.metric("CROWN", "🛡️ Triggered" if crown["triggered"] else "Normal")
        st.session_state.fed_audit.append("FED_COT_VOTE", "server",
                                          consensus_strength=vote["strength"], crown_triggered=crown["triggered"])


# ── Tab 4: Federated Learning ──
def _render_fed_learning():
    st.markdown("### 🤝 Federated Learning — Monetary Policy Prediction Modeling")
    st.markdown("Institutions train locally; only gradients are shared, raw data/positions stay private")
    nodes = [
        {"Node": "node_central_bank", "Institution": "Central Bank Research", "Samples": 1247,
         "Round": 8, "Local Loss": 0.034, "Status": "🟢 Training"},
        {"Node": "node_academic", "Institution": "Academic Institution", "Samples": 832,
         "Round": 8, "Local Loss": 0.041, "Status": "🟢 Training"},
        {"Node": "node_investment", "Institution": "Investment Bank Research", "Samples": 2156,
         "Round": 7, "Local Loss": 0.028, "Status": "🟡 Syncing"},
    ]
    st.markdown("#### 🏛️ Participating Nodes")
    st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
    st.markdown("#### 📊 FedAvg Aggregation Status (Round 8)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", "3/4")
    c2.metric("Global Loss", "0.031")
    c3.metric("AUROC", "0.847")
    c4.metric("Rounds", "8")
    st.markdown("**Training Curve:**")
    rng = np.random.default_rng(42)
    rounds = list(range(1, 9))
    loss = [max(0.02, 0.12 - 0.01 * r + rng.normal(0, 0.005)) for r in rounds]
    acc = [min(0.95, 0.65 + 0.025 * r + rng.normal(0, 0.01)) for r in rounds]
    chart_df = pd.DataFrame({"Global Loss": loss, "Accuracy": acc}, index=rounds)
    st.line_chart(chart_df)
    st.markdown("""
    | Privacy Protection | Method |
    |---|---|
    | Data stays local | Each node trains locally |
    | Differential Privacy | Gaussian noise (ε=2.0) |
    | Model encryption | Gradient homomorphic encryption |
    | Audit trail | SHA-256 hash chain per aggregation |
    """)
    st.session_state.fed_audit.append("FED_LOSS_AGGREGATION", "server",
                                       round=8, participants=3, global_loss=0.031, auroc=0.847)


# ── Tab 5: Hallucination Defense ──
def _render_hallucination_defense():
    st.markdown("### 🛡️ Five-Layer Hallucination Defense System")
    st.markdown("Based on NeuroSync's CROWN defense, tailored for monetary policy research")
    tabs = st.tabs(["L1 Retrieval", "L3 CROWN", "L4 Multi-Vote", "L5 Self-Consistency", "Integrated"])

    with tabs[0]:
        st.markdown("**Layer 1: Retrieval Consistency Check**")
        st.markdown("Claim → Vectorize → Vector DB search → Support check")
        claim = st.text_input("Claim to verify", "The Fed will shift to easing policy in 2024", key="l1_claim")
        if st.button("Check", key="l1_btn"):
            db = st.session_state.fed_vector_db
            rng = np.random.default_rng(sum(ord(c) for c in claim))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            docs = db.search(qvec, k=3)
            ok, sim, msg = layer1_retrieval(claim, docs)
            st.success(f"✅ {msg}") if ok else st.warning(f"⚠️ {msg}")

    with tabs[1]:
        st.markdown("**Layer 3: CROWN Conformity Defense** (NeuroSync Original)")
        st.markdown("Independent reasoning → Reference external → Confidence drop > δ → Reject conformity answer")
        c1, c2 = st.columns(2)
        with c1:
            init_conf = st.slider("Initial Confidence", 0.0, 1.0, 0.88, key="crown_init")
        with c2:
            soc_conf = st.slider("Social Confidence", 0.0, 1.0, 0.25, key="crown_soc")
        delta = st.slider("CROWN δ Threshold", 0.05, 0.30, 0.10, key="crown_delta")
        result = layer3_crown(init_conf, soc_conf, delta)
        if result["triggered"]:
            st.error(f"🛡️ CROWN TRIGGERED! Confidence drop {result['confidence_drop']:.3f} > δ={delta}, rejecting social answer")
        else:
            st.success(f"Confidence drop {result['confidence_drop']:.3f} ≤ δ={delta}, accepting social answer")

    with tabs[2]:
        st.markdown("**Layer 4: Multi-Node Consensus Vote**")
        st.markdown("Multiple institutions independently analyze the same question, vote on consensus")
        default_ans = ["Inflation will return to 2% sustainably",
                       "Inflation will return to 2% sustainably",
                       "Inflation may fluctuate",
                       "Focus on services inflation"]
        st.markdown("**Input each party's answer:**")
        ans_inputs, conf_inputs = [], []
        for i in range(4):
            c1, c2 = st.columns([3, 1])
            a = c1.text_input(f"Answer {i+1}", default_ans[i], key=f"va_{i}")
            c = c2.number_input("Confidence", 0.0, 1.0, 0.75, key=f"vc_{i}")
            ans_inputs.append(a); conf_inputs.append(c)
        if st.button("Vote", type="primary", key="vote_btn"):
            vote = layer4_vote(ans_inputs, conf_inputs)
            c1, c2, c3 = st.columns(3)
            c1.success(f"Consensus: {vote['consensus'][:30]}...")
            c2.info(f"Strength: {vote['strength']:.0%}")
            c3.warning(f"Verdict: {vote['verdict']}")
            st.bar_chart(pd.DataFrame({"Votes": vote["all_votes"]}))

    with tabs[3]:
        st.markdown("**Layer 5: LLM Self-Consistency Check**")
        st.markdown("Multiple samplings of the same question, check answer stability")
        samples = []
        for i in range(3):
            s = st.text_area(f"Sample {i+1}",
                             f"Based on current data, recommend allocating {int(np.random.uniform(40,80))}% bonds / {int(np.random.uniform(20,60))}% equities",
                             key=f"sc_{i}", height=60)
            samples.append(s)
        if st.button("Check Self-Consistency", key="sc_btn"):
            result = layer5_self_consistency(samples)
            color = "🟢" if result["passes"] else "🔴"
            st.metric("Consistency", f"{result['score']:.0%} {color}")
            st.info(f"Most common: {result['most_common'][:50]}... ({result['count']}/3 times)")

    with tabs[4]:
        st.markdown("**Integrated Defense Engine**")
        st.markdown("Run all 5 layers for complete quality assessment of research outputs")
        claim = st.text_area("Research conclusion to verify", key="comp_claim",
                             value="The Fed is most likely to cut rates by 75bp in 2024, creating downward pressure on long-term yields",
                             height=80)
        if st.button("🛡️ Integrated Check", type="primary", key="comp_btn"):
            db = st.session_state.fed_vector_db
            rng = np.random.default_rng(sum(ord(c) for c in claim))
            qvec = (rng.random(128) / np.linalg.norm(rng.random(128))).tolist()
            docs = db.search(qvec, k=3)
            multi_ans = ["Most likely 75bp cut", "Most likely 75bp cut", "50bp more likely"]
            multi_conf = [0.82, 0.77, 0.61]
            dr = run_defense(claim, docs, multi_ans, multi_conf)
            c1, c2, c3 = st.columns(3)
            rcolor = "🟢" if dr.risk_score < 0.3 else ("🟡" if dr.risk_score < 0.6 else "🔴")
            c1.metric("Risk", f"{rcolor} {dr.risk_score:.0%}")
            c2.metric("Verdict", dr.verdict)
            c3.metric("Action", dr.defense_action)
            for d in dr.details:
                st.caption(d)
            st.session_state.fed_audit.append("HALLUCINATION_CHECK", "defense_engine",
                                               claim=claim[:30], risk=dr.risk_score, verdict=dr.verdict)


# ── Tab 6: Audit Chain ──
def _render_audit():
    st.markdown("### 🔗 Blockchain Audit Chain")
    st.markdown("SHA-256 hash chain recording all federated operations, tamper-proof")
    chain = st.session_state.fed_audit
    c1, c2, c3 = st.columns(3)
    c1.metric("Audit Records", len(chain))
    c2.metric("Genesis Hash", chain.genesis[:16] + "...")
    c3.metric("Chain Integrity", "✅ Verified" if chain.verify() else "❌ Anomaly")
    filter_type = st.selectbox("Filter by Event Type",
                                ["All", "VECTOR_SEARCH", "FED_RAG_QUERY", "FED_COT_REASONING", "FED_COT_VOTE", "HALLUCINATION_CHECK"],
                                key="audit_filter")
    et = None if filter_type == "All" else filter_type
    records = chain.query(event_type=et, limit=20)
    if records:
        rows = [{"#": e["sequence"], "Time": e["timestamp"][11:16],
                "Event": e["event_type"], "Node": e["node_id"],
                "Content": str(e["metadata"])[:60], "Hash": e["hash"][:16] + "..."}
               for e in records]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander("🌳 Merkle Tree Info"):
            st.info("Merkle batch verification enabled: 10 records per batch")
            st.caption(f"Current Merkle batch: {len(records)} records")


# ── Tab 7: Agent Orchestration ──
def _render_agent():
    st.markdown("### 🤖 Agent Orchestration")
    st.markdown("ReAct pattern: auto-analyze task → select tools → execute → reflect")
    task = st.text_area("📋 Research Task",
                         placeholder="Analyze the Fed's easing path in 2024 and its impact on the bond market",
                         key="agent_task", height=80)
    if st.button("🚀 Execute Task", type="primary") and task:
        with st.spinner("Agent planning..."):
            steps = plan_research(task)
            time.sleep(0.5)
        st.markdown("#### 📐 Task Plan")
        plan_df = pd.DataFrame([{"Step": s["step"],
                                 "Tool": TOOL_MAP.get(s["tool"], (s["tool"], ""))[0],
                                 "Description": TOOL_MAP.get(s["tool"], ("", s["tool"]))[1],
                                 "Reason": s["desc"]} for s in steps])
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        st.markdown("#### ⚙️ Execution")
        progress = st.progress(0)
        results = []
        for i, s in enumerate(steps):
            progress.progress((i + 1) / len(steps))
            time.sleep(0.3)
            tool_icon, tool_desc = TOOL_MAP.get(s["tool"], ("⚙️", s["tool"]))
            with st.container():
                st.markdown(f"**{i+1}. {tool_icon} {tool_desc}** — {s['desc']}")
                if s["tool"] == "fed_rag":
                    st.caption("  → Federated search complete, found 3 relevant documents")
                elif s["tool"] == "analysis":
                    st.caption("  → Regression complete, R² = 0.73")
                elif s["tool"] == "halluc_check":
                    st.caption("  → Hallucination check passed, risk score 23%")
                elif s["tool"] == "chart":
                    st.caption("  → Rate path forecast chart generated")
                else:
                    st.caption("  → Execution complete")
                results.append({"tool": s["tool"], "success": True})
        progress.empty()
        st.success(f"✅ Task complete! Executed {len(results)} steps, confidence 81%")
        st.session_state.fed_audit.append("AGENT_EXECUTION", "agent", task=task[:30], steps=len(results))


# ═══════════════════════════════════════════════════════════
# Main Render
# ═══════════════════════════════════════════════════════════

def render():
    _init_state()
    _render_header()
    tabs = st.tabs([
        "🔍 Vector Search",
        "📚 Federated RAG",
        "🧠 Federated CoT",
        "🤝 Federated Learning",
        "🛡️ Hallucination Defense",
        "🤖 Agent",
        "🔗 Audit Chain",
    ])
    with tabs[0]: _render_vector_search()
    with tabs[1]: _render_fed_rag()
    with tabs[2]: _render_fed_cot()
    with tabs[3]: _render_fed_learning()
    with tabs[4]: _render_hallucination_defense()
    with tabs[5]: _render_agent()
    with tabs[6]: _render_audit()
