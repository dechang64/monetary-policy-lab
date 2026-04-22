"""
FOMC Sentiment Analysis Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from analysis.nlp_engine import FOMCSentimentEngine
from visualization.charts import sentiment_trajectory


def render():
    st.markdown('<div class="main-header"><h1>💬 FOMC Sentiment Analysis</h1><p>NLP-powered analysis of Federal Reserve communications</p></div>', unsafe_allow_html=True)
    
    # ── Sample FOMC Statements ──
    SAMPLE_STATEMENTS = {
        "2024-12-18 (Hawkish)": """
        The Committee judges that the risks to achieving its employment and inflation 
        goals remain roughly in balance. However, inflation remains somewhat elevated. 
        The Committee does not expect it will be appropriate to reduce the target range 
        until it has gained greater confidence that inflation is moving sustainably 
        toward 2 percent. The economic outlook is uncertain, and the Committee remains 
        attentive to inflation risks. The Committee would be prepared to adjust the 
        stance of monetary policy as appropriate if risks emerge that could impede 
        progress toward the Committee's goals.
        """,
        "2020-03-15 (Emergency Cut)": """
        The Federal Reserve is committed to using its full range of tools to support 
        the U.S. economy in this challenging time. The coronavirus outbreak has harmed 
        communities and disrupted economic activity in many countries, including the 
        United States. The Committee expects to maintain this target range until it 
        is confident that the economy has weathered recent events and is on track to 
        achieve its maximum employment and price stability goals. The Federal Reserve 
        will continue to closely monitor market conditions and is prepared to use its 
        full range of tools to support the flow of credit to households and businesses.
        """,
        "2019-07-31 (Mid-Cycle Adjustment)": """
        Information received since the Federal Open Market Committee met in June 
        indicates that the labor market remains strong and that economic activity 
        has been rising at a moderate rate. Job gains have been solid, on average, 
        in recent months, and the unemployment rate has remained low. Although 
        growth of household spending has picked up from earlier in the year, growth 
        of business fixed investment has been soft. On a 12-month basis, overall 
        inflation and inflation for items other than food and energy are running 
        below 2 percent. In light of the implications of global developments for 
        the economic outlook as well as muted inflation pressures, the Committee 
        decided to lower the target range.
        """,
        "2022-06-15 (Aggressive Tightening)": """
        Inflation remains elevated, reflecting supply and demand imbalances related 
        to the pandemic, higher energy prices, and broader price pressures. The 
        Committee is strongly committed to returning inflation to its 2 percent 
        objective. The Committee decided to raise the target range for the federal 
        funds rate to 1-1/2 to 1-3/4 percent and anticipates that ongoing increases 
        in the target range will be appropriate. In addition, the Committee decided 
        to begin reducing the size of the Federal Reserve's balance sheet on June 1. 
        The Committee is determined to take the measures necessary to restore price 
        stability.
        """,
        "2023-01-25 (Slowing Pace)": """
        The Committee seeks to achieve maximum employment and inflation at the rate 
        of 2 percent over the longer run. In support of these goals, the Committee 
        decided to raise the target range for the federal funds rate to 4-1/2 to 
        4-3/4 percent. The Committee anticipates that ongoing increases in the target 
        range will be appropriate in order to attain a stance of monetary policy that 
        is sufficiently restrictive to return inflation to 2 percent over time. 
        In determining the extent of future increases, the Committee will take into 
        account the cumulative tightening of monetary policy.
        """,
    }
    
    # ── Mode Selection ──
    mode = st.radio("Analysis Mode", ["📝 Text Input", "📚 Sample Statements"], horizontal=True)
    
    engine = FOMCSentimentEngine()
    
    if mode == "📝 Text Input":
        text_input = st.text_area(
            "Paste FOMC statement or any central bank communication:",
            height=200,
            placeholder="Paste text here...",
        )
        
        if text_input and st.button("🔍 Analyze", type="primary"):
            result = engine.analyze(text_input)
            _display_result(result, text_input)
    
    else:
        selected = st.selectbox("Select Sample Statement", list(SAMPLE_STATEMENTS.keys()))
        text = SAMPLE_STATEMENTS[selected]
        
        st.text_area("Statement Text", text, height=200, disabled=True)
        
        if st.button("🔍 Analyze", type="primary"):
            result = engine.analyze(text)
            _display_result(result, text)
    
    # ── Historical Sentiment Trajectory ──
    st.markdown("---")
    st.markdown("### 📈 Historical Sentiment Trajectory")
    st.caption("Sentiment scores for sample FOMC statements")
    
    if st.button("Generate Trajectory", use_container_width=True):
        dates = []
        scores = []
        labels = []
        
        for date_str, text in SAMPLE_STATEMENTS.items():
            result = engine.analyze(text)
            dates.append(date_str.split(" ")[0])
            scores.append(result["sentiment_score"])
            labels.append(result["label"])
        
        trajectory_df = pd.DataFrame({
            "date": dates,
            "sentiment_score": scores,
            "label": labels,
        })
        
        fig = sentiment_trajectory(trajectory_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detail table
        st.markdown("### Detailed Scores")
        detail_df = pd.DataFrame({
            "Date": dates,
            "Sentiment": labels,
            "Score": [round(s, 3) for s in scores],
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    
    # ── Methodology ──
    with st.expander("🔧 NLP Methodology"):
        st.markdown("""
        **Current Mode**: Rule-based keyword matching (fast, no GPU)
        
        **For publication-quality results:**
        
        1. **FinBERT-FOMC** (ProsusAI/finbert fine-tuned on FOMC text)
           ```python
           from transformers import pipeline
           classifier = pipeline("sentiment-analysis", 
                                model="ProsusAI/finbert")
           result = classifier(fomc_text)
           ```
        
        2. **FinBERT-FOMC (Domain-Specific)**
           - Fine-tune FinBERT on labeled FOMC statements
           - Labels: Hawkish / Neutral / Dovish
           - Achieves ~85% accuracy on FOMC text
        
        3. **LLM-Based Analysis** (GPT-4 / Llama-3)
           - Zero-shot: "Rate the hawkishness of this statement on 1-10"
           - Few-shot: Provide examples of hawkish/dovish statements
           - Chain-of-thought: "Explain your reasoning step by step"
        
        4. **Topic Modeling** (BERTopic)
           - Identify key themes in FOMC communications
           - Track topic evolution over time
           - Correlate topics with market reactions
        
        **Recommended**: Start with rule-based for exploration, 
        then validate with FinBERT-FOMC for the paper.
        """)


def _display_result(result, text):
    """Display sentiment analysis results."""
    col1, col2, col3 = st.columns(3)
    
    score = result["sentiment_score"]
    label = result["label"]
    
    color = "#e74c3c" if label == "Hawkish" else ("#27ae60" if label == "Dovish" else "#95a5a6")
    
    with col1:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;background:white;border-radius:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);border-top:4px solid {color};">
            <div style="font-size:0.8rem;color:#888;">Sentiment</div>
            <div style="font-size:1.5rem;font-weight:700;color:{color};">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;background:white;border-radius:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="font-size:0.8rem;color:#888;">Score</div>
            <div style="font-size:1.5rem;font-weight:700;">{score:.3f}</div>
            <div style="font-size:0.7rem;color:#888;">-1 = Dovish, +1 = Hawkish</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;background:white;border-radius:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="font-size:0.8rem;color:#888;">Readability</div>
            <div style="font-size:1.5rem;font-weight:700;">{result['readability']['complexity']}</div>
            <div style="font-size:0.7rem;color:#888;">FK Grade: {result['readability']['flesch_kincaid']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Keyword breakdown
    st.markdown("### Keyword Breakdown")
    kw_col1, kw_col2 = st.columns(2)
    
    with kw_col1:
        st.markdown(f"🔴 **Hawkish Signals** ({len(result['hawkish_found'])} found)")
        if result["hawkish_found"]:
            for kw in result["hawkish_found"]:
                st.markdown(f"- `{kw}`")
        else:
            st.caption("No hawkish keywords detected")
    
    with kw_col2:
        st.markdown(f"🟢 **Dovish Signals** ({len(result['dovish_found'])} found)")
        if result["dovish_found"]:
            for kw in result["dovish_found"]:
                st.markdown(f"- `{kw}`")
        else:
            st.caption("No dovish keywords detected")
    
    # Readability details
    with st.expander("📖 Readability Metrics"):
        r = result["readability"]
        st.json(r)
