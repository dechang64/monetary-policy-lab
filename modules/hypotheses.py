#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hypotheses & Unified Framework page (Proposal v2, Oct 2026).

Presents the six-hypothesis system from Eileen's research proposal v2
alongside the unified theoretical framework (three layers of information,
two transmission channels, one institutional friction), and maps each
proposal hypothesis to the empirical test modules on this site so the
paper-internal numbering no longer clashes with the proposal numbering.
"""
import streamlit as st

RED = "#CC0033"
BLUE = "#1F4E79"
GOLD = "#BF8F00"
GREY = "#5A5A5A"

HYP_CSS = f"""
<style>
    .hyp-card {{
        background: #FFFFFF; border-radius: 10px; padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid {RED};
        margin-bottom: 0.6rem; height: 100%;
    }}
    .hyp-card.blue {{ border-left-color: {BLUE}; }}
    .hyp-card.gold {{ border-left-color: {GOLD}; }}
    .hyp-card h4 {{ margin: 0 0 0.4rem 0; font-size: 1.02rem; }}
    .hyp-card .hyp-q {{ font-size: 0.86rem; color: #262626; margin: 0.25rem 0; }}
    .hyp-card .hyp-model {{ font-size: 0.8rem; color: #333; background: #F7F7F8;
        border-radius: 6px; padding: 0.35rem 0.55rem; margin: 0.45rem 0; }}
    .hyp-card .hyp-anchor {{ font-size: 0.76rem; font-style: italic; color: {GREY}; margin: 0.35rem 0 0 0; }}
    .hyp-card .hyp-where {{ font-size: 0.76rem; margin: 0.45rem 0 0 0; }}
    .fw-card {{ background: #F8F9FB; border-radius: 10px; padding: 0.95rem 1.1rem;
        border: 1px solid #E3E6EA; height: 100%; }}
    .fw-card h5 {{ margin: 0 0 0.45rem 0; font-size: 0.95rem; }}
    .fw-card ul {{ margin: 0; padding-left: 1.1rem; font-size: 0.83rem; color: #333; }}
    .fw-card li {{ margin-bottom: 0.3rem; }}
</style>
"""

FRAMEWORK_HTML = f"""
<div class="fw-card">
<h5 style="color:{BLUE}">Three layers of information</h5>
<ul>
<li><b>Policy action</b> → Target surprise</li>
<li><b>Policy path</b> → Path surprise</li>
<li><b>Economic narrative</b> → Sentiment — <i>not spanned by high-frequency surprises</i></li>
</ul>
<p style="font-size:0.76rem;color:{GREY};margin:0.5rem 0 0 0">Tests H1 · H2 · H3</p>
</div>
"""

FRAMEWORK_HTML2 = f"""
<div class="fw-card">
<h5 style="color:{RED}">Two transmission channels</h5>
<ul>
<li><b>Discount-rate channel</b> — MP shocks reprice assets immediately</li>
<li><b>Information channel</b> — CBI shocks update beliefs; prices move on the day, portfolios follow with a delay</li>
</ul>
<p style="font-size:0.76rem;color:{GREY};margin:0.5rem 0 0 0">Tests H4 · H6</p>
</div>
"""

FRAMEWORK_HTML3 = f"""
<div class="fw-card">
<h5 style="color:{GOLD}">One institutional friction</h5>
<ul>
<li>Decision cycles · risk controls · settlement lags</li>
<li>Institutional rebalancing lags prices → <b>price leads, flow lags</b></li>
</ul>
<p style="font-size:0.76rem;color:{GREY};margin:0.5rem 0 0 0">Tests H5 · H6</p>
</div>
"""


def _card(hid, title, color_class, hypothesis, model, anchor, where):
    cls = f"hyp-card {color_class}" if color_class else "hyp-card"
    st.markdown(
        f"""
<div class="{cls}">
<h4>{hid}: {title}</h4>
<div class="hyp-q">{hypothesis}</div>
<div class="hyp-model">{model}</div>
<p class="hyp-anchor">Theory anchor: {anchor}</p>
<p class="hyp-where">🧪 {where}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render():
    st.markdown(HYP_CSS, unsafe_allow_html=True)

    st.markdown("## 🧩 Hypotheses & Unified Framework")
    st.caption("Proposal v2 (October 2026) · two papers, one system: Part 1 — Words Beyond the Rate (H1–H3) · Part 2 — Information vs. Action (H4–H6)")

    st.markdown("---")

    # ── Unified framework ──
    st.markdown("### A Unified Theoretical Framework")
    st.markdown("Two channels · three layers of information · one institutional friction — every hypothesis sits somewhere on this map.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(FRAMEWORK_HTML, unsafe_allow_html=True)
    with c2:
        st.markdown(FRAMEWORK_HTML2, unsafe_allow_html=True)
    with c3:
        st.markdown(FRAMEWORK_HTML3, unsafe_allow_html=True)

    st.markdown("---")

    # ── Six hypothesis cards ──
    st.markdown("### The Six Hypotheses")

    row1 = st.columns(2)
    with row1[0]:
        _card(
            "H1", "Statement language", "",
            "FOMC statement sentiment is more closely associated with the <b>Target</b> surprise than with the <b>Path</b> surprise.",
            "Model 1: Sentiment = α + β₁·Target + β₂·Path + ε&nbsp;&nbsp;(Wald: β₁ = β₂)<br>H1b: sentiment does not distinguish MP from CBI shocks",
            "GSS (2005); Bauer–Swanson (2023) target dominance",
            "Tested in <b>📊 Research Results (v10.3)</b>",
        )
    with row1[1]:
        _card(
            "H2", "Incremental information", "blue",
            "Statement sentiment explains asset-price responses <b>beyond</b> Target and Path surprises.",
            "Model 2: Return = α + β₁·Target + β₂·Path + β₃·Sentiment + ε",
            "Romer &amp; Romer (2000); Nakamura &amp; Steinsson (2018) — narrative carries central-bank private information",
            "Tested in <b>📊 Research Results (v10.3)</b>",
        )

    row2 = st.columns(2)
    with row2[0]:
        _card(
            "H3", "State dependence", "gold",
            "The incremental effect of statement sentiment varies across policy regimes, particularly under forward guidance (Dec 2008–Dec 2015).",
            "Model 3: R = α + β₁·Target + β₂·Path + β₃·Sentiment + β₄(Sentiment × FG) + β₅·FG + ε",
            "Eggertsson &amp; Woodford (2003); Swanson (2021) — communication substitutes for action at the ZLB",
            "Tested in <b>📊 Research Results (v10.3)</b>",
        )
    with row2[1]:
        _card(
            "H4", "MP vs. CBI", "",
            "Monetary-policy shocks and central-bank-information shocks generate <b>different mutual-fund flow responses</b>.",
            "Model 4: Flow(c, w) = α + β₁·MPₜ + β₂·CBIₜ + γ′Controls + ε&nbsp;&nbsp;(Wald: β₁ = β₂ per asset class, for w = Same, Post, Diff)<br>Zero-rate check: adds MP×ZLB and CBI×ZLB",
            "Jarociński &amp; Karadi (2020); Koijen &amp; Yogo (2019); Froot et al. (2001)",
            "Tested in <b>🎯 Direction 2</b> → §3 MP vs. CBI Differential Effects",
        )

    row3 = st.columns(2)
    with row3[0]:
        _card(
            "H5", "Risk ladder", "blue",
            "MP and CBI shocks produce <b>systematically different reallocations</b> across asset classes ordered by risk.",
            "Model 5: Flow = α + β₁·MPₜ + β₂·CBIₜ + β₃·(Risk×MP) + β₄·(Risk×CBI) + γ′Controls + ε — predict β₃ &lt; 0, β₄ &gt; 0",
            "Risk-taking channel — Adrian &amp; Shin (2010); Borio &amp; Zhu (2012)",
            "Tested in <b>🎯 Direction 2</b> → risk-ladder panel (Risk × MP, Risk × CBI) · §4 substitution matrix (supporting)",
        )
    with row3[1]:
        _card(
            "H6", "Transmission timing", "gold",
            "MP and CBI shocks move prices in <b>opposite directions</b> on the announcement day; the CBI price effect fades before portfolio flows adjust.",
            "Model 6A: R(i,t) = α + β₁·MPₜ + β₂·CBIₜ + ε (same-day)<br>Model 6B: Flow(c,t→t+1) = α + γ₁·MPₜ + γ₂·CBIₜ + δ′Controls + ε (subsequent)",
            "Jarociński &amp; Karadi (2020) sign restrictions; investor frictions → price leads, flow lags",
            "Tested in <b>🎯 Direction 2</b> → §2 Event Windows · §8 Cross-Paper Comparison",
        )

    st.markdown("---")

    # ── Numbering map ──
    st.markdown("### Numbering Map")
    st.markdown(
        "The proposal reorganized the research program into six hypotheses (October 2026). "
        "Paper-internal labels and on-site tests use their own numbering — this table maps them."
    )
    st.markdown(
        """
| Proposal (Oct 2026) | Paper-internal label | Where on this site |
|---|---|---|
| H1 Statement language | Part 1 · Model 1 (+ H1b) | Research Results → H1 |
| H2 Incremental information | Part 1 · Model 2 | Research Results → H2 (Incremental R²) |
| H3 State dependence | Part 1 · Model 4 (FG interaction, Dec 2008–Dec 2015) | Research Results → FG highlight |
| H4 MP vs. CBI flows | D2 paper · differential-effects test | Direction 2 → §3 |
| H5 Risk ladder | D2 paper · risk-ladder / substitution test | Direction 2 → §4 |
| H6 Transmission timing | D2 paper · event-window test | Direction 2 → §2 · §8 |
""",
        unsafe_allow_html=True,
    )
    st.info(
        "ℹ️ Direction 2's on-page section numbers (§2, §3, §4, §8) are this site's presentation "
        "order, and the D2 paper's internal H-labels (H1/H3/H5) refer to its own hypothesis list — "
        "neither conflicts with the proposal numbering above.",
        icon="🧭",
    )
