#!/usr/bin/env python3
"""
Literature Radar — Automated research paper monitoring for Monetary Policy Lab

Searches multiple sources for new papers related to FOMC/monetary policy/NLP,
scores relevance, generates comparison analysis, and maintains a database.

Sources: SSRN, arXiv q-fin, NBER, BIS/IMF/Fed working papers, RePEc
Output: literature_radar.json + daily digest
"""
import json
import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ──
LAB_DIR = Path("/home/z/my-project/monetary-policy-lab")
RADAR_DB = LAB_DIR / "results" / "literature_radar.json"
DIGEST_DIR = LAB_DIR / "results" / "literature_digests"

# Keywords organized by dimension
KEYWORDS = {
    "core": [
        "FOMC", "monetary policy surprise", "central bank communication",
        "federal reserve", "monetary policy shock", "policy rate decision",
    ],
    "method": [
        "sentiment analysis", "NLP", "large language model", "LLM",
        "text analysis", "hawkish dovish", "FinBERT", "GPT",
        "natural language processing", "transformer", "CB-LM",
        "dictionary-based", "bag-of-words",
    ],
    "data": [
        "high-frequency identification", "target path factor", "Kuttner",
        "Gürkaynak Sack Swanson", "GSS", "high-frequency shock",
        "monetary policy surprises", "event study",
    ],
    "extension": [
        "forward guidance", "information shock", "term premium",
        "risk premium channel", "zero lower bound", "ZLB",
        "quantitative easing", "central bank language",
        "monetary policy transmission", "expectations channel",
    ],
}

# Our paper's core findings (for relevance scoring)
OUR_FINDINGS = {
    "target_shock_significant": "Target shock is significant predictor of FOMC sentiment (p=0.017)",
    "path_shock_not_significant": "Path shock is not significant for sentiment (p=0.152)",
    "low_r2": "R² is modest (1.57%), suggesting sentiment captures only a fraction of shock variation",
    "cb_beats_lm": "CB-only score outperforms combined LM+CB (R² 3.90% vs 1.57%)",
    "h4_null": "Forward guidance interaction is null (p=0.991)",
    "dual_channel_null": "Risk premium channel not detectable at daily frequency",
    "forward_looking_mismatch": "Forward-looking sentiment dimension does not improve path shock significance",
    "novelty_weighting": "Statement novelty weighting improves R² by 45%",
}

# Already-cited papers (to avoid duplicates)
OUR_REFERENCES = [
    "Acosta 2022", "Apel Blix 2014", "Bauer Swanson 2023",
    "Blinder et al 2008", "Campbell et al 2012", "Chen Granville Matousek 2025",
    "Christiano et al 1999", "Cieslak et al 2019", "Devlin et al 2019",
    "Federal Reserve Board 2024", "Friedman Schwartz 1963",
    "Gambacorta et al 2024", "Gambacorta et al 2025",
    "Gertler Gilchrist 1994", "Gürkaynak et al 2005",
    "Hansen et al 2018", "Jarociński Karadi 2020",
    "Kuttner 2001", "Loughran McDonald 2011",
    "Nakamura Steinsson 2018", "Newey West 1987",
    "Romer Romer 2004", "Swanson 2021", "Tadle 2022",
    "Weinig 2025", "Yang et al 2020", "Yao Chai 2025",
]


def load_db():
    """Load existing literature radar database."""
    if RADAR_DB.exists():
        with open(RADAR_DB, 'r') as f:
            return json.load(f)
    return {"papers": [], "last_scan": None, "scan_history": []}


def save_db(db):
    """Save literature radar database."""
    RADAR_DB.parent.mkdir(parents=True, exist_ok=True)
    with open(RADAR_DB, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def paper_id(title, authors="", year=""):
    """Generate unique ID for a paper."""
    raw = f"{title}|{authors}|{year}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def score_relevance(title, abstract, source_type="working_paper"):
    """
    Score paper relevance to our research (0-1).
    
    Uses weighted keyword matching with partial credit.
    Core keywords are weighted 3x, method 2x, data 1.5x.
    """
    text = f"{title} {abstract}".lower()
    
    # Flatten all keywords with weights
    weighted_terms = []
    for term in KEYWORDS["core"]:
        weighted_terms.append((term, 3.0))
    for term in KEYWORDS["method"]:
        weighted_terms.append((term, 2.0))
    for term in KEYWORDS["data"]:
        weighted_terms.append((term, 1.5))
    for term in KEYWORDS["extension"]:
        weighted_terms.append((term, 1.0))
    
    # Score: sum of weights for matched terms, normalized
    total_weight = sum(w for _, w in weighted_terms)
    matched_weight = 0
    matched_keywords = []
    
    for term, weight in weighted_terms:
        # Match term or its stem (simple: allow plural/verb forms)
        term_lower = term.lower()
        # Check exact match first
        if term_lower in text:
            matched_weight += weight
            matched_keywords.append(term)
        # Check with common suffixes
        elif term_lower.rstrip('e') + 'ion' in text:  # e.g., communicate -> communication
            matched_weight += weight * 0.8
            matched_keywords.append(f"{term}~")
        elif term_lower + 's' in text:
            matched_weight += weight * 0.9
            matched_keywords.append(f"{term}~")
        elif term_lower + 'ed' in text:
            matched_weight += weight * 0.8
            matched_keywords.append(f"{term}~")
        elif term_lower + 'ing' in text:
            matched_weight += weight * 0.8
            matched_keywords.append(f"{term}~")
    
    # Normalize to 0-1, with diminishing returns (sqrt)
    raw_score = matched_weight / total_weight
    relevance = min(raw_score ** 0.5, 1.0)
    
    # Bonus for working papers (more recent/cutting edge)
    if source_type == "working_paper":
        relevance = min(relevance * 1.15, 1.0)
    
    # Determine impact type
    impact_type = "tangential"
    if any(t in text for t in ["FOMC", "federal reserve", "central bank communication"]):
        if any(t in text for t in ["sentiment", "NLP", "LLM", "text analysis", "language model"]):
            impact_type = "methodology_upgrade"
        elif any(t in text for t in ["surprise", "shock", "high-frequency"]):
            impact_type = "identification_strategy"
        else:
            impact_type = "directly_related"
    elif any(t in text for t in ["sentiment", "NLP", "LLM", "text analysis"]):
        impact_type = "methodology_related"
    
    return round(relevance, 3), impact_type, matched_keywords


def classify_impact(paper, relevance_score):
    """
    Classify how a paper impacts our research.
    Returns: (impact_type, description)
    """
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    
    # Check for direct challenges
    if any(t in text for t in ["narrative surprise", "text-based surprise", "replaces high-frequency"]):
        return ("paradigm_threat", 
                "Challenges our reliance on high-frequency surprise data — text-derived surprises may substitute")
    
    if any(t in text for t in ["risk premium channel", "term premium", "two channels"]):
        return ("methodological_challenge",
                "Suggests our single-equation model misses a risk premium channel")
    
    if any(t in text for t in ["topic decomposition", "topic-specific", "four topics", "dimension"]):
        return ("measurement_improvement",
                "Topic-decomposed sentiment could improve our R² by reducing signal dilution")
    
    if any(t in text for t in ["CB-LM", "central bank language model", "domain-specific model"]):
        return ("upgrade_path",
                "Domain-specific LM offers reproducible upgrade from dictionary approach")
    
    if any(t in text for t in ["uncertainty", "confidence", "calibration"]):
        return ("methodological_improvement",
                "Uncertainty quantification could improve signal-to-noise ratio")
    
    if any(t in text for t in ["cross-country", "ECB", "Bank of Japan", "Bundesbank"]):
        return ("extension_opportunity",
                "Cross-country comparison could test generalizability of our findings")
    
    if any(t in text for t in ["forward guidance", "FG", "zero lower bound", "ZLB"]):
        return ("h4_context",
                "Provides context for our null forward guidance interaction result")
    
    if any(t in text for t in ["GPT-4", "GPT-5", "Claude", "Gemini", "generative AI"]):
        return ("technology_update",
                "New LLM capabilities may improve sentiment measurement")
    
    if relevance_score >= 0.5:
        return ("relevant", "Relevant to our research but no direct methodological challenge")
    
    return ("tangential", "Tangentially related — may be useful for literature review")


def search_ssrn():
    """Search SSRN for new FOMC/monetary policy papers."""
    papers = []
    queries = [
        "FOMC monetary policy sentiment NLP LLM",
        "central bank communication text analysis 2024 2025",
        "monetary policy surprise forward guidance NLP",
        "hawkish dovish sentiment dictionary central bank",
    ]
    
    for query in queries:
        try:
            import subprocess
            result = subprocess.run(
                ["z-ai", "function", "-n", "web_search",
                 "-a", json.dumps({"query": f"site:ssrn.com {query}", "num": 8}),
                 "-o", "/tmp/ssrn_search.json"],
                capture_output=True, text=True, timeout=30
            )
            with open("/tmp/ssrn_search.json") as f:
                data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])
            for r in results:
                if isinstance(r, dict):
                    url = r.get("url", "")
                    if "ssrn.com" in url:
                        papers.append({
                            "title": r.get("name", r.get("title", "")),
                            "url": url,
                            "snippet": r.get("snippet", ""),
                            "source": "SSRN",
                            "source_type": "working_paper",
                        })
        except Exception as e:
            print(f"  SSRN search error: {e}")
        time.sleep(0.5)
    
    return papers


def search_arxiv():
    """Search arXiv q-fin for new papers."""
    papers = []
    queries = [
        "FOMC monetary policy NLP sentiment",
        "central bank communication LLM text analysis",
        "monetary policy surprise identification",
    ]
    
    for query in queries:
        try:
            import subprocess
            result = subprocess.run(
                ["z-ai", "function", "-n", "web_search",
                 "-a", json.dumps({"query": f"site:arxiv.org q-fin {query}", "num": 5}),
                 "-o", "/tmp/arxiv_search.json"],
                capture_output=True, text=True, timeout=30
            )
            with open("/tmp/arxiv_search.json") as f:
                data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])
            for r in results:
                if isinstance(r, dict):
                    url = r.get("url", "")
                    if "arxiv.org" in url:
                        papers.append({
                            "title": r.get("name", r.get("title", "")),
                            "url": url,
                            "snippet": r.get("snippet", ""),
                            "source": "arXiv",
                            "source_type": "working_paper",
                        })
        except Exception as e:
            print(f"  arXiv search error: {e}")
        time.sleep(0.5)
    
    return papers


def search_central_banks():
    """Search BIS, IMF, Fed for new working papers."""
    papers = []
    queries = [
        "BIS working paper monetary policy NLP language model 2024 2025",
        "IMF working paper central bank communication LLM 2025",
        "Federal Reserve FEDS notes FOMC generative AI 2024 2025",
        "Bundesbank monetary policy communication artificial intelligence 2025",
    ]
    
    for query in queries:
        try:
            import subprocess
            result = subprocess.run(
                ["z-ai", "function", "-n", "web_search",
                 "-a", json.dumps({"query": query, "num": 5}),
                 "-o", "/tmp/cb_search.json"],
                capture_output=True, text=True, timeout=30
            )
            with open("/tmp/cb_search.json") as f:
                data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])
            for r in results:
                if isinstance(r, dict):
                    url = r.get("url", "")
                    # Classify source
                    if "bis.org" in url:
                        source = "BIS"
                        stype = "central_bank"
                    elif "imf.org" in url:
                        source = "IMF"
                        stype = "central_bank"
                    elif "federalreserve.gov" in url:
                        source = "Fed"
                        stype = "central_bank"
                    elif "bundesbank.de" in url:
                        source = "Bundesbank"
                        stype = "central_bank"
                    else:
                        source = "Web"
                        stype = "working_paper"
                    papers.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "snippet": r.get("snippet", ""),
                        "source": source,
                        "source_type": stype,
                    })
        except Exception as e:
            print(f"  Central bank search error: {e}")
        time.sleep(0.5)
    
    return papers


def search_journals():
    """Search for newly published journal articles."""
    papers = []
    queries = [
        "Journal of Monetary Economics FOMC sentiment NLP 2024 2025",
        "Journal of Finance central bank communication text 2024 2025",
        "Journal of Financial Economics monetary policy surprise 2024 2025",
        "American Economic Review monetary policy communication 2024 2025",
    ]
    
    for query in queries:
        try:
            import subprocess
            result = subprocess.run(
                ["z-ai", "function", "-n", "web_search",
                 "-a", json.dumps({"query": query, "num": 5}),
                 "-o", "/tmp/journal_search.json"],
                capture_output=True, text=True, timeout=30
            )
            with open("/tmp/journal_search.json") as f:
                data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])
            for r in results:
                if isinstance(r, dict):
                    papers.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", ""),
                        "source": "Journal",
                        "source_type": "top_journal",
                    })
        except Exception as e:
            print(f"  Journal search error: {e}")
        time.sleep(0.5)
    
    return papers


def deduplicate(papers):
    """Remove duplicate papers based on URL and very similar titles."""
    seen_urls = set()
    unique = []
    
    for p in papers:
        # URL dedup
        url = p.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        
        # Skip empty titles
        title = p.get("title", "").strip()
        if not title or len(title) < 10:
            continue
        
        # ID dedup
        pid = paper_id(title, p.get("source", ""))
        p["id"] = pid
        
        # Check if we already have this ID
        existing = [u for u in unique if u["id"] == pid]
        if existing:
            continue
        
        unique.append(p)
    
    return unique


def is_already_cited(title):
    """Check if paper is already in our references."""
    title_lower = title.lower()
    for ref in OUR_REFERENCES:
        ref_parts = ref.lower().split()
        # Check if author name appears in title
        if any(part in title_lower for part in ref_parts if len(part) > 3):
            return True
    return False


def run_scan():
    """Run a full literature radar scan."""
    print("=" * 70)
    print("LITERATURE RADAR — Monetary Policy Lab")
    print(f"Scan started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    db = load_db()
    existing_ids = {p["id"] for p in db["papers"]}
    
    # Search all sources
    print("\n📡 Searching SSRN...")
    ssrn_papers = search_ssrn()
    print(f"   Found {len(ssrn_papers)} results")
    
    print("📡 Searching arXiv...")
    arxiv_papers = search_arxiv()
    print(f"   Found {len(arxiv_papers)} results")
    
    print("📡 Searching central banks (BIS/IMF/Fed)...")
    cb_papers = search_central_banks()
    print(f"   Found {len(cb_papers)} results")
    
    print("📡 Searching journals...")
    journal_papers = search_journals()
    print(f"   Found {len(journal_papers)} results")
    
    # Combine and deduplicate
    all_papers = ssrn_papers + arxiv_papers + cb_papers + journal_papers
    all_papers = deduplicate(all_papers)
    print(f"\n📊 After dedup: {len(all_papers)} unique papers")
    
    # Score and classify
    new_papers = []
    for p in all_papers:
        if p["id"] in existing_ids:
            continue
        
        # Score relevance
        relevance_score, impact_type, matched_kw = score_relevance(
            p.get("title", ""), p.get("snippet", ""), p.get("source_type", "")
        )
        p["relevance_score"] = relevance_score
        p["matched_keywords"] = matched_kw
        
        # Classify impact (use impact_type from score_relevance as primary)
        _, impact_desc = classify_impact(p, p["relevance_score"])
        p["impact_type"] = impact_type
        p["impact_description"] = impact_desc
        
        # Check if already cited
        p["already_cited"] = is_already_cited(p.get("title", ""))
        
        # Add metadata
        p["discovered_date"] = datetime.now().strftime("%Y-%m-%d")
        p["scan_id"] = datetime.now().strftime("%Y%m%d_%H%M")
        
        new_papers.append(p)
    
    # Sort by relevance
    new_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Add to database
    db["papers"].extend(new_papers)
    db["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    db["scan_history"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "new_papers": len(new_papers),
        "high_relevance": len([p for p in new_papers if p["relevance_score"] >= 0.5]),
        "medium_relevance": len([p for p in new_papers if 0.3 <= p["relevance_score"] < 0.5]),
        "low_relevance": len([p for p in new_papers if p["relevance_score"] < 0.3]),
    })
    
    # Keep only last 500 papers
    if len(db["papers"]) > 500:
        db["papers"] = sorted(db["papers"], key=lambda x: x["relevance_score"], reverse=True)[:500]
    
    save_db(db)
    
    # Generate digest
    generate_digest(new_papers)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SCAN COMPLETE")
    print(f"{'=' * 70}")
    print(f"New papers found: {len(new_papers)}")
    
    high = [p for p in new_papers if p["relevance_score"] >= 0.5]
    medium = [p for p in new_papers if 0.3 <= p["relevance_score"] < 0.5]
    
    if high:
        print(f"\n🔴 HIGH RELEVANCE ({len(high)}):")
        for p in high[:10]:
            print(f"   [{p['relevance_score']:.2f}] {p['impact_type']}")
            print(f"   {p['title'][:80]}")
            print(f"   {p['impact_description']}")
            print(f"   {p['url'][:80]}")
            print()
    
    if medium:
        print(f"\n🟡 MEDIUM RELEVANCE ({len(medium)}):")
        for p in medium[:5]:
            print(f"   [{p['relevance_score']:.2f}] {p['title'][:80]}")
            print()
    
    # Alert-worthy papers
    alerts = [p for p in new_papers if p["impact_type"] in 
              ("paradigm_threat", "methodological_challenge", "measurement_improvement")]
    if alerts:
        print(f"\n⚠️  ALERTS ({len(alerts)}):")
        for p in alerts:
            print(f"   {p['impact_type'].upper()}: {p['impact_description']}")
    
    return db


def generate_digest(new_papers):
    """Generate a markdown digest of new papers."""
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    digest_path = DIGEST_DIR / f"digest_{date_str}.md"
    
    lines = [
        f"# Literature Radar Digest — {date_str}",
        "",
        f"New papers found: {len(new_papers)}",
        "",
    ]
    
    # Group by impact type
    impact_groups = {}
    for p in new_papers:
        it = p.get("impact_type", "tangential")
        if it not in impact_groups:
            impact_groups[it] = []
        impact_groups[it].append(p)
    
    # Priority order
    priority = [
        ("paradigm_threat", "🔴 Paradigm Threats"),
        ("methodological_challenge", "🟠 Methodological Challenges"),
        ("measurement_improvement", "🔵 Measurement Improvements"),
        ("upgrade_path", "🟢 Upgrade Paths"),
        ("methodological_improvement", "🟣 Methodological Improvements"),
        ("h4_context", "⚪ H4 Context"),
        ("extension_opportunity", "🔶 Extension Opportunities"),
        ("technology_update", "💻 Technology Updates"),
        ("relevant", "📄 Relevant"),
        ("tangential", "📎 Tangential"),
    ]
    
    for impact_key, label in priority:
        papers = impact_groups.get(impact_key, [])
        if not papers:
            continue
        lines.append(f"## {label} ({len(papers)})")
        lines.append("")
        for p in sorted(papers, key=lambda x: x["relevance_score"], reverse=True):
            cited = " [ALREADY CITED]" if p.get("already_cited") else ""
            lines.append(f"### {p['title']}{cited}")
            lines.append(f"- **Relevance**: {p['relevance_score']:.2f}")
            lines.append(f"- **Source**: {p['source']}")
            lines.append(f"- **Impact**: {p['impact_description']}")
            lines.append(f"- **URL**: {p['url']}")
            if p.get("snippet"):
                lines.append(f"- **Abstract**: {p['snippet'][:300]}...")
            lines.append("")
    
    with open(digest_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"  Digest saved: {digest_path}")


def get_stats():
    """Get literature radar statistics."""
    db = load_db()
    papers = db.get("papers", [])
    
    stats = {
        "total_papers": len(papers),
        "last_scan": db.get("last_scan"),
        "high_relevance": len([p for p in papers if p.get("relevance_score", 0) >= 0.5]),
        "medium_relevance": len([p for p in papers if 0.3 <= p.get("relevance_score", 0) < 0.5]),
        "by_impact_type": {},
        "by_source": {},
    }
    
    for p in papers:
        it = p.get("impact_type", "unknown")
        stats["by_impact_type"][it] = stats["by_impact_type"].get(it, 0) + 1
        src = p.get("source", "unknown")
        stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
    
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats = get_stats()
        print(json.dumps(stats, indent=2))
    else:
        run_scan()
