"""
Step 3: Loughran-McDonald Financial Sentiment Dictionary
No torch/transformers needed - pure Python dictionary approach
"""
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Loughran-McDonald (2011) Master Dictionary - Financial Sentiment
# Subset: most relevant words for monetary policy context
LM_NEGATIVE = {
    "adverse", "against", "bad", "bear", "bearish", "below", "beyond", "bottleneck",
    "break", "breakdown", "breach", "bribery", "bubble", "burden", "caution",
    "cautious", "challenge", "challenged", "collapse", "concern", "concerned",
    "concerns", "constraint", "contagion", "contraction", "correction", "crash",
    "crisis", "critical", "cut", "cutback", "damage", "danger", "dangerous",
    "decline", "declining", "decrease", "decreased", "deficit", "deflation",
    "deflationary", "delay", "delayed", "delinquency", "delinquent", "demand",
    "depressed", "depression", "depreciate", "depreciation", "deteriorate",
    "deterioration", "difficult", "difficulty", "diminish", "diminished", "down",
    "downgrade", "downside", "downturn", "drop", "dropped", "drought", "duty",
    "dysfunction", "excess", "excessive", "fail", "failed", "failure", "fall",
    "falling", "fear", "fears", "flat", "flattened", "flattening", "force",
    "foreclosure", "fraud", "fraught", "freeze", "frozen", "gap", "hardship",
    "hazard", "headwinds", "hesitate", "hesitation", "higher", "highs",
    "hinder", "hindrance", "hit", "hold", "holding", "holdings", "hot",
    "hurricane", "illiquid", "illiquidity", "impact", "impair", "impairment",
    "inadequate", "incidence", "inflate", "inflated", "inflation", "inflationary",
    "insolvency", "insufficient", "interruption", "inventory", "irregularities",
    "jeopardize", "lag", "lagging", "late", "lawsuit", "liabilities",
    "liability", "loss", "losses", "lost", "low", "lower", "lowered",
    "manipulation", "mismatch", "misstatement", "negative", "negatively",
    "negatives", "negotiate", "never", "not", "obstacle", "obstacles",
    "overdue", "overhang", "overheated", "overheating", "overvalued",
    "painful", "penalty", "penalize", "pessimistic", "plunge", "plunged",
    "poor", "poorly", "pressure", "pressured", "problem", "problems",
    "prohibit", "provision", "raise", "raised", "ranging", "rationalize",
    "recession", "recessionary", "reduction", "reductions", "refinance",
    "restructure", "restructuring", "restrict", "restricted", "restriction",
    "restrictions", "revoke", "risk", "risks", "risky", "rivalry", "slow",
    "slowed", "slowing", "slowly", "slowdown", "smaller", "soften",
    "softened", "softening", "squeeze", "stagnant", "stagnation", "strain",
    "strained", "stress", "stressed", "stretches", "struggle", "struggling",
    "subdue", "subdued", "suffer", "suffered", "suffering", "surge",
    "suspend", "suspension", "tension", "tight", "tighten", "tightened",
    "tightening", "too", "tough", "tougher", "trouble", "troubled",
    "uncertain", "uncertainty", "uncertainties", "uncomfortable",
    "undermine", "unemployment", "unease", "uneven", "unexpected",
    "unexpectedly", "unfavorable", "unfounded", "unprecedented", "unstable",
    "volatility", "volatile", "vulnerability", "vulnerable", "weak",
    "weaken", "weakened", "weakening", "weakness", "weaknesses", "worse",
    "worst", "write", "writing", "written",
}

LM_POSITIVE = {
    "achieve", "achieved", "achievement", "advantage", "advantageous",
    "benefit", "beneficial", "better", "bolster", "boost", "boosted",
    "breakthrough", "bullish", "capable", "certainty", "clear", "clarity",
    "comfortable", "commitment", "communicate", "communication", "competent",
    "confident", "confidently", "constructive", "continue", "continued",
    "continuing", "continuity", "control", "controlled", "cooperation",
    "coordination", "correction", "covered", "cure", "curtail", "curtailed",
    "decrease", "decreasing", "dependable", "deploy", "deployed", "deploying",
    "depreciation", "desirable", "develop", "developed", "developing",
    "development", "discipline", "disciplined", "distressed", "diversified",
    "dividend", "dividends", "dollar", "dominate", "dominant", "double",
    "doubled", "durable", "ease", "eased", "easing", "efficient",
    "efficiency", "enhance", "enhanced", "enhancement", "ensure", "ensured",
    "expand", "expanded", "expanding", "expansion", "favorable", "firm",
    "flexibility", "flexible", "flow", "flows", "forecast", "formidable",
    "foundation", "foundations", "gain", "gained", "gains", "growing",
    "growth", "healthy", "higher", "hike", "hiked", "hiking", "improve",
    "improved", "improvement", "improvements", "improving", "increase",
    "increased", "increasing", "increasingly", "innovation", "innovative",
    "intact", "integrate", "integrated", "integrity", "intended", "invest",
    "invested", "investing", "investment", "investments", "investor",
    "investors", "job", "jobs", "lead", "leading", "lean", "leveraged",
    "liquidity", "maintain", "maintained", "manageable", "measured",
    "moderate", "moderated", "moderating", "momentum", "monetary",
    "negotiate", "normalized", "optimal", "optimism", "optimistic",
    "outlook", "outperform", "outperformed", "outperforming", "outperformance",
    "pace", "patient", "patience", "peace", "peaceful", "policy", "positive",
    "positively", "potential", "potentially", "praise", "predictable",
    "preference", "preferred", "prepared", "progress", "progressing",
    "promote", "promoted", "prosper", "prosperity", "prosperous",
    "protect", "protected", "protection", "proven", "prudent", "prudently",
    "quality", "rebalance", "recovery", "recovery", "reduced", "refinance",
    "reform", "reformed", "reforms", "regain", "regained", "reinforce",
    "reiterated", "reliable", "relief", "relieved", "resilience", "resilient",
    "resolve", "resolved", "resource", "resources", "response", "responsive",
    "restore", "restored", "restructuring", "result", "results", "resume",
    "resumed", "retained", "revenue", "revenues", "reward", "rewarding",
    "rise", "rising", "robust", "robustly", "satisfy", "secure", "secured",
    "securities", "security", "shelter", "shield", "significant",
    "significantly", "simplify", "simplified", "simplify", "solid",
    "solution", "solutions", "sought", "source", "sources", "stability",
    "stabilize", "stabilized", "stabilizing", "stable", "steadily",
    "steady", "strength", "strengthen", "strengthened", "strengthening",
    "strengthens", "strong", "stronger", "strongest", "succeed",
    "succeeded", "success", "successful", "successfully", "sufficient",
    "sufficiently", "support", "supported", "supporting", "supportive",
    "sustain", "sustainable", "sustained", "sustaining", "target",
    "targeted", "targets", "tool", "tools", "transparency", "transparent",
    "trend", "trending", "trillion", "trillions", "trust", "trusted",
    "unprecedented", "upgrade", "upgraded", "upgrading", "upside",
    "uptick", "upturn", "upward", "useful", "utilization", "validate",
    "validated", "value", "valued", "values", "vigorous", "well",
    "well-positioned", "willingness", "win", "winning", "worthy",
}

# Central bank specific words (Henry 2008 + custom)
CB_HAWKISH = {
    "tighten", "tightening", "tightened", "restrict", "restricting", "restrictive",
    "hike", "hiking", "hiked", "higher", "elevated", "inflation", "inflationary",
    "overheating", "overheated", "vigilant", "vigilance", "aggressive", "firm",
    "firming", "hawkish", "preemptive", "accelerate", "accelerating", "strong",
    "stronger", "robust", "robustly", "pressures", "upside", "risks",
    "balance", "unwinding", "normalization", "normalize", "normalized",
}

CB_DOVISH = {
    "patient", "patience", "accommodative", "accommodation", "supportive",
    "easing", "eased", "stimulus", "stimulative", "flexible", "flexibility",
    "measured", "gradual", "gradually", "moderate", "moderation", "modest",
    "cautious", "careful", "carefully", "wait", "monitoring", "data",
    "dependent", "uncertainty", "uncertain", "soft", "softening", "subdued",
    "slack", "headwinds", "transitory", "temporary", "downturn", "slowdown",
    "slowing", "stabilize", "stabilizing", "stability", "symmetric",
    "symmetrically", "balanced", "neutral", "appropriate",
}


def compute_lm_sentiment(text):
    """Compute Loughran-McDonald sentiment score for a text."""
    words = text.lower().split()
    words = [w.strip(".,;:!?()[]{}\"'-") for w in words]
    words = [w for w in words if len(w) > 1]
    
    neg_count = sum(1 for w in words if w in LM_NEGATIVE)
    pos_count = sum(1 for w in words if w in LM_POSITIVE)
    total = len(words)
    
    if total == 0:
        return 0.0, 0.0, 0.0, 0
    
    # LM sentiment: (pos - neg) / total
    lm_score = (pos_count - neg_count) / total
    
    # Central bank specific
    hawk_count = sum(1 for w in words if w in CB_HAWKISH)
    dove_count = sum(1 for w in words if w in CB_DOVISH)
    cb_score = (hawk_count - dove_count) / max(total, 1)
    
    # Combined: higher = more hawkish
    combined = 0.5 * lm_score + 0.5 * cb_score
    
    return combined, lm_score, cb_score, total


if __name__ == "__main__":
    test = "The Committee judges that inflation has eased somewhat but remains elevated. The economic outlook is uncertain, and the Committee is attentive to risks. The Committee anticipates that it will be appropriate to maintain a restrictive stance for some time."
    score, lm, cb, n = compute_lm_sentiment(test)
    print(f"Test: {n} words")
    print(f"  LM score: {lm:.4f}")
    print(f"  CB score: {cb:.4f}")
    print(f"  Combined: {score:.4f}")
