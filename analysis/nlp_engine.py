"""
NLP Sentiment Engine for FOMC Communications
=============================================
Analyze FOMC statements, minutes, and press conferences using NLP.

In production: load FinBERT-FOMC (ProsusAI/finbert) fine-tuned on FOMC text.
This module provides the framework and a rule-based fallback.
"""

import numpy as np
import pandas as pd
import re
from typing import Optional


class FOMCSentimentEngine:
    """
    Analyze sentiment of FOMC communications.
    
    Modes:
    - Rule-based: keyword matching (fast, no GPU needed)
    - FinBERT-FOMC: transformer model (accurate, needs GPU)
    - LLM: GPT-4 / Llama for deep semantic analysis
    """
    
    # Hawkish keywords (signal tightening)
    HAWKISH = [
        "inflationary pressures", "elevated inflation", "inflation remains",
        "tighten", "tightening", "restrictive", "restrictive stance",
        "strong labor market", "robust growth", "strong growth",
        "above target", "upside risks", "inflation expectations",
        "need to act", "additional firming", "further rate increases",
        "higher for longer", "no rate cuts", "premature to cut",
        "remain vigilant", "inflation is still too high",
    ]
    
    # Dovish keywords (signal easing)
    DOVISH = [
        "moderate growth", "moderating", "slowing", "slowdown",
        "downside risks", "uncertainty", "uncertainties",
        "accommodative", "support growth", "patient", "patient approach",
        "appropriate to cut", "rate cuts", "easing",
        "labor market cooling", "soft landing", "balanced approach",
        "data dependent", "incoming data", "carefully monitor",
        "downward pressure", "disinflation", "progress on inflation",
    ]
    
    # Forward guidance keywords
    FORWARD_GUIDANCE = [
        "anticipate", "expect to", "projected to", "likely to",
        "appropriate to", "intend to", "prepared to",
        "outlook", "projections", "forward guidance",
        "conditional on", "depending on", "subject to",
    ]
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.mode = "rule-based"
    
    def load_finbert(self):
        """
        Load FinBERT-FOMC model.
        In production: uncomment and ensure transformers + torch installed.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            )
            self.mode = "finbert"
            return True
        except Exception as e:
            print(f"Failed to load FinBERT: {e}. Falling back to rule-based.")
            self.mode = "rule-based"
            return False
    
    def analyze(self, text: str) -> dict:
        """Alias for analyze_statement (backward compat)."""
        return self.analyze_statement(text)

    def analyze_statement(self, text: str) -> dict:
        """
        Analyze a single FOMC statement.
        Returns sentiment score, label, keyword breakdown, and readability.
        """
        if self.mode == "finbert" and self.model:
            result = self._finbert_analyze(text)
        else:
            result = self._rule_based_analyze(text)

        # Add readability
        result["readability"] = self.readability_analysis(text)

        # Compatibility aliases
        result["score"] = result["sentiment_score"]
        result["label"] = result["sentiment_label"]
        result["hawkish_found"] = result.get("hawkish_keywords", [])
        result["dovish_found"] = result.get("dovish_keywords", [])

        return result
    
    def _rule_based_analyze(self, text: str) -> dict:
        """Rule-based sentiment analysis using keyword matching."""
        text_lower = text.lower()
        
        hawkish_count = sum(1 for kw in self.HAWKISH if kw in text_lower)
        dovish_count = sum(1 for kw in self.DOVISH if kw in text_lower)
        fg_count = sum(1 for kw in self.FORWARD_GUIDANCE if kw in text_lower)
        
        total = hawkish_count + dovish_count
        if total == 0:
            score = 0
        else:
            score = (dovish_count - hawkish_count) / total  # -1 to 1
        
        # Weight by forward guidance intensity
        fg_weight = 1 + 0.1 * fg_count
        score *= min(fg_weight, 1.5)
        score = np.clip(score, -1, 1)
        
        # Find matched keywords
        hawkish_found = [kw for kw in self.HAWKISH if kw in text_lower]
        dovish_found = [kw for kw in self.DOVISH if kw in text_lower]
        fg_found = [kw for kw in self.FORWARD_GUIDANCE if kw in text_lower]
        
        return {
            "sentiment_score": round(score, 3),
            "sentiment_label": "Hawkish" if score < -0.15 else ("Dovish" if score > 0.15 else "Neutral"),
            "hawkish_count": hawkish_count,
            "dovish_count": dovish_count,
            "fg_count": fg_count,
            "hawkish_keywords": hawkish_found,
            "dovish_keywords": dovish_found,
            "fg_keywords": fg_found,
            "word_count": len(text.split()),
            "mode": "rule-based",
        }
    
    def _finbert_analyze(self, text: str) -> dict:
        """FinBERT-based sentiment analysis."""
        import torch
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        
        # FinBERT labels: 0=positive, 1=negative, 2=neutral
        label_map = {0: "Dovish", 1: "Hawkish", 2: "Neutral"}
        score_map = {0: 1.0, 1: -1.0, 2: 0.0}
        
        return {
            "sentiment_score": score_map[pred],
            "sentiment_label": label_map[pred],
            "probabilities": {
                "Dovish": round(probs[0][0].item(), 3),
                "Hawkish": round(probs[0][1].item(), 3),
                "Neutral": round(probs[0][2].item(), 3),
            },
            "word_count": len(text.split()),
            "mode": "finbert",
        }
    
    def batch_analyze(self, statements) -> pd.DataFrame:
        """
        Analyze a batch of FOMC statements.

        statements: DataFrame with columns ['date', 'text'],
                    or list of strings (returns list of dicts).
        """
        results = []
        if isinstance(statements, list):
            for text in statements:
                results.append(self.analyze_statement(text))
            return results

        for _, row in statements.iterrows():
            analysis = self.analyze_statement(row["text"])
            analysis["date"] = row["date"]
            results.append(analysis)

        return pd.DataFrame(results).set_index("date")
    
    def sentiment_change_analysis(
        self, sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute sentiment changes between consecutive FOMC meetings.
        Large changes often predict market reactions.
        """
        df = sentiment_df.copy()
        df["prev_score"] = df["sentiment_score"].shift(1)
        df["sentiment_change"] = df["sentiment_score"] - df["prev_score"]
        df["abs_change"] = df["sentiment_change"].abs()
        df["direction"] = df["sentiment_change"].apply(
            lambda x: "More Hawkish" if x < -0.1 else ("More Dovish" if x > 0.1 else "No Change")
        )
        
        return df
    
    def readability_analysis(self, text: str) -> dict:
        """
        Compute readability metrics for FOMC statements.
        Following the literature on central bank communication clarity.
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Syllable count (approximate)
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = "aeiou"
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    count += 1
            if word.endswith("e") and count > 1:
                count -= 1
            return max(count, 1)
        
        total_syllables = sum(count_syllables(w) for w in words)
        
        # Flesch-Kincaid Grade Level
        n_words = len(words)
        n_sentences = max(len(sentences), 1)
        fk_grade = 0.39 * (n_words / n_sentences) + 11.8 * (total_syllables / n_words) - 15.59
        
        # Fog Index
        complex_words = sum(1 for w in words if count_syllables(w) >= 3)
        fog = 0.4 * (n_words / n_sentences + 100 * complex_words / n_words)
        
        return {
            "word_count": n_words,
            "sentence_count": n_sentences,
            "avg_words_per_sentence": round(n_words / n_sentences, 1),
            "flesch_kincaid": round(fk_grade, 1),
            "fog_index": round(fog, 1),
            "complexity": "High" if fk_grade > 15 else ("Medium" if fk_grade > 10 else "Low"),
        }
