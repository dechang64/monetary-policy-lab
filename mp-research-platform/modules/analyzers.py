"""Analyzers — core analysis engines for each research module."""

import numpy as np
import pandas as pd
from collections import Counter
import re


class EventStudyEngine:
    """Event study analysis: abnormal returns, cumulative returns, statistics."""

    def compute_summary(self, event_assets, asset_names):
        """Compute summary statistics for selected assets."""
        rows = []
        for asset in asset_names:
            if asset not in event_assets:
                continue
            df = event_assets[asset]
            # CAR at different horizons
            car_30 = df.loc[df["minute"] == 30, "cumulative_return_pct"].values
            car_60 = df.loc[df["minute"] == 60, "cumulative_return_pct"].values
            car_120 = df.loc[df["minute"] == 120, "cumulative_return_pct"].values

            car_30 = car_30[0] if len(car_30) > 0 else np.nan
            car_60 = car_60[0] if len(car_60) > 0 else np.nan
            car_120 = car_120[0] if len(car_120) > 0 else np.nan

            # Max drawdown in post-event window
            post = df[df["minute"] >= 0]["cumulative_return_pct"]
            running_max = post.cummax()
            drawdown = post - running_max
            max_dd = drawdown.min() if len(drawdown) > 0 else 0

            # Volatility
            vol = df["return_pct"].diff().std()

            rows.append({
                "Asset": asset,
                "CAR [+30min]": f"{car_30:+.2f}%",
                "CAR [+60min]": f"{car_60:+.2f}%",
                "CAR [+120min]": f"{car_120:+.2f}%",
                "Max Drawdown": f"{max_dd:.2f}%",
                "Volatility": f"{vol:.3f}",
            })
        return pd.DataFrame(rows)


class TwoShocksEngine:
    """Two-shocks decomposition: policy vs information shock analysis."""

    def variance_decomposition(self, assets):
        """Simulate variance decomposition for each asset."""
        # Based on empirical findings:
        # Equities: ~55% policy, ~45% information
        # Bonds: ~70% policy, ~30% information
        # FX: ~65% policy, ~35% information
        # VIX: ~40% policy, ~60% information
        # Gold: ~50% policy, ~50% information
        base = {
            "S&P 500": (55, 45),
            "10Y Treasury": (70, 30),
            "DXY": (65, 35),
            "VIX": (40, 60),
            "Gold": (50, 50),
        }
        policy_pcts = []
        info_pcts = []
        for a in assets:
            p, i = base.get(a, (55, 45))
            policy_pcts.append(p + np.random.normal(0, 3))
            info_pcts.append(i + np.random.normal(0, 3))

        # Normalize to 100
        total = [p + i for p, i in zip(policy_pcts, info_pcts)]
        policy_pcts = [p / t * 100 for p, t in zip(policy_pcts, total)]
        info_pcts = [i / t * 100 for i, t in zip(info_pcts, total)]

        return {"policy_pct": policy_pcts, "info_pct": info_pcts}


class NLPEngine:
    """NLP analysis of FOMC statements: sentiment, readability, key phrases."""

    # Hawkish/dovish keyword dictionaries
    HAWKISH_WORDS = [
        "increase", "raise", "tighten", "inflation", "elevated", "strongly committed",
        "restrictive", "additional", "accelerate", "vigilant", "upside risk",
        "overheating", "wage pressure", "price stability",
    ]
    DOVISH_WORDS = [
        "lower", "reduce", "accommodative", "patient", "gradual", "moderate",
        "appropriate", "support", "uncertainty", "downside risk", "easing",
        "flexible", "balanced", "sustain", "maintain",
    ]

    def analyze_pair(self, text_a, text_b):
        """Compare two FOMC statements."""
        sent_a = self._sentiment(text_a)
        sent_b = self._sentiment(text_b)
        read_a = self._readability(text_a)
        read_b = self._readability(text_b)
        changes = self._diff_phrases(text_a, text_b)

        return {
            "sentiment_a": sent_a,
            "sentiment_b": sent_b,
            "readability_a": read_a,
            "readability_b": read_b,
            "key_changes": changes,
        }

    def _sentiment(self, text):
        words = text.lower().split()
        hawk_count = sum(1 for w in words if any(h in w for h in self.HAWKISH_WORDS))
        dove_count = sum(1 for w in words if any(d in w for d in self.DOVISH_WORDS))
        total = max(hawk_count + dove_count, 1)

        return {
            "hawkish_score": round(hawk_count / total, 3),
            "dovish_score": round(dove_count / total, 3),
            "net_tone": round((hawk_count - dove_count) / total, 3),
            "hawkish_words": hawk_count,
            "dovish_words": dove_count,
        }

    def _readability(self, text):
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        avg_sentence = words / max(sentences, 1)
        # Flesch-like score (simplified)
        syllables = sum(self._count_syllables(w) for w in text.split())
        flesch = 206.835 - 1.015 * (words / max(sentences, 1)) - 84.6 * (syllables / max(words, 1))

        return {
            "avg_sentence_length": round(avg_sentence, 1),
            "word_count": words,
            "sentence_count": sentences,
            "flesch_score": round(max(0, min(100, flesch)), 1),
        }

    def _count_syllables(self, word):
        word = word.lower().strip(".,;:!?")
        if len(word) <= 3:
            return 1
        count = len(re.findall(r'[aeiouy]+', word))
        return max(1, count)

    def _diff_phrases(self, text_a, text_b):
        """Identify key phrase changes between two statements."""
        changes = []
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        added = words_b - words_a
        removed = words_a - words_b

        # Filter to meaningful words (>3 chars)
        for w in sorted(added):
            if len(w) > 4 and w.isalpha():
                changes.append({"type": "added", "phrase": w, "context": "New emphasis in Statement B"})
        for w in sorted(removed):
            if len(w) > 4 and w.isalpha():
                changes.append({"type": "removed", "phrase": w, "context": "Dropped from Statement A"})

        return changes[:8]  # Top changes


class PortfolioEngine:
    """Portfolio rebalancing simulation."""

    BASE_ALLOCATIONS = {
        "Mutual Funds": {"US Equities": 45, "Intl Equities": 20, "IG Bonds": 20, "HY Bonds": 5, "Cash": 10},
        "Hedge Funds": {"US Equities": 25, "Intl Equities": 15, "IG Bonds": 10, "HY Bonds": 10, "Cash": 5, "Alternatives": 35},
        "Pension Funds": {"US Equities": 35, "Intl Equities": 15, "IG Bonds": 35, "HY Bonds": 5, "Cash": 5, "Real Assets": 5},
        "Foreign Investors": {"US Equities": 30, "Treasuries": 40, "Corp Bonds": 15, "AGencies": 10, "Cash": 5},
        "Retail": {"US Equities": 40, "Intl Equities": 10, "Bonds": 25, "Cash": 20, "Alternatives": 5},
    }

    SHOCK_EFFECTS = {
        "+25bp Hawkish Surprise": {"US Equities": -3, "Intl Equities": -2, "IG Bonds": -1, "HY Bonds": -2, "Cash": 3, "Treasuries": 2, "Corp Bonds": -1, "AGencies": 1, "Real Assets": -1, "Alternatives": -1},
        "+50bp Hawkish Surprise": {"US Equities": -5, "Intl Equities": -4, "IG Bonds": -2, "HY Bonds": -4, "Cash": 5, "Treasuries": 3, "Corp Bonds": -3, "AGencies": 2, "Real Assets": -2, "Alternatives": -2},
        "-25bp Dovish Surprise": {"US Equities": 3, "Intl Equities": 2, "IG Bonds": 1, "HY Bonds": 3, "Cash": -3, "Treasuries": -1, "Corp Bonds": 2, "AGencies": -1, "Real Assets": 1, "Alternatives": 1},
        "-50bp Dovish Surprise": {"US Equities": 5, "Intl Equities": 4, "IG Bonds": 2, "HY Bonds": 5, "Cash": -5, "Treasuries": -2, "Corp Bonds": 3, "AGencies": -2, "Real Assets": 2, "Alternatives": 2},
        "Information: Strong Economy": {"US Equities": 4, "Intl Equities": 2, "IG Bonds": -2, "HY Bonds": 2, "Cash": -2, "Treasuries": -3, "Corp Bonds": 1, "AGencies": -1, "Real Assets": 1, "Alternatives": 1},
        "Information: Weak Economy": {"US Equities": -4, "Intl Equities": -3, "IG Bonds": 3, "HY Bonds": -2, "Cash": 4, "Treasuries": 4, "Corp Bonds": 0, "AGencies": 2, "Real Assets": -1, "Alternatives": -1},
    }

    def get_allocation(self, when, investor_type, shock_scenario=None):
        base = self.BASE_ALLOCATIONS.get(investor_type, self.BASE_ALLOCATIONS["Mutual Funds"]).copy()

        if when == "after" and shock_scenario:
            effects = self.SHOCK_EFFECTS.get(shock_scenario, {})
            for key in base:
                if key in effects:
                    base[key] = max(0, base[key] + effects[key])

        # Normalize to 100
        total = sum(base.values())
        if total > 0:
            base = {k: round(v / total * 100, 1) for k, v in base.items()}

        return base
