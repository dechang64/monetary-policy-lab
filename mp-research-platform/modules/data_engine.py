"""
Data Engine — generates realistic demo data for the research platform.
All data is simulated based on published empirical patterns.
Replace with real data sources (FRED, WRDS, Bloomberg) for production use.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)


class DataEngine:
    """Generates and manages all research data."""

    def __init__(self):
        self._fomc_cache = {}
        self._asset_cache = {}
        self._shock_cache = {}
        self._statement_cache = {}
        self._sentiment_cache = {}
        self._papers = self._build_papers_db()

    # ─── FOMC Meeting Data ─────────────────────────────────────
    def get_fomc_data(self, start_year=1994, end_year=2025):
        key = (start_year, end_year)
        if key in self._fomc_cache:
            return self._fomc_cache[key]

        # Historical FOMC meeting dates and rate decisions
        meetings = self._generate_fomc_meetings(start_year, end_year)
        df = pd.DataFrame(meetings)
        df["date"] = pd.to_datetime(df["date"])
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)
        self._fomc_cache[key] = df
        return df

    def _generate_fomc_meetings(self, start_year, end_year):
        meetings = []
        # Approximate historical Fed Funds Rate path
        rate_path = {
            1994: 3.50, 1995: 5.50, 1996: 5.25, 1997: 5.50, 1998: 4.75,
            1999: 5.50, 2000: 6.50, 2001: 1.75, 2002: 1.25, 2003: 1.00,
            2004: 2.25, 2005: 4.25, 2006: 5.25, 2007: 4.25, 2008: 0.25,
            2009: 0.25, 2010: 0.25, 2011: 0.25, 2012: 0.25, 2013: 0.25,
            2014: 0.25, 2015: 0.50, 2016: 0.75, 2017: 1.50, 2018: 2.50,
            2019: 1.75, 2020: 0.25, 2021: 0.25, 2022: 4.50, 2023: 5.50,
            2024: 4.75, 2025: 4.25,
        }

        # FOMC meeting months (8 per year typically)
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]

        for year in range(start_year, end_year + 1):
            base_rate = rate_path.get(year, rate_path.get(year - 1, 2.0))
            for month in fomc_months:
                day = random.choice([1, 15, 28]) if month != 2 else 15
                try:
                    date = datetime(year, month, day)
                except ValueError:
                    date = datetime(year, month, 28)

                # Rate change with realistic distribution
                # Most meetings: no change. Some: ±25bp. Rare: ±50bp
                r = random.random()
                if r < 0.65:
                    change = 0
                elif r < 0.80:
                    change = 0.25
                elif r < 0.90:
                    change = -0.25
                elif r < 0.95:
                    change = 0.50
                elif r < 0.98:
                    change = -0.50
                else:
                    change = random.choice([-0.75, 0.75])

                rate_after = max(0, base_rate + change)

                # Surprise = actual change - expected change
                expected = 0 if change == 0 else (change * random.uniform(0.3, 1.0))
                surprise_bp = round((change - expected) * 100)

                meetings.append({
                    "date": date,
                    "rate_before": round(base_rate, 2),
                    "rate_after": round(rate_after, 2),
                    "change_bp": round(change * 100),
                    "surprise_bp": surprise_bp,
                    "meeting_type": random.choice(["Scheduled", "Scheduled", "Scheduled", "Unscheduled"]),
                })

                base_rate = rate_after

        return meetings

    # ─── Asset Price Data ──────────────────────────────────────
    def get_asset_data(self, start_year=1994, end_year=2025):
        key = (start_year, end_year)
        if key in self._asset_cache:
            return self._asset_cache[key]
        # Generated on-demand in get_event_window
        self._asset_cache[key] = True
        return True

    def get_event_window(self, event_idx, window_before, window_after, assets):
        """Generate simulated minute-by-minute asset data around an event."""
        fomc = self.get_fomc_data(1994, 2025)
        event = fomc.iloc[event_idx]
        surprise = event["surprise_bp"]

        # Empirical patterns from literature:
        # 25bp tightening surprise → SPX -1%, 10Y +10bp, DXY +0.5%, Gold -0.5%
        sensitivity = {
            "S&P 500": {"coef": -0.04, "vol": 0.3},       # % per bp surprise
            "NASDAQ": {"coef": -0.05, "vol": 0.5},
            "2Y Treasury": {"coef": -0.015, "vol": 0.05},  # yield change in %
            "10Y Treasury": {"coef": -0.008, "vol": 0.04},
            "DXY": {"coef": 0.02, "vol": 0.15},
            "Gold": {"coef": -0.015, "vol": 0.3},
            "Oil": {"coef": -0.01, "vol": 0.5},
            "VIX": {"coef": 0.03, "vol": 1.0},
        }

        total_minutes = window_before + window_after
        minutes = list(range(-window_before, window_after + 1))

        result = {}
        for asset in assets:
            if asset not in sensitivity:
                continue
            s = sensitivity[asset]
            # Pre-event: random walk with low vol
            pre_event = np.cumsum(np.random.normal(0, s["vol"] * 0.3, window_before))
            # Event jump at t=0
            jump = s["coef"] * surprise + np.random.normal(0, s["vol"] * 0.5)
            # Post-event: partial reversal + drift
            reversal = -jump * random.uniform(0.1, 0.3)
            post_event = jump + np.cumsum(
                np.random.normal(reversal / window_after, s["vol"] * 0.4, window_after + 1)
            )
            prices = np.concatenate([pre_event, post_event])
            result[asset] = pd.DataFrame({
                "minute": minutes,
                "return_pct": prices,
                "cumulative_return_pct": prices,
            })

        return result

    # ─── Shock Data ────────────────────────────────────────────
    def get_shock_data(self, start_year=1994, end_year=2025):
        key = (start_year, end_year)
        if key in self._shock_cache:
            return self._shock_cache[key]

        fomc = self.get_fomc_data(start_year, end_year)
        n = len(fomc)
        # Simulate policy and information shocks
        # Policy shock ≈ 60-75% of total, info shock ≈ 25-40%
        policy_shock = fomc["surprise_bp"].values * np.random.uniform(0.5, 0.8, n)
        info_shock = fomc["surprise_bp"].values * np.random.uniform(-0.3, 0.3, n) + np.random.normal(0, 3, n)

        self._shock_cache[key] = pd.DataFrame({
            "date": fomc["date"].values,
            "policy_shock": policy_shock,
            "info_shock": info_shock,
            "total_shock": fomc["surprise_bp"].values,
        })
        return self._shock_cache[key]

    # ─── FOMC Statements ───────────────────────────────────────
    def get_fomc_statements(self, start_year=1994, end_year=2025):
        key = (start_year, end_year)
        if key in self._statement_cache:
            return self._statement_cache[key]

        # Representative FOMC statement templates across eras
        statements = {
            "2025-03-19": "The Committee decided to maintain the target range for the federal funds rate at 4.25 to 4.50 percent. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance. The economic outlook is uncertain, and the Committee is attentive to risks to both sides of its dual mandate.",
            "2025-01-29": "The Committee decided to lower the target range for the federal funds rate by 25 basis points to 4.25 to 4.50 percent. Inflation has made progress toward the Committee's 2 percent objective but remains somewhat elevated. The Committee expects that further gradual adjustments will be appropriate.",
            "2024-12-18": "The Committee decided to lower the target range by 25 basis points to 4.25 to 4.50 percent. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent. The labor market has moderated from its earlier strong pace.",
            "2024-11-07": "The Committee decided to lower the target range by 25 basis points. Inflation has declined significantly but remains somewhat above the Committee's 2 percent longer-run goal. The Committee will continue to assess incoming information.",
            "2024-09-18": "The Committee decided to lower the target range by 50 basis points to 4.75 to 5.00 percent. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent, and judges that the risks to achieving its employment and inflation goals are roughly in balance.",
            "2024-07-31": "The Committee decided to maintain the target range at 5.25 to 5.50 percent. Inflation has eased over the past year but remains elevated. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent.",
            "2024-06-12": "The Committee decided to maintain the target range at 5.25 to 5.50 percent. Inflation has moderated since last year but remains above the Committee's 2 percent longer-run goal. The economic outlook is uncertain.",
            "2023-12-13": "The Committee decided to maintain the target range at 5.25 to 5.50 percent. Inflation has eased from its highs but remains elevated. The Committee will continue to evaluate additional information and its implications for monetary policy.",
            "2023-07-26": "The Committee decided to raise the target range by 25 basis points to 5.25 to 5.50 percent. The Committee is strongly committed to returning inflation to its 2 percent objective.",
            "2023-03-22": "The Committee decided to raise the target range by 25 basis points. The banking system is sound and resilient. Recent developments are likely to result in tighter credit conditions for households and businesses.",
            "2022-12-14": "The Committee decided to raise the target range by 50 basis points to 4.25 to 4.50 percent. The Committee anticipates that ongoing increases will be appropriate. Inflation remains elevated, reflecting supply and demand imbalances.",
            "2022-06-15": "The Committee decided to raise the target range by 75 basis points to 1.50 to 1.75 percent. The Committee is strongly committed to returning inflation to its 2 percent objective.",
            "2022-03-16": "The Committee decided to raise the target range by 25 basis points. The invasion of Ukraine by Russia is causing tremendous human and economic hardship. The implications for the U.S. economy are highly uncertain.",
            "2021-12-15": "The Committee decided to accelerate the pace of asset purchases tapering. With inflation having exceeded 2 percent for some time, the Committee expects it will be appropriate to maintain an accommodative stance.",
            "2020-12-16": "The Committee decided to maintain the target range at 0 to 0.25 percent. The pandemic continues to weigh on economic activity. The Committee will continue to purchase assets at the current pace.",
            "2020-03-15": "The Committee decided to lower the target range to 0 to 0.25 percent. The effects of the coronavirus will weigh on economic activity in the near term and pose risks to the economic outlook.",
            "2019-07-31": "The Committee decided to lower the target range by 25 basis points. This action is consistent with the Committee's approach of acting as appropriate to sustain the expansion.",
            "2015-12-16": "The Committee decided to raise the target range by 25 basis points to 0.25 to 0.50 percent. The Committee expects that economic conditions will evolve in a manner that will warrant only gradual increases.",
        }

        # Filter by date range
        filtered = {}
        for date_str, text in statements.items():
            year = int(date_str[:4])
            if start_year <= year <= end_year:
                filtered[date_str] = text

        self._statement_cache[key] = filtered
        return filtered

    # ─── Historical Sentiments ─────────────────────────────────
    def get_historical_sentiments(self, start_year=1994, end_year=2025):
        key = (start_year, end_year)
        if key in self._sentiment_cache:
            return self._sentiment_cache[key]

        fomc = self.get_fomc_data(start_year, end_year)
        sentiments = {}
        for _, row in fomc.iterrows():
            surprise = row["surprise_bp"]
            # Hawkish signal increases with positive surprise
            hawkish = max(0, min(1, 0.5 + surprise / 50 + np.random.normal(0, 0.1)))
            dovish = max(0, min(1, 0.5 - surprise / 50 + np.random.normal(0, 0.1)))
            sentiments[row["date_str"]] = {"hawkish": round(hawkish, 3), "dovish": round(dovish, 3)}

        self._sentiment_cache[key] = sentiments
        return sentiments

    # ─── Literature Database ───────────────────────────────────
    def _build_papers_db(self):
        return [
            {"authors": "Cook & Hahn", "year": 1989, "title": "The effect of changes in the federal funds rate target on market interest rates", "journal": "JME", "topic": "Event Studies", "era": "1980s-1990s", "method": "Event study, regression", "finding": "Discount rate announcements systematically affect Treasury yields across maturities", "gap": "Only discount rate changes, not target rate", "short": "Cook-Hahn 89", "color": "#E74C3C"},
            {"authors": "Kuttner", "year": 2001, "title": "Monetary policy surprises and interest rates", "journal": "JME", "topic": "Expectations", "era": "2000s", "method": "Fed funds futures", "finding": "Only unexpected rate changes matter; 25bp surprise → 4bp change in 2Y yield", "gap": "Focus on interest rates only", "short": "Kuttner 01", "color": "#3498DB"},
            {"authors": "Bernanke & Kuttner", "year": 2005, "title": "What explains the stock market's reaction to Fed policy?", "journal": "JF", "topic": "Event Studies", "era": "2000s", "method": "Event study, futures-based surprise", "finding": "25bp surprise → ~1% stock move; discount rate channel dominates (75%)", "gap": "Aggregated equity market only", "short": "B&K 05", "color": "#2ECC71"},
            {"authors": "Gürkaynak, Sack & Swanson", "year": 2005, "title": "Do actions speak louder than words?", "journal": "IJCB", "topic": "Transmission", "era": "2000s", "method": "High-frequency identification", "finding": "FOMC statements have larger market impact than rate decisions themselves", "gap": "Pre-2004 data, pre-communication era", "short": "GSS 05", "color": "#9B59B6"},
            {"authors": "Campbell et al.", "year": 2012, "title": "Measuring the output response to monetary policy", "journal": "AEJ: Macro", "topic": "Transmission", "era": "2010s", "method": "High-frequency identification, VAR", "finding": "Monetary policy shocks have persistent effects on output and inflation", "gap": "Macro aggregates, no asset-level heterogeneity", "short": "Campbell 12", "color": "#E67E22"},
            {"authors": "Nakamura & Steinsson", "year": 2018, "title": "High-frequency identification of monetary non-neutrality", "journal": "QJE", "topic": "Two-Shocks", "era": "2010s", "method": "High-frequency, price rigidity", "finding": "Monetary policy is non-neutral even at high frequency; prices don't adjust instantly", "gap": "Focus on prices, not portfolio flows", "short": "N&S 18", "color": "#1ABC9C"},
            {"authors": "Jarociński & Karadi", "year": 2020, "title": "Deconstructing monetary policy surprises", "journal": "AEJ: Macro", "topic": "Two-Shocks", "era": "2020s", "method": "High-frequency VAR, sign restrictions", "finding": "Information shocks explain significant share; can reverse policy shock effects", "gap": "Broad asset classes only", "short": "J&K 20", "color": "#E74C3C"},
            {"authors": "Ciminelli, Rogers & Wu", "year": 2022, "title": "Effects of US monetary policy on international mutual fund investment", "journal": "JIMF", "topic": "Capital Flows", "era": "2020s", "method": "Fund flow data, event study", "finding": "US monetary policy shocks significantly affect global capital allocation", "gap": "Fund-level, not asset-class level", "short": "CRW 22", "color": "#3498DB"},
            {"authors": "Borio & Zhu", "year": 2012, "title": "Capital regulation, risk-taking and monetary policy", "journal": "BIS Working Paper", "topic": "Risk-Taking", "era": "2010s", "method": "Theoretical framework", "finding": "Low rates encourage risk-taking and search for yield across asset classes", "gap": "Theoretical, limited empirical validation", "short": "B&Z 12", "color": "#2ECC71"},
            {"authors": "Melosi", "year": 2017, "title": "Signaling effects of monetary policy", "journal": "AEJ: Macro", "era": "2010s", "topic": "Two-Shocks", "method": "DSGE model", "finding": "Policy rate signals central bank's view about macroeconomic developments", "gap": "Model-dependent, no cross-asset analysis", "short": "Melosi 17", "color": "#9B59B6"},
            {"authors": "Hanson & Stein", "year": 2015, "title": "Monetary policy and long-term real rates", "journal": "JFE", "topic": "Transmission", "era": "2010s", "method": "Event study, term structure", "finding": "Fed purchases affect long-term rates through duration risk channel", "gap": "Government bonds only", "short": "H&S 15", "color": "#E67E22"},
            {"authors": "Gilchrist et al.", "year": 2014, "title": "Unconventional monetary policy and cross-asset portfolio rebalancing", "journal": "RFS", "topic": "Capital Flows", "era": "2010s", "method": "Event study, portfolio analysis", "finding": "QE triggers significant portfolio rebalancing across asset classes", "gap": "QE-specific, not standard rate decisions", "short": "Gilchrist 14", "color": "#1ABC9C"},
            {"authors": "Lucca & Moench", "year": 2015, "title": "The pre-FOMC announcement drift", "journal": "JF", "topic": "Event Studies", "era": "2010s", "method": "Event study", "finding": "Stock market earns excess returns in 24h before FOMC announcements", "gap": "Pre-announcement only, not post-announcement rebalancing", "short": "L&M 15", "color": "#E74C3C"},
            {"authors": "FinBERT-FOMC Study", "year": 2024, "title": "Sentiment detection in central bank communication using LLMs", "journal": "Working Paper", "topic": "NLP/Communication", "era": "2020s", "method": "NLP, FinBERT, GPT-4", "finding": "LLMs can detect nuanced sentiment shifts in FOMC statements", "gap": "No link to market outcomes", "short": "FinBERT 24", "color": "#3498DB"},
        ]

    def get_papers(self, topics=None, eras=None):
        papers = self._papers
        if topics:
            papers = [p for p in papers if p["topic"] in topics]
        if eras:
            papers = [p for p in papers if p["era"] in eras]
        return sorted(papers, key=lambda x: x["year"])

    def get_all_papers(self):
        return self._papers

    def get_literature_timeline(self, topics=None):
        papers = self.get_papers(topics)
        return papers

    def get_literature_network_data(self, topics=None, eras=None):
        """Build network graph data: papers connected by shared topics/methods."""
        papers = self.get_papers(topics, eras)
        nodes = []
        edges = []
        for i, p in enumerate(papers):
            nodes.append({
                "id": i,
                "label": p["short"],
                "title": f"{p['authors']} ({p['year']})",
                "group": p["topic"],
                "size": 10 + (2025 - p["year"]) * 0.5,
                "color": p["color"],
            })
            for j, q in enumerate(papers):
                if j <= i:
                    continue
                # Connect papers that share topics or methods
                shared = 0
                if p["topic"] == q["topic"]:
                    shared += 2
                if abs(p["year"] - q["year"]) <= 5:
                    shared += 1
                if any(w in p["method"].lower() for w in q["method"].lower().split()):
                    shared += 1
                if shared >= 2:
                    edges.append({"source": i, "target": j, "value": shared})
        return nodes, edges
