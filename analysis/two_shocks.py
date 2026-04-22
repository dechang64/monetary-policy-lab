"""
Two-Shocks Decomposition Engine
================================
Decompose monetary policy surprises into policy shocks and information shocks,
following Jarociński & Karadi (2020).

In production: estimate via SVAR with high-frequency identification.
This module provides both a simplified analytical version and the framework
for full SVAR estimation.
"""

import numpy as np
import pandas as pd
from typing import Optional


class TwoShocksDecomposer:
    """
    Decompose FOMC surprises into:
    1. Policy Shock: unexpected change in the monetary policy stance
    2. Information Shock: revelation about future economic fundamentals
    
    Methods:
    - Simplified: heuristic decomposition based on asset price responses
    - SVAR: structural VAR identification (production)
    - High-Frequency: using intraday data around FOMC (production)
    """
    
    def __init__(self, surprises_df: pd.DataFrame, returns: pd.DataFrame):
        self.surprises = surprises_df
        self.returns = returns
        self.decomposition = None
    
    def simplified_decompose(
        self,
        equity_col: str = "S&P 500",
        bond_col: str = "US 10Y Treasury",
        policy_weight: float = 0.6,
    ) -> pd.DataFrame:
        """
        Simplified two-shocks decomposition.
        
        Logic (Jarociński & Karadi 2020):
        - Policy shock: moves rates and equity in OPPOSITE directions
          (tightening → rates up, equity down)
        - Information shock: moves rates and equity in SAME direction
          (good news → rates up, equity up)
        
        We use the correlation between rate surprise and equity response
        to decompose.
        """
        df = self.surprises.copy()
        
        # Get equity response on FOMC days
        fomc_dates = df.index
        equity_responses = []
        for date in fomc_dates:
            if date in self.returns.index:
                equity_responses.append(self.returns.loc[date, equity_col])
            else:
                equity_responses.append(0)
        
        df["equity_response"] = equity_responses
        
        # Decompose based on sign co-movement
        # If surprise > 0 (tightening) and equity < 0 → pure policy shock
        # If surprise > 0 (tightening) and equity > 0 → information shock dominates
        decomposed = []
        for _, row in df.iterrows():
            surprise = row["surprise"]
            eq_resp = row["equity_response"]
            
            # Heuristic: use equity response to split
            if abs(surprise) < 0.001:
                policy = 0
                info = 0
            else:
                # Negative correlation = policy shock
                # Positive correlation = information shock
                corr_signal = -np.sign(surprise * eq_resp) if eq_resp != 0 else 0
                # Blend with base weight
                w = policy_weight + (1 - policy_weight) * (1 + corr_signal) / 2
                w = np.clip(w, 0.2, 0.8)
                policy = surprise * w
                info = surprise * (1 - w)
            
            decomposed.append({
                "date": row.name,
                "surprise": surprise,
                "policy_shock": round(policy, 4),
                "info_shock": round(info, 4),
                "policy_pct": round(abs(policy) / (abs(policy) + abs(info) + 1e-10) * 100, 1),
                "equity_response": round(eq_resp, 6),
                "dominant": "Policy" if abs(policy) > abs(info) else "Information",
            })
        
        self.decomposition = pd.DataFrame(decomposed).set_index("date")
        return self.decomposition
    
    def asset_response_by_shock(
        self,
        decomposition: pd.DataFrame,
        window_days: int = 5,
    ) -> pd.DataFrame:
        """
        Analyze how different asset classes respond to each shock type.
        This is the core "Two-Shocks Radar" data.
        """
        results = []
        
        for asset in self.returns.columns:
            for shock_type in ["policy_shock", "info_shock"]:
                # Split events by dominant shock
                decomp = decomposition.copy()
                decomp["shock_magnitude"] = decomp[shock_type].abs()
                
                # Top quartile shock events
                threshold = decomp["shock_magnitude"].quantile(0.75)
                strong_events = decomp[decomp["shock_magnitude"] >= threshold].index
                
                # Compute average response
                responses = []
                for event_date in strong_events:
                    for d in range(0, window_days + 1):
                        target = event_date + pd.Timedelta(days=d)
                        if target in self.returns.index:
                            responses.append(self.returns.loc[target, asset])
                
                if responses:
                    avg_response = np.mean(responses)
                    results.append({
                        "asset": asset,
                        "shock_type": "Policy" if "policy" in shock_type else "Information",
                        "avg_response": round(avg_response, 6),
                        "avg_response_pct": round(avg_response * 100, 4),
                        "n_events": len(strong_events),
                        "window_days": window_days,
                    })
        
        return pd.DataFrame(results)
    
    def temporal_evolution(self, decomposition: pd.DataFrame) -> pd.DataFrame:
        """
        Track how the policy/information shock balance evolves over time.
        Useful for identifying regime changes (e.g., forward guidance era).
        """
        df = decomposition.copy()
        df["year"] = df.index.year
        
        yearly = df.groupby("year").agg(
            avg_surprise=("surprise", "mean"),
            avg_policy=("policy_shock", "mean"),
            avg_info=("info_shock", "mean"),
            policy_pct_mean=("policy_pct", "mean"),
            n_meetings=("surprise", "count"),
            policy_dominant_pct=("dominant", lambda x: (x == "Policy").mean() * 100),
        ).reset_index()
        
        return yearly
    
    def shock_correlation_matrix(self, decomposition: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation between shock types and asset responses.
        """
        df = decomposition.copy()
        
        # Add asset responses
        for asset in self.returns.columns:
            responses = []
            for date in df.index:
                if date in self.returns.index:
                    responses.append(self.returns.loc[date, asset])
                else:
                    responses.append(np.nan)
            df[asset] = responses
        
        cols = ["surprise", "policy_shock", "info_shock"] + [
            c for c in self.returns.columns if c in df.columns
        ]
        
        return df[cols].corr().round(3)
