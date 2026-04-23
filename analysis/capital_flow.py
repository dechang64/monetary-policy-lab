"""
Capital Flow Analysis Engine
=============================
Analyze portfolio rebalancing around FOMC events.

In production: use mutual fund holdings data from CRSP/Thomson Reuters.
This module provides the framework with synthetic data for demo.
"""

import numpy as np
import pandas as pd
from typing import Optional


class CapitalFlowAnalyzer:
    """
    Analyze how investors reallocate capital across asset classes
    around monetary policy announcements.
    
    Features:
    - Flow decomposition by asset class
    - Sankey diagram data generation
    - Risk-on / Risk-off regime detection
    - Cross-asset correlation changes around FOMC
    """
    
    # Asset class mapping
    ASSET_CLASSES = {
        "US Large Cap": ["S&P 500"],
        "US Tech": ["NASDAQ"],
        "US Small Cap": ["Russell 2000"],
        "Emerging Markets": ["MSCI EM"],
        "US Treasuries": ["US 2Y Treasury", "US 10Y Treasury", "US 30Y Treasury"],
        "Corporate Bonds": ["Corporate BBB"],
        "Commodities": ["Gold", "Oil (WTI)"],
        "FX": ["DXY (USD)"],
        "Crypto": ["Bitcoin"],
    }
    
    RISK_CATEGORIES = {
        "Risk-On": ["US Large Cap", "US Tech", "US Small Cap", "Emerging Markets", "Crypto"],
        "Risk-Off": ["US Treasuries", "Corporate Bonds"],
        "Alternative": ["Commodities", "FX"],
    }

    def __init__(self, returns: pd.DataFrame, fomc_dates: list):
        self.returns = returns
        self.fomc_dates = [pd.Timestamp(d) for d in fomc_dates]
        # Build reverse mapping: individual asset → category
        self._asset_to_category = {}
        for cat, assets in self.ASSET_CLASSES.items():
            for a in assets:
                self._asset_to_category[a] = cat
        # Resolve RISK_CATEGORIES to actual column names present in returns
        self._resolved_risk = {}
        for group, categories in self.RISK_CATEGORIES.items():
            resolved = []
            for cat in categories:
                resolved.extend(self.ASSET_CLASSES.get(cat, [cat]))
            self._resolved_risk[group] = [
                a for a in resolved if a in self.returns.columns
            ]

    def _get_risk_assets(self, group: str) -> list:
        """Get resolved asset column names for a risk group."""
        return self._resolved_risk.get(group, [])
    
    def compute_flows(
        self,
        pre_window: int = 5,
        post_window: int = 10,
    ) -> pd.DataFrame:
        """
        Estimate capital flows around FOMC events.
        
        Method: Use return differentials between pre and post FOMC windows
        as a proxy for flow direction (following Ciminelli et al. 2022).
        """
        flows = []
        
        for fomc_date in self.fomc_dates:
            pre_returns = {}
            post_returns = {}
            
            for asset in self.returns.columns:
                pre_vals = []
                post_vals = []
                
                for d in range(-pre_window, 0):
                    target = fomc_date + pd.Timedelta(days=d)
                    if target in self.returns.index:
                        pre_vals.append(self.returns.loc[target, asset])
                
                for d in range(1, post_window + 1):
                    target = fomc_date + pd.Timedelta(days=d)
                    if target in self.returns.index:
                        post_vals.append(self.returns.loc[target, asset])
                
                if pre_vals and post_vals:
                    pre_returns[asset] = np.mean(pre_vals)
                    post_returns[asset] = np.mean(post_vals)
            
            # Compute flow as return change
            for asset in pre_returns:
                if asset in post_returns:
                    flow = post_returns[asset] - pre_returns[asset]
                    flows.append({
                        "fomc_date": fomc_date,
                        "asset": asset,
                        "pre_avg_return": round(pre_returns[asset], 6),
                        "post_avg_return": round(post_returns[asset], 6),
                        "flow": round(flow, 6),
                        "flow_pct": round(flow * 100, 4),
                    })
        
        return pd.DataFrame(flows)
    
    def sankey_data(
        self,
        flows_df: pd.DataFrame,
        n_events: Optional[int] = None,
    ) -> dict:
        """
        Generate Sankey diagram data showing capital flows between asset classes.
        
        Returns dict with nodes and links for Plotly Sankey.
        """
        if flows_df.empty:
            return {"nodes": [], "links": []}
        
        # Aggregate flows by asset class
        flows_df = flows_df.copy()
        flows_df["asset_class"] = flows_df["asset"].map(
            lambda x: next((k for k, v in self.ASSET_CLASSES.items() if x in v), x)
        )
        
        class_flows = flows_df.groupby("asset_class")["flow"].mean().sort_values()
        
        # Create nodes: "Pre-FOMC" → Asset Classes → "Post-FOMC"
        nodes = ["Pre-FOMC Allocation"] + list(class_flows.index) + ["Post-FOMC Allocation"]
        node_colors = ["#2c3e50"] + [
            "#e74c3c" if f > 0 else "#3498db" for f in class_flows.values
        ] + ["#2c3e50"]
        
        links = []
        source_idx = 0  # Pre-FOMC
        
        for i, (asset_class, flow) in enumerate(class_flows.items()):
            target_idx = len(class_flows) + 1  # Post-FOMC
            node_idx = i + 1
            
            # Normalize flow to positive value for Sankey
            value = abs(flow) * 1000  # Scale for visibility
            value = max(value, 0.1)
            
            if flow > 0:  # Inflow
                links.append({
                    "source": source_idx,
                    "target": node_idx,
                    "value": value,
                    "color": "#27ae60",
                    "label": f"{asset_class}: +{flow*100:.2f}%",
                })
                links.append({
                    "source": node_idx,
                    "target": target_idx,
                    "value": value,
                    "color": "#27ae60",
                    "label": "",
                })
            else:  # Outflow
                links.append({
                    "source": source_idx,
                    "target": node_idx,
                    "value": value,
                    "color": "#e74c3c",
                    "label": f"{asset_class}: {flow*100:.2f}%",
                })
                links.append({
                    "source": node_idx,
                    "target": target_idx,
                    "value": value,
                    "color": "#e74c3c",
                    "label": "",
                })
        
        return {
            "nodes": nodes,
            "node_colors": node_colors,
            "links": links,
        }
    
    def risk_regime_analysis(
        self,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Detect risk-on / risk-off regime changes around FOMC events.
        """
        results = []
        
        for fomc_date in self.fomc_dates:
            pre_risk_on = []
            post_risk_on = []
            
            for d in range(-window, 0):
                target = fomc_date + pd.Timedelta(days=d)
                if target in self.returns.index:
                    risk_assets = self._get_risk_assets("Risk-On")
                    safe_assets = self._get_risk_assets("Risk-Off")
                    if risk_assets and safe_assets:
                        risk_ret = np.mean([self.returns.loc[target, a] for a in risk_assets])
                        safe_ret = np.mean([self.returns.loc[target, a] for a in safe_assets])
                        pre_risk_on.append(risk_ret - safe_ret)
            
            for d in range(1, window + 1):
                target = fomc_date + pd.Timedelta(days=d)
                if target in self.returns.index:
                    risk_assets = self._get_risk_assets("Risk-On")
                    safe_assets = self._get_risk_assets("Risk-Off")
                    if risk_assets and safe_assets:
                        risk_ret = np.mean([self.returns.loc[target, a] for a in risk_assets])
                        safe_ret = np.mean([self.returns.loc[target, a] for a in safe_assets])
                        post_risk_on.append(risk_ret - safe_ret)
            
            if pre_risk_on and post_risk_on:
                pre_regime = "Risk-On" if np.mean(pre_risk_on) > 0 else "Risk-Off"
                post_regime = "Risk-On" if np.mean(post_risk_on) > 0 else "Risk-Off"
                regime_change = post_regime != pre_regime
                
                results.append({
                    "fomc_date": fomc_date,
                    "pre_regime": pre_regime,
                    "post_regime": post_regime,
                    "regime_change": regime_change,
                    "risk_spread_change": round(
                        np.mean(post_risk_on) - np.mean(pre_risk_on), 6
                    ),
                })
        
        return pd.DataFrame(results)
    
    def correlation_change(
        self,
        pre_window: int = 30,
        post_window: int = 30,
    ) -> pd.DataFrame:
        """
        Analyze how cross-asset correlations change around FOMC events.
        Higher correlations post-FOMC suggest "herding" behavior.
        """
        results = []
        
        for fomc_date in self.fomc_dates:
            pre_data = self.returns.loc[
                fomc_date - pd.Timedelta(days=pre_window):fomc_date
            ].dropna(axis=1, how="all")
            
            post_data = self.returns.loc[
                fomc_date:fomc_date + pd.Timedelta(days=post_window)
            ].dropna(axis=1, how="all")
            
            if len(pre_data) > 10 and len(post_data) > 10:
                pre_corr = pre_data.corr()
                post_corr = post_data.corr()
                
                n_cols = pre_corr.shape[0]
                if n_cols < 2:
                    continue
                
                # Average pairwise correlation
                pre_avg = (pre_corr.values.sum() - n_cols) / (n_cols * (n_cols - 1))
                post_avg = (post_corr.values.sum() - n_cols) / (n_cols * (n_cols - 1))
                
                results.append({
                    "fomc_date": fomc_date,
                    "pre_avg_corr": round(pre_avg, 4),
                    "post_avg_corr": round(post_avg, 4),
                    "corr_change": round(post_avg - pre_avg, 4),
                })
        
        return pd.DataFrame(results)
