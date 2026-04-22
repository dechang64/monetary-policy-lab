"""
Event Study Engine
==================
Core analysis module for computing abnormal returns around FOMC events.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional


class EventStudyEngine:
    """
    Compute event study statistics for FOMC announcements.
    
    Methods:
    - Market Model (traditional)
    - Constant Mean Return Model
    - Fama-French Three-Factor Model (extended)
    
    In production: add GARCH for volatility clustering.
    """
    
    def __init__(self, returns: pd.DataFrame, fomc_dates: list):
        self.returns = returns
        self.fomc_dates = [pd.Timestamp(d) for d in fomc_dates]
        self.event_mask = returns.index.isin(self.fomc_dates)
        self.non_event_returns = returns[~self.event_mask]
        self.results = None
    
    def market_model(
        self,
        asset: str,
        market: str = "S&P 500",
        estimation_window: int = 250,
        event_window_pre: int = 1,
        event_window_post: int = 5,
    ) -> pd.DataFrame:
        """
        Market Model Event Study.
        
        AR_it = R_it - (alpha_i + beta_i * R_mt)
        CAR = sum(AR) over event window
        """
        from scipy import stats
        
        # Estimation: use non-event days
        est_data = self.non_event_returns[[asset, market]].dropna().tail(estimation_window)
        
        if len(est_data) < 50:
            return pd.DataFrame()
        
        slope, intercept, _, _, _ = stats.linregress(
            est_data[market].values, est_data[asset].values
        )
        
        # Residual std for t-stats
        predicted = intercept + slope * est_data[market].values
        residuals = est_data[asset].values - predicted
        sigma = np.std(residuals, ddof=2)
        
        # Event window
        results = []
        for fomc_date in self.fomc_dates:
            ar_series = []
            for d in range(-event_window_pre, event_window_post + 1):
                target = fomc_date + timedelta(days=d)
                if target in self.returns.index:
                    actual = self.returns.loc[target, asset]
                    if market in self.returns.columns and target in self.returns.index:
                        mkt_ret = self.returns.loc[target, market]
                    else:
                        mkt_ret = 0
                    expected = intercept + slope * mkt_ret
                    ar = actual - expected
                    ar_series.append({
                        "date": target,
                        "day_offset": d,
                        "actual_return": actual,
                        "expected_return": expected,
                        "AR": ar,
                    })
            
            if ar_series:
                ar_df = pd.DataFrame(ar_series)
                car = ar_df["AR"].sum()
                n = len(ar_df)
                sar = sigma * np.sqrt(n)  # standard deviation of CAR
                t_stat = car / sar if sar > 0 else 0
                
                results.append({
                    "asset": asset,
                    "fomc_date": fomc_date,
                    "alpha": round(intercept, 6),
                    "beta": round(slope, 4),
                    "sigma": round(sigma, 6),
                    "AR_mean": round(ar_df["AR"].mean(), 6),
                    "AR_std": round(ar_df["AR"].std(), 6),
                    "CAR": round(car, 6),
                    "CAR_pct": round(car * 100, 4),
                    "t_stat": round(t_stat, 3),
                    "event_days": n,
                    "daily_AR": ar_df[["day_offset", "AR"]].to_dict("records"),
                })
        
        return pd.DataFrame(results)
    
    def cross_sectional_analysis(
        self,
        event_window_pre: int = 1,
        event_window_post: int = 5,
    ) -> pd.DataFrame:
        """
        Aggregate event study across all assets.
        Compute AAR (Average Abnormal Return) and CAAR for each event.
        """
        all_results = []
        
        for asset in self.returns.columns:
            if asset == "S&P 500":
                market = "NASDAQ"
            else:
                market = "S&P 500"
            
            res = self.market_model(
                asset, market,
                event_window_pre=event_window_pre,
                event_window_post=event_window_post,
            )
            if not res.empty:
                all_results.append(res)
        
        if not all_results:
            return pd.DataFrame()
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Aggregate by FOMC date
        agg = combined.groupby("fomc_date").agg(
            AAR=("AR_mean", "mean"),
            CAAR=("CAR", "mean"),
            avg_t_stat=("t_stat", "mean"),
            n_assets=("asset", "count"),
            avg_beta=("beta", "mean"),
        ).reset_index()
        
        agg["CAAR_pct"] = (agg["CAAR"] * 100).round(4)
        
        return combined, agg
    
    def cumulative_by_asset(
        self,
        event_window_pre: int = 1,
        event_window_post: int = 5,
    ) -> pd.DataFrame:
        """
        Average CAR across all FOMC events for each asset.
        """
        all_results = []
        
        for asset in self.returns.columns:
            market = "NASDAQ" if asset == "S&P 500" else "S&P 500"
            res = self.market_model(
                asset, market,
                event_window_pre=event_window_pre,
                event_window_post=event_window_post,
            )
            if not res.empty:
                all_results.append(res)
        
        if not all_results:
            return pd.DataFrame()
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Average CAR per asset
        asset_summary = combined.groupby("asset").agg(
            avg_CAR=("CAR", "mean"),
            median_CAR=("CAR", "median"),
            std_CAR=("CAR", "std"),
            avg_t_stat=("t_stat", "mean"),
            n_events=("fomc_date", "count"),
            pct_positive=("CAR", lambda x: (x > 0).mean() * 100),
        ).reset_index()
        
        asset_summary["avg_CAR_pct"] = (asset_summary["avg_CAR"] * 100).round(4)
        asset_summary = asset_summary.sort_values("avg_CAR_pct")
        
        return asset_summary
