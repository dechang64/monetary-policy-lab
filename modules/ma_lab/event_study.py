"""
M&A Event Study Engine
======================
Extends the monetary-policy-lab EventStudyEngine to handle M&A announcement events.
Reuses the market model from analysis/event_study.py with custom event dates.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional
from scipy import stats


class MAEventStudyEngine:
    """
    Compute abnormal returns around M&A announcement dates.
    
    Uses the same market model as FOMC event study (Brown & Warner, 1985):
        AR_it = R_it - (alpha_i + beta_i * R_mt)
        CAR_i = sum AR over event window
        t = CAR / (sigma_AR * sqrt(N))
    
    The only change: replace FOMC dates with M&A announcement dates.
    """
    
    def __init__(self, returns: pd.DataFrame, event_dates: list, labels: Optional[list] = None):
        """
        Args:
            returns: DataFrame of daily asset returns, indexed by date.
            event_dates: list of M&A announcement dates (str or Timestamp).
            labels: optional list of deal names aligned with event_dates.
        """
        self.returns = returns
        self.event_dates = [pd.Timestamp(d) for d in event_dates]
        self.labels = labels
        self.event_mask = returns.index.isin(self.event_dates)
        self.non_event_returns = returns[~self.event_mask]
    
    def market_model(
        self,
        asset: str,
        market: str = "S&P 500",
        estimation_window: int = 250,
        event_window_pre: int = 1,
        event_window_post: int = 5,
    ) -> pd.DataFrame:
        """Market-model abnormal returns for one asset across all M&A events."""
        est_data = self.non_event_returns[[asset, market]].dropna().tail(estimation_window)
        if len(est_data) < 30:
            return pd.DataFrame()
        
        slope, intercept, _, _, _ = stats.linregress(
            est_data[market].values, est_data[asset].values
        )
        predicted = intercept + slope * est_data[market].values
        sigma = np.std(est_data[asset].values - predicted, ddof=2)
        
        results = []
        for i, event_date in enumerate(self.event_dates):
            ar_series = []
            for d in range(-event_window_pre, event_window_post + 1):
                target = event_date + timedelta(days=d)
                if target in self.returns.index:
                    actual = self.returns.loc[target, asset]
                    mkt_ret = self.returns.loc[target, market] if market in self.returns.columns else 0
                    expected = intercept + slope * mkt_ret
                    ar = actual - expected
                    ar_series.append({"date": target, "day_offset": d, "AR": ar})
            
            if ar_series:
                ar_df = pd.DataFrame(ar_series)
                car = ar_df["AR"].sum()
                n = len(ar_df)
                sar = sigma * np.sqrt(n)
                t_stat = car / sar if sar > 0 else 0
                results.append({
                    "asset": asset,
                    "event_date": event_date,
                    "label": self.labels[i] if self.labels else str(event_date.date()),
                    "alpha": round(intercept, 6),
                    "beta": round(slope, 4),
                    "sigma": round(sigma, 6),
                    "AR_mean": round(ar_df["AR"].mean(), 6),
                    "CAR": round(car, 6),
                    "CAR_pct": round(car * 100, 4),
                    "t_stat": round(t_stat, 3),
                    "event_days": n,
                })
        return pd.DataFrame(results)
    
    def cross_sectional(
        self,
        assets: list,
        market: str = "S&P 500",
        event_window_pre: int = 1,
        event_window_post: int = 1,
    ) -> pd.DataFrame:
        """Average CAR across events for each asset (acquirer vs target)."""
        all_results = []
        for asset in assets:
            res = self.market_model(asset, market,
                                    event_window_pre=event_window_pre,
                                    event_window_post=event_window_post)
            if not res.empty:
                all_results.append(res)
        if not all_results:
            return pd.DataFrame()
        combined = pd.concat(all_results, ignore_index=True)
        summary = combined.groupby("asset").agg(
            avg_CAR=("CAR", "mean"),
            median_CAR=("CAR", "median"),
            std_CAR=("CAR", "std"),
            n_events=("event_date", "count"),
            pct_positive=("CAR", lambda x: (x > 0).mean() * 100),
            avg_t_stat=("t_stat", "mean"),
        ).reset_index()
        summary["avg_CAR_pct"] = (summary["avg_CAR"] * 100).round(4)
        return summary.sort_values("avg_CAR_pct")
