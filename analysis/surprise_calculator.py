"""
Kuttner Surprise Calculator
============================
Compute monetary policy surprises using the Kuttner (2001) method.

The surprise is measured as the change in the implied federal funds rate
from the day before to the day of the FOMC meeting, using fed funds
futures prices.

Reference: Kuttner, K.N. (2001). Monetary Policy Surprises and Interest
Rates: Evidence from the Fed Funds Futures Market. JME, 47(3), 523-544.
"""

import numpy as np
import pandas as pd
from typing import Optional


class SurpriseCalculator:
    """
    Calculate Kuttner (2001) monetary policy surprises.

    Methods:
    - futures_based: Standard Kuttner method using fed funds futures
    - target_based: Simple method using target rate changes
    - path_factor: Gürkaynak et al. (2005) path factor decomposition
    """

    def __init__(self, futures_data: pd.DataFrame, fomc_dates: list):
        """
        Args:
            futures_data: DataFrame with date index and 'ff_futures' column
                          (implied fed funds rate from futures contract)
            fomc_dates: List of FOMC meeting date strings
        """
        self.futures = futures_data
        self.fomc_dates = [pd.Timestamp(d) for d in fomc_dates]
        self.surprises = None

    def futures_based(self, futures_col: str = "FF1M Futures") -> pd.DataFrame:
        """
        Compute Kuttner surprises using fed funds futures.

        S_t = (f_{t,d} - f_{t-1,d}) / 100

        where f is the futures price for the month of the FOMC meeting.

        Returns:
            DataFrame with columns: fomc_date, surprise, surprise_bp
        """
        if futures_col not in self.futures.columns:
            return pd.DataFrame()

        results = []
        for fomc_date in self.fomc_dates:
            # Find the trading day before the FOMC meeting
            pre_mask = self.futures.index < fomc_date
            pre_dates = self.futures.index[pre_mask]

            if len(pre_dates) == 0:
                continue

            # Use the closest trading day before FOMC
            t_minus_1 = pre_dates[-1]

            # Find the closest trading day on or after FOMC
            post_mask = self.futures.index >= fomc_date
            post_dates = self.futures.index[post_mask]

            if len(post_dates) == 0:
                continue

            t_0 = post_dates[0]

            f_pre = self.futures.loc[t_minus_1, futures_col]
            f_post = self.futures.loc[t_0, futures_col]

            if pd.isna(f_pre) or pd.isna(f_post):
                continue

            surprise = (f_post - f_pre) / 100.0  # Convert to decimal
            results.append({
                "fomc_date": fomc_date,
                "surprise": surprise,
                "surprise_bp": surprise * 10000,  # Basis points
                "f_pre": f_pre,
                "f_post": f_post,
            })

        df = pd.DataFrame(results)
        df = df.set_index("fomc_date").sort_index() if not df.empty else df
        self.surprises = df
        return df

    def target_based(self, target_col: str = "Effective Fed Funds") -> pd.DataFrame:
        """
        Compute surprises using the actual change in the fed funds rate.
        This is the Cook & Hahn (1989) approach — less precise but simpler.

        Returns:
            DataFrame with columns: fomc_date, surprise, surprise_bp
        """
        if target_col not in self.futures.columns:
            return pd.DataFrame()

        results = []
        for fomc_date in self.fomc_dates:
            pre_mask = self.futures.index < fomc_date
            post_mask = self.futures.index >= fomc_date
            pre_dates = self.futures.index[pre_mask]
            post_dates = self.futures.index[post_mask]

            if len(pre_dates) == 0 or len(post_dates) == 0:
                continue

            t_minus_1 = pre_dates[-1]
            t_0 = post_dates[0]

            r_pre = self.futures.loc[t_minus_1, target_col]
            r_post = self.futures.loc[t_0, target_col]

            if pd.isna(r_pre) or pd.isna(r_post):
                continue

            surprise = (r_post - r_pre) / 100.0
            results.append({
                "fomc_date": fomc_date,
                "surprise": surprise,
                "surprise_bp": surprise * 10000,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("fomc_date").sort_index()
        return df

    def path_factor(
        self,
        short_col: str = "2Y Treasury",
        long_col: str = "10Y Treasury",
    ) -> pd.DataFrame:
        """
        Approximate the Gürkaynak et al. (2005) path factor.

        The path factor is captured by the change in long-term rates
        that is NOT explained by the short-term rate change.

        Returns:
            DataFrame with columns: fomc_date, target_factor, path_factor
        """
        if short_col not in self.futures.columns or long_col not in self.futures.columns:
            return pd.DataFrame()

        results = []
        for fomc_date in self.fomc_dates:
            pre_mask = self.futures.index < fomc_date
            post_mask = self.futures.index >= fomc_date
            pre_dates = self.futures.index[pre_mask]
            post_dates = self.futures.index[post_mask]

            if len(pre_dates) == 0 or len(post_dates) == 0:
                continue

            t_minus_1 = pre_dates[-1]
            t_0 = post_dates[0]

            short_pre = self.futures.loc[t_minus_1, short_col]
            short_post = self.futures.loc[t_0, short_col]
            long_pre = self.futures.loc[t_minus_1, long_col]
            long_post = self.futures.loc[t_0, long_col]

            if any(pd.isna(x) for x in [short_pre, short_post, long_pre, long_post]):
                continue

            target_factor = (short_post - short_pre) / 100.0
            long_change = (long_post - long_pre) / 100.0
            # Path factor = residual long-rate change not explained by short-rate change
            path_factor = long_change - 0.5 * target_factor  # Simplified

            results.append({
                "fomc_date": fomc_date,
                "target_factor": target_factor,
                "path_factor": path_factor,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("fomc_date").sort_index()
        return df

    def summary_stats(self) -> pd.DataFrame:
        """Summary statistics for computed surprises."""
        if self.surprises is None or self.surprises.empty:
            return pd.DataFrame()

        s = self.surprises["surprise_bp"]
        return pd.DataFrame({
            "N": [len(s)],
            "Mean (bp)": [s.mean()],
            "Std (bp)": [s.std(ddof=1)],
            "Min (bp)": [s.min()],
            "Max (bp)": [s.max()],
            "Median (bp)": [s.median()],
            "% Tightening": [(s > 0).mean() * 100],
            "% Easing": [(s < 0).mean() * 100],
            "% Neutral": [(s == 0).mean() * 100],
        }).T
