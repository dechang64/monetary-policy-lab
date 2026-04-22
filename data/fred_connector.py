"""
FRED Data Connector
===================
Real-time data integration with the Federal Reserve Economic Data (FRED) API.

FRED provides 800,000+ economic data series for free.
Get your API key at: https://fred.stlouisfed.org/docs/api/api_key.html

Features:
- Automatic frequency detection per series
- Rate limiting and local caching
- Graceful fallback when no API key
- One-call fetch for all monetary policy data
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
from typing import Optional


class FREDConnector:
    """
    Connect to FRED API and fetch monetary policy related data.

    Usage:
        fred = FREDConnector(api_key="your_key_here")
        df = fred.fetch_all(start="2015-01-01", end="2024-12-31")
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # ── Series registry: name → (fred_id, native_frequency) ──
    SERIES_MAP = {
        # Interest Rates (daily)
        "1M Treasury": ("DGS1MO", "d"),
        "2Y Treasury": ("DGS2", "d"),
        "5Y Treasury": ("DGS5", "d"),
        "10Y Treasury": ("DGS10", "d"),
        "30Y Treasury": ("DGS30", "d"),
        "2Y-10Y Spread": ("T10Y2Y", "d"),
        "Effective Fed Funds": ("DFF", "d"),
        "SOFR": ("SOFR", "d"),

        # Inflation (monthly → daily via FRED aggregation)
        "CPI YoY": ("CPIAUCSL", "m"),
        "Core CPI YoY": ("CPILFESL", "m"),
        "PCE Inflation": ("PCEPI", "m"),
        "Core PCE YoY": ("PCEPILFE", "m"),
        "Breakeven 5Y": ("T5YIE", "d"),
        "Breakeven 10Y": ("T10YIE", "d"),

        # Equity Indices (daily)
        "S&P 500": ("SP500", "d"),
        "NASDAQ": ("NASDAQCOM", "d"),
        "VIX": ("VIXCLS", "d"),

        # FX (daily)
        "DXY": ("DTWEXBGS", "d"),
        "EUR/USD": ("DEXUSEU", "d"),
        "JPY/USD": ("DEXJPUS", "d"),

        # Commodities (daily)
        "Gold": ("GOLDAMGBD228NLBM", "d"),
        "Oil (WTI)": ("DCOILWTICO", "d"),

        # Credit (daily)
        "BAA Corporate": ("BAA", "d"),
        "AAA Corporate": ("AAA", "d"),

        # Labor Market (monthly)
        "Unemployment Rate": ("UNRATE", "m"),
        "Nonfarm Payrolls": ("PAYEMS", "m"),

        # GDP (quarterly)
        "Real GDP": ("GDPC1", "q"),
        "GDP Growth": ("A191RL1Q225SBEA", "q"),

        # Fed Balance Sheet (weekly)
        "Fed Assets Total": ("WALCL", "w"),

        # Expectations (monthly)
        "Michigan 1Y Inflation Exp": ("MICH", "m"),

        # Fed Funds Target (daily, from 2008)
        "Fed Funds Target Upper": ("DFEDTAR", "d"),
        "Fed Funds Target Lower": ("DFEDTARL", "d"),
    }

    # Derived series (computed from primary data)
    DERIVED = {
        "Credit Spread": ("BAA Corporate", "AAA Corporate", "sub"),  # BAA - AAA
    }

    def __init__(self, api_key: str = "", cache_dir: str = None, cache_hours: int = 6):
        self.api_key = api_key.strip() if api_key else ""
        self.cache_hours = cache_hours
        self._memory_cache = {}
        self._last_request = 0
        self.rate_limit_delay = 0.1

        # File cache
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()

    def _cache_path(self, key: str) -> str:
        safe = key.replace("/", "_").replace(" ", "_")
        return os.path.join(self.cache_dir, f"{safe}.json")

    def _load_cache(self, key: str) -> Optional[pd.Series]:
        path = self._cache_path(key)
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if time.time() - mtime < self.cache_hours * 3600:
                try:
                    with open(path) as f:
                        data = json.load(f)
                    s = pd.Series(data["values"], index=pd.to_datetime(data["index"]))
                    s.name = data.get("name", "")
                    return s
                except Exception:
                    pass
        return None

    def _save_cache(self, key: str, series: pd.Series):
        path = self._cache_path(key)
        try:
            data = {
                "index": [str(d) for d in series.index],
                "values": [None if pd.isna(v) else v for v in series.values],
                "name": series.name,
            }
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _fetch_series(
        self,
        series_id: str,
        start: str = "2015-01-01",
        end: str = None,
        frequency: str = "d",
    ) -> pd.Series:
        """Fetch a single FRED series with caching and error handling."""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        cache_key = f"{series_id}_{start}_{end}_{frequency}"

        # Check memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check file cache
        cached = self._load_cache(cache_key)
        if cached is not None:
            self._memory_cache[cache_key] = cached
            return cached

        # No API key → return empty silently
        if not self.api_key:
            return pd.Series(dtype=float, name=series_id)

        self._throttle()

        params = {
            "api_key": self.api_key,
            "series_id": series_id,
            "observation_start": start,
            "observation_end": end,
            "frequency": frequency,
            "file_type": "json",
            "sort_order": "asc",
        }

        try:
            resp = requests.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if "observations" not in data:
                return pd.Series(dtype=float, name=series_id)

            records = []
            for obs in data["observations"]:
                val = obs["value"]
                records.append((obs["date"], np.nan if val == "." else float(val)))

            if not records:
                return pd.Series(dtype=float, name=series_id)

            series = pd.DataFrame(records, columns=["date", "value"]).set_index("date")["value"]
            series.index = pd.to_datetime(series.index)
            series = series[~series.index.duplicated(keep="last")]
            series.name = series_id

            # Cache
            self._memory_cache[cache_key] = series
            self._save_cache(cache_key, series)

            return series

        except requests.exceptions.RequestException:
            return pd.Series(dtype=float, name=series_id)

    def fetch_series(self, name: str, start: str = "2015-01-01", end: str = None) -> pd.Series:
        """Fetch a named series using its native frequency."""
        if name not in self.SERIES_MAP:
            raise ValueError(f"Unknown series: {name}. Available: {list(self.SERIES_MAP.keys())}")

        series_id, freq = self.SERIES_MAP[name]
        return self._fetch_series(series_id, start, end, freq)

    def fetch_all(
        self,
        start: str = "2015-01-01",
        end: str = None,
        series_names: list = None,
    ) -> pd.DataFrame:
        """
        Fetch multiple series, align to daily frequency, compute derived series.

        Returns DataFrame with date index (daily), NaN for non-trading days.
        """
        if series_names is None:
            series_names = list(self.SERIES_MAP.keys())

        # Separate primary vs derived
        derived_names = [s for s in series_names if s in self.DERIVED]
        primary_names = [s for s in series_names if s in self.SERIES_MAP and s not in derived_names]

        # Fetch primary series
        frames = {}
        for name in primary_names:
            s = self.fetch_series(name, start, end)
            if not s.empty:
                frames[name] = s

        if not frames:
            return pd.DataFrame()

        # Align to daily index (forward-fill lower frequency data)
        all_dates = pd.date_range(start, end or datetime.now().strftime("%Y-%m-%d"), freq="D")
        df = pd.DataFrame(index=all_dates)

        for name, series in frames.items():
            df[name] = series.reindex(df.index, method="ffill")

        # Compute derived series
        for name, (s1, s2, op) in self.DERIVED.items():
            if s1 in df.columns and s2 in df.columns:
                if op == "sub":
                    df[name] = df[s1] - df[s2]

        return df

    def compute_returns(self, levels: pd.DataFrame) -> pd.DataFrame:
        """Convert price levels to daily percentage returns."""
        returns = levels.pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan)
        return returns.iloc[1:]

    def test_connection(self) -> bool:
        """Test if the FRED API key is valid."""
        if not self.api_key:
            return False
        try:
            s = self._fetch_series("DFF", frequency="d")
            return not s.empty
        except Exception:
            return False

    def get_available_series(self) -> dict:
        """Return {name: (fred_id, frequency)} for all series."""
        return dict(self.SERIES_MAP)

    def get_series_info(self, series_id: str) -> dict:
        """Get metadata about a FRED series."""
        if not self.api_key:
            return {}
        self._throttle()
        try:
            resp = requests.get(
                f"{self.BASE_URL}/series",
                params={"api_key": self.api_key, "series_id": series_id, "file_type": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("seriess", [{}])[0]
        except Exception:
            return {}
