"""
Step 1: Download all data from FRED API
- Monetary policy surprises (FF1M for Kuttner)
- Asset prices (7 asset classes)
- Control variables (VIX, term spread, credit spread)
"""
import pandas as pd
import numpy as np
from fredapi import Fred
import json, os

FRED_KEY = os.environ.get("FRED_API_KEY", "YOUR_KEY_HERE")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def download_fred_data():
    fred = Fred(api_key=FRED_KEY)
    
    series = {
        # Asset prices
        "SP500": "S&P 500 Daily Close",
        "NASDAQCOM": "NASDAQ Composite",
        "DGS2": "2-Year Treasury Yield",
        "DGS10": "10-Year Treasury Yield",
        "DTWEXBGS": "Trade-Weighted Dollar Index",
        "GOLDAMGBD228NLBM": "Gold Price (London Fixing)",
        "VIXCLS": "CBOE VIX",
        # Fed Funds Futures (for Kuttner surprise)
        "FF1M": "1-Month Fed Funds Futures Rate",
        # Control variables
        "DAAA": "AAA Corporate Bond Yield",
        "DBAA": "BAA Corporate Bond Yield",
        "DGS3MO": "3-Month Treasury Yield",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "FEDFUNDS": "Federal Funds Effective Rate",
    }
    
    data = {}
    for code, name in series.items():
        try:
            df = fred.get_series(code, observation_start="1994-01-01", observation_end="2025-12-31")
            data[code] = df
            print(f"  ✅ {code}: {len(df)} observations ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  ❌ {code}: {e}")
    
    # Save
    combined = pd.DataFrame(data)
    combined.to_csv(os.path.join(DATA_DIR, "fred_data.csv"))
    print(f"\nSaved {len(combined)} rows to fred_data.csv")
    return combined

if __name__ == "__main__":
    print("Downloading FRED data...")
    download_fred_data()
