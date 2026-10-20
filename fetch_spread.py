# -*- coding: utf-8 -*-
"""Fetch BBB-AAA credit spread (monthly) from FRED with retries."""
import requests, pandas as pd, io, time

H = {"User-Agent": "Mozilla/5.0 (research)"}

def fetch(code, tries=6):
    for i in range(tries):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}",
                             headers=H, timeout=60)
            if r.text.startswith("observation_date"):
                return pd.read_csv(io.StringIO(r.text), na_values=".")
            print(f"  {code} try{i+1}: non-CSV", flush=True)
        except Exception as e:
            print(f"  {code} try{i+1}: {type(e).__name__}", flush=True)
        time.sleep(5)
    raise RuntimeError(code)

out = {}
for code in ("BAA10Y",):
    df = fetch(code)
    df.columns = ["date", code]
    df["date"] = pd.to_datetime(df["date"])
    out[code] = df.set_index("date")[code]
    print(code, "rows:", len(df), flush=True)

dgs = pd.read_csv("data/fred_DGS10.csv")
dgs.columns = ["date", "DGS10"]
dgs["date"] = pd.to_datetime(dgs["date"])
dgs = dgs.set_index("date")["DGS10"]
m = out["BAA10Y"].to_frame("BAA").join(dgs.to_frame("DGS10"), how="inner").dropna()
m["spread"] = m["BAA"] - m["DGS10"]
m["ym"] = m.index.to_period("M")
mo = m.groupby("ym")["spread"].last()
print("monthly spread:", len(mo), "months | mean", round(mo.mean(), 3), flush=True)
mo.to_frame("spread").to_csv("data/fred_spread_Baa_Treasury_monthly.csv")
print("saved", flush=True)
