"""
Constants and configuration for Monetary Policy Research Lab.
"""

# ── FOMC Meeting Dates (sample: 2015-2024) ──
FOMC_DATES = [
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-02-01", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (includes emergency meetings on Mar 3 and Mar 15)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
]

# ── Known Rate Changes (for demo/simulation) ──
RATE_CHANGES = {
    "2015-12-16": 0.25,
    "2016-12-14": 0.25,
    "2017-03-15": 0.25,
    "2017-06-14": 0.25,
    "2017-12-13": 0.25,
    "2018-03-21": 0.25,
    "2018-06-13": 0.25,
    "2018-09-26": 0.25,
    "2018-12-19": 0.25,
    "2019-07-31": -0.25,
    "2019-09-18": -0.25,
    "2019-10-30": -0.25,
    "2020-03-03": -0.50,
    "2020-03-15": -1.00,
    "2022-03-16": 0.25,
    "2022-05-04": 0.50,
    "2022-06-15": 0.75,
    "2022-07-27": 0.75,
    "2022-09-21": 0.75,
    "2022-11-02": 0.75,
    "2022-12-14": 0.50,
    "2023-02-01": 0.25,
    "2023-03-22": 0.25,
    "2023-05-03": 0.25,
    "2023-07-26": 0.25,
    "2024-09-18": -0.50,
}

# ── Asset Categories ──
ASSET_CATEGORIES = {
    "S&P 500": {"ticker": "^GSPC", "type": "equity", "risk": "high"},
    "NASDAQ": {"ticker": "^IXIC", "type": "equity", "risk": "high"},
    "Russell 2000": {"ticker": "^RUT", "type": "equity", "risk": "very_high"},
    "MSCI EM": {"ticker": "EEM", "type": "equity", "risk": "very_high"},
    "US 2Y Treasury": {"ticker": "^IRX", "type": "bond", "risk": "low"},
    "US 10Y Treasury": {"ticker": "^TNX", "type": "bond", "risk": "low"},
    "US 30Y Treasury": {"ticker": "^TYX", "type": "bond", "risk": "low"},
    "Corporate BBB": {"ticker": "LQD", "type": "bond", "risk": "medium"},
    "DXY (USD)": {"ticker": "DX-Y.NYB", "type": "fx", "risk": "medium"},
    "Gold": {"ticker": "GC=F", "type": "commodity", "risk": "medium"},
    "Oil (WTI)": {"ticker": "CL=F", "type": "commodity", "risk": "high"},
    "Bitcoin": {"ticker": "BTC-USD", "type": "crypto", "risk": "very_high"},
}

# ── Event Study Windows ──
WINDOWS = {
    "Ultra-Short": {"pre": 30, "post": 120, "unit": "minutes"},
    "Intraday": {"pre": 60, "post": 240, "unit": "minutes"},
    "Short": {"pre": 1, "post": 5, "unit": "days"},
    "Medium": {"pre": 5, "post": 20, "unit": "days"},
    "Long": {"pre": 20, "post": 60, "unit": "days"},
}

# ── FOMC Statement Sample (for NLP demo) ──
FOMC_STATEMENTS_SAMPLE = {
    "2024-12-18": """
    The Federal Open Market Committee decided to lower the target range for the 
    federal funds rate by 1/4 percentage point to 4-1/4 to 4-1/2 percent. The 
    Committee judges that the risks to achieving its employment and inflation 
    goals are roughly in balance. The economic outlook is uncertain, and the 
    Committee is attentive to risks to both sides of its dual mandate.
    
    Inflation has made progress toward the Committee's 2 percent objective but 
    remains somewhat elevated. The Committee intends to continue reducing its 
    holdings of Treasury securities and agency debt and agency mortgage-backed 
    securities.
    """,
    "2024-09-18": """
    The Federal Open Market Committee decided to lower the target range for the 
    federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent. The 
    Committee has gained greater confidence that inflation is moving sustainably 
    toward 2 percent, and judges that the risks to achieving its employment and 
    inflation goals are roughly in balance.
    
    The U.S. economy continues to grow at a solid pace. Job gains have slowed 
    but remain strong, and the unemployment rate has moved up but remains low. 
    Inflation has eased further but remains somewhat elevated.
    """,
    "2024-07-31": """
    The Committee decided to maintain the target range for the federal funds 
    rate at 5-1/4 to 5-1/2 percent. The Committee does not expect it will be 
    appropriate to reduce the target range until it has gained greater confidence 
    that inflation is moving sustainably toward 2 percent.
    
    Inflation remains elevated. The Committee is strongly committed to returning 
    inflation to its 2 percent objective. Job gains have moderated since early 
    last year but remain strong.
    """,
    "2022-06-15": """
    The Committee decided to raise the target range for the federal funds rate 
    by 75 basis points to 1-1/2 to 1-3/4 percent. The Committee is strongly 
    committed to returning inflation to its 2 percent objective.
    
    Inflation remains elevated, reflecting supply and demand imbalances related 
    to the pandemic, higher energy prices, and broader price pressures. The 
    ongoing war in Ukraine is creating additional upward pressure on inflation.
    """,
    "2020-03-15": """
    The Federal Reserve is prepared to use its full range of tools to support 
    the U.S. economy in this challenging time. The Committee decided to lower 
    the target range for the federal funds rate to 0 to 1/4 percent. The 
    Committee expects to maintain this target range until it is confident that 
    the economy has weathered recent events and is on track to achieve its 
    maximum employment and price stability goals.
    
    The coronavirus outbreak has harmed communities and disrupted economic 
    activity in many countries, including the United States.
    """,
}

# ── Colors ──
COLORS = {
    "policy_shock": "#e74c3c",
    "info_shock": "#3498db",
    "hawkish": "#e74c3c",
    "dovish": "#2ecc71",
    "neutral": "#95a5a6",
    "equity": "#27ae60",
    "bond": "#2980b9",
    "fx": "#8e44ad",
    "commodity": "#f39c12",
    "crypto": "#e67e22",
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "bg_dark": "#1a1a2e",
    "bg_mid": "#16213e",
    "bg_light": "#f8f9fa",
}

# ── Classic Papers for Replication ──
PAPERS = {
    "Kuttner (2001)": {
        "title": "Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market",
        "journal": "Journal of Monetary Economics",
        "method": "Fed funds futures surprise",
        "key_result": "Unexpected 25bp tightening → 2Y yield +4-6bp",
    },
    "Bernanke & Kuttner (2005)": {
        "title": "What Explains the Stock Market's Reaction to Federal Reserve Policy?",
        "journal": "Journal of Finance",
        "method": "Campbell-Shiller decomposition + futures surprise",
        "key_result": "75% of stock reaction via discount rate channel",
    },
    "Gürkaynak et al. (2005)": {
        "title": "Do Actions Speak Louder Than Words?",
        "journal": "International Journal of Central Banking",
        "method": "High-frequency identification of target vs path factors",
        "key_result": "Path factor > target factor for long-term rates",
    },
    "Jarociński & Karadi (2020)": {
        "title": "Deconstructing Monetary Policy Surprises",
        "journal": "AEJ: Macroeconomics",
        "method": "Two-shocks: policy vs information",
        "key_result": "Information shocks explain equity puzzle",
    },
    "Nakamura & Steinsson (2018)": {
        "title": "High-Frequency Identification of Monetary Non-Neutrality",
        "journal": "Quarterly Journal of Economics",
        "method": "High-frequency responses of real rates, inflation expectations",
        "key_result": "Monetary policy is non-neutral even at high frequency",
    },
}
