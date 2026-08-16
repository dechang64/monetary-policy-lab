"""
M&A Research Lab — Constants & Sample Data
==========================================
Merger & Acquisition and LBO research constants.
All data is sample/demo for teaching; in production connect to SEC EDGAR + WRDS SDC.
"""

# ── Sample M&A Deals (for demo; in production: SEC EDGAR / SDC Platinum) ──
MA_DEALS = [
    {"date": "2024-12-10", "acquirer": "Microsoft", "target": "Activision Blizzard",
     "deal_value_b": 69.0, "premium_pct": 45, "payment": "cash", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2024-01-08", "acquirer": "Cisco", "target": "Splunk",
     "deal_value_b": 28.0, "premium_pct": 31, "payment": "cash", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2023-10-26", "acquirer": "Chevron", "target": "Hess",
     "deal_value_b": 53.0, "premium_pct": 10.8, "payment": "stock", "industry": "Energy",
     "completed": False, "leverage_ratio": 0.0},
    {"date": "2023-07-11", "acquirer": "Broadcom", "target": "VMware",
     "deal_value_b": 61.0, "premium_pct": 49, "payment": "mixed", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.35},
    {"date": "2022-04-25", "acquirer": "Elon Musk", "target": "Twitter",
     "deal_value_b": 44.0, "premium_pct": 38, "payment": "cash", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.65},
    {"date": "2021-11-30", "acquirer": "Salesforce", "target": "Slack",
     "deal_value_b": 27.7, "premium_pct": 55, "payment": "stock", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2020-09-14", "acquirer": "Nvidia", "target": "Arm (SoftBank)",
     "deal_value_b": 40.0, "premium_pct": 17, "payment": "mixed", "industry": "Tech",
     "completed": False, "leverage_ratio": 0.0},
    {"date": "2020-10-26", "acquirer": "Advanced Micro Devices", "target": "Xilinx",
     "deal_value_b": 35.0, "premium_pct": 25, "payment": "stock", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2019-11-04", "acquirer": "LVMH", "target": "Tiffany",
     "deal_value_b": 16.2, "premium_pct": 37, "payment": "cash", "industry": "Luxury",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2019-06-02", "acquirer": "United Technologies", "target": "Raytheon",
     "deal_value_b": 121.0, "premium_pct": 28, "payment": "stock", "industry": "Defense",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2018-07-16", "acquirer": " Comcast", "target": "Sky plc",
     "deal_value_b": 39.0, "premium_pct": 82, "payment": "cash", "industry": "Media",
     "completed": True, "leverage_ratio": 0.30},
    {"date": "2018-04-25", "acquirer": "T-Mobile", "target": "Sprint",
     "deal_value_b": 26.5, "premium_pct": 18, "payment": "stock", "industry": "Telecom",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2017-09-05", "acquirer": "Disney", "target": "21st Century Fox",
     "deal_value_b": 71.3, "premium_pct": 30, "payment": "stock", "industry": "Media",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2016-10-24", "acquirer": "AT&T", "target": "Time Warner",
     "deal_value_b": 85.4, "premium_pct": 35, "payment": "mixed", "industry": "Media",
     "completed": True, "leverage_ratio": 0.40},
    {"date": "2016-09-26", "acquirer": "Bayer", "target": "Monsanto",
     "deal_value_b": 66.0, "premium_pct": 44, "payment": "cash", "industry": "Pharma",
     "completed": True, "leverage_ratio": 0.25},
    {"date": "2015-11-23", "acquirer": "Pfizer", "target": "Allergan",
     "deal_value_b": 160.0, "premium_pct": 30, "payment": "stock", "industry": "Pharma",
     "completed": False, "leverage_ratio": 0.0},
    {"date": "2015-03-25", "acquirer": "Royal Dutch Shell", "target": "BG Group",
     "deal_value_b": 70.0, "premium_pct": 50, "payment": "mixed", "industry": "Energy",
     "completed": True, "leverage_ratio": 0.20},
    {"date": "2014-02-19", "acquirer": "Facebook", "target": "WhatsApp",
     "deal_value_b": 19.0, "premium_pct": 95, "payment": "mixed", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2013-09-02", "acquirer": "Microsoft", "target": "Nokia Devices",
     "deal_value_b": 7.2, "premium_pct": 40, "payment": "cash", "industry": "Tech",
     "completed": True, "leverage_ratio": 0.0},
    {"date": "2013-02-14", "acquirer": "Berkshire Hathaway + 3G", "target": "Heinz",
     "deal_value_b": 28.0, "premium_pct": 20, "payment": "cash", "industry": "Food",
     "completed": True, "leverage_ratio": 0.70},  # LBO
    {"date": "2009-11-02", "acquirer": "Kohlberg Kravis Roberts", "target": "Dollar General",
     "deal_value_b": 7.3, "premium_pct": 31, "payment": "cash", "industry": "Retail",
     "completed": True, "leverage_ratio": 0.72},  # LBO
    {"date": "2007-07-02", "acquirer": "KKR", "target": "First Data",
     "deal_value_b": 29.0, "premium_pct": 26, "payment": "cash", "industry": "Fintech",
     "completed": True, "leverage_ratio": 0.68},  # LBO
    {"date": "2007-02-09", "acquirer": "Blackstone", "target": "Equity Office Properties",
     "deal_value_b": 39.0, "premium_pct": 35, "payment": "cash", "industry": "REIT",
     "completed": True, "leverage_ratio": 0.65},  # LBO
    {"date": "1989-02-09", "acquirer": "KKR", "target": "RJR Nabisco",
     "deal_value_b": 31.4, "premium_pct": 40, "payment": "cash", "industry": "Food",
     "completed": True, "leverage_ratio": 0.85},  # Legendary LBO "Barbarians at the Gate"
]

# ── LBO Sample Cases ──
LBO_CASES = [
    {"name": "RJR Nabisco (1989)", "sponsor": "KKR", "entry_ebitda_b": 3.1,
     "entry_ev_b": 31.4, "exit_ev_b": 28.0, "hold_years": 4, "debt_pct": 0.85},
    {"name": "Dollar General (2009)", "sponsor": "KKR", "entry_ebitda_b": 0.65,
     "entry_ev_b": 7.3, "exit_ev_b": 9.5, "hold_years": 6, "debt_pct": 0.72},
    {"name": "First Data (2007)", "sponsor": "KKR", "entry_ebitda_b": 2.3,
     "entry_ev_b": 29.0, "exit_ev_b": 18.5, "hold_years": 2, "debt_pct": 0.68},
    {"name": "Hertz (2005)", "sponsor": "CD&R+Carlyle+MLIM", "entry_ebitda_b": 1.9,
     "entry_ev_b": 15.0, "exit_ev_b": 17.5, "hold_years": 5, "debt_pct": 0.78},
    {"name": "Heinz (2013)", "sponsor": "Berkshire+3G", "entry_ebitda_b": 2.5,
     "entry_ev_b": 28.0, "exit_ev_b": 46.0, "hold_years": 5, "debt_pct": 0.70},
    {"name": "Toys 'R' Us (2005)", "sponsor": "KKR+Bain+Vornado", "entry_ebitda_b": 1.0,
     "entry_ev_b": 6.6, "exit_ev_b": 0.0, "hold_years": 12, "debt_pct": 0.80},
]

# ── M&A NLP Keywords (extends FOMC sentiment dictionary) ──
MA_SENTIMENT_DICT = {
    "positive": [
        "synergy", "accretive", "strategic fit", "value creation", "complementary",
        "scale", "efficiency", "growth opportunity", "unlock value", "premium",
        "best offer", "superior proposal", "enhanced", "innovation", "expansion",
    ],
    "negative": [
        "dilutive", "integration risk", "goodwill impairment", "overpayment",
        "regulatory concern", "antitrust", "hostile", "unfavorable", "rejection",
        "termination fee", "break-up fee", "material adverse change",
    ],
    "uncertainty": [
        "conditional", "subject to", "pending", "regulatory approval",
        "shareholder vote", "expected to close", "contingent",
        "MAC clause", "fiduciary out", "go-shop",
    ],
}

# ── Classic M&A / LBO Papers ──
MA_PAPERS = {
    "Andrade et al. (2001)": {
        "title": "New Evidence and Perspectives on Mergers",
        "journal": "Journal of Economic Perspectives",
        "method": "Meta-analysis of event studies",
        "key_result": "Target +3-8% CAR; acquirer ~0%",
    },
    "Moeller et al. (2005)": {
        "title": "Wealth Destruction on a Massive Scale?",
        "journal": "Journal of Finance",
        "method": "Large-sample event study 1991-2001",
        "key_result": "Acquirers lost $240B during 1998-2001 bubble",
    },
    "Kaplan & Strömberg (2009)": {
        "title": "Leveraged Buyouts and Private Equity",
        "journal": "Journal of Economic Perspectives",
        "method": "Survey of LBO/PE industry",
        "key_result": "LBO returns driven by operational improvement + leverage tax shield",
    },
    "Axelson et al. (2013)": {
        "title": "Borrow Cheap, Buy High?",
        "journal": "Quarterly Journal of Economics",
        "method": "Large-sample LBO determinants study",
        "key_result": "LBO leverage countercyclical; cheap debt drives activity",
    },
    "Harford (2005)": {
        "title": "What drives merger waves?",
        "journal": "Journal of Financial Economics",
        "method": "Industry-merger wave analysis",
        "key_result": "Waves cluster with macro shocks, capital liquidity, regulation",
    },
    "Masulis et al. (2007)": {
        "title": "Corporate Governance and Firm Value in Mergers",
        "journal": "Journal of Financial Economics",
        "method": "Acquirer governance × M&A returns",
        "key_result": "Better governed acquirers earn higher CAR",
    },
}

# ── Industries for Filter ──
INDUSTRIES = ["Tech", "Pharma", "Energy", "Media", "Telecom", "Food",
              "Defense", "Luxury", "Retail", "Fintech", "REIT"]

# ── LBO Default Parameters ──
LBO_DEFAULTS = {
    "entry_revenue_b": 5.0,
    "entry_ebitda_margin": 0.20,
    "entry_ev_ebitda": 8.0,
    "debt_pct": 0.65,
    "interest_rate": 0.06,
    "tax_rate": 0.21,
    "revenue_growth": 0.05,
    "margin_expansion": 0.005,  # 50bp per year
    "exit_ev_ebitda": 9.0,
    "hold_years": 5,
}
