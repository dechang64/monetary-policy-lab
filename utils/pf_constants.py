# Personal Finance Lab — Constants

# Federal tax brackets 2024 (single filer)
TAX_BRACKETS_SINGLE = [
    (0, 0.10),
    (11600, 0.12),
    (47150, 0.22),
    (100525, 0.24),
    (191950, 0.32),
    (243725, 0.35),
    (609350, 0.37),
]

# State tax (simplified average)
STATE_TAX = {
    "None (no state tax)": 0.0,
    "Low (~3%)": 0.03,
    "Medium (~5%)": 0.05,
    "High (~7%)": 0.07,
    "California (~9.3%)": 0.093,
    "New York (~6.8%)": 0.068,
}

# Asset class expected returns (historical 50yr avg)
ASSET_EXPECTED_RETURNS = {
    "US Large Cap (S&P 500)": 0.10,
    "US Small Cap": 0.11,
    "International Equity": 0.08,
    "Emerging Markets": 0.09,
    "US Bonds": 0.04,
    "Real Estate (REITs)": 0.07,
    "Cash / T-Bills": 0.02,
    "Gold": 0.05,
}

ASSET_VOLS = {
    "US Large Cap (S&P 500)": 0.16,
    "US Small Cap": 0.20,
    "International Equity": 0.18,
    "Emerging Markets": 0.24,
    "US Bonds": 0.06,
    "Real Estate (REITs)": 0.18,
    "Cash / T-Bills": 0.01,
    "Gold": 0.16,
}

# FICO score factors
FICO_FACTORS = [
    ("Payment History", 0.35),
    ("Credit Utilization", 0.30),
    ("Length of Credit History", 0.15),
    ("Credit Mix", 0.10),
    ("New Credit Inquiries", 0.10),
]

# Credit card typical terms
CC_APR_RANGE = (0.15, 0.29)  # 15% - 29%

# Student loan types
STUDENT_LOAN_TYPES = {
    "Federal Subsidized": 0.0499,
    "Federal Unsubsidized": 0.0499,
    "Federal PLUS": 0.0754,
    "Private (Undergrad)": 0.065,
    "Private (Grad)": 0.075,
}
