"""
Helper utilities for Monetary Policy Research Lab.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_returns(n_days: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic daily returns for demo purposes.
    In production, replace with real data from CRSP/Bloomberg.
    """
    np.random.seed(seed)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    
    assets = {
        "S&P 500": 0.0004, "NASDAQ": 0.0005, "Russell 2000": 0.0003,
        "MSCI EM": 0.0002, "US 2Y Treasury": 0.0001, "US 10Y Treasury": 0.0001,
        "US 30Y Treasury": 0.0001, "Corporate BBB": 0.0002,
        "DXY (USD)": 0.0000, "Gold": 0.0002, "Oil (WTI)": 0.0001,
        "Bitcoin": 0.002,
    }
    vols = {
        "S&P 500": 0.01, "NASDAQ": 0.013, "Russell 2000": 0.015,
        "MSCI EM": 0.014, "US 2Y Treasury": 0.003, "US 10Y Treasury": 0.006,
        "US 30Y Treasury": 0.008, "Corporate BBB": 0.005,
        "DXY (USD)": 0.005, "Gold": 0.01, "Oil (WTI)": 0.02,
        "Bitcoin": 0.04,
    }
    
    # Correlation structure
    n_assets = len(assets)
    corr = np.eye(n_assets)
    # Equities correlated with each other
    for i in range(4):
        for j in range(4):
            corr[i, j] = 0.6 + 0.3 * (i == j)
    # Bonds correlated with each other, negatively with equities
    for i in range(4, 8):
        for j in range(4, 8):
            corr[i, j] = 0.7 + 0.2 * (i == j)
        for j in range(4):
            corr[i, j] = -0.2
    # Crypto weakly correlated
    corr[11, :] = 0.1
    corr[:, 11] = 0.1
    corr[11, 11] = 1.0
    
    L = np.linalg.cholesky(corr)
    z = np.random.randn(n_days, n_assets)
    correlated = z @ L.T
    
    df = pd.DataFrame(index=dates)
    for i, (asset, mu) in enumerate(assets.items()):
        df[asset] = correlated[:, i] * vols[asset] + mu
    
    return df


def generate_fomc_surprises(fomc_dates: list, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic monetary policy surprises.
    In production, use Kuttner (2001) method with Fed funds futures data.
    
    Surprise = Actual Change - Expected Change (from futures)
    """
    np.random.seed(seed)
    
    surprises = []
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        # Most meetings: no change expected
        expected = np.random.normal(0, 0.02)
        # Actual: sometimes matches, sometimes surprises
        actual = expected + np.random.normal(0, 0.08)
        surprises.append({
            "date": date,
            "expected_change": round(expected, 4),
            "actual_change": round(actual, 4),
            "surprise": round(actual - expected, 4),
        })
    
    return pd.DataFrame(surprises).set_index("date")


def generate_two_shocks(surprises_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Decompose surprises into policy shock and information shock.
    Uses a simplified version of Jarociński & Karadi (2020).
    
    In production: estimate via SVAR with high-frequency data.
    """
    np.random.seed(seed)
    
    df = surprises_df.copy()
    # Simplified decomposition: policy ~ 60%, information ~ 40%
    # with some noise
    policy_weight = 0.6 + np.random.normal(0, 0.1, len(df))
    policy_weight = np.clip(policy_weight, 0.3, 0.8)
    
    df["policy_shock"] = df["surprise"] * policy_weight
    df["info_shock"] = df["surprise"] * (1 - policy_weight)
    df["policy_pct"] = policy_weight
    
    return df


def generate_sentiment_scores(fomc_dates: list, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic FOMC sentiment scores.
    In production: use FinBERT-FOMC model on actual statements.
    
    Score range: -1 (very hawkish) to +1 (very dovish)
    """
    np.random.seed(seed)
    
    scores = []
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        # Sentiment drifts over time
        t = (date - pd.Timestamp("2015-01-01")).days / 365.25
        base = 0.1 * np.sin(t * 0.5)  # cyclical component
        noise = np.random.normal(0, 0.15)
        score = np.clip(base + noise, -1, 1)
        
        # 2022 hiking cycle: hawkish
        if pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2023-06-01"):
            score -= 0.4
        # 2020 COVID: dovish
        elif pd.Timestamp("2020-03-01") <= date <= pd.Timestamp("2020-12-31"):
            score += 0.5
        
        scores.append({
            "date": date,
            "sentiment_score": round(score, 3),
            "sentiment_label": "Hawkish" if score < -0.15 else ("Dovish" if score > 0.15 else "Neutral"),
            "word_count": np.random.randint(300, 800),
            "readability": round(np.random.normal(12, 2), 1),  # Flesch-Kincaid
        })
    
    return pd.DataFrame(scores).set_index("date")


def generate_portfolio_flows(fomc_dates: list, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic mutual fund flow data around FOMC events.
    In production: use Thomson Reuters / CRSP mutual fund data.
    
    Shows net flows (in $B) by asset category in event windows.
    """
    np.random.seed(seed)
    
    categories = ["US Large Cap", "US Small Cap", "International DM", "EM", 
                  "US Gov Bond", "US Corp Bond", "High Yield", "Money Market",
                  "Commodities", "Real Estate"]
    
    rows = []
    for date_str in fomc_dates:
        date = pd.Timestamp(date_str)
        # Base flows
        base = np.random.normal(0, 2, len(categories))
        # FOMC effect: rate hikes → flows from bonds to equities
        surprise = np.random.normal(0, 0.1)
        fomc_effect = np.array([
            surprise * 3,   # Large Cap: benefit from hikes (initially)
            surprise * -1,  # Small Cap: hurt by hikes
            surprise * -2,  # Int'l DM: hurt by USD
            surprise * -4,  # EM: hurt most
            surprise * -5,  # Gov Bond: hurt by rate hikes
            surprise * -3,  # Corp Bond: hurt
            surprise * -6,  # High Yield: hurt most
            surprise * 2,   # Money Market: benefit
            surprise * -1,  # Commodities: mixed
            surprise * -2,  # Real Estate: hurt
        ])
        
        for i, cat in enumerate(categories):
            rows.append({
                "fomc_date": date,
                "category": cat,
                "flow_pre_b": round(base[i] + np.random.normal(0, 1), 2),
                "flow_post_b": round(base[i] + fomc_effect[i] + np.random.normal(0, 1), 2),
                "flow_change_b": round(fomc_effect[i] + np.random.normal(0, 0.5), 2),
            })
    
    return pd.DataFrame(rows)


def compute_event_study_stats(
    returns: pd.DataFrame,
    fomc_dates: list,
    window_pre: int = 1,
    window_post: int = 5,
) -> pd.DataFrame:
    """
    Compute event study statistics: AR, CAR, t-stats.
    Simplified market model approach.
    """
    results = []
    
    for asset in returns.columns:
        # Market model: use S&P 500 as market proxy (skip for S&P itself)
        if asset == "S&P 500":
            market = returns["NASDAQ"]  # use NASDAQ as proxy
        else:
            market = returns["S&P 500"]
        
        # Estimate market model on non-event days
        event_mask = returns.index.isin([pd.Timestamp(d) for d in fomc_dates])
        non_event = returns[~event_mask]
        
        if len(non_event) < 100:
            continue
        
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            market.values, returns[asset].values
        )
        
        # Compute AR and CAR for each event
        for date_str in fomc_dates:
            date = pd.Timestamp(date_str)
            ar_list = []
            
            for d in range(-window_pre, window_post + 1):
                target_date = date + timedelta(days=d)
                if target_date in returns.index:
                    actual = returns.loc[target_date, asset]
                    if target_date in market.index:
                        expected = intercept + slope * market.loc[target_date]
                    else:
                        expected = intercept
                    ar = actual - expected
                    ar_list.append(ar)
            
            if ar_list:
                car = sum(ar_list)
                n = len(ar_list)
                # Simplified t-stat
                t_stat = car / (np.std(ar_list) / np.sqrt(n)) if np.std(ar_list) > 0 else 0
                
                results.append({
                    "asset": asset,
                    "fomc_date": date,
                    "AR_mean": round(np.mean(ar_list), 6),
                    "CAR": round(car, 6),
                    "t_stat": round(t_stat, 3),
                    "n_days": n,
                })
    
    return pd.DataFrame(results)


def format_pct(val: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    sign = "+" if val > 0 else ""
    return f"{sign}{val * 100:.{decimals}f}%"


def format_bp(val: float) -> str:
    """Format a decimal as basis points."""
    sign = "+" if val > 0 else ""
    return f"{sign}{val * 10000:.0f}bp"
