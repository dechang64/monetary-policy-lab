# -*- coding: utf-8 -*-
"""
WRDS Connector — Mutual Fund Flow Data
For Direction 2: Portfolio Rebalancing and Cross-Asset Contagion

Fetches CRSP Mutual Fund data from WRDS:
- Fund header (tna, fund_name, objective code)
- Fund returns (monthly)
- Constructs net flows for 7 asset classes

7 Asset Classes (per Eileen Zhang's proposal):
1. Large-cap equities      (CRSP obj: 'EDYG' or lipper_class containing 'Large-Cap')
2. Small-cap equities       (CRSP obj: 'EDYS' or lipper_class containing 'Small-Cap')
3. Emerging market equities (CRSP obj: 'EDYE' or lipper_class containing 'Emerging')
4. Developed market equities (CRSP obj: 'EDYD' or lipper_class containing 'International')
5. Real assets              (CRSP obj: 'REIT' or lipper_class containing 'Real Estate')
6. Corporate bonds          (CRSP obj: 'CBDI' or lipper_class containing 'Corporate')
7. Government bonds         (CRSP obj: 'GBDI' or lipper_class containing 'Government')

Requirements:
    pip install wrds pandas

Usage:
    from wrds_connector import WRDSConnector
    conn = WRDSConnector(wrds_username='your_username')
    flows = conn.fetch_fund_flows(start='2006-01-01', end='2022-12-31')
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent dir for audit_chain
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_chain import AuditChain


class WRDSConnector:
    """WRDS data connector for CRSP Mutual Fund data."""
    
    # 7 asset class mappings using CRSP objective codes
    # Reference: CRSP Mutual Fund database objective codes
    ASSET_CLASS_MAP = {
        'large_cap_equity': {
            'crsp_obj': ['EDYG'],  # Equity - Domestic - Large Growth
            'lipper_keywords': ['Large-Cap', 'Large Cap', 'Large Growth', 'Large Value', 'Large Blend'],
        },
        'small_cap_equity': {
            'crsp_obj': ['EDYS'],  # Equity - Domestic - Small
            'lipper_keywords': ['Small-Cap', 'Small Cap', 'Small Growth', 'Small Value', 'Small Blend'],
        },
        'emerging_market_equity': {
            'crsp_obj': ['EDYE'],  # Equity - International - Emerging
            'lipper_keywords': ['Emerging Markets', 'Emerging Market', 'Diversified Emerging'],
        },
        'developed_market_equity': {
            'crsp_obj': ['EDYD', 'EDYF'],  # Equity - International - Developed
            'lipper_keywords': ['International', 'Foreign', 'Global', 'International Large'],
        },
        'real_assets': {
            'crsp_obj': ['REIT', 'EDYR'],
            'lipper_keywords': ['Real Estate', 'REIT', 'Natural Resources', 'Commodity'],
        },
        'corporate_bonds': {
            'crsp_obj': ['CBDI', 'CBDG', 'CBDV'],
            'lipper_keywords': ['Corporate Bond', 'Corporate Debt', 'High Yield', 'Investment Grade Bond'],
        },
        'government_bonds': {
            'crsp_obj': ['GBDI', 'GBDM'],
            'lipper_keywords': ['Government Bond', 'Treasury', 'US Government', 'Mortgage'],
        },
    }
    
    def __init__(self, wrds_username, audit_chain=None):
        """
        Initialize WRDS connection.
        
        Args:
            wrds_username: WRDS username (Eileen's Rutgers or Dechang's account)
            audit_chain: AuditChain instance for logging
        """
        self.username = wrds_username
        self.db = None
        self.audit = audit_chain or AuditChain("direction2")
    
    def connect(self):
        """Connect to WRDS PostgreSQL."""
        import wrds
        self.db = wrds.Connection(wrds_username=self.username)
        self.audit.log_data_access(
            source="WRDS",
            query="CONNECT",
            metadata={"username": self.username}
        )
        print(f"✅ Connected to WRDS as {self.username}")
    
    def fetch_fund_header(self, start_date='2006-01-01', end_date='2022-12-31'):
        """
        Fetch CRSP Mutual Fund header data including fund-level controls.
        
        Combines three tables:
        - fund_hdr: basic fund info (name, dates, delisting)
        - fund_style: objective codes + lipper class (latest record per fund)
        - fund_summary2: expense ratio, mgmt fee, TNA (latest record per fund)
        """
        query = f"""
            WITH latest_style AS (
                SELECT DISTINCT ON (crsp_fundno)
                    crsp_fundno, crsp_obj_cd, 
                    lipper_class_name, lipper_obj_name
                FROM crsp_q_mutualfunds.fund_style
                WHERE crsp_fundno IS NOT NULL
                ORDER BY crsp_fundno, begdt DESC
            ),
            latest_summary AS (
                SELECT DISTINCT ON (crsp_fundno)
                    crsp_fundno, 
                    tna_latest,
                    exp_ratio, mgmt_fee
                FROM crsp_q_mutualfunds.fund_summary2
                WHERE crsp_fundno IS NOT NULL
                  AND exp_ratio IS NOT NULL
                ORDER BY crsp_fundno, caldt DESC
            )
            SELECT 
                h.crsp_fundno, h.fund_name, 
                s.crsp_obj_cd, s.lipper_class_name, s.lipper_obj_name,
                h.first_offer_dt, h.end_dt,
                h.delist_cd,
                su.tna_latest as latest_tna,
                su.exp_ratio, su.mgmt_fee
            FROM crsp_q_mutualfunds.fund_hdr h
            LEFT JOIN latest_style s ON h.crsp_fundno = s.crsp_fundno
            LEFT JOIN latest_summary su ON h.crsp_fundno = su.crsp_fundno
            WHERE h.first_offer_dt <= '{end_date}'
              AND (h.end_dt >= '{start_date}' OR h.end_dt IS NULL)
              AND h.crsp_fundno IS NOT NULL
        """
        
        self.audit.log_data_access(source="WRDS", query="fund_hdr+style+summary2 fetch")
        
        df = self.db.raw_sql(query)
        print(f"✅ Fetched {len(df)} fund header records")
        return df
    
    def fetch_fund_returns(self, start_date='2006-01-01', end_date='2022-12-31'):
        """
        Fetch CRSP Mutual Fund monthly returns and TNA.
        
        Returns: fundno, date, mret, mtna (monthly total net assets)
        """
        query = f"""
            SELECT 
                crsp_fundno, caldt, mret, mtna
            FROM crsp_q_mutualfunds.monthly_tna_ret_nav
            WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
              AND crsp_fundno IS NOT NULL
            ORDER BY crsp_fundno, caldt
        """
        
        self.audit.log_data_access(source="WRDS", query="monthly_tna_ret_nav fetch")
        
        df = self.db.raw_sql(query)
        print(f"✅ Fetched {len(df)} fund-month return records")
        return df
    
    def classify_funds(self, fund_header_df):
        """
        Classify funds into 7 asset classes using CRSP objective codes
        and Lipper class names.
        """
        classified = []
        
        for asset_class, mapping in self.ASSET_CLASS_MAP.items():
            # Match by CRSP objective code
            mask = fund_header_df['crsp_obj_cd'].isin(mapping['crsp_obj'])
            
            # Also match by Lipper keywords
            for kw in mapping['lipper_keywords']:
                lipper_mask = fund_header_df['lipper_class_name'].str.contains(
                    kw, case=False, na=False
                )
                mask = mask | lipper_mask
            
            matched = fund_header_df[mask].copy()
            matched['asset_class'] = asset_class
            classified.append(matched)
        
        result = pd.concat(classified, ignore_index=True)
        
        # Remove duplicates (a fund might match multiple classes — keep first)
        result = result.drop_duplicates(subset=['crsp_fundno'], keep='first')
        
        self.audit.log_human_decision(
            f"Classified {len(result)} funds into 7 asset classes. "
            f"Distribution: {result['asset_class'].value_counts().to_dict()}",
            author="ai"
        )
        
        print(f"✅ Classified {len(result)} funds into 7 asset classes")
        print(result['asset_class'].value_counts())
        return result
    
    def compute_fund_flows(self, returns_df, fund_header_df, 
                           fomc_dates=None, window_days=5,
                           event_window='same'):
        """
        Compute net fund flows following FOMC announcements.
        
        Flow formula (standard CRSP approach, Chevalier & Ellison 1997):
            flow_t = (TNA_t - TNA_{t-1} * (1 + r_t)) / TNA_{t-1}
        
        Optimization: Added fund-level control variables per Fecht & Kellers
        (2026) and Blanco et al. (2025):
          - log_tna: fund size control (log of TNA)
          - flow_vol_12m: fragility proxy (12-month rolling std of flow)
          - ret_12m_lag: return-chasing proxy (12-month lagged return)
          - exp_ratio: expense ratio (median-filled for missing values)
        
        Args:
            returns_df: Monthly fund returns from fetch_fund_returns()
        
        Args:
            returns_df: Monthly fund returns from fetch_fund_returns()
            fund_header_df: Classified fund header from classify_funds()
            fomc_dates: List of FOMC announcement dates
            window_days: Days after FOMC (unused with monthly data, kept for API)
            event_window: Flow measurement window:
                'same' = FOMC month flow (default)
                'post' = FOMC+1 month flow (cleaner post-shock)
                'diff' = (FOMC+1 month) - (FOMC-1 month) flow difference
        
        Returns:
            DataFrame with columns: fomc_date, asset_class, net_flow_pct,
                                    log_tna, flow_vol_12m, ret_12m_lag, exp_ratio
        """
        # Merge returns with asset class and fund-level controls
        control_cols = ['crsp_fundno', 'asset_class']
        available_controls = [c for c in ['latest_tna', 'exp_ratio', 'mgmt_fee'] 
                              if c in fund_header_df.columns]
        control_cols = control_cols + available_controls
        
        merged = returns_df.merge(
            fund_header_df[control_cols], 
            on='crsp_fundno', 
            how='inner'
        )
        
        # Ensure caldt is datetime (WRDS may return strings)
        merged['caldt'] = pd.to_datetime(merged['caldt'])
        
        # Compute monthly flow
        merged = merged.sort_values(['crsp_fundno', 'caldt'])
        merged['mtna_prev'] = merged.groupby('crsp_fundno')['mtna'].shift(1)
        
        # flow_t = (TNA_t - TNA_{t-1} * (1 + r_t)) / TNA_{t-1}
        merged['flow'] = (
            merged['mtna'] - merged['mtna_prev'] * (1 + merged['mret'])
        ) / merged['mtna_prev']
        
        # Clean infinite/NaN values from flow computation (e.g., mtna_prev=0)
        merged['flow'] = merged['flow'].replace([np.inf, -np.inf], np.nan)
        
        # ── FIX 1: Winsorize fund-level flows at 1%/99% before aggregation ──
        # Extreme outliers (e.g., 383% from fund mergers/liquidations) distort
        # asset-class aggregates. Hard-filter |flow| > 50% as data errors,
        # then winsorize the remaining at 1%/99%.
        n_before = merged['flow'].notna().sum()
        merged.loc[merged['flow'].abs() > 50, 'flow'] = np.nan  # hard filter
        flow_vals = merged['flow'].dropna()
        if len(flow_vals) > 0:
            p01, p99 = flow_vals.quantile([0.01, 0.99])
            merged['flow'] = merged['flow'].clip(lower=p01, upper=p99)
        print(f"   Winsorized fund flows (1%/99%), filtered {n_before - merged['flow'].notna().sum()} extreme obs")

        # Construct fund-level controls
        # 1. log_tna: log of current TNA (fund size)
        merged['log_tna'] = np.log(merged['mtna'].clip(lower=1))

        # 2. flow_vol_12m: 12-month rolling std of flow (fragility proxy)
        merged['flow_vol_12m'] = (
            merged.groupby('crsp_fundno')['flow']
            .rolling(12, min_periods=6)
            .std()
            .reset_index(level=0, drop=True)
        )
        # ── FIX 2: Fix flow_vol_12m zeros (moved AFTER column creation) ──
        # Zero volatility is nonsensical; replace with NaN so median-fill handles it
        merged['flow_vol_12m'] = merged['flow_vol_12m'].replace(0, np.nan)

        # 3. ret_12m_lag: 12-month lagged cumulative return (return-chasing proxy)
        merged['ret_12m_lag'] = (
            merged.groupby('crsp_fundno')['mret']
            .rolling(12, min_periods=6)
            .sum()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        
        # 4. exp_ratio: from fund header (median-fill missing values)
        if 'exp_ratio' in merged.columns:
            median_exp = merged['exp_ratio'].median()
            merged['exp_ratio'] = merged['exp_ratio'].fillna(median_exp)
        
        # ── FIX 3: TNA-weighted aggregation instead of simple mean ──
        # Simple mean washes out the signal (outflows averaged with inflows).
        # Standard approach: dollar_flow = flow × TNA_prev, then
        # aggregated_flow = sum(dollar_flow) / sum(TNA_prev)
        # This properly accounts for fund size and preserves net flow sign.
        merged['dollar_flow'] = merged['flow'] * merged['mtna_prev']
        
        monthly_flows = merged.groupby([
            'asset_class', pd.Grouper(key='caldt', freq='ME')
        ]).agg({
            'dollar_flow': 'sum',
            'mtna_prev': 'sum',
            'log_tna': 'mean',
            'flow_vol_12m': 'mean',
            'ret_12m_lag': 'mean',
            'exp_ratio': 'mean',
        }).reset_index()
        
        # TNA-weighted net flow percentage
        monthly_flows['net_flow_pct'] = (
            monthly_flows['dollar_flow'] / monthly_flows['mtna_prev'] * 100
        )
        monthly_flows.drop(columns=['dollar_flow', 'mtna_prev'], inplace=True)
        monthly_flows.rename(columns={'caldt': 'month'}, inplace=True)
        
        # Fill NaN controls with median
        for col in ['flow_vol_12m', 'ret_12m_lag', 'exp_ratio', 'log_tna']:
            if col in monthly_flows.columns:
                monthly_flows[col] = monthly_flows[col].fillna(monthly_flows[col].median())
        
        # If FOMC dates provided, create event-window flows
        if fomc_dates is not None:
            control_cols = [c for c in ['log_tna', 'flow_vol_12m', 'ret_12m_lag', 'exp_ratio']
                            if c in monthly_flows.columns]
            event_flows = []
            for fomc_date in fomc_dates:
                fomc_month = pd.Timestamp(fomc_date).to_period('M')
                post_month = fomc_month + 1
                pre_month = fomc_month - 1
                for asset_class in self.ASSET_CLASS_MAP.keys():
                    row_same = monthly_flows[
                        (monthly_flows['asset_class'] == asset_class) &
                        (monthly_flows['month'].dt.to_period('M') == fomc_month)
                    ]
                    row_post = monthly_flows[
                        (monthly_flows['asset_class'] == asset_class) &
                        (monthly_flows['month'].dt.to_period('M') == post_month)
                    ]
                    row_pre = monthly_flows[
                        (monthly_flows['asset_class'] == asset_class) &
                        (monthly_flows['month'].dt.to_period('M') == pre_month)
                    ]
                    
                    flow_same = row_same['net_flow_pct'].mean()
                    flow_post = row_post['net_flow_pct'].mean()
                    flow_pre = row_pre['net_flow_pct'].mean()
                    
                    if event_window == 'post':
                        flow = flow_post
                        row_data = row_post
                    elif event_window == 'diff':
                        # ── FIX 7: diff窗口NaN处理 ──
                        # 旧代码：缺失侧替换为0.0，污染diff（变成纯post）
                        # 新代码：任一侧缺失，整个diff为NaN
                        if pd.isna(flow_post) or pd.isna(flow_pre):
                            flow = np.nan
                        else:
                            flow = flow_post - flow_pre
                        row_data = row_post
                    else:  # 'same'
                        flow = flow_same
                        row_data = row_same
                    
                    entry = {
                        'fomc_date': fomc_date,
                        'asset_class': asset_class,
                        # ── FIX 7b: Keep NaN as NaN, don't replace with 0.0 ──
                        # 旧代码把NaN替换成0.0，把"无数据"变成"零流动"，污染回归
                        'net_flow_pct': flow if (not pd.isna(flow) and not np.isinf(flow)) else np.nan,
                    }
                    for col in control_cols:
                        val = row_data[col].mean()
                        entry[col] = val if (not pd.isna(val) and not np.isinf(val)) else 0.0
                    
                    event_flows.append(entry)
            
            result = pd.DataFrame(event_flows)
            self.audit.log_data_access(
                source="computed",
                query="fund flows around FOMC",
                rows_returned=len(result)
            )
            return result
        
        return monthly_flows
    
    def close(self):
        """Close WRDS connection."""
        if self.db:
            self.db.close()
            print("✅ WRDS connection closed")


# =============================================================================
# Risk ranking for H4 (Risk-Ladder Substitution)
# =============================================================================
RISK_RANKING = {
    'small_cap_equity': 7,           # Highest risk
    'emerging_market_equity': 6,
    'developed_market_equity': 5,
    'large_cap_equity': 4,
    'real_assets': 3,
    'corporate_bonds': 2,
    'government_bonds': 1,           # Lowest risk
}


if __name__ == "__main__":
    print("=" * 60)
    print("WRDS Connector — Direction 2 Fund Flow Analysis")
    print("=" * 60)
    print("\nThis script requires WRDS credentials.")
    print("Usage:")
    print("  1. Set WRDS_USERNAME environment variable")
    print("  2. Run: python wrds_connector.py")
    print("\nOr in Python:")
    print("  from wrds_connector import WRDSConnector")
    print("  conn = WRDSConnector(wrds_username='your_username')")
    print("  conn.connect()")
    print("  header = conn.fetch_fund_header()")
    print("  returns = conn.fetch_fund_returns()")
    print("  classified = conn.classify_funds(header)")
    print("  flows = conn.compute_fund_flows(returns, classified)")
