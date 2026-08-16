# -*- coding: utf-8 -*-
"""
Master Pipeline — Direction 2
Run end-to-end: WRDS fetch → Flow computation → H1-H4 → Robustness → Figures

Usage:
    python run_pipeline.py --wrds-username YOUR_USERNAME
    
    # Or set environment variable:
    export WRDS_USERNAME=your_username
    python run_pipeline.py
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_chain import AuditChain
from contribution_tracker import ContributionTracker
from wrds_connector import WRDSConnector, RISK_RANKING
from h1_h4_regression import (
    h1_risk_off, h2_risk_on, h3_asymmetry, h4_risk_ladder,
    run_all_hypotheses
)
from load_phase1_shocks import load_phase1_shocks


def main():
    parser = argparse.ArgumentParser(description="Direction 2 Pipeline")
    parser.add_argument("--wrds-username", default=os.environ.get("WRDS_USERNAME"),
                        help="WRDS username")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2022-12-31")
    parser.add_argument("--skip-wrds", action="store_true",
                        help="Skip WRDS fetch (use cached data)")
    parser.add_argument("--event-window", default="same",
                        choices=["same", "post", "diff"],
                        help="Flow measurement: same=FOMC month, post=FOMC+1, diff=post-pre")
    args = parser.parse_args()
    
    # Initialize audit chain
    chain = AuditChain("direction2", base_dir="..")
    tracker = ContributionTracker("direction2", base_dir="..")
    
    # Ensure output directories exist
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../docs", exist_ok=True)
    os.makedirs("../audit_chain", exist_ok=True)
    
    chain.log_prompt(
        f"Direction 2 pipeline started. WRDS user: {args.wrds_username or 'cached'}. "
        f"Date range: {args.start_date} to {args.end_date}."
    )
    
    print("=" * 60)
    print("Direction 2 — Portfolio Rebalancing Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch WRDS data
    if not args.skip_wrds:
        if not args.wrds_username:
            print("❌ WRDS username required. Set WRDS_USERNAME or use --wrds-username")
            sys.exit(1)
        
        print("\n[Step 1] Fetching CRSP Mutual Fund data from WRDS...")
        conn = WRDSConnector(args.wrds_username, audit_chain=chain)
        conn.connect()
        
        fund_header = conn.fetch_fund_header(args.start_date, args.end_date)
        fund_returns = conn.fetch_fund_returns(args.start_date, args.end_date)
        classified = conn.classify_funds(fund_header)
        
        # Save cached data
        fund_header.to_csv("../results/fund_header.csv", index=False)
        fund_returns.to_csv("../results/fund_returns.csv", index=False)
        classified.to_csv("../results/fund_classified.csv", index=False)
        
        conn.close()
    else:
        print("\n[Step 1] Using cached WRDS data...")
        fund_header = pd.read_csv("../results/fund_header.csv")
        fund_returns = pd.read_csv("../results/fund_returns.csv")
        classified = pd.read_csv("../results/fund_classified.csv")
    
    # Step 2: Load Phase 1 shocks
    print("\n[Step 2] Loading Phase 1 JK-decomposed shocks...")
    shocks_df = load_phase1_shocks("../results/minutes_sentiment_corrected.csv")
    
    chain.log_data_access(
        source="Phase 1",
        query="JK-decomposed shocks (mp_shock, cbi_shock, path_shock)",
        rows_returned=len(shocks_df)
    )
    
    # Step 3: Compute fund flows
    print(f"\n[Step 3] Computing fund flows (window={args.event_window})...")
    fomc_dates = shocks_df['date'].tolist() if len(shocks_df) > 0 else []
    
    conn = WRDSConnector("__cached__", audit_chain=chain)
    flows = conn.compute_fund_flows(
        fund_returns, classified, 
        fomc_dates=fomc_dates, window_days=5,
        event_window=args.event_window
    )
    # Rename fomc_date → date to match H1-H4 regression merge key
    if 'fomc_date' in flows.columns:
        flows = flows.rename(columns={'fomc_date': 'date'})
    # Ensure date columns are datetime on both sides for merge
    flows['date'] = pd.to_datetime(flows['date'])
    shocks_df['date'] = pd.to_datetime(shocks_df['date'])
    flows.to_csv("../results/fund_flows.csv", index=False)
    
    # Step 4: Run H1-H4 regressions
    print("\n[Step 4] Running H1-H4 regressions...")
    if len(flows) > 0 and len(shocks_df) > 0:
        results = run_all_hypotheses(flows, shocks_df, audit_chain=chain)

        # Save results (include event_window in filename for robustness comparison)
        results_file = f"../results/h1_h4_results_{args.event_window}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {results_file}")
    else:
        print("⚠️  No data available for regression. Skipping H1-H4.")
        results = {}
    
    # Step 5: Generate contribution report
    print("\n[Step 5] Generating contribution report...")
    tracker.record_file("code/run_pipeline.py", author="ai",
                        model="claude-sonnet-4",
                        note="AI-generated master pipeline")
    report = tracker.generate_report("../docs/contribution_report.md")
    
    # Step 6: Verify audit chain
    valid, msg = chain.verify_chain()
    print(f"\n[Step 6] Audit chain verification: {msg}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Results: ../results/")
    print(f"  Audit chain: ../audit_chain/")
    print(f"  Contribution report: ../docs/contribution_report.md")
    print("=" * 60)
    
    chain.log_ai_response(
        f"Pipeline complete. Results saved. Audit chain verified: {valid}",
        model="pipeline"
    )


if __name__ == "__main__":
    main()
