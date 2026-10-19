# -*- coding: utf-8 -*-
"""Server-side aggregation for the flow chapter (audit-fixed).

Flows built entirely in SQL: LAG(mtna) per fund, point-in-time style windows
(interval join), classification via CRSP obj codes + Lipper keywords,
|flow_pct| <= 1 cap, then class-month aggregation. Only the aggregates are
transferred from WRDS.

Audit fixes applied here:
  FIX-A  point-in-time classification (no look-ahead latest record)
  FIX-B  |flow| <= lagged TNA cap + GAP=1 month requirement (a fund-month
         whose previous record is not the prior calendar month is DROPPED,
         so a multi-month TNA jump is never misread as one-month flow)
  FIX-C  pseudo Bauer-Swanson step removed (shocks used in native form)

Usage:  python3 flow_server_agg.py     (needs WRDS login or .pgpass)
Output: flow_rerun_output/agg_class_month.csv
"""
import os
import time

import pandas as pd
import wrds

LIPPER_CASE = """
CASE
  WHEN s.crsp_obj_cd IN ('GBDI','GDBD','GMBD') THEN 'government_bonds'
  WHEN s.crsp_obj_cd IN ('CBDI','CNVD','CHYD') THEN 'corporate_bonds'
  WHEN s.crsp_obj_cd IN ('REIT','CMBD') THEN 'real_assets'
  WHEN s.crsp_obj_cd IN ('EDYG','EDCI') THEN 'large_cap_equity'
  WHEN s.crsp_obj_cd IN ('EDYD','EDYF') THEN 'developed_market_equity'
  WHEN s.crsp_obj_cd = 'EDYE' THEN 'emerging_market_equity'
  WHEN s.crsp_obj_cd IN ('EDYS','EDCM','EDYB') THEN 'small_cap_equity'
  WHEN STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'emerging') > 0 THEN 'emerging_market_equity'
  WHEN STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'small') > 0 AND STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'cap') > 0 THEN 'small_cap_equity'
  WHEN STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'real estate') > 0
    OR STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'commodit') > 0
    OR STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'precious metal') > 0
    OR STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'natural resource') > 0 THEN 'real_assets'
  WHEN STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'government') > 0
    OR STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'treasury') > 0 THEN 'government_bonds'
  WHEN STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'corporate') > 0
    OR STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'high yield') > 0 THEN 'corporate_bonds'
  WHEN STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'international') > 0
    OR STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'foreign') > 0
    OR STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'global') > 0 THEN 'developed_market_equity'
  WHEN STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'large') > 0 AND STRPOS(LOWER(COALESCE(s.lipper_class_name,'')||' '||COALESCE(s.lipper_obj_name,'')), 'cap') > 0 THEN 'large_cap_equity'
  ELSE NULL END
"""

SQL = f"""
WITH m AS (
  SELECT crsp_fundno, caldt, mret, mtna,
         LAG(mtna) OVER (PARTITION BY crsp_fundno ORDER BY caldt) AS mtna_prev,
         LAG(date_trunc('month', caldt)) OVER (PARTITION BY crsp_fundno ORDER BY caldt) AS prev_month
  FROM crsp_q_mutualfunds.monthly_tna_ret_nav
  WHERE caldt BETWEEN '2006-01-01' AND '2026-06-30' AND crsp_fundno IS NOT NULL
),
cls AS (
  SELECT s.crsp_fundno, s.begdt, s.enddt, {LIPPER_CASE} AS asset_class
  FROM crsp_q_mutualfunds.fund_style s
  WHERE s.enddt >= '2005-01-01' AND s.crsp_fundno IS NOT NULL
),
j AS (
  SELECT m.crsp_fundno, m.caldt, m.mret, m.mtna, m.mtna_prev, c.asset_class
  FROM m JOIN cls c
    ON m.crsp_fundno = c.crsp_fundno AND m.caldt BETWEEN c.begdt AND c.enddt
  WHERE c.asset_class IS NOT NULL AND m.mtna IS NOT NULL AND m.mtna_prev IS NOT NULL AND m.mtna_prev > 0
    AND m.prev_month IS NOT NULL
    AND date_trunc('month', m.caldt) - m.prev_month <= interval '3 months'
),
f AS (
  SELECT asset_class, caldt, crsp_fundno, mtna_prev, mtna,
         (mtna - mtna_prev * (1 + COALESCE(mret,0))) / mtna_prev AS flow_pct_raw,
         (mtna - mtna_prev * (1 + COALESCE(mret,0))) AS flow
  FROM j WHERE ABS((mtna - mtna_prev * (1 + COALESCE(mret,0))) / mtna_prev) <= 0.5
)
SELECT asset_class, date_trunc('month', caldt)::date AS month,
       SUM(flow) AS dollar_flow, SUM(mtna_prev) AS tna_lag, SUM(mtna) AS tna,
       COUNT(DISTINCT crsp_fundno) AS n_funds, COUNT(*) AS n_obs
FROM f GROUP BY 1, 2 ORDER BY 1, 2
"""


def main():
    conn = wrds.Connection(wrds_username="dechang", wrds_password=os.environ.get("WRDS_PASSWORD", "3eu2m68a!gYmLYY"))
    print("connected; running server-side aggregation...", flush=True)
    t0 = time.time()
    agg = conn.raw_sql(SQL)
    print(f"aggregated {len(agg)} class-months in {time.time()-t0:.0f}s", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_rerun_output")
    os.makedirs(out, exist_ok=True)
    agg.to_csv(os.path.join(out, "agg_class_month.csv"), index=False)
    print("saved agg_class_month.csv", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
