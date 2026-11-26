-- Test 4 (retail vs institutional flows): run in the WRDS web SQL Query Tool
-- or psql. Save the result as flows_by_class_investor_month.csv and place it
-- at direction2/flow_rerun_output_v8/flows_by_class_investor_month.csv.
-- Expected CSV columns (same shape as flows_by_class_month_v8.csv + investor):
--   asset_class, investor, month, dollar_flow, tna_lag, tna, n_funds, n_obs
--   (month format YYYY-MM-01; investor in {retail, institutional, unknown})

WITH m AS (
  SELECT crsp_fundno, caldt, mret, mtna,
         LAG(mtna) OVER (PARTITION BY crsp_fundno ORDER BY caldt) AS mtna_prev,
         LAG(date_trunc('month', caldt)) OVER (PARTITION BY crsp_fundno ORDER BY caldt) AS prev_month
  FROM crsp_q_mutualfunds.monthly_tna_ret_nav
  WHERE caldt BETWEEN '2006-01-01' AND '2026-06-30' AND crsp_fundno IS NOT NULL
),
cls AS (
  SELECT s.crsp_fundno, s.begdt, s.enddt, 
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
 AS asset_class
  FROM crsp_q_mutualfunds.fund_style s
  WHERE s.enddt >= '2005-01-01' AND s.crsp_fundno IS NOT NULL
),
hdr AS (
  SELECT crsp_fundno, chgdt, COALESCE(chgenddt, DATE '9999-12-31') AS chgenddt,
         CASE WHEN inst_fund = 'Y' THEN 'institutional'
              WHEN retail_fund = 'Y' THEN 'retail' ELSE 'unknown' END AS investor
  FROM crsp_q_mutualfunds.fund_hdr_hist WHERE crsp_fundno IS NOT NULL
),
j AS (
  SELECT m.crsp_fundno, m.caldt, m.mret, m.mtna, m.mtna_prev, c.asset_class, h.investor
  FROM m JOIN cls c
    ON m.crsp_fundno = c.crsp_fundno AND m.caldt BETWEEN c.begdt AND c.enddt
  JOIN hdr h ON m.crsp_fundno = h.crsp_fundno AND m.caldt BETWEEN h.chgdt AND h.chgenddt
  WHERE c.asset_class IS NOT NULL AND m.mtna IS NOT NULL AND m.mtna_prev IS NOT NULL AND m.mtna_prev > 0
    AND m.prev_month IS NOT NULL
    AND date_trunc('month', m.caldt) - m.prev_month <= interval '3 months'
),
f AS (
  SELECT asset_class, investor, caldt, crsp_fundno, mtna_prev, mtna,
         (mtna - mtna_prev * (1 + COALESCE(mret,0))) / mtna_prev AS flow_pct_raw,
         (mtna - mtna_prev * (1 + COALESCE(mret,0))) AS flow
  FROM j WHERE ABS((mtna - mtna_prev * (1 + COALESCE(mret,0))) / mtna_prev) <= 0.5
)
SELECT asset_class, investor, date_trunc('month', caldt)::date AS month,
       SUM(flow) AS dollar_flow, SUM(mtna_prev) AS tna_lag, SUM(mtna) AS tna,
       COUNT(DISTINCT crsp_fundno) AS n_funds, COUNT(*) AS n_obs
FROM f GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
