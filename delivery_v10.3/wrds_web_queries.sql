-- ============================================================
-- Words Beyond the Rate v10.3 — WRDS 网页端 SQL 查询
-- ============================================================
-- 
-- 使用方法:
--   1. 登录 https://wrds.wharton.upenn.edu/
--   2. 进入 Query Tools → SQL Query
--   3. 逐个复制下面的查询，运行，下载 CSV
--   4. 保存到 wrds_reproduction_output/ 目录
--
-- Eileen: 每个查询对应论文中的一个数据源
-- 下载后用 reproduce_from_wrds.py 或 reproduce_all_tables_figures.py 跑回归
-- ============================================================


-- ============================================================
-- 查询1: CRSP 日度市场指数 (Table 4, Table 5 的收益数据)
-- ============================================================
-- 数据库: crsp.dsi
-- 说明: 这是论文中 CRSP VW/EW/S&P 500 收益的原始数据源
--       vwretd = value-weighted return (含分红), 小数格式
--       ewretd = equal-weighted return (含分红), 小数格式  
--       sprtrn = S&P 500 total return, 小数格式
--       ⚠️ 小数格式! 0.01 = 1%, 论文中需要 ×100 转百分比

SELECT date, vwretd, ewretd, sprtrn
FROM crsp.dsi
WHERE date >= '2006-01-01'
  AND date <= '2022-12-31'
ORDER BY date;


-- ============================================================
-- 查询2: CME 联邦基金期货 (Kuttner 2001 surprise)
-- ============================================================
-- 数据库: cme.ff
-- 说明: 用于计算 Kuttner (2001) 货币政策意外
--       FF1 = 当月合约, FF2 = 下月合约
--       settle = 结算价 (100 - implied rate)
--       Kuttner surprise = -(settle_t - settle_{t-1}) × 100 bp

SELECT date, symbol, settle, volume, open_interest
FROM cme.ff
WHERE date >= '2006-01-01'
  AND date <= '2022-12-31'
  AND symbol IN ('FF1', 'FF2', 'FF3', 'FF4', 'FF5', 'FF6')
ORDER BY date, symbol;


-- ============================================================
-- 查询3: CME Eurodollar 期货 (Gürkaynak path factor)
-- ============================================================
-- 数据库: cme.ef
-- 说明: 用于 Gürkaynak, Sack, Swanson (2005) target-path 分解
--       ED1-ED8 = 不同到期日的 Eurodollar 期货
--       Path factor = 第2主成分 (target factor 之后的共同变化)

SELECT date, symbol, settle, volume, open_interest
FROM cme.ef
WHERE date >= '2006-01-01'
  AND date <= '2022-12-31'
  AND symbol IN ('ED1', 'ED2', 'ED3', 'ED4', 'ED5', 'ED6', 'ED7', 'ED8')
ORDER BY date, symbol;


-- ============================================================
-- 查询4: FRED 国债收益率 (Table 4 中 Treasury 收益)
-- ============================================================
-- 数据库: fred (via WRDS)
-- 说明: DGS10 = 10年期国债, DGS3MO = 3个月国库券
--       日频变化 = FOMC 当天收益率变化 (百分点)

SELECT observation_date AS date, dgs10, dgs3mo
FROM fred.dgs10
FULL OUTER JOIN fred.dgs3mo USING (observation_date)
WHERE observation_date >= '2006-01-01'
  AND observation_date <= '2022-12-31'
ORDER BY observation_date;


-- ============================================================
-- 查询5: CRSP 金融行业收益 (补充分析)
-- ============================================================
-- 数据库: crsp.dsf + crsp.dsenames
-- 说明: 金融行业对货币政策更敏感 (Bernanke & Kuttner 2005)
--       这里拉取金融行业等权/值权收益

SELECT d.date, 
       COUNT(DISTINCT d.permno) AS n_stocks,
       SUM(d.ret * d.shrout * ABS(d.prc)) / SUM(d.shrout * ABS(d.prc)) AS fin_vw_ret,
       AVG(d.ret) AS fin_ew_ret
FROM crsp.dsf d
INNER JOIN crsp.dsenames n
  ON d.permno = n.permno
  AND d.date BETWEEN n.namedt AND COALESCE(n.nameendt, '9999-12-31')
WHERE d.date >= '2006-01-01'
  AND d.date <= '2022-12-31'
  AND n.siccd BETWEEN 6000 AND 6999  -- 金融行业 SIC 代码
  AND n.shrcd IN (10, 11)
  AND n.exchcd IN (1, 2, 3)
  AND d.ret IS NOT NULL
GROUP BY d.date
ORDER BY d.date;


-- ============================================================
-- 查询6: VIX 指数 (Table 1, 控制变量)
-- ============================================================
-- 数据库: optionm (或直接用 CBOE 数据)
-- 说明: VIX 是波动率控制变量
--       如果 optionm 不可用，可以用 yfinance 获取

-- 方法1: 从 OptionMetrics 构造 (复杂)
-- 方法2: 直接用 CBOE VIX 数据 (推荐)
SELECT date, vix_open, vix_high, vix_low, vix_close
FROM cboe.vix_daily
WHERE date >= '2006-01-01'
  AND date <= '2022-12-31'
ORDER BY date;


-- ============================================================
-- 查询7: 验证 — CRSP 数据完整性检查
-- ============================================================
-- 说明: 检查 CRSP 数据是否有缺失值、异常值

SELECT 
    COUNT(*) AS total_days,
    COUNT(vwretd) AS vwretd_count,
    COUNT(ewretd) AS ewretd_count,
    COUNT(sprtrn) AS sprtrn_count,
    MIN(date) AS first_date,
    MAX(date) AS last_date,
    AVG(vwretd) AS vwretd_mean,
    STDDEV(vwretd) AS vwretd_std
FROM crsp.dsi
WHERE date >= '2006-01-01'
  AND date <= '2022-12-31';


-- ============================================================
-- 注意事项:
-- 
-- 1. 查询1-3 是核心，必须跑。查询4-6 是补充。
-- 2. CRSP 收益是小数格式 (0.01 = 1%)，论文用百分比要 ×100
-- 3. CME 期货结算价 = 100 - implied rate
--    settle 上升 = 隐含利率下降 = 鸽派信号
-- 4. Kuttner surprise 计算:
--    surprise_bp = -(settle_today - settle_yesterday) × 100
--    正值 = 鹰派意外 (利率预期上升)
-- 5. GSS target/path shocks 来自 Acosta (2022)，不在 WRDS 中
--    需要单独下载或从论文作者获取
-- 6. Sentiment 指标 (LM/CB dictionary) 不在 WRDS 中
--    需要自己计算或使用已提供的 CSV
-- ============================================================
