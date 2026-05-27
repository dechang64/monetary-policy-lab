-- ============================================================
-- WRDS Data Extraction for FOMC Monetary Policy Research
-- Phase 2: CME Fed Funds Futures (Kuttner Surprise)
-- ============================================================
-- 在 WRDS 网页端运行: https://wrds.wharton.upenn.edu/pages/support/data-overview/
-- 路径: Query Tools → SQL Query → 粘贴运行 → 下载 CSV

-- ============================================================
-- 1. CME Fed Funds Futures (Kuttner 2001 surprise measure)
-- 表: cme.ff
-- 字段: date, symbol, open, high, low, close, settle, volume, oi
-- ============================================================
SELECT date, symbol, settle, volume, oi
FROM cme.ff
WHERE date >= '1990-01-01'
  AND date <= '2025-12-31'
  AND symbol IN ('FF1', 'FF2', 'FF3', 'FF4', 'FF5', 'FF6')
ORDER BY date, symbol;

-- ============================================================
-- 2. CME Eurodollar Futures (Gürkaynak path factor)
-- 表: cme.ef
-- ============================================================
SELECT date, symbol, settle, volume, oi
FROM cme.ef
WHERE date >= '1990-01-01'
  AND date <= '2025-12-31'
  AND symbol IN ('ED1', 'ED2', 'ED3', 'ED4', 'ED5', 'ED6',
                 'ED7', 'ED8')
ORDER BY date, symbol;

-- ============================================================
-- 3. CRSP Daily Stock Returns (market index)
-- 表: crsp.dsi
-- 字段: date, vwretd, ewretd, sprtrn (S&P 500 return)
-- ============================================================
SELECT date, vwretd, ewretd, sprtrn
FROM crsp.dsi
WHERE date >= '1990-01-01'
  AND date <= '2025-12-31'
ORDER BY date;

-- ============================================================
-- 4. CRSP Daily Stock File (individual stock returns)
-- 表: crsp.dsf
-- 用于计算 FOMC window 的个股异常收益
-- 先拉 S&P 500 成分股即可
-- ============================================================
SELECT d.date, d.permno, d.ret, d.prc, d.shrout, d.vol
FROM crsp.dsf d
INNER JOIN crsp.dsenames n
  ON d.permno = n.permno
  AND d.date BETWEEN n.namedt AND COALESCE(n.nameendt, '9999-12-31')
WHERE d.date >= '2024-01-01'
  AND d.date <= '2025-12-31'
  AND n.shrcd IN (10, 11)
  AND n.exchcd IN (1, 2, 3)
ORDER BY d.date, d.permno;

-- ============================================================
-- 5. IBES Analyst Forecasts (验证用)
-- 表: ibes.statsum_epsus
-- ============================================================
SELECT ticker, fpedats, statpers, meanest, numest, medest
FROM ibes.statsum_epsus
WHERE fpedats >= '2020-01-01'
  AND fpedats <= '2025-12-31'
  AND ticker IN ('SPY', 'JPM', 'BAC', 'GS', 'MS')
ORDER BY ticker, fpedats, statpers;

-- ============================================================
-- 注意事项:
-- 1. 每个查询单独运行，下载为 CSV
-- 2. 查询 4 (crsp.dsf) 数据量很大，建议分年份拉取
-- 3. 优先跑查询 1-3，这是 H1-H3 验证的核心数据
-- 4. CME 期货数据是 Kuttner surprise 的正确数据源
--    (之前用 DFF 代理导致 H1 R² 只有 0.39% vs 论文 2.76%)
-- ============================================================
