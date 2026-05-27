# Monetary Policy Lab — WRDS 升级方案

## 一、当前技术栈盘点

### 1. 数据层（现状 vs 缺口）

| 模块 | 当前数据源 | 数据质量 | 缺口 |
|------|-----------|---------|------|
| **Kuttner Surprise** | FRED DFF（日频代理） | ❌ 不是期货价格，surprise 不显著 | 需 CME FF 期货（`cme.ff`） |
| **Path Factor** | 无（代码写了简化版） | ❌ 纯占位 | 需 CME Eurodollar 期货（`cme.ef`） |
| **股票收益** | yfinance（`^GSPC`等） | ⚠️ 无退市调整、MultiIndex 问题 | 需 CRSP（`crsp.dsf`/`crsp.dsi`） |
| **Two-Shocks** | 日频 yfinance 数据 | ❌ 无法做高频识别 | 需 TAQ 日内数据（`taqmsec.ctm_*`） |
| **Sentiment** | LM + CB 词典 | ⚠️ 方差太小（0.003 vs 论文 0.035） | 需 FinBERT 或扩充词典 |
| **信息冲击验证** | 无 | ❌ 无法验证 | 需 IBES（`ibes.statsum_epsus`）+ SPF（`philfed.spf`） |
| **资本流动** | 合成数据 | ❌ 代码注释写了 "In production: use CRSP/Thomson Reuters" | 需 CRSP Mutual Fund |
| **隐含波动率** | 无 | ❌ 缺失 | 需 OptionMetrics（`optionm.opprcd*`） |
| **FOMC 声明** | 自建爬虫（155/157 成功） | ✅ 可用 | — |
| **利率数据** | FRED DGS10/DGS3MO | ✅ 可用 | — |

### 2. 计算层

| 组件 | 技术栈 | 状态 |
|------|--------|------|
| **SurpriseCalculator** | Python, pandas | ⚠️ 框架在，数据源不对 |
| **TwoShocksDecomposer** | Python, numpy | ⚠️ 简化版，需 SVAR 实现 |
| **CapitalFlowAnalyzer** | Python, pandas | ❌ 合成数据 |
| **RegressionEngine** | Python, statsmodels | ✅ 可用 |
| **NLPEngine** | Python, LM/CB 词典 | ⚠️ 方差太小 |
| **EventStudy** | Python | ✅ 框架在 |
| **FREDConnector** | Python, requests | ✅ 可用 |
| **WRDSConnector** | Python, wrds 3.5.0 | 🆕 已写框架，待连接测试 |

### 3. 展示层

| 组件 | 技术栈 | 状态 |
|------|--------|------|
| **Streamlit App** | Python, Streamlit | ✅ 可用 |
| **Charts** | Python, matplotlib/plotly | ✅ 可用 |
| **Docker** | Dockerfile + docker-compose | ✅ 可用 |

### 4. 跨项目协同资产

| 项目 | 与 monetary-policy-lab 的关系 |
|------|------------------------------|
| **ewa-fed** | 熵加权聚合框架，可增强 Two-Shocks 的 SVAR 估计 |
| **unified-fl-backend** | Rust HNSW + gRPC，可做 sentiment embedding 的语义检索 |
| **claw-med-gps** | RAG 引擎 + LLM 集成经验，可复用到 FOMC 文本分析 |
| **FundFL** | Rust 金融数据管线，期货数据处理可复用 |

---

## 二、WRDS 数据库 → 研究假设映射

### 核心论文假设与所需 WRDS 数据

| 假设 | 论文声称 | 当前结果 | WRDS 数据库 | 预期改善 |
|------|---------|---------|------------|---------|
| **H1**: Sentiment → 资产价格 | R²=2.76% | R²=0.39% | `crsp.dsf`（退市调整收益）+ `cme.ff`（正确 surprise） | surprise 变量修正后 R² 应显著提升 |
| **H2**: Surprise → 短期利率 | p=0.033 | 不显著 | `cme.ff`（FF 期货价格） | 直接计算 Kuttner surprise，应显著 |
| **H3**: Residual → 非利率成分 | 97.2% | 99.6% | `cme.ef`（ED 期货，path factor） | 分离 target + path factor 后残差占比下降 |
| **JK 识别**: 政策冲击 vs 信息冲击 | sign restriction | 无法做 | `taqmsec.ctm_*`（TAQ 日内）+ `crsp.dsf` | 30min 窗口内 S&P500+2Y Treasury 协动 |

---

## 三、分阶段升级方案

### Phase 1：修正核心变量（WRDS 批准后立即可做，1-2 天）

**目标**：让 H1、H2 从不显著变显著

```python
# 1. 安装 wrds
pip install wrds

# 2. 连接
import wrds
db = wrds.Connection(wrds_username='your_username')

# 3. 获取 CME Fed Funds 期货
ff_futures = db.raw_sql("""
    SELECT date, symbol, open, high, low, close, volume, oi
    FROM cme.ff
    WHERE date >= '1990-01-01'
    ORDER BY date
""")

# 4. 获取 CME Eurodollar 期货
ed_futures = db.raw_sql("""
    SELECT date, symbol, open, high, low, close, volume, oi
    FROM cme.ef
    WHERE date >= '1990-01-01'
    ORDER BY date
""")

# 5. 计算 Kuttner Surprise
# surprise = (FF_rate_post - FF_rate_pre) × 100  (bp)
# FF_rate = 100 - FF_futures_price
```

**交付物**：
- `WRDSConnector` 完整实现（`fetch_fed_funds_futures()`, `fetch_eurodollar_futures()`）
- `SurpriseCalculator` 替换 FRED DFF → CME FF 期货
- `PathFactorDecomposer` 新增（target factor + path factor）
- 重新跑 H1-H3 回归，对比结果

**预期**：
- H2 surprise 应从"不显著"变为 p<0.05
- H1 R² 应从 0.39% 提升到 1-3% 区间

---

### Phase 2：高频 Two-Shocks 识别（3-5 天）

**目标**：复刻 Jarociński & Karadi (2020) 的 sign restriction 识别

```python
# TAQ 日内数据：FOMC 前后 30 分钟窗口
# 注意：TAQ 表按日期分表 taqmsec.ctm_{YYYYMMDD}
fomc_dates = ['20240320', '20240501', ...]  # FOMC 会议日期

for fomc_date in fomc_dates:
    trades = db.raw_sql(f"""
        SELECT date, time_m, sym_root, sym_suffix, price, size
        FROM taqmsec.ctm_{fomc_date}
        WHERE sym_root IN ('SPY', 'TLT')
          AND time_m BETWEEN '13:30:00' AND '15:00:00'
        ORDER BY time_m
    """)
```

**交付物**：
- `HighFrequencyIdentifier` 类（TAQ 数据获取 + 窗口计算）
- `SignRestrictionSVAR` 类（简化版 SVAR 估计）
- `TwoShocksDecomposer` 升级（从简化版 → 高频版）
- FOMC 窗口内资产响应图

**注意**：
- TAQ 数据量巨大（单日数 GB），需按日期分批下载
- 建议先做 2020-2025 的 FOMC 日期，验证流程后再扩展
- 云 VM 无 GPU 不影响此阶段（纯 CPU 计算）

---

### Phase 3：信息冲击验证 + 稳健性（3-5 天）

**目标**：用外部预期数据验证信息冲击的真实性

```python
# IBES 分析师预测修正
ibes = db.raw_sql("""
    SELECT ticker, cusip, fpi, statpers, medest, meanest, numest
    FROM ibes.statsum_epsus
    WHERE statpers >= '2000-01-01'
    ORDER BY statpers
""")

# SPF 专业预测者调查
spf = db.raw_sql("""
    SELECT date, inflation_mean, gdp_mean, rgdp_mean
    FROM philfed.spf
    WHERE date >= '1990-01-01'
    ORDER BY date
""")

# OptionMetrics 隐含波动率
options = db.raw_sql("""
    SELECT date, symbol, exdate, cp_flag, strike_price, best_bid, best_offer, impl_volatility
    FROM optionm.opprcd2023
    WHERE symbol = 'SPY'
    ORDER BY date
""")
```

**交付物**：
- `InformationShockValidator` 类
- FOMC 前后分析师预测修正统计
- SPF 预期变化与信息冲击的相关性
- VIX/隐含波动率在 FOMC 前后的变化

---

### Phase 4：资本流动真实数据（2-3 天）

**目标**：替换 `CapitalFlowAnalyzer` 的合成数据

```python
# CRSP Mutual Fund 数据
crsp_mf = db.raw_sql("""
    SELECT fundno, date, crsp_fundno, fund_name, tna, flow
    FROM crsp.fund_hdr
    WHERE date >= '2000-01-01'
""")

# CRSP Stock Returns（含退市调整）
crsp_dsf = db.raw_sql("""
    SELECT permno, date, ret, vwretd, ewretd, sprtrn
    FROM crsp.dsf
    WHERE date >= '2000-01-01'
    ORDER BY date
""")
```

**交付物**：
- `CapitalFlowAnalyzer` 升级（真实基金流动数据）
- FOMC 前后资金跨资产类别流动分析
- Risk-on/Risk-off regime 检测

---

### Phase 5：Sentiment 增强（与 Phase 2-4 并行）

**目标**：解决 sentiment 方差太小的问题

**方案 A（无需 GPU）**：扩充 CB 词典
- 从 FOMC 声明中提取 bigram/trigram
- 用 TF-IDF 筛选区分度高的词组
- 目标：sentiment std 从 0.003 提升到 0.01+

**方案 B（需 GPU）**：FinBERT
- 本地 5060 Ti 推理（需先解决驱动问题）
- 或用 MiniMax API 做 batch sentiment scoring
- 目标：sentiment std 达到论文水平 0.035

---

## 四、技术架构升级

### 数据管线重构

```
当前：
  FRED API → FREDConnector → pandas DataFrame → Streamlit

升级后：
  WRDS PostgreSQL ──┐
  FRED API ─────────┤→ DataHarmonizer → Parquet Cache → Analysis Engine → Streamlit
  FOMC Scraper ─────┤
  CME Futures ──────┘
```

**关键设计**：
1. **Parquet 缓存层**：WRDS 查询慢（跨太平洋 PostgreSQL），首次下载后存本地 Parquet
2. **DataHarmonizer**：统一 FRED + WRDS + CME 的日期对齐、频率对齐
3. **增量更新**：只拉取新数据，不重复下载

### 新增文件结构

```
monetary-policy-lab/
├── data/
│   ├── fred_connector.py      # 现有
│   ├── wrds_connector.py      # 🆕 已写框架
│   ├── data_harmonizer.py     # 🆕 日期/频率对齐
│   ├── parquet_cache.py       # 🆕 本地缓存
│   └── cache/                 # 🆕 Parquet 文件
├── analysis/
│   ├── surprise_calculator.py # 升级：CME FF 期货
│   ├── path_factor.py         # 🆕 target + path factor
│   ├── two_shocks.py          # 升级：高频版
│   ├── hf_identifier.py       # 🆕 TAQ 日内识别
│   ├── sign_restriction.py    # 🆕 SVAR sign restriction
│   ├── info_shock_validator.py# 🆕 IBES + SPF 验证
│   ├── capital_flow.py        # 升级：真实数据
│   ├── nlp_engine.py          # 升级：扩充词典/FinBERT
│   ├── regression_engine.py   # 现有
│   └── event_study.py         # 现有
└── modules/                   # Streamlit UI 模块
    ├── two_shocks.py          # 升级 UI
    ├── capital_flow.py        # 升级 UI
    └── ...
```

---

## 五、预期成果对比

| 指标 | 当前（FRED + yfinance） | Phase 1 后 | Phase 2-3 后 | 论文水平 |
|------|------------------------|-----------|-------------|---------|
| H1 R² | 0.39% | 1-2% | 2-3% | 2.76% |
| H2 p-value | 不显著 | <0.05 | <0.01 | 0.033 |
| H3 residual | 99.6% | 98-99% | 96-98% | 97.2% |
| Sentiment std | 0.003 | 0.005-0.01 | 0.02-0.035 | 0.035 |
| Two-Shocks | 无 | 无 | ✅ sign restriction | ✅ |
| 资本流动 | 合成 | 合成 | ✅ 真实数据 | ✅ |

---

## 六、WRDS 申请注意事项

1. **申请入口**：https://wrds-www.wharton.upenn.edu/register/
2. **机构选择**：XJTLU（如果列表里有）或通过 Wharton 合作渠道
3. **需要的数据库名称**（申请时勾选）：
   - CME (Chicago Mercantile Exchange) — FF + ED 期货
   - CRSP — 股票收益 + 基金数据
   - TAQ (Trade and Quote) — 日内数据
   - IBES — 分析师预测
   - OptionMetrics — 期权数据
   - Philadelphia Fed — SPF 调查
   - Compustat — 公司基本面（稳健性检验用）
4. **审批时间**：通常 1-3 个工作日
5. **连接方式**：`pip install wrds` → `wrds.Connection()` → 首次输入密码存 `~/.wrds.cfg`

---

*方案版本：v1.0 | 2026-05-20 | 曼卿*
