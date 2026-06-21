# WRDS 完整复现指南

Eileen: 这个指南帮你从 WRDS 原始数据开始，一步步复现论文中每个数字。

## 两种方式

### 方式A: Python 脚本（推荐）
直接运行 `reproduce_from_wrds.py`，自动连接 WRDS、拉数据、跑回归。

```bash
# 1. 安装依赖
pip install wrds statsmodels pandas numpy

# 2. 运行脚本
python3 reproduce_from_wrds.py

# 3. 首次运行会提示输入 WRDS 用户名和密码
#    密码输入后自动保存到 ~/.wrds.cfg，下次不用再输
#    ⚠️ 如果学校有 Duo MFA，需要在手机上 Approve
```

### 方式B: WRDS 网页端 + 本地 Python
如果 Python 连接 WRDS 有问题（Duo MFA 等），可以分两步：

1. **在 WRDS 网页端跑 SQL** → 下载 CSV
2. **本地跑回归** → 用下载的 CSV

```bash
# 第1步: 在 WRDS 网页端运行 wrds_web_queries.sql 中的查询
#   登录 https://wrds.wharton.upenn.edu/
#   Query Tools → SQL Query → 粘贴查询 → 运行 → 下载 CSV
#   保存到 wrds_reproduction_output/ 目录

# 第2步: 本地跑回归
python3 reproduce_all_tables_figures.py
```

## 数据源说明

| 数据 | WRDS 表 | 论文用途 | 替代方案 |
|------|---------|----------|----------|
| CRSP 市场收益 | `crsp.dsi` | Table 4, 5 | 已提供 CSV |
| 联邦基金期货 | `cme.ff` | Kuttner surprise | 已提供 CSV |
| Eurodollar 期货 | `cme.ef` | GSS path factor | 已提供 CSV |
| 国债收益率 | `fred.dgs10/dgs3mo` | Table 4 Treasury | FRED API |
| GSS shocks | — | Table 2-6 | 已提供 CSV (Acosta 2022) |
| Sentiment | — | Table 2-6 | 已提供 CSV (LM+CB) |

**关键**: GSS shocks 和 Sentiment 不在 WRDS 中，已提供 CSV 文件。

## 每个 Table 背后的计算

### Table 1: Summary Statistics
- **计算**: `df.describe()` — 均值、标准差、最小值、最大值
- **数据**: 全部来自 `minutes_sentiment_corrected.csv`
- **无需 WRDS**

### Table 2: Sentiment ~ Target + Path (H1)
- **回归**: `sentiment = α + β₁·target_shock + β₂·path_shock + ε`
- **标准误**: Newey-West HAC(4)
- **数据**: GSS shocks + Sentiment (已提供)
- **无需 WRDS**

### Table 3: Surprise Measure Comparison
- **三个回归**:
  1. `sentiment ~ rate_change` (利率变化)
  2. `sentiment ~ kuttner_bp` (Kuttner surprise)
  3. `sentiment ~ target_shock + path_shock` (GSS, 同 Table 2)
- **Kuttner surprise**: 需要从 CME 期货计算 → **需要 WRDS (cme.ff)**
- **或者**: 直接用已提供的 `kuttner_bp` 列

### Table 4: Asset Returns ~ Shocks (H2)
- **回归**: `asset_return = α + β_T·target_shock + β_P·path_shock + ε`
- **资产**: CRSP VW, CRSP EW, S&P 500, NASDAQ, Gold, 10Y Treasury, 13W T-bill
- **数据**: CRSP 收益 → **需要 WRDS (crsp.dsi)**
- **⚠️ CRSP 收益是小数格式**: 0.01 = 1%，论文用百分比要 ×100

### Table 5: Forward Guidance Interaction (H4)
- **回归**: `asset_return = α + β₁·target + β₂·path + β₃·sentiment + β₄·(sentiment×FG) + ε`
- **数据**: 同 Table 4 + FG 期间指示变量
- **需要 WRDS (crsp.dsi)**

### Table 6: Alternative Sentiment Measures
- **回归**: 各种 sentiment 指标 ~ target + path
- **数据**: Statement + Minutes sentiment (已提供)
- **无需 WRDS**

### Figure 2: Sentiment vs Shocks 散点图
- **数据**: Table 2 的回归数据
- **无需 WRDS**

### Figure 3: Asset Return Responses
- **数据**: Table 4 的回归系数
- **需要 WRDS (crsp.dsi)**

## WRDS 连接问题排查

### 问题1: Duo MFA
- WRDS 强制 Duo 多因素认证
- 运行 `wrds.Connection()` 后手机会收到推送
- **必须在手机上 Approve**，否则连接超时
- Approve 后 30 天内免 MFA

### 问题2: 密码输入
- 首次运行提示输入用户名和密码
- 密码保存到 `~/.wrds.cfg`，下次自动读取
- 如果密码过期，删除 `~/.wrds.cfg` 重新输入

### 问题3: 连接超时
- WRDS 服务器在美国，国内连接可能较慢
- 如果超时，尝试:
  1. 使用 VPN
  2. 用方式B (网页端 + 本地 Python)
  3. 减小查询日期范围

## 验证清单

跑完后对照这些数字，确认复现成功：

| 指标 | 论文值 | 你的值 |
|------|--------|--------|
| N | 117 | ___ |
| β_Target (Table 2) | 0.000577 | ___ |
| p_Target (Table 2) | 0.017 | ___ |
| β_Path (Table 2) | 0.000633 | ___ |
| p_Path (Table 2) | 0.152 | ___ |
| R² (Table 2) | 1.57% | ___ |
| CRSP VW β_T (Table 4) | -0.435 | ___ |
| NASDAQ β_T (Table 4) | -0.282 | ___ |
| CRSP VW Sent×FG p (Table 5) | 0.836 | ___ |
| NASDAQ Sent×FG p (Table 5) | 0.739 | ___ |
| Wald test p (H3) | 0.90 | ___ |

如果数字有微小差异（小数点后2-3位），可能是：
- 标准误计算方法不同 (NW lag 选择)
- 数据版本差异 (CRSP 定期更新)
- 样本微调 (是否排除极端值)

**核心结论应该一致**: target 显著, path 不显著, interaction 不显著。
