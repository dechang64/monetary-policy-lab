# 数据字典: minutes_sentiment_corrected.csv

## 基本信息
- 样本量: N = 117
- 时间范围: 2006-01-31 至 2022-07-27
- 每行 = 一次 FOMC 会议

## 变量说明

### 日期与决策
| 变量 | 说明 | 类型 |
|------|------|------|
| date | FOMC 会议日期 | YYYY-MM-DD |
| decision | 利率决策类型 | conventional / forward_guidance / normalization |
| rate_before | 会议前联邦基金利率 (%) | 连续 |
| rate_after | 会议后联邦基金利率 (%) | 连续 |
| rate_change | 利率变化 (百分点) | rate_after - rate_before |
| expected_change | 市场预期利率变化 (百分点) | 来自期货市场 |
| surprise | 利率意外 | rate_change - expected_change |
| chair | 美联储主席 | Bernanke / Yellen / Powell |
| regime | 货币政策时期 | conventional / forward_guidance / normalization |

### 货币政策冲击
| 变量 | 说明 | 来源 |
|------|------|------|
| target_shock | GSS 目标因子 (当前利率意外) | Acosta (2022) |
| path_shock | GSS 路径因子 (未来政策预期修正) | Acosta (2022) |
| kuttner_bp | Kuttner (2001) 意外 (基点) | 联邦基金期货 |
| ns_shock | Nakamura-Steinsson 冲击 | 高频识别 |

### 情感指标
| 变量 | 说明 | 构造方法 |
|------|------|----------|
| sentiment | Combined sentiment (论文主变量) | 0.5 × LM + 0.5 × CB |
| lm_score | Loughran-McDonald 情感分数 | LM 正面词 - LM 负面词 / 总词数 |
| cb_score | Central-Bank 情感分数 | CB 鹰派词 - CB 鸽派词 / 总词数 |
| word_count | 声明词数 | 计数 |

### 金融资产收益 (会议日)
| 变量 | 说明 | 数据源 | 单位 |
|------|------|--------|------|
| sp500_ret | S&P 500 日收益 | yfinance | 百分比 |
| nasdaq_ret | NASDAQ 日收益 | yfinance | 百分比 |
| gold_ret | 黄金日收益 | yfinance | 百分比 |
| ty10_chg | 10年期国债收益率变化 | FRED DGS10 | 百分点 |
| tb13w_chg | 13周国库券收益率变化 | FRED DGS3MO | 百分点 |
| vix | VIX 指数 | yfinance | 指数 |
| term_spread | 期限利差 (10Y-3M) | FRED | 百分点 |

### CRSP 收益 (会议日, 退市调整)
| 变量 | 说明 | 数据源 | 单位 |
|------|------|--------|------|
| vwretd_day | CRSP VW 日收益 | WRDS | **小数** (×100 = 百分比) |
| ewretd_day | CRSP EW 日收益 | WRDS | **小数** (×100 = 百分比) |
| sprtrn_day | S&P 500 日收益 (含退市) | WRDS | **小数** (×100 = 百分比) |
| vwretd_pre | CRSP VW 会前收益 | WRDS | 小数 |
| vwretd_post | CRSP VW 会后收益 | WRDS | 小数 |
| vwretd_2d | CRSP VW 2日收益 | WRDS | 小数 |

### Forward Guidance 交互项
| 变量 | 说明 | 构造方法 |
|------|------|----------|
| fg_period | FG 期间指示变量 | 1 = 2008-12 至 2015-12, 0 = 其他 |
| sentiment_x_fg | Sentiment × FG 交互项 | sentiment × fg_period |

### FOMC Minutes 情感
| 变量 | 说明 | 构造方法 |
|------|------|----------|
| min_sentiment | Minutes combined sentiment | 0.5 × min_lm + 0.5 × min_cb |
| min_lm_score | Minutes LM 情感分数 | LM dictionary |
| min_cb_score | Minutes CB 情感分数 | CB dictionary |
| min_word_count | Minutes 词数 | 计数 |
| min_cb_x_fg | Minutes CB × FG 交互项 | min_cb_score × fg_period |

## 重要注意事项

1. **CRSP 收益单位**: vwretd_day / ewretd_day / sprtrn_day 是**小数**格式 (0.01 = 1%)
   - 论文中使用百分比，需要 ×100
   - nasdaq_ret / sp500_ret / gold_ret 已经是百分比格式

2. **标准误**: 论文全部使用 Newey-West HAC(4) 标准误
   - statsmodels: `fit(cov_type='HAC', cov_kwds={'maxlags': 4})`

3. **FG 期间定义**: 2008-12-16 至 2015-12-15 (ZIRP 期间)
   - fg_period = 1 的会议有 57 次 (不是 56，因为 2008-12-16 的会议)

4. **Sentiment 构造**: Combined = 0.5 × LM + 0.5 × CB
   - LM score 始终为正 (正面词多于负面词)
   - CB score 多数为负 (鹰派词多于鸽派词)
