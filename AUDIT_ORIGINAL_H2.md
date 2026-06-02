# Original H2-H4 审计报告
## 日期: 2026-06-02

---

## 一、代码审计结论

### ✅ 通过的检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| VIF 多重共线性 | ✅ 全部 < 1.5 | FG: 1.14-1.33, Non-FG: 1.00-1.01 |
| OLS vs NW HAC | ✅ 一致 | SE 比率 = 1.00，HAC 未膨胀 |
| 离群值 | ✅ 无极端 | CRSP VW 无 3×IQR 离群值 |
| Leave-one-out | ✅ 稳健 | 最大 |Δβ| = 0.54（2011-08-09），删除后仍显著 (p=0.025) |
| FG 定义 | ✅ 稳健 | Strict ZLB 和 Extended 定义结果一致 |
| Permutation test | ✅ 极显著 | 1000次随机分割，|β|≥2.60 的 p = 0.000 |

### ⚠️ 需要注意的问题

| 问题 | 严重程度 | 详情 |
|------|---------|------|
| **Non-FG 包含 COVID** | 🟡 中 | 19/60 = 32% 的 non-FG 是 COVID 时期，但排除 COVID 后 non-FG 仍不显著 (p=0.472) |
| **COVID 子样本显著** | 🟡 中 | COVID 时期 β_S = -0.75** (p=0.012), R²=53.3%——但 N=19 太小 |
| **FG 时期 sentiment 方差小** | 🟡 中 | FG std=0.0019 vs non-FG std=0.0078 (4x)，但 FG 反而更显著——不是统计功效问题 |
| **FG 时期 corr(S,Target) 更高** | 🟡 中 | FG: 0.34 vs non-FG: 0.07——sentiment 在 FG 时期与 target shock 更相关 |

### 🔴 关键发现：Partial Correlation

| | FG 时期 | Non-FG 时期 |
|---|---|---|
| corr(Sentiment, Return) | **-0.454** | -0.046 |
| partial corr(S,R \| Target,Path) | **-0.364** (p=0.005) | -0.033 (p=0.804) |

**即使控制了 target 和 path shock，sentiment 在 FG 时期仍有显著的偏相关 (p=0.005)**。这不是 sentiment 代理了 shock 信息——是独立的语言通道。

---

## 二、结果稳健性总结

### H2: Sentiment 增量解释力（全样本）

| 资产 | β_S | p | ΔR² | 判定 |
|------|-----|---|-----|------|
| S&P 500 | 19.46 | 0.088* | +0.66% | 边际显著 |
| 10Y Treasury | 1.24 | 0.098* | +1.22% | 边际显著 |
| CRSP VW | -0.20 | 0.185 | +0.74% | 不显著 |
| CRSP EW | -0.10 | 0.566 | +0.21% | 不显著 |

**全样本结论**：弱证据，不足以单独支撑"语言有增量信息"的论点。

### H4: FG 时期 Sentiment 更强（核心发现）

| 子样本 | CRSP VW β_S | p | R² |
|--------|------------|---|-----|
| **FG 时期 (N=57)** | **-2.60** | **0.004\*\*\*** | **30.6%** |
| Non-FG (N=60) | -0.05 | 0.727 | 5.6% |
| Non-FG excl COVID (N=41) | 0.82 | 0.472 | 11.1% |
| COVID (N=19) | -0.75 | 0.012** | 53.3% |

**交互项**：β_{S×FG} = -2.87*** (p=0.005) for CRSP VW

**稳健性**：
- ✅ Leave-one-out: 删除最 influential obs 后仍显著 (p=0.025)
- ✅ Permutation: 1000次随机分割，p = 0.000
- ✅ Alternative FG definitions: strict/extended 一致
- ✅ VIF < 1.5: 无多重共线性
- ✅ Partial correlation: 控制shocks后仍显著 (p=0.005)

---

## 三、文献对比

### 直接相关文献

| 论文 | 核心发现 | 与我们的关系 |
|------|---------|------------|
| **GSS (2005)** | Path factor 独立于 target factor 影响长期利率 | 我们的基础框架，但 GSS 没有检验 sentiment |
| **Hansen & McMahon (2016)** | FOMC 声明语言有宏观经济效应，FG 段落有独立影响 | 最接近——他们也发现语言有独立效应，但没有做 regime 分层 |
| **Jegadeesh & Wu (2013/2021)** | Minutes sentiment 与联邦基金期货和美元显著相关 | Minutes 而非 Statement，没有做增量 R² 检验 |
| **Swanson (2021)** | FG 和 LSAP 有显著资产价格效应 | 三因子框架（target+FG+LSAP），我们没有 LSAP 因子 |
| **Cieslak et al. (2025)** | Press conference 比 statement 影响更大 | 支持我们的"不同沟通渠道不同信息角色"假说 |
| **FF (2025)** | LLM narrative surprise 比市场 HF 更干净 | 方法层面，没有检验 sentiment 增量效应 |

### 我们的独特贡献

**没有文献做过以下组合**：
1. Sentiment → Returns **controlling for** HF shocks（增量 R² 检验）
2. **FG 时期 vs 非 FG 时期**的 sentiment 增量效应对比
3. Partial correlation 证明 sentiment 不是 shock 的代理变量

Hansen & McMahon (2016) 最接近，但他们：
- 用的是 topic model（LDA），不是 sentiment score
- 没有控制 HF shocks
- 没有做 regime 分层

---

## 四、审计结论

### FG 时期 R²=30.6% 是否可信？

**可信，但需要谨慎解读**：

1. **统计上稳健**：permutation test p=0.000, leave-one-out 稳健, VIF 正常
2. **经济上合理**：ZLB 时期利率工具受限，语言成为主要传导渠道——这与 Hansen & McMahon (2016) 的理论一致
3. **但有两个 caveats**：
   - FG 时期 sentiment 方差很小 (std=0.0019)，β_S=-2.60 意味着 1个标准差的 sentiment 变化 → -0.5% 的 VW return 变化，经济量级合理
   - COVID 时期也显著 (R²=53.3%)，说明"非常规时期语言更重要"可能不仅限于 ZLB

### 全样本 H2 不显著怎么办？

**不矛盾**——全样本不显著 + FG 时期高度显著 = **regime-dependent effect**。这比"全样本显著"更有趣，因为它回答了"什么时候语言重要"——答案是：当利率工具受限时。

---

## 五、对论文的影响

### 当前论文的 FG null result 需要重新解读

| | 当前论文 | 修正后 |
|---|---|---|
| FG 交互项 | β_{P×FG} 不显著 | β_{S×FG} 高度显著*** |
| 解读 | "FG 不强化 path shock" | "FG 强化 **sentiment 通道**" |
| 含义 | 弱信息效应 | **ZLB 时期语言成为主要传导渠道** |

### 建议新增一节

**Section 6.X: Does Sentiment Have Incremental Explanatory Power?**

1. 全样本：弱增量效应（S&P 500 p=0.088, 10Y p=0.098）
2. FG 时期：强增量效应（CRSP VW β=-2.60***, R²=30.6%）
3. Partial correlation 证明不是 shock 代理
4. 经济解读：ZLB 时期语言替代利率成为信号载体

---

*审计日期: 2026-06-02*
*审计者: 曼卿*
*数据: minutes_sentiment_corrected.csv (N=117)*
