# Phase 2-4 研究计划调整建议（v4 — 终版）
## 基于 Eileen 六份原始文档 + FF (2025) + 当前论文 v10.3

---

## 一、原始假设 vs 当前论文假设（关键偏离）

### Phase 1 执行方案定义的四个假设

| # | 原始假设 | 检验方法 | 预期结果 |
|---|---------|---------|---------|
| H1 | FOMC 声明 hawkish/dovish 情绪与 Kuttner surprise 正相关但不完全重叠 | 相关系数 < 1 | β₁ > 0, R² ≈ 0.4-0.6 |
| H2 | 控制利率 surprise 后，语言情绪对资产价格仍有显著解释力 | 增量 R² | β₂ 显著, 增量 R² ≈ 5-15% |
| H3 | 语言情绪与 Two-Shocks 框架中的 **information shock** 正相关 | 回归分析 | β₂(info) > β₂(policy) |
| H4 | 语言情绪的预测力在 **forward guidance 时期（2008-2015）更强** | 交互项回归 | β₃ > 0 |

### 当前论文 v10.3 的四个假设

| # | 当前假设 | 检验方法 | 实际结果 |
|---|---------|---------|---------|
| H1 | Target shock 显著预测 statement sentiment | NW HAC 回归 | β_T 显著 (p=0.012), β_P 不显著 |
| H2 | Target shock 显著预测 asset returns | NW HAC 回归 | β_T 显著 (p=0.043), β_P 不显著 |
| H3 | Target = Path? | Wald 检验 | 不能拒绝 (p=0.90) |
| H4 | FG 时期 path shock 更重要 | 交互项回归 | 不显著 |

### 偏离分析

| 维度 | 原始设计 | 当前论文 | 偏离程度 |
|------|---------|---------|---------|
| **核心问题** | 语言是否包含**增量信息**（beyond the rate） | 语言反映 **Implementation 还是 Revelation** | 🔴 重定向 |
| **H1** | Sentiment 与 surprise **正相关但不完全重叠** | Target shock **显著预测** sentiment | 🟡 方向一致但问题不同 |
| **H2** | 控制 surprise 后 sentiment 对**资产价格**有增量解释力 | Target shock 对**资产回报**有显著效应 | 🔴 被解释变量不同 |
| **H3** | 情绪更多反映 **information shock** | Target = Path? (Wald test) | 🔴 完全不同 |
| **H4** | FG 时期语言**更重要** | FG 时期 path 更重要 | 🟡 方向一致但 framing 不同 |
| **冲击框架** | Kuttner + Two-Shocks (policy vs info) | GSS (target vs path) | 🟡 框架不同但有关联 |
| **被解释变量** | Sentiment → Asset prices (两步) | Sentiment + Asset returns (并列) | 🟡 结构不同 |

**核心偏离**：原始设计问的是"语言有没有增量信息"（affirmative），当前论文问的是"语言反映什么"（diagnostic）。原始 H3 预期 information shock 更重要，但实际数据不支持——这正是论文重新定位的原因。

---

## 二、四个 Research Direction 的完整理论支撑

### Direction 1: The Information Content of Central Bank Communication
**核心问题**：FOMC 声明的语言特征是否包含超越利率决策的增量信息？

| 学科 | 理论 | 在 Direction 1 中的角色 |
|------|------|----------------------|
| 经济学 | Rational Inattention (Sims 2003) | 投资者注意力有限，同一声明不同解读 |
| 经济学 | Signaling Theory (Spence 1973) | 声明语言是信号载体，利率是行动载体 |
| 金融科技 | NLP Sentiment Analysis | 文本情绪量化为数值指标 |
| 传播学 | Framing Theory (Entman 1993) | "data dependent" vs "patient" 是框架操作 |
| 新闻学 | Second-Level Agenda-Setting (McCombs 1997) | 媒体对 FOMC 声明的二次框架化 |

**当前论文覆盖**：Signaling + NLP ✅ | Rational Inattention ❌ | Framing ❌ | Agenda-Setting ❌

### Direction 2: Portfolio Rebalancing and Cross-Asset Contagion
**核心问题**：货币政策公告后，投资者的跨资产再平衡行为如何产生传染效应？

| 学科 | 理论 | 在 Direction 2 中的角色 |
|------|------|----------------------|
| 经济学 | Portfolio Balance Channel (Tobin 1969) | MP 改变资产相对供给，迫使再平衡 |
| 经济学 | Risk-Taking Channel (Borio & Zhu 2012) | 宽松政策鼓励追逐收益，压缩风险溢价 |
| 金融科技 | Network Analysis + DCC-GARCH | 跨资产相关性动态建模 |
| 传播学 | Information Cascade (Bikhchandani 1992) | 投资者模仿他人行为，放大初始冲击 |
| 新闻学 | Contagion vs. Interdependence (Forbes & Rigobon 2002) | 区分真正的传染和正常关联 |

**Eileen 原始 Proposal 的核心假设**：
- H1 (Risk-Off): 紧缩 MP shock → 流出高风险资产
- H2 (Risk-On): 正面 Info shock → 流入高风险资产
- H3 (不对称): MP ≠ Info 对 fund flows 的影响
- H4 (Risk-Ladder): 投资者沿风险阶梯替代，非二元切换

### Direction 3: Media Amplification and Social Transmission
**核心问题**：金融媒体和社交网络如何放大或扭曲央行信号？

| 学科 | 理论 | 在 Direction 3 中的角色 |
|------|------|----------------------|
| 经济学 | Attention Economics (Hirshleifer & Teoh 2003) | 有限注意力决定哪些信号被放大 |
| 金融科技 | NLP + Sentiment Contagion | 媒体情绪传播建模 |
| 传播学 | Social Amplification of Risk (Kasperson 1988) | 媒体放大/衰减风险信号 |
| 传播学 | Two-Step Flow (Lazarsfeld 1944) | 意见领袖→大众的信号传递 |
| 新闻学 | Agenda-Setting (McCombs & Shaw 1972) | 媒体决定公众关注什么 |

### Direction 4: Regime-Dependent Effects
**核心问题**：以上效应如何随政策环境、经济周期和制度框架变化？

| 学科 | 理论 | 在 Direction 4 中的角色 |
|------|------|----------------------|
| 经济学 | Lucas Critique (1976) | 政策制度变化→参数不稳定 |
| 经济学 | Time-Varying Parameters | TVP-VAR 捕捉动态效应 |
| 金融科技 | Markov-Switching + ML | 自动识别 regime |
| 传播学 | Framing Contingency (Hallahan 2008) | 框架效应随语境变化 |
| 新闻学 | News Value Theory (Galtung & Ruge 1965) | 什么环境下新闻更有影响力 |

---

## 三、当前论文在原始框架中的定位

当前论文（Words Beyond the Rate）= **Direction 1 的第一轮产出**，但发生了重要的研究问题演化：

```
原始 H3: 情绪更多反映 information shock（预期：Info > Policy）
    ↓ 数据不支持
实际发现: Target (policy) 显著，Path (info) 不显著
    ↓ 重新定位
当前论文: 语言主要反映 policy implementation（证据偏向 Implementation）
```

**这个演化是健康的**——原始假设 H3 被数据否定后，论文诚实地报告了发现，并重新定位了问题。这正是学术研究的正常过程。

**但原始 H2（增量解释力）还没有被检验**——这是当前论文最大的缺口：

> **控制 surprise 后，语言情绪对资产价格是否有增量解释力？**

当前论文只检验了 shocks → sentiment 和 shocks → returns，没有检验 **sentiment → returns (controlling for shocks)**。

---

## 四、Phase 2-4 调整方案

### Phase 2: Direction 1 深化 + Direction 2 启动

**Direction 1 深化（补全原始 H2 + 方法升级）**

| 任务 | 对应原始假设 | 优先级 | 工作量 |
|------|------------|--------|--------|
| **补做 H2**: Sentiment → Returns (controlling for shocks) | 原始 H2 | 🔴 最高 | 1天 |
| JK BVAR sign restriction（替代简化版） | 原始 H3 深化 | 🔴 高 | 3-5天 |
| B-S 完整正交化 + FF narrative IV | 外生性验证 | 🔴 高 | 2-3天 |
| Statement vs Minutes vs Press Conf 三层对比 | Framing Theory | 🟡 中 | 3-4天 |
| Sentiment 方法论对比（dict vs LLM） | NLP 方法论 | 🟡 中 | 2-3天 |

**关键**：原始 H2 是当前论文**最大的缺失**——"语言有没有增量信息"这个问题还没回答。当前论文只证明了"shocks 预测 sentiment"，但没有证明"sentiment 预测 returns beyond shocks"。

**Direction 2 启动（Fund Flows + Risk-Ladder）**

| 任务 | 对应 Eileen Proposal | 优先级 | 工作量 |
|------|---------------------|--------|--------|
| CRSP Mutual Fund 7类资产流动数据 | H1 Risk-Off | 🔴 高 | 2-3天 |
| JK 分解 → MP vs Info shock | H1-H3 基础 | 🔴 高 | Phase 1 完成 |
| Fund flows 回归（MP shock → 流出高风险） | H1 Risk-Off | 🔴 高 | 2天 |
| Fund flows 回归（Info shock → 流入高风险） | H2 Risk-On | 🔴 高 | 1天 |
| Risk-Ladder 检验（沿风险阶梯替代） | H4 Risk-Ladder | 🟡 中 | 2天 |
| DCC-GARCH 跨资产相关性 | Cross-Asset Contagion | 🟡 中 | 2天 |

### Phase 3: Direction 4（Regime-Dependent Effects）

| 任务 | 理论 | 优先级 | 工作量 |
|------|------|--------|--------|
| Markov-Switching 识别 regime | Lucas Critique | 🔴 高 | 3天 |
| Regime-Dependent 回归（Direction 1 + 2） | TVP | 🔴 高 | 2天 |
| 跨 Fed Chair 对比 | 制度变化 | 🟡 中 | 1天 |
| ML regime detection（随机森林/梯度提升） | 金融科技 | 🟢 可选 | 2天 |

### Phase 4: Direction 3（Media Amplification）

| 任务 | 理论 | 优先级 | 工作量 |
|------|------|--------|--------|
| 金融新闻抓取 + 框架分析 | Agenda-Setting | 🔴 高 | 3-5天 |
| Media Amplification Index (MAI) | Social Amplification | 🔴 高 | 2天 |
| Two-Step Flow 检验 | Two-Step Flow | 🟡 中 | 2天 |
| 社交媒体情绪（Twitter/Reddit） | Attention Economics | 🟢 可选 | 3-5天 |

---

## 五、最紧迫的发现：原始 H2 未检验

当前论文最大的缺口不是 JK 或 B-S，而是**原始 H2**：

> **控制 surprise 后，语言情绪对资产价格是否有增量解释力？**

这是 Direction 1 的核心问题——"语言有没有增量信息"。当前论文只做了：
- Shocks → Sentiment ✅
- Shocks → Returns ✅
- **Sentiment → Returns (controlling for shocks)** ❌

这个回归模型应该是：
```
Asset_Return_t = α + β₁ · Target_Shock_t + β₂ · Path_Shock_t + β₃ · Sentiment_t + ε_t
```

如果 β₃ 显著且增量 R² > 0，则证明语言包含超越利率决策的增量信息——这正是原始 H2 的核心。

**建议**：立即补做这个回归，作为 v10.4 的核心新增结果。

---

## 六、核心逻辑总结

**原始设计的四个 Direction 是递进因果链：**

```
Direction 1: 信息如何产生（央行说什么）
    → Signaling + NLP + Framing + Rational Inattention
Direction 2: 信息如何传播（投资者怎么动）
    → Portfolio Balance + Risk-Taking + Network + Risk-Ladder
Direction 3: 信息如何被放大（媒体怎么加工）
    → Social Amplification + Agenda-Setting + Two-Step Flow
Direction 4: 以上效应如何随环境变化
    → Lucas Critique + TVP + Markov-Switching + ML
```

**当前论文 = Direction 1 的第一轮产出**，但原始 H2（增量解释力）尚未检验。

**Phase 2-4 调整**：
1. **立即补做原始 H2**（Sentiment → Returns controlling for shocks）——这是当前论文最大的缺口
2. Phase 2: Direction 1 深化（JK+B-S+IV+三层文档）+ Direction 2 启动（Fund flows + Risk-Ladder）
3. Phase 3: Direction 4（Markov-Switching + ML regime）
4. Phase 4: Direction 3（Media Amplification + 社交媒体）

**FF (2025) 只影响了 Direction 1 的方法选择，四个 Direction 的核心问题他都没碰。**

每个 Direction 独立可发表：
- Direction 1 → JMP / JFE（当前论文 + v2 深化）
- Direction 2 → JFE / JF（Fund flows + Risk-Ladder）
- Direction 3 → JF / RFS（Media Amplification）
- Direction 4 → JME / AEJ:Macro（Regime-Dependent Effects）

---

*分析日期: 2026-06-02*
*基于: Eileen 六份原始文档 + FF (2025) + 当前论文 v10.3*
*理论框架: Signaling → Portfolio Balance → Social Amplification → Lucas Critique*
