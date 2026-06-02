# Phase 2-4 研究计划调整建议（v3 — 终版）
## 基于 Eileen 四份原始文档 + FF (2025) + 当前论文进展

---

## 一、Eileen 的完整研究架构（四份文档整合）

### 文档清单

| 文档 | 日期 | 核心内容 |
|------|------|---------|
| Research Proposal | Feb 2026 | 核心问题：MP shocks → Fund flows → Risk-Ladder |
| Literature Review (FOMC) | 早期版 | 100篇文献，7个渠道，cross-asset/cross-country |
| Literature Review (Revised) | Apr 2026 | 44条引用，10章，从理论到开放问题的完整综述 |
| Research Plan (4 Directions) | — | 四方向跨学科架构：信息→再平衡→放大→Regime |
| 5-Year Proposal | — | 扩展版：+REIT异质性 +国际溢出 +LLM识别 |

### 原始核心问题

> **Through what channels do monetary policy announcements affect asset prices, and how do investors reallocate their portfolios in response?**

这个问题有两个层次：
1. **What** — 哪些资产受影响？方向和幅度？
2. **How** — 投资者如何据此调整组合？传导渠道是什么？

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

**当前论文覆盖了**：Signaling Theory + NLP Sentiment
**当前论文未覆盖**：Rational Inattention、Framing Theory、Agenda-Setting

### Direction 2: Portfolio Rebalancing and Cross-Asset Contagion
**核心问题**：货币政策公告后，投资者的跨资产再平衡行为如何产生传染效应？

| 学科 | 理论 | 在 Direction 2 中的角色 |
|------|------|----------------------|
| 经济学 | Portfolio Balance Channel (Tobin 1969) | 政策改变资产相对供给，迫使再平衡 |
| 经济学 | Risk-Taking Channel (Borio & Zhu 2012) | 宽松政策鼓励追逐收益，压缩风险溢价 |
| 金融科技 | Network Analysis (DCC-GARCH) | 跨资产相关性动态建模 |
| 金融科技 | LLM Classification | 基金流动的语义分类 |
| 传播学 | Contagion Theory | 跨资产传染的传播学类比 |

**Eileen 原始 Proposal 的 4 个假设**：
- **H1 Risk-Off**：紧缩冲击 → 资金流出高风险资产
- **H2 Risk-On**：信息冲击 → 资金流入高风险资产
- **H3 不对称性**：MP shock ≠ Info shock 的组合调整
- **H4 Risk-Ladder**：投资者沿风险阶梯替代，非二元切换

**当前论文完全未涉及此方向**。

### Direction 3: Media Amplification and Social Transmission
**核心问题**：财经媒体和社交网络如何放大、扭曲或过滤 FOMC 信号？

| 学科 | 理论 | 在 Direction 3 中的角色 |
|------|------|----------------------|
| 传播学 | Social Amplification of Risk (Kasperson 1988) | 媒体放大初始信号 |
| 传播学 | Two-Step Flow (Lazarsfeld 1944) | 意见领袖中介传播 |
| 新闻学 | Media Agenda-Setting (McCombs 1972) | 媒体选择强调什么 |
| 金融科技 | Sentiment Contagion Detection | 新闻→社交媒体的情绪传染 |
| 经济学 | Noise Trader Theory (De Long 1990) | 媒体噪声放大价格波动 |

**当前论文完全未涉及此方向**。

### Direction 4: Regime-Dependent Effects
**核心问题**：以上所有效应如何随政策框架、经济周期和市场环境变化？

| 学科 | 理论 | 在 Direction 4 中的角色 |
|------|------|----------------------|
| 经济学 | Lucas Critique (1976) | 政策框架变化→参数不稳定 |
| 经济学 | Time-Varying Parameter Models | 滚动窗口/Markov-Switching |
| 金融科技 | ML Regime Detection | 无监督学习识别 regime |
| 传播学 | Framing Variability | 不同 regime 下框架策略变化 |
| 新闻学 | Comparative Media Analysis | 跨主席/跨 regime 媒体报道差异 |

**当前论文部分涉及**：FG period 交互检验、Chair 固定效应，但只是二元分组，未用 Markov-Switching 或 ML。

---

## 三、四方向的递进关系

```
Direction 1: 信息如何产生（央行说什么）
    ↓ 信号产生
Direction 2: 信息如何传播（投资者怎么动）
    ↓ 行为传播
Direction 3: 信息如何被放大（媒体怎么加工）
    ↓ 放大扭曲
Direction 4: 以上效应如何随环境变化
    ↓ 条件依赖
```

每个方向**独立可发表**，但四个方向合在一起是完整的跨学科研究体系。

---

## 四、当前论文 vs 原始设计的定位

| 维度 | 原始 Proposal 核心 | 当前论文 | 差距 |
|------|-------------------|---------|------|
| 冲击分解 | **JK MP vs Info** | GSS target vs path | ⚠️ 不同分解框架 |
| 被解释变量 | **Fund flows (7类资产)** | Statement sentiment | ⚠️ 完全不同 |
| 核心问题 | **Portfolio reallocation** | Implementation vs Revelation | ⚠️ 不同问题 |
| 理论假设 | **Risk-Off/Risk-On/Risk-Ladder** | H1-H4 (sentiment/returns) | ⚠️ 不同假设体系 |
| 资产焦点 | **REIT + 7类基金** | Broad equity market | ⚠️ 不同资产类别 |
| 跨学科 | **5学科×4方向** | 经济学+NLP | ⚠️ 缺传播学/新闻学 |

**但当前论文不是偏离，而是 Direction 1 的深化**：
- 原始 Direction 1 问"语言有没有增量信息"→ 当前论文用 GSS shocks 回答了"语言反映什么类型的增量信息"
- 当前论文的 Implementation vs Revelation 是 Signaling Theory 的具体化
- 当前论文的 cross-asset evidence 是 Direction 2 的前期探索

---

## 五、FF (2025) 对四个 Direction 的影响

| Direction | FF 覆盖了？ | 影响 | 调整 |
|-----------|----------|------|------|
| 1 信息内容 | ⚠️ 方法覆盖 | LLM 冲击识别比我们好 | 不比方法，比问题；用 FF 数据做 IV |
| 2 组合再平衡 | ❌ | 无影响 | **优先推进**——FF 完全没碰 |
| 3 媒体放大 | ❌ | 无影响 | 保持原计划 |
| 4 Regime 变化 | ⚠️ 二元 regime | 他只做了 FG vs Normal | 升级为 Markov-Switching + ML |

---

## 六、调整后的 Phase 2-4

### Phase 2 → Direction 1 深化 + Direction 2 启动（2-3个月）

**Direction 1 深化（当前论文 v2）**：

| 任务 | 理论支撑 | 优先级 | 工作量 |
|------|---------|--------|--------|
| JK 分解升级（BVAR sign restriction） | JK (2020) | 🔴 高 | 3-5天 |
| B-S 正交化升级（完整控制变量集） | B-S (2023) | 🔴 高 | 2-3天 |
| FF narrative surprise 作为 IV | 外生性验证 | 🔴 高 | 1-2天 |
| Statement vs Minutes vs Press Conf | Framing Theory | 🔴 高 | 3-4天 |
| Sentiment 方法论对比（dict vs LLM） | NLP 方法论 | 🟡 中 | 2-3天 |
| Rational Inattention 框架引入 | Sims (2003) | 🟡 中 | 1-2天 |

**Direction 2 启动**：

| 任务 | 理论支撑 | 优先级 | 工作量 |
|------|---------|--------|--------|
| JK MP vs Info 分解（替代 GSS） | JK (2020) | 🔴 高 | 3-5天 |
| 7类基金流动数据获取 | Portfolio Balance | 🔴 高 | WRDS 权限 |
| H1 Risk-Off 检验 | Borio & Zhu | 🔴 高 | 2天 |
| H2 Risk-On 检验 | Information Effect | 🔴 高 | 2天 |
| H3 不对称性检验 | Two-Shocks | 🟡 中 | 1天 |
| H4 Risk-Ladder 检验 | Novel | 🟡 中 | 2-3天 |
| DCC-GARCH 跨资产相关性 | Network Analysis | 🟡 中 | 3天 |

### Phase 3 → Direction 4（3-4个月）

| 任务 | 理论支撑 | 优先级 | 工作量 |
|------|---------|--------|--------|
| Markov-Switching regime 检测 | Lucas Critique | 🔴 高 | 3-5天 |
| ML 无监督 regime 识别 | FinTech | 🟡 中 | 3天 |
| 跨主席对比（Greenspan→Powell） | Framing Variability | 🟡 中 | 2天 |
| ZLB vs Normal 完整对比 | Time-Varying | 🟡 中 | 2天 |
| Direction 1+2 的 regime 条件分析 | 综合 | 🔴 高 | 3天 |

### Phase 4 → Direction 3（4-6个月）

| 任务 | 理论支撑 | 优先级 | 工作量 |
|------|---------|--------|--------|
| 财经新闻抓取+框架分析 | Agenda-Setting | 🔴 高 | 5-7天 |
| MAI (Media Amplification Index) | Social Amplification | 🔴 高 | 3天 |
| 社交媒体情绪传染检测 | Two-Step Flow | 🟡 中 | 5天 |
| 伦理审查 | — | 🔴 高 | 1-2周 |
| Direction 1+2+4 的媒体调节效应 | 综合 | 🔴 高 | 3天 |

---

## 七、关键决策点

### 1. 冲击分解框架：GSS vs JK？

| | GSS target/path | JK MP/Info |
|---|---|---|
| 当前论文 | ✅ 已用 | ❌ 只做了简化版 |
| Direction 1 | 合适（Implementation vs Revelation） | 升级版（区分纯政策vs信息） |
| Direction 2 | 不合适（target≠MP, path≠Info） | **必须用**（Risk-Off/Risk-On 的理论基础） |
| 文献综述 | GSS 是基础 | JK 是前沿 |

**建议**：Direction 1 继续用 GSS（保持当前论文一致性），Direction 2 切换到 JK（理论要求）。两个框架在论文中明确对比。

### 2. 被解释变量：Sentiment vs Fund Flows？

| | Sentiment | Fund Flows |
|---|---|---|
| Direction 1 | ✅ 核心变量 | 辅助验证 |
| Direction 2 | 辅助变量 | ✅ 核心变量 |
| 数据可得性 | 已有 | 需 WRDS CRSP MF |
| 发表潜力 | 已有论文 | **新论文** |

**建议**：两个变量分别支撑两个 Direction，不冲突。

### 3. REIT 异质性：纳入还是独立？

Eileen 五年计划把 REIT 作为 Year 2 重点，但当前论文用的是 broad equity。

**建议**：REIT 作为 Direction 2 的子分析（Mortgage REIT 对 MP shocks 最敏感），不独立成 Phase。

---

## 八、核心逻辑总结

**四个 Direction 是递进因果链，不是平行技术升级：**

```
Direction 1 (信息产生): 央行说什么 → Signaling + NLP + Framing
    ↓
Direction 2 (行为传播): 投资者怎么动 → Portfolio Balance + Risk-Taking + Network
    ↓
Direction 3 (信号放大): 媒体怎么加工 → Social Amplification + Agenda-Setting
    ↓
Direction 4 (条件依赖): 环境如何调节 → Lucas Critique + TVP + ML
```

**当前论文 = Direction 1 的第一轮产出**，覆盖了 Signaling + NLP，还缺 Rational Inattention + Framing + Agenda-Setting。

**Phase 2-4 调整**：
- Phase 2: Direction 1 深化（JK+B-S+IV+三层文档）+ Direction 2 启动（Fund flows + Risk-Ladder）
- Phase 3: Direction 4（Markov-Switching + ML regime）
- Phase 4: Direction 3（Media Amplification + 社交媒体）

**FF (2025) 只影响了 Direction 1 的方法选择，四个 Direction 的核心问题他都没碰。**

每个 Direction 独立可发表：
- Direction 1 → JMP / JFE（当前论文 + v2 深化）
- Direction 2 → JFE / JF（Fund flows + Risk-Ladder）
- Direction 3 → JF / RFS（Media Amplification）
- Direction 4 → JME / AEJ:Macro（Regime-Dependent Effects）

---

*分析日期: 2026-06-02*
*基于: Eileen 四份原始文档 + FF (2025) + 当前论文 v10.3*
*理论框架: Signaling → Portfolio Balance → Social Amplification → Lucas Critique*
