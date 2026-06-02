# Phase 2-4 研究计划调整建议（v2 — 基于原始研究设计）
## 基于 Eileen 原始 Research Plan + FF (2025) + 文献综述

---

## 一、原始四个 Research Direction（正确版本）

原始设计是**四个跨学科研究方向**，每个方向有独立的理论支撑、平台模块和可发表成果：

### Direction 1: The Information Content of Central Bank Communication
**核心问题**：FOMC 声明的语言特征是否包含超越利率决策的增量信息？

| 学科 | 理论 | 如何支撑 |
|------|------|---------|
| 经济学 | Rational Inattention (Sims 2003) | 投资者注意力有限，同一声明不同解读 |
| 经济学 | Signaling Theory (Spence 1973) | 声明语言是信号载体，利率是行动载体 |
| 金融科技 | NLP Sentiment Analysis | 文本情绪量化为数值指标 |
| 传播学 | Framing Theory (Entman 1993) | "data dependent" vs "patient" 是框架操作 |
| 新闻学 | Second-Level Agenda-Setting (McCombs 1997) | 媒体对 FOMC 声明的二次框架化 |

**当前论文 = Direction 1 的产出**（Words Beyond the Rate）

### Direction 2: Portfolio Rebalancing and Cross-Asset Contagion
**核心问题**：货币政策公告后，投资者的跨资产再平衡行为如何产生传染效应？

| 学科 | 理论 | 如何支撑 |
|------|------|---------|
| 经济学 | Portfolio Balance Channel (Tobin 1969) | 货币政策改变资产相对供给，迫使再平衡 |
| 经济学 | Risk-Taking Channel (Borio & Zhu 2012) | 宽松政策鼓励追逐收益，压缩风险溢价 |
| 金融科技 | Network Contagion Models | 量化跨资产传染的路径和强度 |
| 传播学 | Social Amplification of Risk (Kasperson 1988) | 风险信号通过社会网络被放大 |
| 新闻学 | Information Cascade (Bikhchandani 1992) | 公告后的"羊群效应" |

### Direction 3: Media Amplification and the Social Transmission of Monetary Policy
**核心问题**：财经媒体和社交媒体如何放大或扭曲货币政策公告的市场影响？

| 学科 | 理论 | 如何支撑 |
|------|------|---------|
| 经济学 | Noise Trader Risk (De Long et al. 1990) | 媒体驱动的噪声交易放大价格反应 |
| 金融科技 | Social Media Analytics | Twitter/Reddit 数据作为情绪代理变量 |
| 新闻学 | News Framing & Priming | "Fed hikes rates" vs "Fed fights inflation" |
| 传播学 | Two-Step Flow (Katz 1955) | 金融分析师/财经KOL是中间节点 |
| 传播学 | Spiral of Silence (Noelle-Neumann 1974) | 市场共识形成中的信息损失 |

### Direction 4: Time-Varying Monetary Policy Transmission and Regime Detection
**核心问题**：货币政策公告对资产价格的影响是否随制度环境、市场状态、央行领导层变化而变化？

| 学科 | 理论 | 如何支撑 |
|------|------|---------|
| 经济学 | Lucas Critique (1976) | 不同 Fed 主席时期，市场反应模式不同 |
| 经济学 | Markov Regime-Switching (Hamilton 1989) | 识别高/低波动、紧缩/宽松 regime |
| 金融科技 | ML for Regime Detection | HMM/聚类算法自动发现 regime |
| 传播学 | Cultivation Theory (Gerbner 1976) | 不同 regime 下投资者信息处理方式不同 |
| 新闻学 | Media Ecology (McLuhan 1964) | 社交媒体时代 vs 传统媒体时代的传导差异 |

### 四个方向的逻辑关系

```
Direction 1 (Communication)     Direction 2 (Portfolio)
    信息如何产生                    信息如何传播
         │                              │
         └──────────┐   ┌───────────────┘
                    │   │
                    ▼   ▼
         Direction 3 (Media)  ←  中间放大环节
         信息如何被媒体放大
                    │
                    ▼
         Direction 4 (Regime)  ←  条件变量
         以上效应如何随环境变化
```

---

## 二、当前论文的定位

**当前论文（Words Beyond the Rate）= Direction 1 的产出**

但当前论文只覆盖了 Direction 1 的一个子集：
- ✅ Signaling Theory：target vs path 的信号区分
- ✅ NLP Sentiment：dictionary-based 情绪分析
- ❌ Rational Inattention：没有检验投资者注意力异质性
- ❌ Framing Theory：没有检验措辞框架效应
- ❌ Agenda-Setting：没有检验媒体二次框架化

---

## 三、FF (2025) 对四个 Direction 的冲击

### Direction 1：FF 部分覆盖
- FF 用 LLM 做了更好的冲击识别（multi-agent ex ante 预期）
- FF 用 LLM 做了更好的 sentiment（R² 12.4% vs 我们 1.57%）
- **但 FF 不关心语言本身的信息内容**——他关心的是造更好的尺子，不是量语言
- **差异化**：我们的 Implementation vs Revelation 是 Direction 1 的核心问题，FF 完全没碰

### Direction 2：FF 没有涉及
- Portfolio rebalancing 和 cross-asset contagion 是 FF 完全没覆盖的
- Eileen 原始 proposal 的核心（7类基金资金流动 + Risk-Ladder）仍然有效
- **但需要 WRDS CRSP Mutual Fund 数据**

### Direction 3：FF 没有涉及
- 媒体放大效应是 FF 完全没覆盖的
- 需要新闻/社交媒体数据（Bloomberg/Reuters/Twitter/Reddit）
- **最复杂的方向，伦理审查+数据获取**

### Direction 4：FF 部分覆盖
- FF 做了 regime 分析（Table 10: ZLB vs non-ZLB）
- **但 FF 的 regime 是二元的（ZLB/非ZLB），我们原始设计是 Markov-Switching + ML**
- **差异化**：跨 Fed 主席对比 + ML regime detection

---

## 四、调整后的 Phase 2-4

### Phase 2（Direction 2）：Portfolio Rebalancing and Cross-Asset Contagion

**理论核心不变**：货币政策公告后，投资者如何跨资产再平衡？

| 任务 | 优先级 | 工作量 | 数据需求 |
|------|--------|--------|---------|
| CRSP Mutual Fund 7类资产资金流动 | 🔴 高 | 3-5天 | WRDS CRSP MF |
| DCC-GARCH 跨资产动态相关 | 🔴 高 | 3-5天 | 现有 FRED 数据 |
| Network contagion（minimum spanning tree） | 🟡 中 | 2-3天 | 现有数据 |
| Risk-Off / Risk-On / Risk-Ladder 假设检验 | 🔴 高 | 2-3天 | Phase 2 前置任务完成 |
| 按市场状态分组（VIX 高/低） | 🟡 中 | 1-2天 | 现有数据 |

**与当前论文的衔接**：Direction 1 发现 target shock 显著 → Direction 2 追问"显著之后投资者怎么行动"

### Phase 3（Direction 4）：Time-Varying Transmission and Regime Detection

**理论核心不变**：以上效应如何随环境变化？

| 任务 | 优先级 | 工作量 | 数据需求 |
|------|--------|--------|---------|
| Markov-Switching VAR 识别隐含 regime | 🔴 高 | 5-7天 | 现有数据 |
| 分 regime 估计 surprise → asset response | 🔴 高 | 2-3天 | Phase 1-2 完成 |
| 跨 Fed 主席对比（G/B/Y/P 四时期） | 🟡 中 | 2-3天 | 现有数据 |
| ML regime detection（HMM / clustering） | 🟡 中 | 3-5天 | 现有数据 |
| "Powell put" vs "Greenspan put" 量化 | 🟢 可选 | 2天 | 现有数据 |

**为什么先做 Direction 4 再做 Direction 3**：原始计划建议 Direction 4 在 Direction 3 之前，因为方法论更成熟、数据现成。Direction 3 需要爬取社交媒体数据+伦理审查，最复杂。

### Phase 4（Direction 3）：Media Amplification and Social Transmission

**理论核心不变**：媒体如何放大或扭曲货币政策信号？

| 任务 | 优先级 | 工作量 | 数据需求 |
|------|--------|--------|---------|
| 新闻框架分析（Bloomberg/Reuters 标题 NLP） | 🔴 高 | 5-7天 | 新闻数据获取 |
| 媒体放大指数（MAI）构建 | 🔴 高 | 2-3天 | 新闻数据 |
| MAI 调节效应检验 | 🔴 高 | 2-3天 | Phase 1 完成 |
| 社交媒体数据（Twitter/Reddit） | 🟡 中 | 5-7天 | API + 伦理审查 |
| Two-Step Flow 验证（KOL → 公众） | 🟢 可选 | 3-5天 | 社交媒体数据 |

---

## 五、Direction 1 的深化（当前论文的扩展）

当前论文是 Direction 1 的第一轮产出，但还有未覆盖的理论：

| 未覆盖理论 | 升级方案 | 优先级 |
|-----------|---------|--------|
| Rational Inattention | 用 FF narrative surprise 做外部验证 | 🟡 中 |
| Framing Theory | Statement vs Minutes vs Press Conf 三层对比 | 🔴 高 |
| Agenda-Setting | 新闻标题框架分析（与 Direction 3 衔接） | Phase 4 |
| JK 分解 | 验证 target = MP + CBI（已做简化版） | 🟡 中 |
| B-S 正交化 | 验证外生性（已做简化版） | 🟡 中 |

这些深化可以在 Phase 2-4 推进的同时并行完成，作为 Direction 1 的 v2 论文。

---

## 六、核心逻辑总结

**原始设计的四个 Direction 是递进的因果链：**

```
Direction 1: 信息如何产生（央行说什么）
    → Direction 2: 信息如何传播（投资者怎么动）
        → Direction 3: 信息如何被放大（媒体怎么加工）
            → Direction 4: 以上效应如何随环境变化
```

**FF (2025) 只影响了 Direction 1 的方法选择，没有替代任何 Direction 的核心问题。**

| Direction | FF 覆盖？ | 我们的独特性 |
|-----------|----------|------------|
| 1 信息内容 | ⚠️ 方法覆盖，问题未碰 | Implementation vs Revelation |
| 2 组合再平衡 | ❌ | Risk-Ladder + 7类基金流动 |
| 3 媒体放大 | ❌ | MAI + Two-Step Flow |
| 4 Regime 变化 | ⚠️ 二元 regime | Markov-Switching + ML + 跨主席 |

**调整原则**：
1. **保持原始四方向架构**——这是 Eileen 的研究设计，理论根基扎实
2. **Direction 1 深化**——JK + B-S + 三层文档对比，作为 v2 论文
3. **Direction 2 优先推进**——与 Eileen 原始 proposal 的核心一致，且 FF 完全没碰
4. **Direction 4 先于 Direction 3**——方法论更成熟，数据现成
5. **FF 的 narrative surprise 作为工具变量**——服务于 Direction 1 的外生性验证

---

*分析日期: 2026-06-02*
*基于: Eileen 原始 Research Plan + Research Proposal + Literature Review + FF (2025)*
*理论框架: Signaling → Portfolio Balance → Social Amplification → Lucas Critique*
