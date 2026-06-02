# Phase 2-4 研究计划调整建议
## 基于 Fernández-Fuertes (2025) 及最新文献调研

---

## 一、四个 Phase 背后的四个基本理论

原始设计不是技术升级路线图，而是**四层理论验证架构**——每一层回答一个递进的理论问题：

### Phase 1（已完成）：GSS 分解 + Sentiment → Implementation vs Revelation
**理论**：Gürkaynak-Sack-Swanson (2005) 的双因子分解
- **核心问题**：FOMC 语言反映的是当前决策（target）还是未来政策预期（path）？
- **答案**：Target 显著，Path 不显著 → 偏 Implementation
- **局限**：GSS path factor 不是纯信息冲击，可能混入 FG 承诺效应

### Phase 2：JK Sign Restriction → 纯政策冲击 vs 信息冲击
**理论**：Jarociński-Karadi (2020) 的符号约束识别
- **核心问题**：Target shock 里的"显著"到底是纯政策效应，还是央行信息效应（CBI）？
- **理论逻辑**：如果加息+股市涨 → CBI（央行看好经济）；如果加息+股市跌 → 纯 MP（紧缩）
- **我们的初步结果**：Sentiment 不区分 MP/CBI（p=0.87），但 asset returns 强烈区分（R² 从 9%→36%）
- **意义**：语言反映"做了什么"不反映"为什么做"——这比 Phase 1 的结论更深一层

### Phase 3：B-S 正交化 + 外部预期验证 → 冲击的外生性
**理论**：Bauer-Swanson (2023) 的可预测性批评 + Miranda-Agrippino & Ricco (2021) 的内生性
- **核心问题**：HF shocks 是真正外生的，还是被 pre-FOMC 宏观信息污染了？
- **理论逻辑**：如果 shocks 可被公开信息预测，那"显著"可能反映的是内生信息而非外生政策
- **我们的初步结果**：Target 在 sentiment 中失去显著性（p: 0.012→0.108），但在 returns 中增强（p: 0.043→0.005）
- **意义**：外生性问题对 sentiment 和 returns 的影响不对称——这本身是重要发现

### Phase 4：资本流动 + 异质性传导 → 传导机制
**理论**：Gertler-Gilchrist (1994) 的异质性货币传导 + portfolio rebalancing
- **核心问题**：MP shocks 如何通过融资渠道和组合再平衡影响不同规模企业？
- **理论逻辑**：小企业更依赖外部融资 → 对 MP shocks 更敏感 → EW > VW
- **我们的初步结果**：EW 确实比 VW 更显著（H2），但缺乏真实资金流动数据验证传导渠道
- **意义**：从"有没有效应"推进到"效应通过什么渠道传导"

---

## 二、四层理论的递进关系

```
Phase 1: 语言反映什么？→ Implementation（target 显著，path 不显著）
    ↓ 但 target 里的显著是纯政策还是信息？
Phase 2: Target = MP + CBI？→ Sentiment 不区分，但 returns 区分
    ↓ 但 shocks 本身是外生的吗？
Phase 3: Shocks 可预测吗？→ 部分可预测，对 sentiment 和 returns 影响不对称
    ↓ 但效应通过什么渠道传导？
Phase 4: 传导渠道是什么？→ 融资渠道（EW > VW），但缺真实资金流数据
```

每一层都在前一层的基础上深化：Phase 1 发现现象 → Phase 2 分解机制 → Phase 3 验证外生性 → Phase 4 追踪传导。

---

## 三、FF (2025) 对四层理论的冲击

### Phase 2（JK 分解）：FF 做了完整版

| 维度 | 我们的简化版 | FF 的完整版 |
|------|------------|-----------|
| 方法 | 符号分类（sign of shock × sign of return） | BVAR sign restriction + 脉冲响应 |
| 分类 | MP=69, CBI=48 | 连续概率分布 |
| Sentiment 结果 | β_MP ≈ β_CBI，均不显著 | 未做（他关心的是 shock 构造，不是语言） |
| Returns 结果 | β_MP = -1.03***, β_CBI = +0.84*** | 类似，且通过纯 MP 检验 |

**评估**：FF 做了更好的 JK 分解，但他**没有把 JK 分解和 sentiment 联系起来**。我们的核心发现——"sentiment 不区分 MP/CBI 但 returns 区分"——他完全没有。**Phase 2 的理论问题仍然有效，但需要升级方法。**

**调整**：
- ✅ 保留 Phase 2 的理论问题
- 🔄 方法升级：从简化符号分类 → BVAR sign restriction（用 Python `bvar` 或 `statsmodels`）
- 🆕 新增：用 FF 的 narrative surprise 作为外部验证——他的 surprise 在 JK 分解中通过纯 MP 检验，如果我们用他的 surprise 替代 GSS shocks，结论是否一致？

### Phase 3（B-S 正交化）：FF 做了更完整版

| 维度 | 我们的简化版 | FF 的完整版 |
|------|------------|-----------|
| 控制变量 | 4个（lagged return, VIX, term spread, rate change） | 完整 B-S 变量集（NFP surprise, Greenbook forecast 等） |
| Target 可预测性 | R² = 10.5% | 类似 |
| Path 可预测性 | R² = 13.8% | 类似 |
| Sentiment 结果 | Target 失去显著性 | 未做 |

**评估**：FF 的 B-S 验证更完整，但同样**没有把正交化后的 shocks 和 sentiment 联系起来**。我们的不对称发现（sentiment 失去显著性但 returns 增强）是独有的。

**调整**：
- ✅ 保留 Phase 3 的理论问题
- 🔄 升级控制变量集：加入 NFP surprise、Greenbook 预测偏差（如果有 WRDS IBES/SPF 权限）
- 🆕 新增：用 FF 的 narrative surprise（已通过 B-S predictability test）作为 IV，估计 sentiment 回归的因果效应

### Phase 4（资本流动）：FF 没有涉及

**评估**：Phase 4 是 FF 完全没有覆盖的领域。异质性传导是我们的独特贡献。

**调整**：
- ✅ 保留 Phase 4 的理论问题
- ⚠️ 但优先级降低——Phase 2-3 的理论深化更紧迫
- 🔄 如果 WRDS 权限有限，可以先用 CRSP size-sorted portfolios 替代真实基金流动

---

## 四、调整后的 Phase 2-4

### Phase 2（升级）：JK 分解 + FF 验证

**理论核心不变**：Target shock 里的显著是纯政策还是信息效应？

| 任务 | 优先级 | 工作量 | 数据需求 |
|------|--------|--------|---------|
| BVAR sign restriction（替代简化符号分类） | 🔴 高 | 3-5天 | 现有数据 |
| 用 FF narrative surprise 替代 GSS shocks 重跑 H1-H4 | 🔴 高 | 1-2天 | FF 数据（已下载） |
| Statement vs Minutes vs Press Conference 三层对比 | 🔴 高 | 3-4天 | Press Conf 抓取 |
| Sentiment 方法论对比（dict vs LLM） | 🟡 中 | 2-3天 | MiniMax API |

**为什么三层对比是 Phase 2**：三层对比直接服务于 JK 分解的理论问题——如果 Statement 不区分 MP/CBI，但 Minutes 区分了，那说明**不同沟通渠道承载不同信息类型**，这是对 JK 分解的深化，不是独立问题。

### Phase 3（升级）：B-S 正交化 + IV 估计

**理论核心不变**：Shocks 是外生的吗？对 sentiment 和 returns 的影响是否对称？

| 任务 | 优先级 | 工作量 | 数据需求 |
|------|--------|--------|---------|
| B-S 完整控制变量集升级 | 🔴 高 | 2-3天 | FRED NFP + Greenbook |
| FF narrative surprise 作为 IV | 🔴 高 | 1-2天 | FF 数据 |
| 2SLS 估计：sentiment 的因果效应 | 🟡 中高 | 2天 | Phase 2 完成 |
| 不对称性的理论解释 | 🟡 中 | 1-2天 | — |

**为什么 IV 是 Phase 3**：IV 估计直接回应 B-S 批评——如果 shocks 有内生性问题，用 FF 的外生 surprise 做工具变量可以恢复因果效应。这是 Phase 3 外生性验证的自然延伸。

### Phase 4（保留但降优先级）：异质性传导

**理论核心不变**：MP shocks 通过什么渠道传导？

| 任务 | 优先级 | 工作量 | 数据需求 |
|------|--------|--------|---------|
| CRSP size-sorted portfolio 回归 | 🟡 中 | 1-2天 | WRDS CRSP |
| 行业层面异质性（金融 vs 非金融） | 🟢 可选 | 2天 | WRDS CRSP |
| CRSP Mutual Fund 真实流动 | ⚪ 降级 | 2-3天 | WRDS MF |

---

## 五、核心逻辑总结

**四个 Phase 对应四个递进的理论问题，FF 没有替代任何一个：**

| Phase | 理论问题 | FF 覆盖了？ | 我们的独特性 |
|-------|---------|------------|------------|
| 1 | 语言反映什么？ | ❌ 他不关心语言 | 核心问题本身 |
| 2 | 显著的是政策还是信息？ | ⚠️ 做了方法，没连语言 | Sentiment 不区分 MP/CBI |
| 3 | Shocks 是外生的吗？ | ⚠️ 做了方法，没连语言 | 不对称性发现 |
| 4 | 通过什么渠道传导？ | ❌ 完全没涉及 | EW > VW + 融资渠道 |

**FF 解决了"如何更好地测量冲击"，我们解决的是"冲击和语言的关系是什么"。**

调整方向：不是砍 Phase，而是**每个 Phase 都用 FF 的方法/数据升级我们的验证**，同时保持我们的问题优势。FF 的 narrative surprise 是我们的工具变量，不是我们的竞争者。

---

*分析日期: 2026-06-02*
*基于: FF (2025) JMP 153页 + Literature Radar 43篇扫描*
*理论框架: GSS (2005) → JK (2020) → B-S (2023) → GG (1994)*
