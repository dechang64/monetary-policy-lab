# Phase 2-4 研究计划调整建议
## 基于 Fernández-Fuertes (2025) 及最新文献调研

---

## 当前 Phase 2-4 计划回顾

| Phase | 原计划 | 状态 |
|-------|--------|------|
| Phase 2 | 高频 Two-Shocks 识别（TAQ + JK sign restriction） | ❌ 未启动（TAQ 需 WRDS 高级权限） |
| Phase 3 | 信息冲击验证 + 稳健性（SPF + VIX + IV） | ❌ 未启动 |
| Phase 4 | 资本流动真实数据（CRSP Mutual Fund） | ❌ 未启动 |
| Phase 5 | Sentiment 增强（FinBERT / CB 词典扩充） | ❌ 未启动 |

---

## FF (2025) 对我们的冲击评估

### FF 已经做了什么（我们原计划要做但被他抢先的）

| 原计划 | FF 的对应实现 | 我们的处境 |
|--------|-------------|-----------|
| Phase 2: JK sign restriction | ✅ Table 10: JK decomposition，完整 BVAR 版本 | 我们只做了简化版（符号分类），他做了完整版 |
| Phase 2: 高频识别升级 | ✅ Multi-agent LLM ex ante 预期，比 HF 更干净 | 完全不同的路线，他绕过了 HF 的污染问题 |
| Phase 3: B-S 正交化 | ✅ Table 5: predictability test，完整 B-S 控制变量 | 我们做了简化版（4个控制变量），他用完整 B-S 变量集 |
| Phase 5: Sentiment 增强 | ✅ LLM 概率提取，R² = 12.4% vs 我们 1.57% | 他用 GPT-4 multi-agent，我们用 dictionary |

### FF 没做什么（我们的差异化空间）

| 领域 | FF | 我们 |
|------|-----|------|
| 核心问题 | 造更好的尺子 | 语言承载什么（Implementation vs Revelation） |
| Statement vs Minutes 分层 | 所有文档扔进一个 pipeline | 分层分析，发现 Minutes 中 path 显著 |
| FG 交互检验 | 无 | 有（p=0.836，不显著） |
| Wald 检验 | 无 | 有（p=0.90，不能拒绝 β_T=β_P） |
| 透明性/可复现性 | GPT-4 API 依赖 | Dictionary + OLS，任何人可复现 |
| 跨资产异质性 | 有限 | 6类资产 + EW>VW 发现 |

---

## 调整建议

### 🔴 降级/取消的项目

**1. Phase 2: TAQ 日内高频识别 → 取消**
- FF 用 LLM 绕过了 HF 污染问题，不需要 TAQ 数据
- TAQ 需要 WRDS 高级权限 + 巨量数据下载（单日数 GB）
- 投入产出比极低：即使拿到 TAQ，也只是改进识别精度，不改变核心结论
- **替代**：我们已做的简化版 JK 分解（符号约束）足够支撑论文

**2. Phase 4: CRSP Mutual Fund 资本流动 → 降级为可选**
- 资本流动与我们的核心问题（Implementation vs Revelation）关系弱
- FF 完全没做资本流动，说明这不是该领域的核心问题
- 如果审稿人要求，可以作为 Online Appendix

**3. Phase 5: FinBERT Sentiment → 调整方向**
- 原计划用 FinBERT 提升 sentiment 方差
- FF 已经证明 LLM 方法远超 dictionary，但他的方法不可复现
- **新方向**：不做"更好的 sentiment"，而是做"sentiment 方法论对比"——dictionary vs LLM，展示两者在 H1-H4 上是否给出一致结论。这本身就是贡献

### 🟡 保留但调整的项目

**4. Phase 2: JK 分解 → 升级为完整 BVAR 版本**
- 当前简化版（符号分类）已跑通，但审稿人可能要求完整版
- 完整版需要：BVAR 估计 + sign restriction + 脉冲响应
- Python 可用 `bvar` 或 `statsmodels` 实现
- **优先级**：中——如果投顶刊，审稿人大概率会问

**5. Phase 3: B-S 正交化 → 升级控制变量集**
- 当前只有4个控制变量（lagged returns, VIX, term spread, rate changes）
- B-S (2023) 原文用 Greenbook 预测误差、NFP surprise 等
- **升级路径**：
  - 加入 S&P 500 3-month return（我们已有）
  - 加入 Nonfarm Payrolls surprise（需 FRED 数据，可获取）
  - 加入 Greenbook 预测（需 Philadelphia Fed SPF 数据）
- **优先级**：高——B-S 是当前最大的方法论挑战

**6. Phase 3: IV 估计 → 新增**
- FF 没做 IV，但这是解决 endogeneity 的标准方法
- 用 lagged shocks 作为 IV，或用 FF 的 narrative surprise 作为 IV
- **优先级**：中高——如果 B-S 正交化后 target 仍不显著，IV 是最后手段

### 🟢 新增项目

**7. FF Narrative Surprise 作为 IV（NEW）**
- FF 的 LLM surprise 是外生于 announcement-day 信息的
- 可以用他的 surprise 作为我们回归的 IV
- 如果 IV 估计后 target 仍显著，说明结果不是 HF 污染驱动的
- **数据**：FF 论文公开了数据（需联系作者或从 SSRN 下载）
- **优先级**：高——直接回应 B-S 批评

**8. Statement vs Minutes vs Press Conference 三层对比（NEW）**
- 我们已有 Statement vs Minutes 的发现
- FF 处理了 Press Conference 但没有分层对比
- 加入 Press Conference transcript 的 sentiment 分析
- **数据**：Press Conference transcripts 可从 Fed 网站抓取
- **优先级**：高——这是 FF 没有的独特贡献

**9. 跨央行对比（NEW）**
- Literature Radar 发现多篇 ECB 相关论文（relevance 0.32-0.38）
- 如果 Implementation vs Revelation 在 ECB 也成立，外部有效性大幅提升
- **数据**：ECB statements 公开可获取
- **优先级**：中——投顶刊的加分项

---

## 调整后的 Phase 2-4

| Phase | 内容 | 优先级 | 预计时间 | 依赖 |
|-------|------|--------|---------|------|
| **Phase 2** | B-S 正交化升级（完整控制变量集） | 🔴 高 | 2-3天 | FRED NFP 数据 |
| **Phase 2** | FF Narrative Surprise 作为 IV | 🔴 高 | 1-2天 | FF 数据获取 |
| **Phase 2** | Statement vs Minutes vs Press Conference | 🔴 高 | 3-4天 | Press Conf 抓取 |
| **Phase 3** | IV 估计（lagged shocks / FF surprise IV） | 🟡 中高 | 2天 | Phase 2 完成 |
| **Phase 3** | JK BVAR 完整版 | 🟡 中 | 3-5天 | Python BVAR 库 |
| **Phase 3** | Sentiment 方法论对比（dict vs LLM） | 🟡 中 | 2-3天 | MiniMax API |
| **Phase 4** | 跨央行对比（ECB） | 🟢 可选 | 3-5天 | ECB 数据 |
| **Phase 4** | CRSP Mutual Fund 资本流动 | ⚪ 降级 | 2-3天 | WRDS 权限 |

---

## 核心逻辑

**不要和 FF 比方法，和他比问题。**

FF 的核心贡献是"更好的尺子"。我们的核心贡献是"用尺子回答他没问的问题"。

调整后的 Phase 2-4 应该强化我们的问题优势：
1. **三层文档对比**（Statement vs Minutes vs Press Conf）——FF 没有分层
2. **B-S 完整正交化 + IV**——回应最大的方法论挑战
3. **Sentiment 方法论对比**——不是做更好的 sentiment，而是证明 dictionary 和 LLM 给出一致结论

这样定位：FF 解决了"如何更好地测量冲击"，我们解决了"冲击和语言的关系是什么"。两个工作互补，不竞争。

---

*分析日期: 2026-06-02*
*基于: FF (2025) JMP 153页 + Literature Radar 43篇扫描*
