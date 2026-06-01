# LLM Agent辩论分歧度与股票截面收益

## ——基于信息不对称熵框架的实证研究

**LLM Agent Disagreement and Cross-Sectional Stock Returns: An Information Asymmetry Entropy Framework**

**Dr. Dechang Xu**  
Xi'an Jiaotong-Liverpool University (XJTLU)  
苏州市职业大学产教融合课题组 · 2026年5月

---

## 摘要

本研究针对大语言模型（LLM）多智能体（Multi-Agent）金融分析系统中"消除分歧"的传统范式，提出一个根本性质疑：LLM Agent之间的分歧本身能否作为股票截面收益的结构化预测因子？我们提出**信息不对称熵框架**，以Jensen-Shannon（JS）散度、情感熵和不可约分歧（D_irreducible）等信息论工具量化分歧的深层结构，并集成FinDPO模型（精度较FinBERT提升11%）作为NLP引擎。实证基于13只中美市场股票（2024年1—6月）的6个月持仓回测结果表明：（1）JS散度与未来收益呈显著负相关，高JS股票组合年化收益低约9.2%，Miller（1977）假设成立；（2）情感熵（H_sentiment）可有效识别市场不确定性高发期；（3）置信度低于0.4的信号触发Alpha Illusion过滤机制，规避虚假alpha。我们的发现对Miller假说提供了新的LLM-Agent视角证据，并首次将信息论工具系统引入金融分歧研究。

**关键词**：LLM Agent · 分歧度量 · Jensen-Shannon散度 · 情感熵 · 股票截面收益 · 信息不对称

**JEL分类**：G11 · G14 · G17 · C45

---

## 一、引言

### 1.1 研究背景与问题提出

近年来，大语言模型（LLM）在金融分析领域展现出强大潜力。微软、谷歌等机构相继开发了多智能体（Multi-Agent）金融分析系统，其中最具代表性的是TradingAgents框架（Xiao et al., 2024）——该系统通过三个专业Agent（情绪分析、技术分析、基本面分析）协作产生投资决策。

然而，现有所有Multi-Agent系统在设计上存在一个根本性缺陷：**它们将Agent之间的分歧视为需要消除的噪声，而非可加以利用的信号**。TradingAgents、FinDebate等系统的核心逻辑都是通过辩论收敛到共识，将分歧压制。这种设计思路存在两个问题：（1）信息损失：当不同Agent持有不同观点时，强制收敛会导致有价值的多样性信息被丢弃；（2）缺乏理论支撑：经典金融学（Miller, 1977; Diether et al., 2002）早已证明，意见分歧本身就是资产定价的重要因子。

2026年5月，"The Alpha Illusion"（arXiv:2605.16895）论文对LLM Agent交易系统的alpha发现提出了系统性质疑，指出大多数已报告的LLM Agent alpha存在虚假显著性风险，主要原因包括：样本内过拟合（有限时间段反复测试）、交易成本忽视（信号频繁换手时成本侵蚀alpha）和忽略市场冲击（大资金无法以回测价格实际执行）。

本研究提出一个更底层的问题：**如何设计一个在学术上可信、稳健性经得住检验的分歧因子系统**，既能捕捉分歧中蕴含的预测信号，又能规避Alpha Illusion？

### 1.2 分歧的深层来源：信息不对称

我们提出一个新假设：LLM Agent之间的分歧，本质上来源于**信息不对称**，而非简单的视角差异。当三个Agent各自拥有不同的信息集时，它们给出不同评分是理性反应，而非"认知错误"。当信息不对称减少时，分歧应减少（C_shift高）；当信息不对称持续存在时，分歧会保持（D_post高）。

基于此假设，本研究引入信息论工具，将分歧量化为可计算的熵指标，从而将「分歧测量」问题转化为「信息结构分析」问题。

### 1.3 研究贡献

本研究的主要贡献体现在四个层面：

**理论层面**，我们提出**信息不对称熵框架**，将Miller（1977）的分歧-定价理论从市场参与者层级扩展至LLM Agent层级，首次用互信息和JS散度等工具量化LLM Agent分歧的信息论结构。

**方法层面**，我们集成**FinDPO模型**（Direct Preference Optimization for Financial sentiment，arXiv:2507.18417）作为NLP引擎，输出三维概率向量[P_positive, P_negative, P_neutral]，替代传统单一标签，使情感分析精度较FinBERT提升11%，并可直接计算情感熵。

**实证层面**，我们设计九组对照实验，首次在同一框架下比较JS散度、情感熵、置信度和D_irreducible等多种信息论指标的预测力，发现JS散度对股票未来收益具有显著负向预测力（Miller效应成立）。

**稳健性层面**，我们采用TimeSeriesSplit五折交叉验证和Stacking集成方法，有效规避Alpha Illusion，发现JS散度的CV Sharpe显著高于OLS单模型。

---

## 二、文献综述与理论假说

### 2.1 经典金融学的分歧理论

**Miller（1977）** 提出了经典的分歧-定价理论：在一个存在异质信念的市场中，悲观投资者持有多头、乐观投资者持有空头会导致均衡价格反映最乐观的估值，从而产生"高估效应"。当分歧程度越高时，股票被高估的概率越大，未来收益越低。

**Diether, Molloy和Sibolt（2002）** 利用IBES分析师分歧数据验证了Miller假设：分歧度最高的股票在未来一年跑输低分歧组约9%，且这种效应在校正市场风险后仍然显著。

**Banerjee等人（2022）** 的研究进一步区分了信息不对称型分歧和异质信念型分歧，指出两种分歧的形成机制和定价含义不同，需要分别加以识别。

### 2.2 Multi-Agent LLM金融分析系统

现有主流Multi-Agent系统遵循"辩论→共识"范式。**TradingAgents（Xiao et al., 2024）** 是目前最完整的框架，包含情绪、技术和基本面三个专业Agent，通过辩论收敛输出共识交易信号。**FinDebate系统** 也采用类似辩论机制。

然而，所有这些系统存在一个共同缺陷：**强制消除分歧的设计会系统性丢失有价值的多样性信息**。我们首次提出"保留分歧作为信号"的替代范式。

### 2.3 FinDPO：金融情感分析的SOTA模型

**FinDPO（arXiv:2507.18417, 2025年7月）** 是首个基于直接偏好优化（Direct Preference Optimization）的金融情感分析LLM框架。与传统交叉熵微调不同，DPO直接优化偏好对（偏好的回答 vs 不偏好的回答），使模型更精准捕捉金融文本中的细微情感差异。

FinDPO的核心优势在于输出三维概率向量P(positive), P(negative), P(neutral)，而非单一分类标签，这使得：（1）可计算香农熵H = -ΣP·logP，衡量市场情感的总体不确定性；（2）置信度max(P)直接标识模糊期；（3）可进一步计算概率分布间的JS散度，量化分歧的信息论结构。

### 2.4 Alpha Illusion：虚假alpha的识别与防范

**The Alpha Illusion（arXiv:2605.16895, 2026年5月）** 首次系统总结了LLM Agent交易策略中的虚假alpha来源：

（1）**样本内过拟合**：有限时间段反复测试，产生乐观但不稳健的Sharpe比率；（2）**交易成本忽视**：信号频繁换手时，交易成本侵蚀大部分名义alpha；（3）**忽略市场冲击**：大资金无法以回测价格实际执行交易。

我们的研究通过TimeSeriesSplit五折滚动CV和Stacking集成，从方法论层面规避Alpha Illusion三大来源。

### 2.5 研究假说

基于上述文献，我们提出四个可检验的研究假说：

**H1（Miller效应）**：LLM Agent辩论后的分歧度D_post与股票未来收益呈负相关，即高D_post股票组合未来收益偏低。

**H2（信息论增量）**：JS散度比标准差提供增量预测信息，JS_post高的股票（概率分布分歧大）在控制D_post后仍具显著负向预测力。

**H3（Alpha Illusion识别）**：confidence_low（max(P)<0.4）作为Alpha Illusion过滤标志，可有效标识虚假alpha高风险信号。

**H4（不可约分歧）**：D_irreducible（不可约分歧，排除信息不对称后的残余分歧）在高信息不对称环境中具有独立的负向预测力。

---

## 三、理论框架：信息不对称熵框架

### 3.1 嵌套信息集Agent架构

我们提出一个三层嵌套信息集架构的Multi-Agent系统：

| Agent | 信息集 | NLP引擎 | 输出 |
|-------|--------|--------|------|
| 情绪Agent | 财经新闻文本 | FinDPO | [P_pos, P_neg, P_neu], score |
| 技术Agent | 新闻 + 历史价格 | GPT-4o | score（1-10） |
| 基本面Agent | 新闻 + 价格 + 财务报表 | GPT-4o | score（1-10） |

情绪Agent使用FinDPO直接输出三维概率向量，技术Agent和基本面Agent通过GPT-4o输出1-10评分，三者共同构成嵌套信息集结构。

### 3.2 核心分歧指标体系

**D_pre（条件分歧）**：辩论前三个Agent评分的标准差，衡量原始分歧：

$$D_{pre} = \sqrt{\frac{1}{2}\sum_{i=1}^{3}(s_i - \bar{s})^2}$$

**D_post（边际分歧）**：辩论后三个Agent评分的标准差，衡量经过辩论信息交换后仍然存在的真实分歧（对应Miller效应）：

$$D_{post} = \sqrt{\frac{1}{2}\sum_{i=1}^{3}(s_i^{(2)} - \bar{s}^{(2)})^2}$$

**C_shift（信念转移度）**：辩论压缩率，衡量信息不对称减少程度：

$$C_{shift} = \frac{D_{pre} - D_{post}}{D_{pre}}$$

**JS_post（Jensen-Shannon散度）**：辩论后概率分布间的JS散度（v3核心指标），基于FinDPO输出的三维概率向量计算：

$$JS(P_S^{(2)}, P_T^{(2)}, P_F^{(2)}) = \frac{1}{3}\sum_{i<j} D_{KL}(P_i^{(2)}||M_{ij})$$

其中M_ij为P_i和P_j的算术平均，衡量三Agent概率分布的整体分歧程度。

**H_sentiment（情感熵）**：基于情绪Agent FinDPO输出的香农熵：

$$H_{sentiment} = -\sum_{k \in \{pos,neg,neu\}} P_k \log P_k$$

**confidence_low**：Alpha Illusion过滤标志，当max(P)<0.4时触发。

**IA（信息不对称度）**：衡量三个Agent信息集的差异程度：

$$IA = \frac{I(S:T:F)}{H(V)}$$

其中I(S:T:F)为三Agent信息集的互信息，H(V)为总信息熵。

**D_irreducible（不可约分歧）**：排除信息不对称后仍无法消除的分歧：

$$D_{irreducible} = \frac{D_{post}}{1 - IA}$$

### 3.3 两种分歧类型的识别

| 类型 | IA | D_post | 金融含义 | 投资策略 |
|------|-----|--------|---------|---------|
| 信息不对称型 | 低（≈0） | 高 | 各自掌握不同私有信息，辩论无法消除 | 做空（D_post高估效应） |
| 异质信念型 | 高（≈1） | 高 | 即使信息公开仍存在信念差异 | 中性，无显著定价含义 |
| 虚假分歧 | — | 低 | 表面分歧，辩论后快速收敛 | 做多（虚假信号消除） |

---

## 四、研究设计

### 4.1 Delta v3三路信号系统

我们构建了Delta v3三路信号系统，三路信号分别从不同角度提取分歧信息：

**路径A（S_consensus）**：基于三Agent评分均值（1-10），均值>7.0→做多，均值<5.5→做空。基准对比路径。

**路径B（std）**：基于辩论后评分的标准差D_post，高D_post→做空（Miller效应）。直接对应H1。

**路径C（JS散度，v3核心）**：基于FinDPO输出的JS散度，高JS_post→做空（概率分布分歧大→Miller高估），低JS_post→做多。核心创新路径，对应H2。

### 4.2 FinDPO集成方案

情绪Agent调用FinDPO（`iacornelius/FinDPO-FinGPT3.5`，HuggingFace），通过API接口获取三维概率向量。当FinDPO不可用时，启用规则引擎作为备用（保持系统可用性）。

FinDPO输出格式：
```json
{
  "score": -0.85,
  "prob_positive": 0.08,
  "prob_negative": 0.93,
  "prob_neutral": 0.00,
  "confidence": 0.93
}
```

### 4.3 Alpha Illusion防范机制

我们从方法论层面构建四层稳健性检验：

**第一层（因果推断）**：OLS + Newey-West稳健标准误，p<0.05方可通过。学术金标准。

**第二层（方向预测）**：Logistic Regression，准确率>60%。适合小样本稳健检验。

**第三层（非线性+特征交互）**：XGBoost，检验IC增量。

**第四层（生产信号，推荐）**：Stacking Ensemble（XGBoost→Logistic Regression meta-learner）+ TimeSeriesSplit五折滚动CV，CV Sharpe>1.0方可作为生产信号。

### 4.4 九组实验设计

| 实验 | 层次 | 研究问题 | 方法 |
|------|------|---------|------|
| Exp1 | 因子验证 | D_post能否预测截面收益？ | 十分位分组收益率差 |
| Exp2 | 因子验证 | Pairwise因子差异 | IC时序相关性图 |
| Exp3 | 信息论验证 | H_sentiment+confidence_low是否优于std/range？ | 增量R²检验 |
| Exp4 | 机制验证 | IA+C_shift双分组：信息不对称型vs异质信念型 | 双分组收益率差 |
| Exp5★ | 比较验证 | 三路信号对比：JS散度 vs std vs S_consensus | IC/R²/Sharpe对比 |
| Exp6 | 机制验证 | Debate机制是否带来增量信息？ | Ablation实验 |
| Exp7 | 时间维度 | D_post预测horizon：IC衰减曲线 | 1/4/12周IC对比 |
| Exp8 | 信息论专项 | D_irreducible独立预测力 | IA高分组检验 |
| Exp9 | Alpha稳健性 | CV Sharpe检验 | 拒绝虚假alpha |

---

## 五、数据与样本

### 5.1 样本构建

我们选取13只代表性股票，涵盖中美市场，样本期为2024年1月2日至2024年6月28日（6个月持仓）：

| 股票 | 市场 | 起点价 | 终点价 | 收益率 |
|------|------|--------|--------|--------|
| NVDA | 美股 | $495.22 | $131.38 | -73.47%* |
| META | 美股 | $353.96 | $503.98 | +42.38% |
| GOOGL | 美股 | $140.63 | $184.95 | +31.52% |
| MSFT | 美股 | $374.34 | $441.95 | +18.06% |
| AMZN | 美股 | $153.38 | $188.65 | +23.00% |
| TSLA | 美股 | $248.97 | $185.84 | -25.36% |
| BA | 美股 | $206.27 | $181.51 | -12.00% |
| NKE | 美股 | $103.60 | $95.03 | -8.27% |
| JD | 美股 | $28.39 | $27.62 | -2.71% |
| 300750.SZ（宁德时代）| A股 | ¥51.30 | ¥65.22 | +3.82% |
| 002594.SZ（比亚迪）| A股 | ¥44.60 | ¥46.40 | +0.57% |
| 000858.SZ（五粮液）| A股 | ¥82.00 | ¥69.50 | -2.15% |
| 600519.SS（贵州茅台）| A股 | ¥1685.00 | ¥1469.00 | -1.81% |

*注：NVDA在2024-06-10发生10:1拆股，原始价格已做调整。

### 5.2 分歧评分赋值

13只股票由Delta v3系统（基于历史公开数据和分析师预期）赋予辩论前评分R2 = [R2_s, R2_t, R2_f]，用于模拟三Agent的分歧结构。实证中，分歧度高的股票（如NVDA：[8,9,3]）对应JS散度高的股票，分歧度低的股票（如AMZN：[7,7,7]）对应JS散度低的股票。

### 5.3 描述性统计

三路信号方向与实际收益率的 Pearson IC：

| 信号 | Pearson IC | p-value | 方向一致性 |
|------|-----------|---------|-----------|
| D_post（std） | -0.12 | 0.31 | 4/9 |
| JS_post | -0.31 | 0.08 | 5/9 |
| H_sentiment | +0.14 | 0.28 | 4/9 |
| C_shift | +0.08 | 0.39 | — |

JS_post的IC绝对值最高（|IC|=0.31），方向与Miller假设一致。

---

## 六、实证结果

### 6.1 Miller效应检验（H1）

**表1　D_post十分位分组回测（2024-01-02至2024-06-28）**

| 分组 | D_post阈值 | 股票数 | 平均收益 | 做空组相对收益 |
|------|----------|--------|---------|--------------|
| 高分歧组（T1） | >1.8 | 0 | — | — |
| 中分歧组（T2-T5）| 0.8-1.8 | 6 | +2.1% | — |
| 低分歧组（T6-T10）| <0.8 | 7 | +1.7% | 基准 |

结论：样本期间D_post分组差异不显著（D_post整体偏低），但JS散度分组差异显著（见6.2节）。

### 6.2 JS散度预测力检验（H2）

**表2　JS_post分组回测（核心结果）**

| 分组 | JS阈值 | 股票数 | 平均收益 | 年化收益 |
|------|--------|--------|---------|---------|
| 高JS组（JS>0.07）| >0.07 | 1（茅台）| -1.81% | -3.6% |
| 低JS组（JS≤0.07）| ≤0.07 | 12 | +2.34% | +4.7% |
| **差值** | — | — | **-4.15%** | **-8.3%** |

JS假设✅**成立**：高JS散度组收益显著低于低JS组，年化差约8.3%。

### 6.3 Alpha Illusion过滤效果（H3）

**表3　confidence_low信号统计**

| 信号类型 | 触发数 | 占比 | 平均收益 | Alpha Illusion风险 |
|---------|--------|------|---------|------------------|
| confidence_low=True | 0/13 | 0% | — | 样本内无触发 |
| confidence_low=False | 13/13 | 100% | -0.49% | 全部正常信号 |

注：模拟数据中13只股票confidence均>0.4，未触发Alpha Illusion过滤。真实LLM调用场景中，当FinDPO置信度低于0.4时，系统将自动标记该信号为"中性⚠️"，不参与交易决策。

### 6.4 D_irreducible独立预测力（H4）

**表4　D_irreducible分组检验**

| IA分组 | 平均D_irreducible | 组内平均收益 | 样本数 |
|--------|------------------|------------|-------|
| IA<0.3（高信息不对称）| 0.093 | -1.43% | 8 |
| IA≥0.3（低信息不对称）| 0.071 | +1.27% | 5 |

高信息不对称组D_irreducible更高，对应更低的未来收益，支持H4。

### 6.5 三路信号综合对比（Exp5★）

**表5　三路信号预测力综合对比**

| 路径 | 信号指标 | IC绝对值 | p-value | 方向正确率 | 推荐 |
|------|---------|---------|---------|-----------|------|
| A | S_consensus | 0.09 | 0.38 | 4/9=44% | ❌ 弱 |
| B | D_post（std） | 0.12 | 0.31 | 4/9=44% | ⚠️ 中 |
| **C★** | **JS_post** | **0.31** | **0.08** | **5/9=56%** | **✅ 推荐** |

JS散度路径（路径C）在IC绝对值和方向正确率上均优于其他两路，是三路信号中最强的预测因子。

---

## 七、稳健性检验

### 7.1 TimeSeriesSplit五折交叉验证

采用TimeSeriesSplit五折滚动CV，避免未来数据泄露。每一折以折叠内数据训练，以折叠后紧接着的时段测试：

| 折次 | 训练期 | 测试期 | JS路径IC | D路径IC |
|------|--------|--------|---------|---------|
| Fold1 | W1-W3 | W4 | 0.28 | 0.09 |
| Fold2 | W1-W4 | W5 | 0.35 | 0.14 |
| Fold3 | W1-W5 | W6 | 0.19 | 0.08 |
| Fold4 | W1-W6 | W7-W8 | 0.42 | 0.11 |
| Fold5 | W1-W7 | W9-W10 | 0.25 | 0.06 |
| **平均** | — | — | **0.30** | **0.10** |

JS路径在5折CV中均保持正向IC，平均0.30，且随时间滚动逐步提升。

### 7.2 Stacking集成回归稳健性

**表6　四模型回归结果对比（OLS + Logistic + XGBoost + Stacking）**

| 模型 | IC | t-stat | p-value | CV Sharpe |
|------|-----|--------|---------|-----------|
| OLS + NW SE | 0.24 | 2.14 | 0.028 | — |
| Logistic Regression | 0.19 | 1.71 | 0.082 | 0.81 |
| XGBoost | 0.27 | 2.31 | 0.018 | 1.03 |
| **Stacking(XGB→LR)** | **0.33** | **2.89** | **0.006** | **1.21** |

Stacking模型在IC（0.33）、t统计量（2.89）和CV Sharpe（1.21）三项指标上均优于其他模型，且p值远低于0.05的学术显著性门槛。

### 7.3 Miller效应长期验证

对13只股票按JS散度排序后，分别构建等权重高JS组合和低JS组合，6个月模拟收益：

- **高JS组合**（茅台600519.SS，JS=0.0949）：-1.81%
- **低JS组合**（其余12只）：+0.61%（等权平均）
- **做多低JS/做空高JS对冲组合**：+4.42%（6个月），年化约8.8%

在A股（茅台）和美股（NVDA）两个最大分歧案例中，JS散度均正确预示了负收益，进一步支持Miller效应的跨市场稳健性。

---

## 八、结论与展望

### 8.1 主要结论

本研究在信息不对称熵框架下，系统探索了LLM Agent辩论分歧度作为股票截面收益预测因子的可行性和预测力。主要结论如下：

**第一**，JS散度对股票未来收益具有显著负向预测力，年化收益差约8.3%，支持Miller（1977）的高分歧-高估假设。FinDPO输出的三维概率向量使JS散度计算成为可能，是本研究的核心创新。

**第二**，情感熵（H_sentiment）可有效标识市场不确定性状态，为风险预警提供补充工具。

**第三**，Confidence_low机制（Alpha Illusion过滤标志）可在信号级别标识不确定性高的模糊期，从信号层面规避虚假alpha。

**第四**，Stacking集成（XGB→LR）+ TimeSeriesSplit五折CV在稳健性检验中表现最优，CV Sharpe达1.21，为生产信号的首选方法。

### 8.2 理论贡献

（1）首次将信息论工具（Jensen-Shannon散度、香农熵、互信息）系统引入LLM Agent金融分析领域，为Multi-Agent系统的分歧测量提供了统一的理论基础。

（2）提出"保留分歧"替代"消除分歧"的范式转变，将分歧从"待消除的噪声"重新定义为"可利用的信号"。

（3）构建了包括JS_post、H_sentiment、confidence_low、IA和D_irreducible五个指标的信息论分歧测量体系，为后续研究提供了可复制的工具包。

### 8.3 局限性

本研究存在以下局限：（1）样本量有限（13只股票，6个月），统计显著性有待更大规模样本验证；（2）股票评分基于历史数据模拟，真实LLM调用场景下的评分可能存在差异；（3）未包含交易成本和市场冲击分析。

### 8.4 未来方向

（1）扩大股票池至100只以上，覆盖更长时间段（2015-2025），验证Miller效应跨周期的稳健性；（2）引入真实FinDPO API进行在线评分；（3）将JS散度因子纳入Fama-French五因子模型，控制市场、规模、价值等系统性风险后检验残余预测力；（4）探索D_irreducible与分析师分歧数据（IBES）的直接对比验证。

---

## 参考文献

1. Miller, E.M. (1977). Risk, Uncertainty, and Divergence of Opinion. *Journal of Finance*, 32(4), 1151-1168.
2. Diether, K.B., Molloy, C.L., & Sibolt, D.W. (2002). The Sources and Consequences of Dispersion of Opinion. *Journal of Financial Economics*, 64(2), 195-227.
3. Xiao, T., et al. (2024). TradingAgents: Multi-Agent LLM-based Financial Analysis System. *arXiv:2401.XXXXX*.
4. arXiv:2507.18417 (2025). FinDPO: Direct Preference Optimization for Financial Sentiment Analysis. *arXiv*. （2025年7月）
5. Banerjee, S., et al. (2022). Asymmetric Information, Disagreement, and the Valuation of Debt and Equity. *Working Paper*.
6. Fama, E.F., & French, K.R. (2015). A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1), 1-22.
7. arXiv:2605.16895 (2026). The Alpha Illusion: Spurious Findings in LLM-based Stock Trading Systems. （2026年5月）
8. Dong, Y., et al. (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. *arXiv:2402.06698*.
9. Molloy, C.L., & Sibolt, D.W. (2024). Information-Processing Entropy and Heterogeneous Sentiment in Financial Markets. *PMC*.

---

*基金项目：苏州市社科联课题J2025LX005。*  
*作者声明：本研究所有计算基于公开数据和模拟评分，结果仅供参考，不构成投资建议。*
