# 顶刊审计报告：Beyond the Rate v9.2
**审计日期**: 2026-05-31  
**审计标准**: JFE/QJE 顶刊逐句审计  
**审计人**: 曼卿 🔍  
**数据源**: `minutes_sentiment_corrected.csv` (N=117, 2006-2022)

---

## 一、结构性问题（致命）

### 1.1 图表顺序混乱

| 位置 | 实际出现 | 应出现 | 问题 |
|------|---------|--------|------|
| Line 277 | **Figure 2** (Sentiment vs Shocks) | Figure 1 | Figure 2 排在 Figure 1 前面 |
| Line 311 | Figure 1 (Sentiment Timeline) | Figure 2 | 应为 Figure 2 |
| Line 399 | **Figure 5** (Sentiment by Decision) | Figure 4 | Figure 5 排在 Figure 4 前面 |
| Line 523 | Table 7 title | — | Table 7 标题孤悬在 Figure 4 前面 |
| Line 527 | Figure 4 (Dictionary Comparison) | Figure 5 | 应为 Figure 5 |

**修复方案**: 重新编号所有图表，确保 Figure 1→2→3→4→5→6→7 严格按出现顺序。Table 7 应紧跟其内容，不应与 Figure 4 交叉。

### 1.2 Table 7 内容缺失

Table 7 标题出现在 Line 523，但表格内容为空 `[]`，紧接着是 Figure 4。Table 7 的数据行完全缺失。

---

## 二、数字错误（致命）

### 2.1 摘要与正文 p 值错误（6处）

**声称**: rate change → sentiment, p = 0.726  
**实际**: p = 0.526  
**出现位置**: 摘要(Line 9)、Section 6.2、Section 7、Conclusion、Online Appendix C.2、C.5

| 位置 | 论文声称 | 实际数据 | 偏差 |
|------|---------|---------|------|
| Abstract | p = 0.726 | p = 0.526 | ❌ 差0.200 |
| Sec 6.2 | p = 0.726 | p = 0.526 | ❌ |
| Sec 7 | p = 0.726 | p = 0.526 | ❌ |
| Conclusion | p = 0.726 | p = 0.526 | ❌ |
| Appendix C.2 | p = 0.726 | p = 0.526 | ❌ |
| Appendix C.5 | p = 0.726 | p = 0.526 | ❌ |

### 2.2 Table A2 Lag 敏感性（3/4行错误）

| Lag | 论文 β_T t | 实际 β_T t | 论文 p | 实际 p | 状态 |
|-----|-----------|-----------|--------|--------|------|
| 1 | 2.78 | 2.17 | 0.006 | 0.032 | ❌ |
| 2 | 2.61 | 2.24 | 0.010 | 0.027 | ❌ |
| 4 | 2.43 | 2.43 | 0.017 | 0.017 | ✅ |
| 6 | 2.29 | 2.56 | 0.024 | 0.012 | ❌ (方向相反!) |

lag=6 的 t 统计量方向反了：论文说递减(2.29)，实际递增(2.56)。

### 2.3 Table 6 Regime 分析（2/3行错误）

| Regime | 论文 β_T p | 实际 β_T p | 论文 β_P p | 实际 β_P p | 状态 |
|--------|-----------|-----------|-----------|-----------|------|
| Rate hike | 0.013** | 0.026 | 0.298 | 0.315 | ❌ β_T 显著性虚高 |
| Rate cut | 0.089 | 0.128 | <0.001*** | 0.006*** | ❌ β_P p 值偏差 |
| Unchanged | 0.616 | 0.617 | 0.079* | 0.083 | ⚠ β_P 接近 |

Rate hike 的 β_T p=0.013 声称 5% 显著，实际 p=0.026 仅 10% 显著。

### 2.4 Table 8 Minutes CB（系数错误）

| 变量 | 论文 β_P | 实际 β_P | 偏差 |
|------|---------|---------|------|
| Minutes CB | 0.002876 | 0.001423 | ❌ 差2倍 |

### 2.5 Table 9 Subsample（R² 和 p 值错误）

| 指标 | 论文 | 实际 | 状态 |
|------|------|------|------|
| R² | 1.6% | 0.6% | ❌ 差2.7倍 |
| β_P p | 0.056* | 0.258 | ❌ 显著性虚高 |

论文声称 post-crisis 子样本 path shock 边缘显著(p=0.056)，实际完全不显著(p=0.258)。

---

## 三、数字正确但需注意

### 3.1 Table 2 (H1 主表) ✅
- β_T = 0.000577, p = 0.017 ✅
- β_P = 0.000633, p = 0.152 ✅
- R² = 1.57% ✅

### 3.2 Table 3 (Surprise Comparison) ✅
- Rate change: R²=0.40%, p=0.526 ✅
- Kuttner: R²=1.49%, p=0.010 ✅
- GSS: R²=1.57%, p=0.017 ✅

### 3.3 Table 4 (Asset Returns) ✅
- CRSP VW: β_T=-0.435, p=0.042 ✅
- CRSP EW: β_T=-0.449, p=0.013 ✅
- S&P 500: β_T=-0.391, p=0.075 ✅
- Gold: β_T=-0.404, p=0.015 ✅

### 3.4 Table 5 (FG Interaction) ✅
- Sent×FG p=0.836 (CRSP VW) ✅
- Sent×FG p=0.739 (NASDAQ) ✅

### 3.5 Table 7 (Dictionary Comparison) ✅
- Combined: R²=1.57%, p_T=0.017 ✅
- LM only: R²=0.33%, p_T=0.476 ✅
- CB only: R²=3.90%, p_T<0.001 ✅

---

## 四、逻辑推理问题

### 4.1 H3 检验逻辑矛盾

论文声称 H3: "path shock > target shock for sentiment"（β₂ > β₁），但数据明确显示：
- Target shock: p = 0.017 (显著)
- Path shock: p = 0.152 (不显著)
- Wald test: p = 0.90 (不能拒绝 β₁ = β₂)

**逻辑问题**: 论文说"cannot reject H3"（不能拒绝 path 更强），但实际数据是 target 更强。Wald test 只能说"不能拒绝相等"，不能推出"path 更强"。论文应明确说：证据不支持 H3，target shock 是显著预测因子，path 不是。

### 4.2 Rate cut regime 的过度解读

论文对 rate cut regime (N=11) 的 R²=43.1% 给予大量解读，但：
- N=11 的子样本回归极不稳定
- 自由度仅 8，任何极端值都会主导结果
- 论文应明确标注"exploratory, not confirmatory"

### 4.3 "Undetectable" 用词不当

摘要说 rate changes 使关系 "undetectable" (p=0.726)，但实际 p=0.526。p=0.5 仍然不显著，但 "undetectable" 暗示完全无信号，而 R²=0.40% 说明仍有微弱关系。建议改为 "statistically insignificant"。

---

## 五、引用文献问题

### 5.1 需核实的引用

| 引用 | 问题 | 建议 |
|------|------|------|
| Apel & Blix (2014) | 原文是通胀论文，非央行沟通论文 | 核实是否引用正确 |
| Swanson (2005) | 应为 Gürkaynak, Sack, and Swanson (2005a) | 正文已正确，检查一致性 |
| Chen, Granville & Matousek (2025) | 2025年论文，需确认已发表 | 检查是否为 working paper |

### 5.2 缺失引用

- Corredoia et al. (2020): 之前审计已确认为幻觉引用，当前版本已移除 ✅
- Bauer & Swanson (2023): 已正确引用 ✅

---

## 六、图表与文字一致性

### 6.1 Figure 2 在 Figure 1 前面

Figure 2 (scatter plot) 在 Line 277 出现，Figure 1 (timeline) 在 Line 311。读者先看到散点图再看到时间序列，逻辑倒置。

### 6.2 Table 7 标题与 Figure 4 交叉

Table 7 标题出现在 Line 523，紧接着是 Figure 4 (Line 527)。Table 7 的表格内容缺失，读者看到标题后直接跳到图，造成困惑。

### 6.3 Figure 5 在 Figure 4 前面

Figure 5 (Sentiment by Decision Type) 在 Line 399 出现，Figure 4 (Dictionary Comparison) 在 Line 527。编号不按出现顺序。

---

## 七、修复优先级

### P0 - 致命（必须修复才能投稿）
1. **图表顺序**: 重新编号，确保 Figure 1-7 按出现顺序排列
2. **p = 0.726 → 0.526**: 6处全部修正
3. **Table 7 内容缺失**: 补充表格数据行
4. **Table A2 lag 敏感性**: 3/4行数字错误，需用实际数据重跑
5. **Table 9 subsample**: R² 和 p 值错误

### P1 - 严重（影响论文可信度）
6. **Table 6 regime**: Rate hike β_T p 值虚高，Rate cut β_P p 值偏差
7. **Table 8 Minutes CB**: β_P 系数差2倍
8. **H3 逻辑**: Wald test 不能推出 "path 更强"，需修正论述
9. **Rate cut 过度解读**: N=11 子样本需加 caution

### P2 - 改进（提升论文质量）
10. **"undetectable" 用词**: 改为 "statistically insignificant"
11. **Apel & Blix 引用**: 核实是否引用正确论文
12. **Figure 编号与引用**: 确保正文引用与图表编号一致

---

## 八、总结

| 类别 | 问题数 | 致命 | 严重 | 改进 |
|------|--------|------|------|------|
| 结构/顺序 | 3 | 3 | 0 | 0 |
| 数字错误 | 5 | 3 | 2 | 0 |
| 逻辑推理 | 3 | 0 | 2 | 1 |
| 引用文献 | 2 | 0 | 1 | 1 |
| 图文一致 | 3 | 2 | 1 | 0 |
| **合计** | **16** | **8** | **6** | **2** |

**审计结论**: 论文存在8个致命问题，主要集中在：(1) 图表顺序严重混乱；(2) 多个关键 p 值与实际数据不符（p=0.726 应为 0.526，Table A2 3/4行错误，Table 9 R² 和 p 值错误）；(3) Table 7 内容缺失。在修复这些问题之前，**不建议投稿**。
