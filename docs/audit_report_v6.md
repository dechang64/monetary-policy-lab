# 顶刊标准审核报告 — Beyond the Rate JMP v6

审核日期: 2026-05-28  
审核人: 曼卿  
审核标准: JPE/AER/QJE 顶刊审稿标准

---

## 🔴 CRITICAL — 必须修复 (5 项)

### 1. Table 9 所有子期间数据无法复现
| 期间 | 论文 R² | 实际 R² | 论文 N | 实际 N | 匹配 |
|------|---------|---------|--------|--------|------|
| Pre-ZLB (2006-2007) | 6.8% | 9.08% | 15 | 7 | ✗ |
| Financial Crisis (2008-2009) | 12.3% | 11.56% | 16 | 13 | ✗ |
| ZLB/FG (2010-2015) | 1.2% | 7.53% | 42 | 48 | ✗ |
| Normalization (2016-2019) | 3.8% | 13.34% | 32 | 30 | ✗ |
| COVID+ (2020-2022) | 8.5% | 3.56% | 12 | 19 | ✗ |

**问题**: N 不对、R² 不对、p 值不对。这些数字无法从当前数据管线复现。可能来自早期版本 (v5) 的不同数据集或不同情感变量。

**修复**: 用当前 v6.1 管线重新跑所有子期间回归，替换 Table 9。

### 2. Table 8 Post-2010 结果严重不符
- 实际: target p=0.369, path p=0.108
- 论文: target p=0.234, path p=0.058
- **path shock 在 Post-2010 子样本中不显著**，论文声称显著

**修复**: 用实际数据替换。

### 3. Target-Path 相关系数错误
- 论文声称 "essentially uncorrelated (r = -0.03)"
- 实际 (N=117): r = +0.14
- 完整 Acosta 样本 (N=220): r = +0.015
- GSS 分解只保证正交化，不保证子样本内零相关

**修复**: 改为 "weakly correlated (r = 0.14 in our sample, r = 0.02 in the full Acosta sample)"。

### 4. "Swanson (2005)" 幽灵引用
- 论文同时引用 "Gürkaynak, Sack, and Swanson (2005)" 和 "Swanson (2005)"
- Swanson 2005 年的唯一论文就是 GSS (2005a)
- 不存在独立的 "Swanson (2005)"

**修复**: 删除 "Swanson (2005)" 引用，或替换为正确的 Swanson 论文（如 Swanson 2021）。

### 5. Acosta 引用年份不一致
- 正文: "Acosta (2024)"
- 参考文献: "Acosta, M. (2022)"
- USMPD 数据来自: "Acosta et al. (2025)" WP 2025-30
- 实际: "The Perceived Causes" 是 2022 WP，2026 R&R at JPE

**修复**: 统一为 "Acosta (2022)" 并在脚注说明 "R&R at Journal of Political Economy"。USMPD 数据单独引用 "Acosta et al. (2025)"。

---

## 🟡 MAJOR — 应该修复 (7 项)

### 6. Sentiment-Rate 相关系数偏差 61%
- 论文: r = 0.18
- 实际: r = 0.29

### 7. GSS (2005) 参考文献列错论文
- 参考文献列: "The sensitivity of long-term interest rates to economic news" (GSS 2005b)
- 论文引用的是 target/path 分解，来自 GSS (2005a)
- 正确: "Do Actions Speak Louder Than Words?" IJCB 1(1), 55-93

### 8. Apel and Blix (2014) 参考文献列错论文
- 参考文献列: "How is inflation affected by globalisation?"
- 论文引用的是央行沟通文本分析
- 正确: Apel, M. and Blix Grimaldi, I. (2014). "How Much Information Do Monetary Policy Committees Disclose?"

### 9. Cieslak et al. (2019) 期刊名错误
- 参考文献写: "Journal of Financial Economics"
- 实际发表: "Journal of Finance" 74(5), 2201-2248

### 10. Karadi (2020) 引用不完整
- 正文写 "Karadi (2020)"
- 论文是 Jarociński and Karadi (2020)
- 应写 "Jarociński and Karadi (2020)"

### 11. Kuttner surprise 系数单位问题
- 论文: β = 0.000028, R² = 1.95%
- 实际 (bp): β = 0.023, R² = 1.49%
- 系数差 3 个数量级

### 12. "primary driver" 措辞过度
- R² = 4.06% 意味着 95.94% 的情感变异无法被 shock 解释
- "primary driver" 暗示 shock 是主要解释因素，但实际只解释 4%
- 应改为 "statistically significant predictor"

---

## 🔵 LOGICAL — 论证逻辑问题 (8 项)

### 13. 因果推断缺陷
- OLS 回归不能建立因果方向
- 虽然 HF shock 是 predetermined，但仍需明确说明这是 reduced-form 关联
- 建议: 用 "is associated with" 替代 "drives"

### 14. H3 缺少正式检验
- 论文比较 |t_path| = 2.012 vs |t_target| = 1.887
- 没有正式 Wald test: H0: β_target = β_path
- 如果不能拒绝 H0，H3 不成立

### 15. 多重检验问题
- 4 个假设 + 多个稳健性检验，未调整显著性水平
- Bonferroni 调整后 (4 tests)，阈值变为 0.0125
- path shock (p=0.047) 将不再显著

### 16. 遗漏变量
- H1 回归没有控制变量 (lagged sentiment, 经济状况, 通胀预期)
- 审稿人会要求加入控制变量

### 17. 样本选择偏差
- 排除 1995-2005 需要明确说明原因 (FOMC 声明文本可得性)
- 117 个观测值偏少

### 18. 测量误差
- 词典方法无法捕捉语境
- 没有验证构念效度 (与人工标注对比)

### 19. 外部效度
- 仅用美国数据，未讨论是否适用于其他央行

### 20. 低 R² 的解释不足
- 4.06% 的 R² 需要诚实讨论为什么这么低
- 可能原因: 声明模板化、遗漏变量、测量噪声

---

## 🟢 MINOR — 建议修复 (8 项)

### 21. "marginally significant" 措辞
- p = 0.062 应称 "significant at the 10% level"
- "marginally significant" 削弱论证

### 22. 词典数量不一致
- §3.2: "36 to 203 terms"
- Appendix A: "97 hawkish terms, 106 dovish terms, and 50 bigram phrases"
- 203 unigrams + 50 bigrams = 253 total，不是 203

### 23. JEL 分类不完整
- 缺少 D83 (信息与知识)、E43 (利率)、C80 (计算方法)

### 24. Table 5 CRSP VW path β 微小差异
- 实际: β_p = -0.186 (t = -0.849)
- 论文: β_p = -0.194 (t = -0.712)
- 可能是 Newey-West vs OLS SE 差异

### 25. Table 8 No COVID 微小差异
- 实际: target p=0.075, path p=0.042
- 论文: target p=0.065, path p=0.050

### 26. 缺少数据可得性声明
- 顶刊要求: "Data availability statement"
- 需说明哪些数据公开、哪些需要 WRDS 权限

### 27. 缺少利益冲突声明
- 顶刊要求: "Conflict of interest disclosure"

### 28. 缺少复制包说明
- 顶刊要求: 说明复制包的位置和内容

---

## 总结

| 严重程度 | 数量 | 说明 |
|----------|------|------|
| 🔴 CRITICAL | 5 | 数据无法复现、引用错误、事实错误 |
| 🟡 MAJOR | 7 | 数据偏差、引用错误、措辞过度 |
| 🔵 LOGICAL | 8 | 因果推断、假设检验、遗漏变量 |
| 🟢 MINOR | 8 | 措辞、一致性、格式 |
| **合计** | **28** | |

**最紧急**: Issue #1 (Table 9) 和 #2 (Table 8 Post-2010) — 这些是数据层面的硬伤，审稿人跑一遍代码就能发现。必须在提交前修复。
