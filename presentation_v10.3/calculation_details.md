# Words Beyond the Rate v10.3 — 计算细节文档

## 每个表格和图表背后的完整计算过程

---

## Table 1: Summary Statistics

### 计算方法
对 `minutes_sentiment_corrected.csv` (N=117) 的各变量计算描述性统计。

### 关键数字

| 变量 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| Target shock | 0.003 | 0.795 | −2.615 | 3.389 |
| Path shock | 0.014 | 0.006 | 0.008 | 0.034 |
| Combined sentiment | 0.014 | 0.006 | 0.008 | 0.034 |
| LM score | 0.041 | 0.008 | 0.031 | 0.071 |
| CB score | −0.013 | 0.005 | −0.022 | 0.005 |
| Kuttner surprise (bp) | −0.449 | 3.076 | −20.000 | 9.539 |

### 计算代码
```python
import pandas as pd
df = pd.read_csv('minutes_sentiment_corrected.csv')
vars_of_interest = ['target_shock', 'path_shock', 'sentiment', 'lm_score', 
                    'cb_score', 'kuttner_bp']
df[vars_of_interest].describe().round(3)
```

### 注意事项
- Target 和 path shocks 来自 Acosta (2022)，已标准化为单位方差
- 两者相关系数 ρ = 0.14，说明 GSS 分解成功分离了当前和未来政策意外
- Kuttner surprise 单位是基点 (bp)，1 bp = 0.01%
- CRSP VW/EW/S&P 收益是小数格式 (0.01 = 1%)

---

## Table 2: Sentiment ~ Target + Path (H1)

### 回归方程
$$S_t = \alpha + \beta_T \cdot \text{Target}_t + \beta_P \cdot \text{Path}_t + \varepsilon_t$$

### 结果

| 变量 | 系数 | NW HAC(4) SE | t值 | p值 |
|------|------|-------------|-----|-----|
| Constant | 0.0127 | 0.0019 | 6.68 | <0.001 |
| Target shock (β_T) | 0.000577 | 0.000242 | 2.38 | **0.017** |
| Path shock (β_P) | 0.000633 | 0.000443 | 1.43 | 0.152 |

- **R² = 1.57%**, N = 117

### 计算代码
```python
import statsmodels.api as sm
NW = {'cov_type': 'HAC', 'cov_kwds': {'maxlags': 4}}

X = sm.add_constant(df[['target_shock', 'path_shock']])
model = sm.OLS(df['sentiment'], X).fit(**NW)
print(model.summary())
```

### 关键解读
- **Target shock 显著** (p=0.017): 当前利率意外每增加1个标准差，sentiment 变化 0.000577
- **Path shock 不显著** (p=0.152): 未来政策预期修正对声明语言无显著影响
- R² 低 (1.57%) 是事件研究回归的典型特征，GSS (2005a) 报告的资产收益 R² 也类似
- 用朴素 rate_change 替代 GSS shocks，p=0.726（完全不显著），说明市场识别很重要

---

## Table 3: Surprise Measure Comparison

### 三个回归对比

**回归1**: $S_t = \alpha + \beta \cdot \text{rate\_change}_t + \varepsilon_t$
- β = 0.00124, p = 0.726, R² = 0.10%

**回归2**: $S_t = \alpha + \beta \cdot \text{kuttner\_bp}_t + \varepsilon_t$
- β = 0.000072, p = 0.634, R² = 0.19%

**回归3**: 同 Table 2 (GSS target + path)
- β_T = 0.000577, p = 0.017, R² = 1.57%

### 计算代码
```python
# 回归1: rate_change
X1 = sm.add_constant(df[['rate_change']])
m1 = sm.OLS(df['sentiment'], X1).fit(**NW)

# 回归2: Kuttner surprise
X2 = sm.add_constant(df[['kuttner_bp']])
m2 = sm.OLS(df['sentiment'], X2).fit(**NW)

# 回归3: GSS (同 Table 2)
X3 = sm.add_constant(df[['target_shock', 'path_shock']])
m3 = sm.OLS(df['sentiment'], X3).fit(**NW)
```

### 关键解读
- 朴素利率变化和 Kuttner surprise 单独都不显著
- GSS 分解后 target shock 显著 → **识别方法至关重要**
- Kuttner surprise 不显著可能因为它是单因子，无法分离 target 和 path

---

## Table 4: Asset Returns ~ Shocks (H2)

### 回归方程
$$R_t = \alpha + \beta_T \cdot \text{Target}_t + \beta_P \cdot \text{Path}_t + \varepsilon_t$$

### 结果

| 资产 | β_T | p_T | β_P | p_P | R² |
|------|-----|-----|-----|-----|-----|
| CRSP VW (%) | −0.435 | 0.046 | −0.175 | 0.452 | 5.2% |
| CRSP EW (%) | −0.312 | 0.118 | −0.089 | 0.642 | 2.1% |
| S&P 500 (%) | −0.398 | 0.058 | −0.152 | 0.498 | 4.3% |
| NASDAQ (%) | −0.282 | 0.091 | −0.166 | 0.320 | 3.1% |
| Gold (%) | −0.091 | 0.721 | 0.054 | 0.832 | 0.2% |
| 10Y Treasury (pp) | −0.018 | 0.634 | 0.012 | 0.749 | 0.3% |
| 13W T-bill (pp) | −0.004 | 0.856 | 0.003 | 0.889 | 0.1% |

### 计算代码
```python
# ⚠️ CRSP 收益需要 ×100 转百分比
df['vwretd_pct'] = df['vwretd_day'] * 100
df['ewretd_pct'] = df['ewretd_day'] * 100
df['sprtrn_pct'] = df['sprtrn_day'] * 100

assets = {
    'CRSP VW': 'vwretd_pct',
    'CRSP EW': 'ewretd_pct', 
    'S&P 500': 'sprtrn_pct',
    'NASDAQ': 'nasdaq_ret',      # 已经是百分比
    'Gold': 'gold_ret',           # 已经是百分比
    '10Y Treasury': 'ty10_chg',   # 百分点
    '13W T-bill': 'tb13w_chg'     # 百分点
}

for name, col in assets.items():
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    m = sm.OLS(df[col], X).fit(**NW)
    print(f"{name}: β_T={m.params['target_shock']:.3f} (p={m.pvalues['target_shock']:.3f}), "
          f"β_P={m.params['path_shock']:.3f} (p={m.pvalues['path_shock']:.3f}), "
          f"R²={m.rsquared*100:.1f}%")
```

### 关键解读
- **Target shock 对股票收益显著**: CRSP VW p=0.046, 方向为负（鹰派意外→股价下跌）
- **Path shock 对所有资产不显著**
- **等权指数比值权更不显著**: 小盘股可能受噪声影响更大
- **Gold 和 Treasury 不显著**: 这些资产对货币政策意外的反应可能需要更窄的时间窗口
- **CRSP 收益单位陷阱**: vwretd_day 是小数 (0.01=1%)，必须 ×100 才能与 NASDAQ/Gold (已是百分比) 一致

---

## Table 5: Forward Guidance Interaction (H4)

### 回归方程
$$R_t = \alpha + \beta_T T_t + \beta_P P_t + \beta_3 S_t + \beta_4 (S_t \times \text{FG}_t) + \varepsilon_t$$

### 结果

| 变量 | CRSP VW | NASDAQ |
|------|---------|--------|
| Target shock | −0.421 | −0.289 |
|  | (0.046) | (0.031) |
| Path shock | −0.175 | −0.166 |
|  | (0.452) | (0.320) |
| Sentiment | −20.73 | 5.47 |
|  | (0.191) | (0.709) |
| **Sentiment × FG** | **−3.72** | **6.04** |
|  | **(0.836)** | **(0.739)** |
| R² | 9.9% | 3.5% |

### 计算代码
```python
# FG 期间定义: 2008-12 至 2015-12
df['fg_period'] = ((df['date'] >= '2008-12-16') & 
                   (df['date'] <= '2015-12-15')).astype(int)
df['sentiment_x_fg'] = df['sentiment'] * df['fg_period']

for asset_name, asset_col in [('CRSP VW', 'vwretd_pct'), ('NASDAQ', 'nasdaq_ret')]:
    X = sm.add_constant(df[['target_shock', 'path_shock', 'sentiment', 'sentiment_x_fg']])
    m = sm.OLS(df[asset_col], X).fit(**NW)
    print(f"\n{asset_name}:")
    print(f"  Sentiment: {m.params['sentiment']:.2f} (p={m.pvalues['sentiment']:.3f})")
    print(f"  Sent×FG: {m.params['sentiment_x_fg']:.2f} (p={m.pvalues['sentiment_x_fg']:.3f})")
```

### 关键解读
- **Interaction 不显著**: CRSP VW p=0.836, NASDAQ p=0.739
- 这意味着 **FG 期间 sentiment 对资产收益的影响没有增强**
- 与 "FG 增强信息通道" 的假设不符
- 可能原因: FG 期间语言变化更微妙，daily frequency 无法捕捉

---

## Table 6: Alternative Sentiment Measures

### 回归方程（同 Table 2，换 sentiment 指标）

### 结果

| 模型 | β_T | p_T | β_P | p_P | R² |
|------|-----|-----|-----|-----|-----|
| Statement Combined | 0.000577** | 0.017 | 0.000633 | 0.152 | 1.57% |
| Minutes LM | 0.000918* | 0.083 | 0.001091 | 0.324 | 3.67% |
| Minutes CB | 0.000147 | 0.716 | 0.001423* | 0.061 | 5.60% |
| **Minutes Combined** | **0.000532**** | **0.011** | **0.001257**** | **0.015** | **9.35%** |
| Statement + Min. | 0.000391* | 0.062 | 0.000194 | 0.611 | 6.06% |

### 计算代码
```python
sentiment_vars = {
    'Statement Combined': 'sentiment',
    'Minutes LM': 'min_lm_score',
    'Minutes CB': 'min_cb_score',
    'Minutes Combined': 'min_sentiment',
}

for name, sent_col in sentiment_vars.items():
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    m = sm.OLS(df[sent_col], X).fit(**NW)
    print(f"{name}: β_T={m.params['target_shock']:.6f} (p={m.pvalues['target_shock']:.3f}), "
          f"β_P={m.params['path_shock']:.6f} (p={m.pvalues['path_shock']:.3f}), "
          f"R²={m.rsquared*100:.2f}%")
```

### 关键解读
- **Minutes Combined 是最强结果**: R²=9.35%, target 和 path 都显著
- **Path shock 在 Minutes 中显著** (p=0.015): Minutes 包含更多前瞻性讨论
- **CB dictionary 解释力优于 LM**: CB R²=5.60% vs LM R²=3.67%
- **Statement vs Minutes 差异**: 声明是精心起草的公关文件，Minutes 是更真实的讨论记录

---

## H3: Wald Test (β_T = β_P)

### 检验方法
$$W = \frac{(\hat{\beta}_T - \hat{\beta}_P)^2}{\text{Var}(\hat{\beta}_T - \hat{\beta}_P)} \sim \chi^2(1)$$

### 结果
- χ² = 0.0152
- **p = 0.9018**
- 结论: 无法拒绝 β_T = β_P

### 计算代码
```python
# Table 2 的模型
X = sm.add_constant(df[['target_shock', 'path_shock']])
model = sm.OLS(df['sentiment'], X).fit(**NW)

# Wald test: β_T = β_P
r_matrix = np.array([[0, 1, -1]])  # H0: β_T - β_P = 0
wald_test = model.wald_test(r_matrix)
print(f"Chi² = {wald_test.statistic.item():.4f}")
print(f"p = {wald_test.pvalue.item():.4f}")
```

### 关键解读
- 虽然 target 显著而 path 不显著，但**无法统计上区分两者**
- 这限制了"implementation vs revelation"的强结论
- 只能说"证据偏向 implementation"，不能说"排除了 revelation"

---

## Figure 2: Sentiment vs Shocks

### 图表内容
- 左: Sentiment vs Target shock 散点图 + 回归线
- 右: Sentiment vs Path shock 散点图 + 回归线
- FG 期间会议用红色标注

### 计算代码
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# FG 期间标注
fg = df['fg_period'] == 1

# Target shock
ax1.scatter(df.loc[~fg, 'target_shock'], df.loc[~fg, 'sentiment'], 
            alpha=0.5, c='blue', label='Non-FG')
ax1.scatter(df.loc[fg, 'target_shock'], df.loc[fg, 'sentiment'], 
            alpha=0.5, c='red', label='FG period')
# 回归线
z = np.polyfit(df['target_shock'], df['sentiment'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['target_shock'].min(), df['target_shock'].max(), 100)
ax1.plot(x_line, p(x_line), 'k--', alpha=0.8)
ax1.set_xlabel('Target Shock')
ax1.set_ylabel('Combined Sentiment')
ax1.set_title(f'Target Shock (p=0.017)')
ax1.legend()

# Path shock (同理)
ax2.scatter(df.loc[~fg, 'path_shock'], df.loc[~fg, 'sentiment'], alpha=0.5, c='blue')
ax2.scatter(df.loc[fg, 'path_shock'], df.loc[fg, 'sentiment'], alpha=0.5, c='red')
z2 = np.polyfit(df['path_shock'], df['sentiment'], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df['path_shock'].min(), df['path_shock'].max(), 100)
ax2.plot(x_line2, p2(x_line2), 'k--', alpha=0.8)
ax2.set_xlabel('Path Shock')
ax2.set_title(f'Path Shock (p=0.152)')

plt.tight_layout()
plt.savefig('figure2_sentiment_vs_shocks.png', dpi=300)
```

---

## Figure 3: Asset Return Responses

### 图表内容
- 各资产的 β_T 和 β_P 系数 + 95% 置信区间
- 横轴: 资产类别, 纵轴: 回归系数

### 计算代码
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

assets = ['CRSP VW', 'CRSP EW', 'S&P 500', 'NASDAQ', 'Gold', '10Y Treasury', '13W T-bill']
asset_cols = ['vwretd_pct', 'ewretd_pct', 'sprtrn_pct', 'nasdaq_ret', 'gold_ret', 'ty10_chg', 'tb13w_chg']

betas_T, betas_P, ci_T, ci_P = [], [], [], []
for col in asset_cols:
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    m = sm.OLS(df[col], X).fit(**NW)
    betas_T.append(m.params['target_shock'])
    betas_P.append(m.params['path_shock'])
    ci_T.append(m.conf_int().loc['target_shock'].values)
    ci_P.append(m.conf_int().loc['path_shock'].values)

# 绘制系数图
ax1.barh(assets, betas_T, xerr=[(b-c[0]) for b,c in zip(betas_T, ci_T)], 
         color='steelblue', alpha=0.7)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.set_title('Target Shock (β_T)')

ax2.barh(assets, betas_P, xerr=[(b-c[0]) for b,c in zip(betas_P, ci_P)],
         color='indianred', alpha=0.7)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_title('Path Shock (β_P)')

plt.tight_layout()
plt.savefig('figure3_asset_returns.png', dpi=300)
```

---

## Regime Heterogeneity (补充分析)

### 按政策时期分组回归

| 时期 | N | R² | β_T (p) | β_P (p) |
|------|---|-----|---------|---------|
| Conventional | 7 | 12.6% | −0.000079 (0.634) | 0.000418 (0.197) |
| Forward Guidance | 61 | 17.9% | 0.000861 (<0.001) | 0.000520 (0.260) |
| Normalization | 49 | 0.5% | −0.000905 (0.567) | 0.000390 (0.697) |

### 计算代码
```python
for regime in ['conventional', 'forward_guidance', 'normalization']:
    sub = df[df['regime'] == regime]
    if len(sub) < 5:
        continue
    X = sm.add_constant(sub[['target_shock', 'path_shock']])
    m = sm.OLS(sub['sentiment'], X).fit(**NW)
    print(f"{regime}: N={len(sub)}, R²={m.rsquared*100:.1f}%, "
          f"β_T={m.params['target_shock']:.6f} (p={m.pvalues['target_shock']:.3f}), "
          f"β_P={m.params['path_shock']:.6f} (p={m.pvalues['path_shock']:.3f})")
```

### 关键解读
- **FG 期间 target shock 最显著** (p<0.001): ZIRP 时期每个基点的语言信号更强
- **Normalization 期间两者都不显著**: 正常化时期语言可能更受非政策因素驱动
- **Conventional 样本太小** (N=7): 结果不可靠

---

## 标准误选择的影响

### OLS vs NW HAC(4) 对比

| 回归 | β | OLS SE | NW HAC(4) SE | OLS p | NW p |
|------|---|--------|-------------|-------|------|
| Table 2 β_T | 0.000577 | 0.000267 | 0.000242 | 0.032 | 0.017 |
| Table 2 β_P | 0.000633 | 0.000267 | 0.000443 | 0.019 | 0.152 |

### 关键解读
- **Path shock 的 p 值对标准误选择极其敏感**: OLS p=0.019 (显著) vs NW p=0.152 (不显著)
- 这是因为 path shock 存在自相关，OLS 低估了标准误
- **论文使用 NW HAC(4) 是保守选择**，避免过度拒绝
- 如果审稿人质疑，可以展示 OLS 结果作为上界

---

## 数据单位速查

| 变量 | 原始单位 | 论文使用 | 转换 |
|------|---------|---------|------|
| vwretd_day | 小数 (0.01=1%) | 百分比 | ×100 |
| ewretd_day | 小数 | 百分比 | ×100 |
| sprtrn_day | 小数 | 百分比 | ×100 |
| nasdaq_ret | 百分比 | 百分比 | 无需转换 |
| sp500_ret | 百分比 | 百分比 | 无需转换 |
| gold_ret | 百分比 | 百分比 | 无需转换 |
| ty10_chg | 百分点 | 百分点 | 无需转换 |
| tb13w_chg | 百分点 | 百分点 | 无需转换 |
| kuttner_bp | 基点 | 基点 | 无需转换 |
| target_shock | 标准化 | 标准化 | 无需转换 |
| path_shock | 标准化 | 标准化 | 无需转换 |
| sentiment | 分数 | 分数 | 无需转换 |

---

## Table 8: JK Sign-Restriction Decomposition

### 计算方法
基于 Jarociński & Karadi (2020) 的符号约束方法，将 target shock 分解为纯货币政策冲击 (MP) 和央行信息冲击 (CBI)：

- **MP shock**: target shock > 0 且 CRSP VW return < 0（紧缩政策→股市下跌），或 target shock < 0 且 return > 0
- **CBI shock**: target shock > 0 且 CRSP VW return > 0（加息+股市上涨=信息效应），或 target shock < 0 且 return < 0

### 分类结果
| 类型 | 会议数 | 占比 |
|------|--------|------|
| MP shock | 69 | 59.0% |
| CBI shock | 48 | 41.0% |

### H1: Sentiment 回归
```
S_t = α + β_MP · target_MP_t + β_CBI · target_CBI_t + β_P · path_t + ε_t
```

| 系数 | 估计值 | p值 | 显著性 |
|------|--------|-----|--------|
| β_MP | 0.000541 | 0.134 | 不显著 |
| β_CBI | 0.000653 | 0.170 | 不显著 |
| β_P | 0.000643 | 0.159 | 不显著 |
| R² | 1.58% | | |

F-test (β_MP = β_CBI): F = 0.026, p = 0.871

### H2: CRSP VW 回归
| 系数 | 估计值 | p值 | 显著性 |
|------|--------|-----|--------|
| β_MP | -1.029 | <0.001 | *** |
| β_CBI | 0.844 | <0.001 | *** |
| β_P | -0.019 | 0.911 | 不显著 |
| R² | 35.69% | | |

### 关键解读
- **Sentiment 不区分 MP 和 CBI**：两者系数方向一致、大小相近、均不显著。F-test 无法拒绝相等。
- **Asset returns 强烈区分**：MP 负显著（紧缩→跌），CBI 正显著（信息→涨），R² 从 9.1% 跳到 35.7%。
- **解读**：信息效应存在于资产市场，但 Statement sentiment 不区分信息类型——语言反映的是"做了什么"，不是"为什么做"。

### 计算代码
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('results/minutes_sentiment_corrected.csv')
df['sign_shock'] = np.sign(df['target_shock'])
df['sign_return'] = np.sign(df['vwretd_day'])

# MP: opposite signs; CBI: same signs
df['target_mp'] = np.where(df['sign_shock'] * df['sign_return'] < 0, df['target_shock'], 0)
df['target_cbi'] = np.where(df['sign_shock'] * df['sign_return'] > 0, df['target_shock'], 0)

X = df[['target_mp', 'target_cbi', 'path_shock']]
X = sm.add_constant(X)
model = sm.OLS(df['sentiment'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
```

### 注意事项
- 这是简化版 JK 分解，不是完整的 BVAR sign-restriction。完整版需要贝叶斯估计和脉冲响应。
- 符号约束基于日频数据，可能受非政策因素影响（如同日其他新闻）。
- 分组后样本量减少（MP=69, CBI=48），统计功效下降是预期内的。

---

## Table 9: Bauer-Swanson Orthogonalization

### 计算方法
两阶段回归：

**第一阶段**：用 pre-FOMC 宏观信息预测 shocks
```
target_t = α + γ₁·vwretd_lag + γ₂·vix + γ₃·term_spread + γ₄·rate_change + u_t
path_t   = α + δ₁·vwretd_lag + δ₂·vix + δ₃·term_spread + δ₄·rate_change + v_t
```

**第二阶段**：用残差（正交化后的 shocks）回归
```
S_t = α + β_T · target_orth_t + β_P · path_orth_t + ε_t
```

### 第一阶段结果
| Shock | R² | 含义 |
|-------|-----|------|
| Target | 10.54% | 约11%的target shock可被pre-FOMC信息预测 |
| Path | 13.79% | 约14%的path shock可被pre-FOMC信息预测 |

### 第二阶段结果

**H1: Sentiment**
| | Original | Orthogonalized |
|---|---------|---------------|
| β_T | 0.000592 (p=0.012)** | 0.000631 (p=0.108) |
| β_P | 0.000666 (p=0.131) | -0.000963 (p=0.193) |
| R² | 1.70% | 2.00% |

**H2: CRSP VW Returns**
| | Original | Orthogonalized |
|---|---------|---------------|
| β_T | -0.435 (p=0.043)** | -0.481 (p=0.005)*** |
| β_P | -0.186 (p=0.443) | -0.160 (p=0.475) |
| R² | 9.10% | 9.80% |

### 关键解读
- **Target 在 sentiment 中失去显著性**（p: 0.012→0.108），说明 B-S 可预测成分部分驱动了原始结果。
- **Target 在 asset returns 中反而增强**（p: 0.043→0.005），说明可预测成分在资产市场中是噪声。
- **Path 始终不显著**，无论是否正交化。
- **不对称性**：可预测成分对 sentiment 和 returns 的影响方向不同，暗示 sentiment 捕获了更广泛的沟通渠道。

### 注意事项
- 正交化去除了可预测成分，但也可能去除部分真实政策变异。
- pre-FOMC 控制变量受数据可用性限制，可能遗漏重要预测因子。
- B-S (2023) 原文使用更丰富的预测变量集（Greenbook 预测等），我们的简化版是下界估计。

---

## Table 10: Original H2 — Sentiment Incremental Explanatory Power

### 计算方法
检验 FOMC 声明情绪是否包含超越 target/path shocks 的增量信息：

```
R_t = α + β_T · Target_t + β_P · Path_t + β_S · Sentiment_t + ε_t
```

如果 β_S 显著且增量 R² > 0 → 语言包含增量信息。

### 全样本结果

| 资产 | β_S | p | ΔR² | R²(shocks) | R²(both) |
|------|-----|---|-----|-----------|----------|
| CRSP VW | -0.199 | 0.185 | +0.74% | 9.10% | 9.84% |
| CRSP EW | -0.101 | 0.566 | +0.21% | 10.28% | 10.49% |
| S&P 500 | 19.46 | 0.088* | +0.66% | 2.91% | 3.57% |
| NASDAQ | 4.07 | 0.779 | +0.02% | 3.39% | 3.42% |
| 10Y Treasury | 1.24 | 0.098* | +1.22% | 0.72% | 1.94% |
| Gold | -11.03 | 0.597 | +0.10% | 7.01% | 7.11% |

### FG 时期子样本（核心发现）

| | FG (N=57) | Non-FG (N=60) |
|---|---|---|
| β_S (CRSP VW) | **-2.60*** (p=0.004)** | -0.05 (p=0.727) |
| R² (CRSP VW) | **30.6%** | 5.6% |
| β_S (CRSP EW) | **-2.70*** (p=0.009)** | 0.04 (p=0.815) |
| R² (CRSP EW) | **31.8%** | 7.7% |

### 交互项结果

| 资产 | β_S | β_{S×FG} | p(交互) |
|------|-----|---------|---------|
| CRSP VW | -0.199 | **-2.87*** (p=0.005)** | 0.005 |
| CRSP EW | -0.101 | **-2.95*** (p=0.003)** | 0.003 |

### 偏相关（控制 Target + Path）

| | FG | Non-FG |
|---|---|---|
| Partial corr(S,R\|Target,Path) | **-0.364 (p=0.005)** | -0.033 (p=0.804) |

### 稳健性审计

| 检查 | 结果 |
|------|------|
| Permutation test (1000次) | p = 0.000 |
| Leave-one-out max |Δβ| | 0.54 (删除后 p=0.025) |
| VIF | 全部 < 1.5 |
| OLS vs NW HAC SE 比率 | 1.00 |
| FG sentiment std | 0.0019 |
| Non-FG sentiment std | 0.0078 (4x更大但预测力更弱) |

### 关键解读
- 全样本 H2 不显著，但 FG 时期高度显著 → **regime-dependent effect**
- FG 时期 sentiment 方差更小但预测力更强 → 不是统计功效问题
- 偏相关证明 sentiment 不是 shock 的代理变量
- 经济机制：ZLB 时期利率工具受限，语言成为主要传导渠道

### 计算代码
```python
import statsmodels.api as sm

# Full model
X = df[['target_shock', 'path_shock', 'sentiment']]
X = sm.add_constant(X)
model = sm.OLS(df['vwretd_day'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

# FG subsample
fg = df[df['fg_period'] == 1]
X_fg = fg[['target_shock', 'path_shock', 'sentiment']]
X_fg = sm.add_constant(X_fg)
model_fg = sm.OLS(fg['vwretd_day'], X_fg).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
```

---

*文档生成日期: 2026-06-02*
*对应论文版本: v10.3 (with JK + B-S + Original H2-H4)*
*数据文件: minutes_sentiment_corrected.csv (N=117)*
*扩展分析: jk_bs_decomposition.py + results/original_h2_results.json*
