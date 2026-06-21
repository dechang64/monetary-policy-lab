#!/usr/bin/env python3
"""
=============================================================================
Words Beyond the Rate v10.3 — WRDS 完整复现脚本
=============================================================================

Eileen: 这个脚本从 WRDS 原始数据库开始，一步步复现论文中每个 Table 和 Figure。

前提条件:
  1. 有 WRDS 账号 (https://wrds.wharton.upenn.edu/)
  2. 安装 wrds 包: pip install wrds
  3. 首次运行会提示输入密码，之后自动保存在 ~/.wrds.cfg

运行方式:
  python3 reproduce_from_wrds.py

数据流程:
  WRDS 原始表 → SQL 查询 → 本地 CSV → 合并 → 回归 → 输出

使用的 WRDS 数据库:
  - crsp.dsi  : CRSP 日度市场指数 (vwretd, ewretd, sprtrn)
  - cme.ff    : CME 联邦基金期货 (Kuttner surprise)
  - cme.ef    : CME Eurodollar 期货 (Gürkaynak path factor)

外部数据 (非 WRDS):
  - GSS target & path shocks: 来自 Acosta (2022)，已提供 CSV
  - FOMC 声明 sentiment: LM + CB dictionary 计算，已提供 CSV
  - FRED DGS10/DGS3MO: 国债收益率 (可用 wrds-fred 或直接 API)

标准误: 全部使用 Newey-West HAC(4)，与论文一致
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 全局设置 ──
NW_KWARGS = {'cov_type': 'HAC', 'cov_kwds': {'maxlags': 4}}
OUTPUT_DIR = 'wrds_reproduction_output'

# ── WRDS 连接 ──
def connect_wrds():
    """连接 WRDS。首次需要输入密码，之后自动保存。"""
    try:
        import wrds
        print("正在连接 WRDS...")
        print("（首次运行会提示输入密码，输入后自动保存到 ~/.wrds.cfg）")
        db = wrds.Connection()
        print("✅ WRDS 连接成功！")
        return db
    except ImportError:
        print("❌ 请先安装 wrds 包: pip install wrds")
        sys.exit(1)
    except Exception as e:
        print(f"❌ WRDS 连接失败: {e}")
        print("提示: 如果是 Duo MFA 问题，请在手机上 Approve")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# 第1步: 从 WRDS 拉取原始数据
# ══════════════════════════════════════════════════════════════════════

def fetch_crsp_index(db, start='2006-01-01', end='2022-12-31'):
    """
    从 WRDS 拉取 CRSP 日度市场指数。
    
    表: crsp.dsi
    关键字段:
      - vwretd: Value-weighted return (含分红), 小数格式 (0.01 = 1%)
      - ewretd: Equal-weighted return (含分红), 小数格式
      - sprtrn: S&P 500 total return, 小数格式
    
    这是论文 Table 4 和 Table 5 中 CRSP VW/EW/S&P 收益的数据源。
    CRSP 的优势: 退市调整 (yfinance 不提供)
    """
    print("\n── 拉取 CRSP 日度市场指数 (crsp.dsi) ──")
    query = f"""
    SELECT date, vwretd, ewretd, sprtrn
    FROM crsp.dsi
    WHERE date >= '{start}'
      AND date <= '{end}'
    ORDER BY date
    """
    df = db.raw_sql(query)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  ✅ 拉取 {len(df)} 条记录")
    print(f"  日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"  vwretd 均值: {df['vwretd'].mean():.6f} (小数格式, ×100 = 百分比)")
    return df


def fetch_cme_fed_funds(db, start='2006-01-01', end='2022-12-31'):
    """
    从 WRDS 拉取 CME 联邦基金期货。
    
    表: cme.ff
    关键字段:
      - date: 交易日期
      - symbol: 合约代码 (FF1=当月, FF2=下月, ...)
      - settle: 结算价 (100 - implied rate)
    
    用途: 计算 Kuttner (2001) surprise
      Kuttner surprise = (FF1_settle_t - FF1_settle_{t-1}) × 100 (基点)
      如果 FOMC 在月末，用 FF2 (下月合约) 更准确
    """
    print("\n── 拉取 CME 联邦基金期货 (cme.ff) ──")
    query = f"""
    SELECT date, symbol, settle, volume, open_interest
    FROM cme.ff
    WHERE date >= '{start}'
      AND date <= '{end}'
      AND symbol IN ('FF1', 'FF2', 'FF3', 'FF4', 'FF5', 'FF6')
    ORDER BY date, symbol
    """
    df = db.raw_sql(query)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  ✅ 拉取 {len(df)} 条记录")
    print(f"  合约类型: {df['symbol'].unique()}")
    return df


def fetch_cme_eurodollar(db, start='2006-01-01', end='2022-12-31'):
    """
    从 WRDS 拉取 CME Eurodollar 期货。
    
    表: cme.ef
    关键字段:
      - date: 交易日期
      - symbol: 合约代码 (ED1-ED8)
      - settle: 结算价
    
    用途: Gürkaynak, Sack, Swanson (2005) path factor 分解
      Target factor = 第1主成分 (FF1 + ED1-4 的日变化)
      Path factor = 第2主成分
    """
    print("\n── 拉取 CME Eurodollar 期货 (cme.ef) ──")
    query = f"""
    SELECT date, symbol, settle, volume, open_interest
    FROM cme.ef
    WHERE date >= '{start}'
      AND date <= '{end}'
      AND symbol IN ('ED1', 'ED2', 'ED3', 'ED4', 'ED5', 'ED6', 'ED7', 'ED8')
    ORDER BY date, symbol
    """
    df = db.raw_sql(query)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  ✅ 拉取 {len(df)} 条记录")
    return df


# ══════════════════════════════════════════════════════════════════════
# 第2步: 计算 Kuttner Surprise
# ══════════════════════════════════════════════════════════════════════

def compute_kuttner_surprise(cme_ff_df, fomc_dates):
    """
    计算 Kuttner (2001) 货币政策意外。
    
    方法:
      1. 找到 FOMC 会议当天 FF1 (当月合约) 的结算价变化
      2. 如果 FOMC 在月末 (合约到期前7天内), 用 FF2 (下月合约)
      3. Surprise = -(settle_change) × 100 基点
         (结算价上升 = 隐含利率下降 = 鸽派意外)
    
    注意: 这是简化版。完整的 Kuttner 方法需要区分:
      - 宽窗口: 全天变化 (用于日频回归)
      - 窄窗口: FOMC 公告前后30分钟 (用于高频识别)
    """
    print("\n── 计算 Kuttner Surprise ──")
    
    # Pivot: 每日每个合约的结算价
    pivot = cme_ff_df.pivot_table(index='date', columns='symbol', values='settle')
    
    # 日度变化
    pivot_diff = pivot.diff()
    
    surprises = []
    for fomc_date in fomc_dates:
        fomc_date = pd.Timestamp(fomc_date)
        
        # 检查是否在月末 (当月合约到期前7天)
        days_to_month_end = (fomc_date + pd.offsets.MonthEnd(0) - fomc_date).days
        
        if days_to_month_end <= 7 and 'FF2' in pivot_diff.columns:
            # 用 FF2 (下月合约)
            contract = 'FF2'
        else:
            contract = 'FF1'
        
        if fomc_date in pivot_diff.index:
            settle_change = pivot_diff.loc[fomc_date, contract]
            if not pd.isna(settle_change):
                # Kuttner surprise: 结算价上升 = 利率预期下降 = 鸽派
                # 论文中 target shock 正值 = 鹰派意外
                # 所以 surprise = -settle_change × 100
                surprise_bp = -settle_change * 100
                surprises.append({
                    'date': fomc_date,
                    'kuttner_bp': surprise_bp,
                    'contract_used': contract
                })
    
    result = pd.DataFrame(surprises)
    print(f"  ✅ 计算了 {len(result)} 个 FOMC 日的 Kuttner surprise")
    print(f"  均值: {result['kuttner_bp'].mean():.2f} bp")
    print(f"  标准差: {result['kuttner_bp'].std():.2f} bp")
    return result


# ══════════════════════════════════════════════════════════════════════
# 第3步: 合并数据集
# ══════════════════════════════════════════════════════════════════════

def build_analysis_dataset(crsp_df, gss_shocks_file, sentiment_file, kuttner_df=None):
    """
    合并所有数据源，构建分析数据集。
    
    数据源:
      1. CRSP 市场指数 (来自 WRDS)
      2. GSS target & path shocks (来自 Acosta 2022 CSV)
      3. FOMC 声明 sentiment (LM + CB dictionary CSV)
      4. Kuttner surprise (从 CME 期货计算，或直接用已有数据)
      5. FRED 国债收益率 (DGS10, DGS3MO)
    """
    print("\n── 合并分析数据集 ──")
    
    # 加载 GSS shocks
    gss = pd.read_csv(gss_shocks_file)
    gss.columns = [c.lower().replace(' ', '_') for c in gss.columns]
    if 'date' not in gss.columns:
        gss = gss.rename(columns={gss.columns[0]: 'date'})
    gss['date'] = pd.to_datetime(gss['date'])
    print(f"  GSS shocks: {len(gss)} 条, {gss['date'].min()} 至 {gss['date'].max()}")
    
    # 加载 sentiment
    sent = pd.read_csv(sentiment_file)
    sent.columns = [c.lower().replace(' ', '_') for c in sent.columns]
    if 'date' not in sent.columns:
        sent = sent.rename(columns={sent.columns[0]: 'date'})
    sent['date'] = pd.to_datetime(sent['date'])
    print(f"  Sentiment: {len(sent)} 条")
    
    # 合并 shocks + sentiment
    df = pd.merge(gss, sent, on='date', how='inner')
    print(f"  Shocks × Sentiment: {len(df)} 条")
    
    # 合并 CRSP 收益
    crsp_df['date'] = pd.to_datetime(crsp_df['date'])
    df = pd.merge(df, crsp_df, on='date', how='left')
    print(f"  + CRSP: {df['vwretd'].notna().sum()} 条有 CRSP 数据")
    
    # 合并 Kuttner surprise (如果有)
    if kuttner_df is not None and len(kuttner_df) > 0:
        kuttner_df['date'] = pd.to_datetime(kuttner_df['date'])
        df = pd.merge(df, kuttner_df[['date', 'kuttner_bp']], on='date', how='left')
        print(f"  + Kuttner: {df['kuttner_bp'].notna().sum()} 条有 Kuttner 数据")
    
    # Forward Guidance 期间
    df['fg_period'] = ((df['date'] >= '2008-12-16') & 
                       (df['date'] <= '2015-12-15')).astype(int)
    df['sentiment_x_fg'] = df['sentiment'] * df['fg_period']
    
    # CRSP 收益转百分比
    for col in ['vwretd', 'ewretd', 'sprtrn']:
        if col in df.columns:
            df[f'{col}_pct'] = df[col] * 100
    
    # 筛选到论文样本
    df = df[(df['date'] >= '2006-01-01') & (df['date'] <= '2022-07-31')]
    df = df.dropna(subset=['target_shock', 'path_shock', 'sentiment'])
    
    print(f"\n  ✅ 最终数据集: N = {len(df)}")
    print(f"  日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"  FG 期间会议: {df['fg_period'].sum()}")
    
    return df


# ══════════════════════════════════════════════════════════════════════
# 第4步: 复现所有回归 (与 reproduce_all_tables_figures.py 相同)
# ══════════════════════════════════════════════════════════════════════

def run_all_regressions(df):
    """运行论文中所有回归，输出每个 Table 的结果。"""
    
    print("\n" + "=" * 70)
    print("回归分析")
    print("=" * 70)
    
    # ── Table 1: Summary Statistics ──
    print("\n── Table 1: Summary Statistics ──")
    vars_t1 = ['target_shock', 'path_shock', 'sentiment', 'lm_score', 'cb_score']
    if 'vwretd_pct' in df.columns:
        vars_t1 += ['vwretd_pct', 'nasdaq_ret', 'gold_ret']
    
    stats_df = df[vars_t1].describe().T
    stats_df.columns = ['N', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    print(stats_df[['Mean', 'Std', 'Min', 'Max']].to_string())
    
    # ── Table 2: Sentiment ~ Target + Path ──
    print("\n── Table 2: Sentiment ~ Target + Path (H1) ──")
    X = sm.add_constant(df[['target_shock', 'path_shock']])
    m = sm.OLS(df['sentiment'], X).fit(**NW_KWARGS)
    print(f"  β_Target = {m.params['target_shock']:.6f}  (t = {m.tvalues['target_shock']:.2f}, p = {m.pvalues['target_shock']:.3f})")
    print(f"  β_Path   = {m.params['path_shock']:.6f}  (t = {m.tvalues['path_shock']:.2f}, p = {m.pvalues['path_shock']:.3f})")
    print(f"  R² = {m.rsquared*100:.2f}%, N = {int(m.nobs)}")
    
    # ── Table 3: Surprise Measure Comparison ──
    print("\n── Table 3: Surprise Measure Comparison ──")
    
    # Rate change only
    if 'rate_change' in df.columns:
        X_rc = sm.add_constant(df[['rate_change']].dropna())
        y_rc = df.loc[X_rc.index, 'sentiment']
        m_rc = sm.OLS(y_rc, X_rc).fit(**NW_KWARGS)
        print(f"  Rate change: β = {m_rc.params['rate_change']:.6f}, p = {m_rc.pvalues['rate_change']:.3f}, R² = {m_rc.rsquared*100:.2f}%")
    
    # Kuttner only
    if 'kuttner_bp' in df.columns and df['kuttner_bp'].notna().sum() > 10:
        valid = df.dropna(subset=['kuttner_bp'])
        X_ku = sm.add_constant(valid[['kuttner_bp']])
        m_ku = sm.OLS(valid['sentiment'], X_ku).fit(**NW_KWARGS)
        print(f"  Kuttner:     β = {m_ku.params['kuttner_bp']:.6f}, p = {m_ku.pvalues['kuttner_bp']:.3f}, R² = {m_ku.rsquared*100:.2f}%")
    
    # GSS (same as Table 2)
    print(f"  GSS target:  β = {m.params['target_shock']:.6f}, p = {m.pvalues['target_shock']:.3f}, R² = {m.rsquared*100:.2f}%")
    
    # ── Table 4: Asset Returns ──
    print("\n── Table 4: Asset Returns and Monetary Policy Shocks (H2) ──")
    assets = {
        'CRSP VW': 'vwretd_pct',
        'CRSP EW': 'ewretd_pct',
        'S&P 500': 'sprtrn_pct',
        'NASDAQ': 'nasdaq_ret',
        'Gold': 'gold_ret',
    }
    
    print(f"  {'Asset':<12} {'β_T':>8} {'t_T':>7} {'p_T':>7} {'β_P':>8} {'p_P':>7} {'R²':>7}")
    print("  " + "-" * 65)
    
    for name, col in assets.items():
        if col not in df.columns:
            continue
        valid = df.dropna(subset=[col])
        if len(valid) < 20:
            continue
        X_a = sm.add_constant(valid[['target_shock', 'path_shock']])
        m_a = sm.OLS(valid[col], X_a).fit(**NW_KWARGS)
        sig_t = "***" if m_a.pvalues['target_shock'] < 0.01 else "**" if m_a.pvalues['target_shock'] < 0.05 else "*" if m_a.pvalues['target_shock'] < 0.1 else ""
        sig_p = "***" if m_a.pvalues['path_shock'] < 0.01 else "**" if m_a.pvalues['path_shock'] < 0.05 else "*" if m_a.pvalues['path_shock'] < 0.1 else ""
        print(f"  {name:<12} {m_a.params['target_shock']:>8.3f} {m_a.tvalues['target_shock']:>7.2f} {m_a.pvalues['target_shock']:>7.3f}{sig_t} "
              f"{m_a.params['path_shock']:>8.3f} {m_a.pvalues['path_shock']:>7.3f}{sig_p} {m_a.rsquared*100:>6.1f}%")
    
    # ── Table 5: Forward Guidance Interaction (H4) ──
    print("\n── Table 5: Forward Guidance Period Interaction (H4) ──")
    
    for name, col in [('CRSP VW', 'vwretd_pct'), ('NASDAQ', 'nasdaq_ret')]:
        if col not in df.columns:
            continue
        valid = df.dropna(subset=[col])
        X_h4 = sm.add_constant(valid[['target_shock', 'path_shock', 'sentiment', 'sentiment_x_fg']])
        m_h4 = sm.OLS(valid[col], X_h4).fit(**NW_KWARGS)
        
        print(f"\n  {name}:")
        for var in ['target_shock', 'path_shock', 'sentiment', 'sentiment_x_fg']:
            sig = "***" if m_h4.pvalues[var] < 0.01 else "**" if m_h4.pvalues[var] < 0.05 else "*" if m_h4.pvalues[var] < 0.1 else ""
            print(f"    {var:<18} β = {m_h4.params[var]:>8.3f}, t = {m_h4.tvalues[var]:>5.2f}, p = {m_h4.pvalues[var]:.3f}{sig}")
        print(f"    R² = {m_h4.rsquared*100:.1f}%, N = {int(m_h4.nobs)}")
    
    # ── Table 6: Alternative Sentiment Measures ──
    print("\n── Table 6: Alternative Sentiment Measures ──")
    
    sentiment_specs = {
        'Statement ~ Shocks': ('sentiment', ['target_shock', 'path_shock']),
    }
    
    # Add Minutes specs if available
    for min_col, label in [('min_lm_score', 'Minutes LM ~ Shocks'), 
                            ('min_cb_score', 'Minutes CB ~ Shocks')]:
        if min_col in df.columns:
            sentiment_specs[label] = (min_col, ['target_shock', 'path_shock'])
    
    # Minutes Combined
    if 'min_lm_score' in df.columns and 'min_cb_score' in df.columns:
        df['min_combined'] = 0.5 * df['min_lm_score'] + 0.5 * df['min_cb_score']
        sentiment_specs['Minutes Combined ~ Shocks'] = ('min_combined', ['target_shock', 'path_shock'])
    
    # Statement ~ Shocks + Minutes Combined
    if 'min_combined' in df.columns:
        sentiment_specs['Statement ~ Shocks + Min Comb.'] = ('sentiment', ['target_shock', 'path_shock', 'min_combined'])
    
    print(f"  {'Model':<35} {'β_T':>10} {'p_T':>7} {'β_P':>10} {'p_P':>7} {'R²':>7}")
    print("  " + "-" * 80)
    
    for label, (dep_var, indep_vars) in sentiment_specs.items():
        valid = df.dropna(subset=[dep_var] + indep_vars)
        X_s = sm.add_constant(valid[indep_vars])
        m_s = sm.OLS(valid[dep_var], X_s).fit(**NW_KWARGS)
        
        beta_t = m_s.params.get('target_shock', np.nan)
        p_t = m_s.pvalues.get('target_shock', np.nan)
        beta_p = m_s.params.get('path_shock', np.nan)
        p_p = m_s.pvalues.get('path_shock', np.nan)
        
        sig_t = "***" if p_t < 0.01 else "**" if p_t < 0.05 else "*" if p_t < 0.1 else ""
        sig_p = "***" if p_p < 0.01 else "**" if p_p < 0.05 else "*" if p_p < 0.1 else ""
        
        print(f"  {label:<35} {beta_t:>10.6f}{sig_t} {p_t:>7.3f} {beta_p:>10.6f}{sig_p} {p_p:>7.3f} {m_s.rsquared*100:>6.2f}%")
    
    # ── H3: Wald Test ──
    print("\n── H3: Wald Test β_Target = β_Path ──")
    X_w = sm.add_constant(df[['target_shock', 'path_shock']])
    m_w = sm.OLS(df['sentiment'], X_w).fit(**NW_KWARGS)
    r_matrix = np.array([[0, 1, -1]])  # H0: β_T = β_P
    wald_test = m_w.wald_test(r_matrix)
    print(f"  Chi² = {wald_test.statistic.item():.4f}")
    print(f"  p = {wald_test.pvalue.item():.4f}")
    print(f"  结论: {'无法拒绝' if wald_test.pvalue.item() > 0.05 else '拒绝'} β_T = β_P")


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Words Beyond the Rate v10.3 — WRDS 完整复现")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ── 1. 连接 WRDS ──
    db = connect_wrds()
    
    # ── 2. 拉取 CRSP 数据 ──
    crsp_df = fetch_crsp_index(db, start='2006-01-01', end='2022-12-31')
    crsp_df.to_csv(f'{OUTPUT_DIR}/crsp_dsi.csv', index=False)
    print(f"  💾 保存到 {OUTPUT_DIR}/crsp_dsi.csv")
    
    # ── 3. 拉取 CME 期货数据 ──
    cme_ff = fetch_cme_fed_funds(db, start='2006-01-01', end='2022-12-31')
    cme_ff.to_csv(f'{OUTPUT_DIR}/cme_ff.csv', index=False)
    print(f"  💾 保存到 {OUTPUT_DIR}/cme_ff.csv")
    
    cme_ef = fetch_cme_eurodollar(db, start='2006-01-01', end='2022-12-31')
    cme_ef.to_csv(f'{OUTPUT_DIR}/cme_ef.csv', index=False)
    print(f"  💾 保存到 {OUTPUT_DIR}/cme_ef.csv")
    
    # ── 4. 计算 Kuttner Surprise ──
    # FOMC 日期来自 GSS shocks 文件
    gss_file = 'gss_target_path_acosta_method.csv'
    if os.path.exists(gss_file):
        gss_dates = pd.read_csv(gss_file)
        fomc_dates = pd.to_datetime(gss_dates.iloc[:, 0]).tolist()
        kuttner_df = compute_kuttner_surprise(cme_ff, fomc_dates)
        kuttner_df.to_csv(f'{OUTPUT_DIR}/kuttner_surprise.csv', index=False)
        print(f"  💾 保存到 {OUTPUT_DIR}/kuttner_surprise.csv")
    else:
        print("  ⚠️ 未找到 GSS shocks 文件，跳过 Kuttner 计算")
        kuttner_df = None
    
    # ── 5. 关闭 WRDS 连接 ──
    db.close()
    print("\n✅ WRDS 数据拉取完成，连接已关闭")
    
    # ── 6. 合并数据集 ──
    sentiment_file = 'minutes_sentiment_corrected.csv'
    if not os.path.exists(sentiment_file):
        # 尝试其他路径
        for path in ['results/minutes_sentiment_corrected.csv', 
                     'delivery_v10.3/minutes_sentiment_corrected.csv']:
            if os.path.exists(path):
                sentiment_file = path
                break
    
    if os.path.exists(gss_file) and os.path.exists(sentiment_file):
        df = build_analysis_dataset(crsp_df, gss_file, sentiment_file, kuttner_df)
        df.to_csv(f'{OUTPUT_DIR}/analysis_dataset_from_wrds.csv', index=False)
        print(f"  💾 保存到 {OUTPUT_DIR}/analysis_dataset_from_wrds.csv")
        
        # ── 7. 运行所有回归 ──
        run_all_regressions(df)
    else:
        print("\n⚠️ 缺少必要文件，无法合并数据集")
        print(f"  需要: {gss_file} 和 {sentiment_file}")
        print("  但 CRSP 和 CME 数据已保存，可以手动合并")
    
    print("\n" + "=" * 70)
    print("WRDS 复现完成!")
    print(f"所有输出保存在: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
