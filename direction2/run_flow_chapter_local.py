# -*- coding: utf-8 -*-
"""
run_flow_chapter_local.py
Direction 2 flow-chapter re-run with all Oct-2026 audit fixes (Eileen Zhang).

Run on YOUR LOCAL machine (Windows PowerShell is fine):

    pip install wrds pandas numpy statsmodels scipy
    python run_flow_chapter_local.py

What it does
------------
1. Logs in to WRDS interactively (username, password, then APPROVE the Duo
   push on your phone). After one approval you get 30 days of MFA trust.
2. Downloads CRSP Survivor-Bias-Free Mutual Fund data, 2006-01 .. 2026-06:
   full style history, fund header, monthly TNA/returns.
3. Builds monthly net flows for 7 asset classes with the audit fixes:
   FIX-A point-in-time classification (no look-ahead "latest record")
   FIX-B outlier handling (winsorize 1%/99% per class; |flow| > TNA dropped)
   FIX-C pseudo Bauer-Swanson step REMOVED - shocks used in native form
4. Runs the flow regressions, writes everything to flow_rerun_output/:
   H4 (MP vs CBI, Same/Post/Diff windows, Wald beta1=beta2),
   H5 (risk-ladder panel), zero-rate check (MP x ZLB, CBI x ZLB),
   plus the flows panel and a run log.
5. Send the whole flow_rerun_output/ folder back for the website update.

All numbers freshly computed - nothing copied from old files.
"""
import io, json, os, sys
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import wrds
except ImportError:
    sys.exit("wrds not installed. Run:  pip install wrds pandas numpy statsmodels scipy")
try:
    import statsmodels.api as sm
except ImportError:
    sys.exit("statsmodels not installed. Run:  pip install statsmodels scipy")

FOMC_DATA = r"""date,target_shock,path_shock,sentiment,crsp_vw
2006-01-31,0.398749,0.704626,0.031401,-0.1529
2006-03-28,0.260054,1.24088,0.0264151,-0.5286
2006-05-10,0.228922,1.35152,0.0320285,-0.1461
2006-06-29,-0.365342,-0.51095,0.0328467,2.362
2006-08-08,-0.784777,-0.0151809,0.0483092,-0.3901
2006-09-20,-0.249318,0.800554,0.0461538,0.5183
2006-10-25,0.0973805,-0.149183,0.0479798,0.4353
2006-12-12,0.24998,-0.915391,0.0480769,-0.2021
2007-01-31,-0.0599761,-0.0556278,0.0493827,0.6358
2007-03-21,0.502351,-2.41994,0.0460526,1.6761
2007-05-09,-0.0306561,1.13973,0.0430464,0.3691
2007-08-07,0.385036,-0.774853,0.0518135,0.6577
2007-09-18,-4.95514,-0.623547,0.022541,2.9282
2007-10-31,-0.0882206,2.06715,0.0259516,1.3188
2007-12-11,1.94292,-0.285878,0.0149813,-2.5421
2008-01-30,-2.88779,-0.245969,0.00847458,-0.5359
2008-03-18,2.69066,-1.15248,0.0218182,3.9596
2008-04-30,-1.63256,-0.544293,0.0171821,-0.1543
2008-06-25,-0.395276,-0.610601,0.0357143,0.6273
2008-08-05,-0.0930438,-0.542722,0.0344828,2.4519
2008-09-16,3.1478,-0.224211,0.0191388,1.5211
2008-10-29,-1.03368,-1.10966,0.00919118,0.1444
2008-12-16,-4.21078,-2.49145,0.0207792,5.2898
2009-01-28,0.285564,-0.233993,0.0195228,3.3821
2009-03-18,0.0715357,-1.13252,0.0173267,2.2385
2009-04-29,-0.0178909,0.543586,0.0183824,2.454
2009-06-24,-0.0136991,1.23921,0.0179063,0.8957
2009-08-12,0.0900052,-0.345876,0.0222772,1.2283
2009-09-23,0.0862709,-2.31583,0.0307882,-1.0386
2009-11-04,0.0325881,-0.367814,0.0326577,0.0887
2009-12-16,-0.296588,0.477109,0.0255941,0.3319
2010-01-27,-0.110548,0.926007,0.0208333,0.4186
2010-03-16,-0.0505709,-0.802096,0.0280374,0.8745
2010-04-28,-0.00747579,-0.0683958,0.0280612,0.524
2010-06-23,0.0840857,0.211936,0.0248538,-0.2917
2010-08-10,0.163317,-0.641679,0.0204545,-0.7992
2010-09-21,0.207575,-0.246455,0.0231144,-0.3097
2010-11-03,0.387102,-0.209677,0.0273109,0.3439
2010-12-14,0.187709,-0.0571919,0.0306554,0.0271
2011-01-26,0.129425,-0.129575,0.0298329,0.7332
2011-03-15,0.0588929,0.1273,0.0280269,-1.1151
2011-04-27,-0.0959304,-0.178533,0.0290828,0.615
2011-06-22,0.0460773,0.235676,0.0214932,-0.5482
2011-08-09,0.221126,-0.738116,0.0167653,5.1244
2011-09-21,0.33101,1.19779,0.0224913,-2.9444
2011-11-02,-0.00157027,0.417355,0.0266525,1.7082
2011-12-13,0.138412,0.241758,0.0277108,-1.1256
2012-01-25,0.153878,-0.449881,0.0136139,0.9947
2012-03-13,0.106593,0.362004,0.0156627,1.7621
2012-04-25,-0.0981111,0.170754,0.0176887,1.3825
2012-06-20,0.231352,0.399785,0.0217391,-0.1471
2012-08-01,0.500636,0.126953,0.0193182,-0.4475
2012-09-13,0.385744,0.204573,0.0313653,1.528
2012-10-24,-0.00882954,0.0726861,0.0275142,-0.2612
2012-12-12,0.172982,-0.0807666,0.0287009,0.0255
2013-01-30,0.0770675,-0.0215364,0.0241935,-0.3846
2013-03-20,0.157227,0.118745,0.0226904,0.7248
2013-05-01,0.0903841,0.0744718,0.0302795,-1.0607
2013-06-19,0.142901,0.0930314,0.0274889,-1.3406
2013-07-31,0.122532,-0.113192,0.0266272,0.0199
2013-09-18,0.0819552,-0.271844,0.0278137,1.2556
2013-10-30,0.0713799,0.0863418,0.0265957,-0.5665
2013-12-18,0.109388,0.0626017,0.0228873,1.4462
2014-01-29,0.0903841,0.0744718,0.0213936,-0.993
2014-03-19,-0.00485654,0.601762,0.0226481,-0.6779
2014-04-30,0.0775686,0.182848,0.0244975,0.3405
2014-06-18,0.038903,0.341424,0.0261814,0.6963
2014-07-30,0.121998,-0.0850383,0.0257794,0.054
2014-09-17,0.0536304,0.550276,0.0262258,0.0797
2014-10-29,0.0555615,0.772812,0.0249288,-0.2246
2014-12-17,0.237263,-1.60381,0.0169014,2.1789
2015-01-28,0.122532,-0.113192,0.0264117,-1.4065
2015-03-18,0.0557724,-1.62689,0.0195035,1.208
2015-04-29,0.0898496,0.102626,0.0166359,-0.3923
2015-06-17,-0.0428053,-0.896924,0.019962,0.1723
2015-07-29,-0.0155245,-0.0743079,0.0192308,0.7577
2015-09-17,-1.4844,-0.378707,0.0158172,-0.0704
2015-10-28,0.0616923,1.08995,0.0142857,1.3657
2015-12-16,0.490547,0.413408,0.0243056,1.4887
2016-01-27,-0.0798186,-0.639944,0.0239852,-1.0546
2016-03-16,-0.59442,-1.49406,0.0244123,0.7493
2016-04-27,-0.105124,0.450354,0.0232558,0.2651
2016-06-15,-0.177783,-0.737435,0.0219048,-0.0492
2016-07-27,-0.00683916,0.267872,0.0236794,-0.1242
2016-09-21,-0.905742,0.0217036,0.0212585,1.1864
2016-11-02,0.168658,0.145458,0.0231959,-0.768
2016-12-14,-0.160183,1.25784,0.022158,-0.9619
2017-02-01,-0.116421,-0.313731,0.0269151,0.0282
2017-03-15,0.210101,-0.991012,0.0267857,1.0125
2017-05-03,0.29917,0.257268,0.0219561,-0.2573
2017-06-14,0.541875,0.16719,0.0225141,-0.2289
2017-07-26,-0.0596643,-0.154374,0.0224949,-0.0085
2017-09-20,-0.0826259,1.2203,0.0184109,0.0919
2017-11-01,0.0576069,0.204193,0.0142566,0.0748
2017-12-13,0.196192,-0.162658,0.0193133,0.0541
2018-01-31,0.128209,0.260674,0.021197,0.0149
2018-03-21,0.00222126,0.0428667,0.0245327,0.0421
2018-05-02,0.0968643,-0.120732,0.0273632,-0.5574
2018-06-13,-0.159283,0.751594,0.0345395,-0.3612
2018-08-01,0.0714911,-0.0451827,0.0376712,-0.1296
2018-09-26,0.416775,-0.234547,0.0272727,-0.4024
2018-11-08,0.041377,0.193833,0.0381944,-0.2936
2018-12-19,0.825103,0.0873195,0.0333333,-1.4833
2019-01-30,0.015904,-0.632908,0.0335463,1.4357
2019-03-20,0.281741,-0.948392,0.0258359,-0.3146
2019-05-01,-0.333764,-0.612488,0.0245902,-0.7552
2019-06-19,1.03288,-2.61472,0.025,0.3742
2019-07-31,1.22372,0.331881,0.0246114,-0.9979
2019-09-18,0.758104,0.367666,0.0311653,-0.0724
2019-10-30,0.0341945,0.407649,0.0286145,0.278
2019-12-11,0.168204,-0.0563025,0.0320946,0.274
2020-01-29,0.46963,-0.0135053,0.0286195,-0.116
2020-04-29,-0.748082,0.446546,0.0291971,2.8865
2020-06-10,0.1032,-0.0339042,0.031401,-0.6659
2020-07-29,0.0824107,-0.112056,0.0309633,1.3428
2020-09-16,0.0251507,0.0902615,0.0345395,-0.3147
2020-11-05,-0.0282665,0.0305842,0.0333333,2.1074
2020-12-16,-0.00485283,0.172161,0.03125,0.1387
2021-01-27,0.240292,-0.0451702,0.0309917,-2.4763
2021-03-17,-0.196886,0.0588019,0.0309917,0.357
2021-04-28,-0.0520767,0.106872,0.0309917,0.0142
2021-06-16,0.426747,0.333174,0.0309917,-0.4693
2021-07-28,-0.02217,0.157911,0.0340909,0.2969
2021-09-22,0.143278,0.128873,0.0310219,0.9965
2021-11-03,-0.0480739,0.215239,0.0317919,0.7317
2021-12-15,-0.0445076,0.962592,0.0424901,1.5057
2022-01-26,-0.0473444,0.0459382,0.0348259,-0.3579
2022-03-16,-0.671592,3.38944,0.0281065,2.5417
2022-05-04,-0.541011,-0.272046,0.028169,2.7983
2022-06-15,1.03081,1.29595,0.0210084,1.4988
2022-07-27,-0.393068,0.434245,0.0288889,2.5315
2022-09-21,-1.04266,1.88951,0.0290179,-1.5952
2022-11-02,-0.223428,-1.30264,0.040636,-2.5073
2022-12-14,-0.0950665,1.65101,0.040493,-0.5473
2023-02-01,-0.324385,0.284838,0.0390335,1.1137
2023-03-22,1.07742,-1.27243,0.0431655,-1.642
2023-05-03,0.577189,0.10365,0.0371094,-0.5189
2023-06-14,-0.781999,1.71634,0.0418251,0.0057
2023-07-26,0,0.145551,0.0389105,0.0799
2023-09-20,0,1.33185,0.042471,-0.8244
2023-11-01,-0.107862,-0.242073,0.042471,0.9644
2023-12-13,-0.17957,-1.88807,0.0423077,1.62
2024-01-31,0.0521332,0.236162,0.0288809,-1.5799
2024-03-20,0,-0.56619,0.0288809,1.0377
2024-05-01,0,-0.0910722,0.0267857,-0.2178
2024-06-12,0,0.358763,0.0291971,0.9215
2024-07-31,0,0.284838,0.0296167,1.4456
2024-09-18,-4.95266,-1.86525,0.027027,-0.2449
2024-11-07,0,0.205958,0.0275591,0.7115
2024-12-18,-0.248635,1.71142,0.0284698,-3.1161
2025-01-29,0.208533,0.363718,0.0313725,
2025-03-19,0,-0.799155,0.0334395,
2025-05-07,0,-0.497186,0.0323194,
2025-06-18,0.260666,-0.589618,0.0309917,
2025-07-30,0.104266,-0.205094,0.027881,
2025-09-17,0.48123,-0.955635,0.024911,
2025-10-29,1.04266,-0.000478327,0.0274725,
2025-12-10,-0.153917,-0.243894,0.0248447,
2026-01-28,0,0.0432433,0.0283401,
2026-03-18,0.248635,-0.237118,0.0230769,
2026-04-29,0.104266,0.429047,0.0273038,"""

WRDS_START, WRDS_END, OUT_DIR = "2006-01-01", "2026-06-30", "flow_rerun_output"
RISK_RANKING = {
    "government_bonds": 1, "corporate_bonds": 2, "real_assets": 3,
    "large_cap_equity": 4, "developed_market_equity": 5,
    "emerging_market_equity": 6, "small_cap_equity": 7,
}
OBJ_CLASS = {
    "GBDI": "government_bonds", "GDBD": "government_bonds", "GMBD": "government_bonds",
    "CBDI": "corporate_bonds", "CNVD": "corporate_bonds", "CHYD": "corporate_bonds",
    "REIT": "real_assets", "CMBD": "real_assets",
    "EDYG": "large_cap_equity", "EDCI": "large_cap_equity",
    "EDYD": "developed_market_equity", "EDYF": "developed_market_equity",
    "EDYE": "emerging_market_equity",
    "EDYS": "small_cap_equity", "EDCM": "small_cap_equity", "EDYB": "small_cap_equity",
}
LIPPER_KEYWORDS = [
    ("emerging", "emerging_market_equity"),
    ("small-cap", "small_cap_equity"), ("small cap", "small_cap_equity"),
    ("real estate", "real_assets"), ("commodit", "real_assets"),
    ("precious metal", "real_assets"), ("gold", "real_assets"),
    ("natural resource", "real_assets"),
    ("government", "government_bonds"), ("treasury", "government_bonds"),
    ("corporate", "corporate_bonds"), ("high yield", "corporate_bonds"),
    ("international", "developed_market_equity"), ("foreign", "developed_market_equity"),
    ("global", "developed_market_equity"),
    ("large-cap", "large_cap_equity"), ("large cap", "large_cap_equity"),
]


def load_fomc():
    df = pd.read_csv(io.StringIO(FOMC_DATA), parse_dates=["date"]).set_index("date")
    df = df[df.index >= WRDS_START]
    # FIX-C note: JK split uses only the SAME-DAY CRSP return; flows never
    # explain that return, so the day-0 circularity from the audit does not
    # apply to this chapter.
    mp = np.where(np.sign(df["target_shock"]) != np.sign(df["crsp_vw"]), df["target_shock"], 0.0)
    cbi = np.where(np.sign(df["target_shock"]) == np.sign(df["crsp_vw"]), df["target_shock"], 0.0)
    df["mp_shock"], df["cbi_shock"] = mp, cbi
    df["zlb_2020_2022"] = ((df.index >= "2020-03-15") & (df.index <= "2022-01-26")).astype(int)
    return df


def connect_wrds():
    print("=" * 70)
    print("WRDS login - username, password, then APPROVE the Duo push on your phone")
    print("(one approval gives 30 days of MFA trust)")
    print("=" * 70)
    conn = wrds.Connection()
    print("Connected.")
    return conn


def fetch_data(conn):
    print("Downloading CRSP mutual fund data (a few minutes)...")
    log = {}
    style = conn.raw_sql(f"""
        SELECT crsp_fundno, begdt, enddt, crsp_obj_cd, lipper_class_name, lipper_obj_name
        FROM crsp_q_mutualfunds.fund_style WHERE crsp_fundno IS NOT NULL""")
    log["style_rows"] = len(style)
    print(f"  style history: {len(style):,} rows")
    hdr = conn.raw_sql(f"""
        SELECT crsp_fundno, fund_name, first_offer_dt, end_of_period
        FROM crsp_q_mutualfunds.fund_hdr WHERE crsp_fundno IS NOT NULL""")
    log["hdr_rows"] = len(hdr)
    print(f"  fund header: {len(hdr):,} rows")
    monthly = conn.raw_sql(f"""
        SELECT crsp_fundno, caldt, mret, mtna
        FROM crsp_q_mutualfunds.monthly_tna_ret_nav
        WHERE caldt BETWEEN '{WRDS_START}' AND '{WRDS_END}' AND crsp_fundno IS NOT NULL""")
    log["monthly_rows"] = len(monthly)
    print(f"  monthly TNA/ret: {len(monthly):,} rows")
    return style, hdr, monthly, log


def classify_point_in_time(style):
    """FIX-A: month t uses the style record current at t."""
    s = style.copy()
    s["begdt"] = pd.to_datetime(s["begdt"])
    s["enddt"] = pd.to_datetime(s["enddt"], errors="coerce").fillna(pd.Timestamp("2030-01-01"))
    s = s.dropna(subset=["begdt"])

    def classify(o, lc, lo):
        o = str(o or "").strip().upper()
        if o in OBJ_CLASS:
            return OBJ_CLASS[o]
        text = f"{lc or ''} {lo or ''}".lower()
        for kw, cls in LIPPER_KEYWORDS:
            if kw in text:
                return cls
        return None

    s["asset_class"] = [classify(o, lc, lo) for o, lc, lo in
                        zip(s["crsp_obj_cd"], s["lipper_class_name"], s["lipper_obj_name"])]
    return s[s["asset_class"].notna()][["crsp_fundno", "begdt", "enddt", "asset_class"]]


def attach_class_at(monthly, style_pit):
    """Vectorized per-fund interval join for point-in-time class."""
    m = monthly.copy()
    m["caldt"] = pd.to_datetime(m["caldt"])
    st = style_pit.sort_values(["crsp_fundno", "begdt"])
    pieces = []
    for fundno, g in m.sort_values("caldt").groupby("crsp_fundno", sort=False):
        w = st[st["crsp_fundno"] == fundno]
        if w.empty:
            continue
        idx = pd.IntervalIndex.from_arrays(w["begdt"].values, w["enddt"].values, closed="both")
        pos = idx.get_indexer(g["caldt"].values)
        g = g.copy()
        g["asset_class"] = [w["asset_class"].values[p] if p >= 0 else None for p in pos]
        pieces.append(g)
    res = pd.concat(pieces, ignore_index=True)
    return res[res["asset_class"].notna()]


def build_flows(m):
    """FIX-B: flow construction + outlier handling."""
    m = m.sort_values(["crsp_fundno", "caldt"]).copy()
    m["mtna_prev"] = m.groupby("crsp_fundno")["mtna"].shift(1)
    for c in ("mret", "mtna", "mtna_prev"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["flow"] = m["mtna"] - m["mtna_prev"] * (1 + m["mret"].fillna(0))
    m["flow_pct"] = m["flow"] / m["mtna_prev"]
    m = m[np.isfinite(m["flow_pct"]) & (m["mtna_prev"] > 0)]
    m = m[m["flow_pct"].abs() <= 1.0]  # impossible obs -> data error

    def wins(g):
        if len(g) > 10:
            lo, hi = g["flow_pct"].quantile([0.01, 0.99])
            g["flow_pct"] = g["flow_pct"].clip(lo, hi)
        return g

    m = m.groupby("asset_class", group_keys=False).apply(wins)
    return m


def aggregate(m):
    agg = m.groupby(["asset_class", "caldt"]).agg(
        dollar_flow=("flow", "sum"), tna_lag=("mtna_prev", "sum"),
        tna=("mtna", "sum"), n_funds=("crsp_fundno", "nunique")).reset_index()
    agg["flow_pct"] = agg["dollar_flow"] / agg["tna_lag"]
    agg["log_tna"] = np.log(agg["tna_lag"].replace(0, np.nan))
    agg = agg.sort_values(["asset_class", "caldt"])
    agg["flow_vol_12m"] = agg.groupby("asset_class")["flow_pct"].transform(
        lambda s: s.rolling(12, min_periods=6).std())
    return agg


def ols_row(y, X):
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, Xc, missing="drop").fit()
    out = {"n": int(res.nobs), "r2": float(res.rsquared)}
    for k in res.params.index:
        out[k] = {"beta": float(res.params[k]), "se": float(res.bse[k]), "p": float(res.pvalues[k])}
    return out, res


def wald_equal(res, a, b):
    try:
        return float(res.t_test(f"{a} = {b}").pvalue)
    except Exception:
        return None


def run_h4(agg, fomc):
    """H4: Flow(c,w) = a + b1*MP + b2*CBI + controls; Wald b1=b2; w = same/post/diff."""
    controls = ["log_tna", "flow_vol_12m"]
    results = {}
    fm = fomc.reset_index().rename(columns={"date": "key"})
    fm["key"] = fm["key"].dt.to_period("M")
    for wname, shift in (("same", 0), ("post", 1)):
        block = {}
        for cls in RISK_RANKING:
            a = agg[agg["asset_class"] == cls].copy()
            a["key"] = (a["caldt"] - pd.DateOffset(months=shift)).dt.to_period("M")
            merged = a.merge(fm, on="key", how="inner")
            if len(merged) < 15:
                continue
            y = merged["flow_pct"] * 100  # percent of lagged TNA
            X = merged[["mp_shock", "cbi_shock"] + [c for c in controls if c in merged.columns]]
            row, res = ols_row(y, X)
            row["wald_mp_eq_cbi_p"] = wald_equal(res, "mp_shock", "cbi_shock")
            row["mp_beta"], row["mp_p"] = row["mp_shock"]["beta"], row["mp_shock"]["p"]
            row["cbi_beta"], row["cbi_p"] = row["cbi_shock"]["beta"], row["cbi_shock"]["p"]
            block[cls] = row
        results[wname] = block
    # diff = post - same month flow, same regression
    a0 = agg.copy(); a0["w"] = a0["caldt"]
    a1 = agg.copy(); a1["w"] = a1["caldt"] - pd.DateOffset(months=1)
    dm = a1.merge(a0[["asset_class", "w", "flow_pct"]].rename(columns={"flow_pct": "f0"}),
                  on=["asset_class", "w"], how="inner")
    dm["flow_pct"] = dm["flow_pct"] - dm["f0"]
    fm = fomc.reset_index().rename(columns={"date": "key"})
    fm["key"] = fm["key"].dt.to_period("M")
    block = {}
    for cls in RISK_RANKING:
        dmc = dm[dm["asset_class"] == cls].copy()
        dmc["key"] = dmc["w"].dt.to_period("M")
        merged = dmc.merge(fm, on="key", how="inner")
        if len(merged) < 15:
            continue
        X = merged[["mp_shock", "cbi_shock", "log_tna", "flow_vol_12m"]]
        row, res = ols_row(merged["flow_pct"] * 100, X)
        row["wald_mp_eq_cbi_p"] = wald_equal(res, "mp_shock", "cbi_shock")
        row["mp_beta"], row["mp_p"] = row["mp_shock"]["beta"], row["mp_shock"]["p"]
        row["cbi_beta"], row["cbi_p"] = row["cbi_shock"]["beta"], row["cbi_shock"]["p"]
        block[cls] = row
    results["diff"] = block
    return results


def run_h5_panel(agg, fomc, shift=0, zlb=False):
    a = agg.copy()
    a["key"] = (a["caldt"] - pd.DateOffset(months=shift)).dt.to_period("M")
    fm = fomc.reset_index().rename(columns={"date": "key"})
    fm["key"] = fm["key"].dt.to_period("M")
    pan = a.merge(fm, on="key", how="inner")
    pan["risk"] = pan["asset_class"].map(RISK_RANKING)
    pan["risk_x_mp"] = pan["risk"] * pan["mp_shock"]
    pan["risk_x_cbi"] = pan["risk"] * pan["cbi_shock"]
    cols = ["mp_shock", "cbi_shock", "risk_x_mp", "risk_x_cbi", "log_tna", "flow_vol_12m"]
    if zlb:
        pan["mp_x_zlb"] = pan["mp_shock"] * pan["zlb_2020_2022"]
        pan["cbi_x_zlb"] = pan["cbi_shock"] * pan["zlb_2020_2022"]
        cols += ["mp_x_zlb", "cbi_x_zlb"]
    pan = pan.dropna(subset=["flow_pct"] + cols)
    row, res = ols_row(pan["flow_pct"] * 100, pan[cols])
    row["n_meeting_months"] = int(pan["key"].nunique())
    row["asset_classes"] = int(pan["asset_class"].nunique())
    row["wald_riskxmp_eq_riskxcbi_p"] = wald_equal(res, "risk_x_mp", "risk_x_cbi")
    return row


def main():
    t0 = datetime.now()
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def log(msg):
        print(msg)
        lines.append(str(msg))

    fomc = load_fomc()
    log(f"[1/6] FOMC data: {len(fomc)} meetings {fomc.index.min().date()} .. {fomc.index.max().date()}")

    conn = connect_wrds()
    style, hdr, monthly, dlog = fetch_data(conn)
    conn.close()
    log(f"[2/6] WRDS download: {dlog}")

    style_pit = classify_point_in_time(style)
    log(f"[3/6] point-in-time classified funds: {style_pit['crsp_fundno'].nunique():,}")

    mf = attach_class_at(monthly, style_pit)
    log(f"[4/6] fund-months with class: {len(mf):,}")

    agg = aggregate(build_flows(mf))
    log(f"[5/6] class-month aggregates: {len(agg):,} "
        f"({agg['asset_class'].nunique()} classes, {agg['caldt'].nunique()} months)")

    h4 = run_h4(agg, fomc)
    h5 = run_h5_panel(agg, fomc, 0, False)
    h5_post = run_h5_panel(agg, fomc, 1, False)
    zero = run_h5_panel(agg, fomc, 0, True)

    results = {
        "meta": {
            "run_at": t0.isoformat(),
            "audit_fixes": ["point-in-time classification", "winsorize 1/99 + |flow|<=TNA cap",
                            "pseudo B-S removed", "single-run outputs"],
            "wrds_download": dlog,
            "fomc_meetings": int(len(fomc)),
            "fomc_range": [str(fomc.index.min().date()), str(fomc.index.max().date())],
            "flow_window": [WRDS_START, WRDS_END],
        },
        "H4_mp_vs_cbi": h4,
        "H5_risk_ladder_same": h5,
        "H5_risk_ladder_post": h5_post,
        "zero_rate_check": zero,
    }
    with open(os.path.join(OUT_DIR, "flow_results_rerun.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    agg.to_csv(os.path.join(OUT_DIR, "flows_by_class_month.csv"), index=False)
    fomc.reset_index().to_csv(os.path.join(OUT_DIR, "fomc_shocks_used.csv"), index=False)

    log("[6/6] regressions done")
    log("")
    log("=== QUICK READOUT (flow in % of lagged TNA, same-month window) ===")
    for cls, row in h4["same"].items():
        wv = row["wald_mp_eq_cbi_p"]
        log(f"  {cls:26s} MP {row['mp_beta']:+8.3f} (p={row['mp_p']:.3f})  "
            f"CBI {row['cbi_beta']:+8.3f} (p={row['cbi_p']:.3f})  "
            f"Wald MP=CBI p={wv if wv is not None else float('nan'):.3f}  N={row['n']}")
    log(f"  H5 panel(same): Risk x MP {h5['risk_x_mp']['beta']:+.3f} (p={h5['risk_x_mp']['p']:.3f}), "
        f"Risk x CBI {h5['risk_x_cbi']['beta']:+.3f} (p={h5['risk_x_cbi']['p']:.3f}), N={h5['n']}")
    log(f"  Zero-rate: MP x ZLB {zero['mp_x_zlb']['beta']:+.3f} (p={zero['mp_x_zlb']['p']:.3f}), "
        f"CBI x ZLB {zero['cbi_x_zlb']['beta']:+.3f} (p={zero['cbi_x_zlb']['p']:.3f}), N={zero['n']}")

    with open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + f"\n\nfinished: {datetime.now().isoformat()}\n")
    print(f"\nAll outputs in ./{OUT_DIR}/ - send that folder back.")


if __name__ == "__main__":
    main()
