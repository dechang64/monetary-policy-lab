"""
build_dataset_2026.py
Rebuild the FOMC meeting-level dataset from January 2006 to the latest available
meeting, with the corrections from "monetary-policy-lab: Corrections Needed":

  1. Statement text = statement body only (no website menus/footer)
  2. Every text is checked to be an FOMC statement (not another press release)
  3. Same-day returns and yield changes (meeting day vs previous trading day)
  4. Sample built from statements + surprises, not from minutes
 13. One unit (percent) for all returns; FG and 2020-22 zero-rate periods coded separately

Run from the repository root:
    python build_dataset_2026.py                # full run: scrape missing statements, WRDS, FRED
    python build_dataset_2026.py --rescrape     # recommended first run: re-download every statement
    python build_dataset_2026.py --offline      # only files already in the repo

Needs: pandas, numpy, requests, beautifulsoup4; wrds (for CRSP; asks for your WRDS login).
Output: results/fomc_dataset_2006_latest.csv plus printed sanity checks.
"""
import argparse
import io
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "mp_sentiment", os.path.join(REPO, "mp-research-platform", "data", "sentiment.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_lm_sentiment = _mod.compute_lm_sentiment  # the dictionary scoring used in the paper

STATEMENTS_JSON = os.path.join(REPO, "mp-research-platform", "data", "fomc_statements_all.json")
SHOCKS_CSV = os.path.join(REPO, "data", "combined_shocks_acosta_usmpd.csv")  # Acosta to Jul 2022, USMPD-based after
CRSP_LOCAL = os.path.join(REPO, "data", "wrds", "crsp_dsi_index.csv")
OUT_CSV = os.path.join(REPO, "results", "fomc_dataset_2006_latest.csv")
CLEAN_JSON = os.path.join(REPO, "data", "fomc_statements_clean.json")
START = "2006-01-01"
FED = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; monetary-policy-lab)"}


# ---------------------------------------------------------------- statements
def page_title(raw):
    """Title of the press release, e.g. 'Federal Reserve issues FOMC statement'."""
    i = raw.find("Please enable JavaScript")
    if i < 0:
        return ""
    seg = raw[i + 110:i + 400]
    m = re.search(r"\d{4} (.*?) (For (immediate )?release|For release at)", seg)
    return m.group(1) if m else ""


def body_only(raw):
    """Fix 1: keep the statement body; drop menus, header block, footer and the implementation note."""
    i = raw.find("Please enable JavaScript")
    if i >= 0:
        k = raw.find(" Share ", i)
        if k > 0:
            raw = raw[k + len(" Share "):]
    for marker in ("Implementation Note issued", "Last Update:"):
        k = raw.find(marker)
        if k > 0:
            raw = raw[:k]
    raw = raw.replace("\ufeff", " ")
    # Strip the header block ("December 16, 2008 FOMC statement For immediate release Share ...")
    m = re.search(r"The (?:Federal Open Market )?Committee\b", raw)
    if m and m.start() < 600:
        raw = raw[m.start():]
    else:
        for tok in ("For immediate release", "Share"):
            raw = raw.replace(tok, " ")
        raw = re.sub(r"^[^A-Za-z]*[A-Z][a-z]+ \d{1,2}, \d{4}[^A-Za-z]*", " ", raw)
        raw = re.sub(r"\bFOMC statement\b", " ", raw, count=1)
    return re.sub(r"\s+", " ", raw).strip()


def is_statement(title, text):
    """Fix 2: reject auction results, discount-rate actions and other releases."""
    if title:
        return ("FOMC statement" in title) or ("Federal Open Market Committee" in title)
    return "Committee" in text and ("Voting for" in text or "federal funds rate" in text)


def scrape_statement(date):
    """Download the statement for one date, trying each URL the Fed has used."""
    import requests
    from bs4 import BeautifulSoup
    d = date.strftime("%Y%m%d")
    for suffix in ("a", "b", "c", "a1"):
        url = f"{FED}/newsevents/pressreleases/monetary{d}{suffix}.htm"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.find("h3", class_="title")
        title = title.get_text(" ", strip=True) if title else (soup.title.get_text(" ", strip=True) if soup.title else "")
        art = soup.find("div", id="article")
        if art is None:
            continue
        for tag in art(["script", "style", "nav"]):
            tag.decompose()
        text = art.get_text(" ", strip=True)
        text = body_only(text)
        if is_statement(title, text):
            time.sleep(1)
            return text, url
        time.sleep(1)
    return None, None


def load_statements(dates, offline, rescrape=False):
    raw = json.load(open(STATEMENTS_JSON))
    # The cleaned cache (from previous rescrape runs) wins over the repo file,
    # so an --offline run never overwrites or loses scraped statements.
    if os.path.exists(CLEAN_JSON):
        raw.update(json.load(open(CLEAN_JSON)))
    clean, log = {}, []
    for dt in dates:
        key = dt.strftime("%Y-%m-%d")
        text, src = None, "repo"
        if rescrape and not offline:
            text, src = scrape_statement(dt)
            log.append(f"{key}: scraped from {src}" if text else f"{key}: no statement found online")
        if text is None and key in raw:
            t = raw[key] if isinstance(raw[key], str) else raw[key].get("text", "")
            title, body = page_title(t), body_only(t)
            if is_statement(title, body):
                text = body
            else:
                log.append(f"{key}: repo text is not an FOMC statement ({title[:60]})")
        if text is None and not offline:
            text, src = scrape_statement(dt)
            log.append(f"{key}: scraped from {src}" if text else f"{key}: no statement found online")
        if text:
            clean[key] = text
    json.dump(clean, open(CLEAN_JSON, "w"), indent=1)
    return clean, log


# ---------------------------------------------------------------- returns
def crsp_daily(offline, end):
    """Fix 3: CRSP daily index returns are already close(t-1) to close(t): same-day by construction."""
    if not offline:
        try:
            import wrds
            db = wrds.Connection()
            q = f"select date, vwretd, ewretd, sprtrn from crsp.dsi where date >= '{START}' and date <= '{end}'"
            df = db.raw_sql(q, date_cols=["date"])
            db.close()
            return df.set_index("date")
        except Exception as e:
            print(f"  WRDS not available ({e}); using the CRSP file in the repo")
    df = pd.read_csv(CRSP_LOCAL, parse_dates=["date"]).set_index("date")
    return df[["vwretd", "ewretd", "sprtrn"]]


def fred_series(code):
    """Treasury yields from FRED (no key needed); prefers the local snapshot data/fred_<code>.csv."""
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"fred_{code}.csv")
    if os.path.exists(local):
        s = pd.read_csv(local, na_values=".")
        s.columns = ["date", code]
        s["date"] = pd.to_datetime(s["date"])
        s = s.set_index("date")[code].dropna()
        return s[s.index >= START]
    import requests
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={START}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    s = pd.read_csv(io.StringIO(r.text), na_values=".")
    s.columns = ["date", code]
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")[code].dropna()


def same_day_change(series, dates):
    """Meeting-day level minus the previous available day's level."""
    out = {}
    idx = series.index
    for d in dates:
        if d not in idx:
            out[d] = np.nan
            continue
        i = idx.get_loc(d)
        out[d] = series.iloc[i] - series.iloc[i - 1] if i > 0 else np.nan
    return pd.Series(out)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use only files in the repo")
    ap.add_argument("--rescrape", action="store_true", help="re-download every statement (recommended once)")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    a = ap.parse_args()

    # Fix 4: the sample is every meeting with a surprise, not every meeting with minutes
    shocks = pd.read_csv(SHOCKS_CSV, index_col=0, parse_dates=True)
    shocks = shocks[(shocks.index >= START) & (shocks.index <= a.end)]
    print(f"Meetings with surprises: {len(shocks)} ({shocks.index.min().date()} to {shocks.index.max().date()})")

    clean, log = load_statements(shocks.index, a.offline, a.rescrape)
    for line in log:
        print("  " + line)

    rows = []
    for dt in shocks.index:
        key = dt.strftime("%Y-%m-%d")
        if key not in clean:
            continue
        comb, lm, cb, wc = compute_lm_sentiment(clean[key])
        rows.append(dict(date=dt, sentiment=comb, lm_score=lm, cb_score=cb, word_count=wc))
    df = pd.DataFrame(rows).set_index("date").join(shocks)

    crsp = crsp_daily(a.offline, a.end) * 100  # percent, fix 13
    df = df.join(crsp.rename(columns={"vwretd": "crsp_vw", "ewretd": "crsp_ew", "sprtrn": "sp500"}))

    # FRED yields: local snapshot first (data/fred_<code>.csv), network fallback.
    for code, name in [("DGS10", "ty10_chg"), ("DGS2", "ty2_chg"), ("DGS3MO", "tb3m_chg")]:
        try:
            df[name] = same_day_change(fred_series(code), df.index)
        except Exception as e:
            print(f"  FRED {code} failed: {e}")

    # Regimes (fix 13): 2008-15 forward guidance and 2020-22 zero rates kept separate
    df["fg_2008_2015"] = ((df.index >= "2008-12-16") & (df.index <= "2015-10-28")).astype(int)  # ends before Dec 2015 liftoff
    df["zlb_2020_2022"] = ((df.index >= "2020-03-15") & (df.index <= "2022-01-26")).astype(int)
    df["post_2022"] = (df.index > "2022-07-27").astype(int)

    # Daily sign split, kept for comparison only (circular on same-day CRSP; see fix 7)
    s = np.sign(df["target_shock"]) * np.sign(df["crsp_vw"])
    df["mp_shock_daily"] = np.where(s < 0, df["target_shock"], 0.0)
    df["cbi_shock_daily"] = np.where(s > 0, df["target_shock"], 0.0)
    df.loc[df["crsp_vw"].isna(), ["mp_shock_daily", "cbi_shock_daily"]] = np.nan

    df.to_csv(OUT_CSV)

    # ---------------------------------------------------------- sanity checks
    print("\nSANITY CHECKS")
    print(f"  Rows written: {len(df)} -> {OUT_CSV}")
    print(f"  Last meeting with tone: {df.index.max().date()}; last with CRSP: {df['crsp_vw'].last_valid_index().date()}")
    print("  Meetings per year:", df.groupby(df.index.year).size().to_dict())
    print(f"  Median statement length: {df.word_count.median():.0f} words (should be a few hundred)")
    print(f"  Tone vs word count correlation: {df.sentiment.corr(df.word_count):.2f} (was -0.83 with the web-page text; earlier statements were shorter, so some link remains)")
    print(f"  CRSP S&P vs CRSP VW same-day correlation: {df.sp500.corr(df.crsp_vw):.2f} (should be about 0.99)")
    for key in [k for k in clean if k in df.index.strftime('%Y-%m-%d')][:3]:
        print(f"  {key}: {clean[key][:150]}...")


if __name__ == "__main__":
    main()
