# -*- coding: utf-8 -*-
"""Build data/jk_shocks_meeting.csv from Marek Jarociński's official updated
MP/CBI decomposition (github.com/marekjarocinski/jkshocks_update_fed).

Download both files first (raw.githubusercontent.com, branch `main`;
if raw fails, use the GitHub contents API which returns base64):
  shocks_fed_jk_t.csv  - per-announcement, columns: start, pc1, SP500,
                         MP_pm, CBI_pm, MP_median, CBI_median
  shocks_fed_jk_m.csv  - monthly aggregates

We use MP_pm / CBI_pm (the post-meeting-window decomposition) as
`mp_jk` / `cbi_jk`, the column names robustness_tests.py expects
(JK_DATE_COL=date, JK_MP_COL=mp_jk, JK_CBI_COL=cbi_jk).

Audit note (2026-11-26): t.csv has 7 duplicate dates (two announcement
windows the same day in 1991 and 2005-05-03). All are before 2006, so they
cannot collide with our 161-meeting sample (2006-2026, matched 161/161).
keep=first is therefore safe; assert added so a future re-download that
changes this is caught loudly.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "jk", "shocks_fed_jk_t.csv")
OUT = os.path.join(ROOT, "data", "jk_shocks_meeting.csv")
OUR_MEETINGS = os.path.join(ROOT, "results", "fomc_dataset_2006_latest.csv")


def main():
    t = pd.read_csv(SRC)
    t["date"] = pd.to_datetime(t["start"]).dt.normalize()

    dup_dates = t.loc[t.duplicated("date", keep=False), "date"].unique()
    if len(dup_dates):
        assert t[t["date"].isin(dup_dates)]["date"].max() < pd.Timestamp("2006-01-01"), \
            "duplicate JK dates now overlap our 2006+ sample - inspect before deduping!"

    t = t.drop_duplicates("date", keep="first")
    out = t[["date", "MP_pm", "CBI_pm"]].rename(columns={"MP_pm": "mp_jk", "CBI_pm": "cbi_jk"})
    out.to_csv(OUT, index=False)

    ours = pd.read_csv(OUR_MEETINGS)
    ours["date"] = pd.to_datetime(ours["date"]).dt.normalize()
    matched = ours["date"].isin(out["date"]).sum()
    print(f"saved {OUT}: {len(out)} rows, {out['date'].min().date()} -> {out['date'].max().date()}")
    print(f"our meetings matched: {matched} / {len(ours)}")
    assert matched == len(ours), "meeting-date coverage dropped below 100%!"


if __name__ == "__main__":
    main()
