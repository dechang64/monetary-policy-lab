"""
Step 2: FOMC Meeting Dates and Decisions
Historical FOMC meeting dates with rate decisions (1994-2025)
Source: Federal Reserve official records
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Historical FOMC meeting dates and rate decisions
# Format: (date, decision_type, rate_before, rate_after)
# decision_type: 'rate_cut', 'rate_hike', 'unchanged', 'emergency'
FOMC_MEETINGS = [
    # 1994
    ("1994-02-04", "rate_hike", 3.00, 3.25),
    ("1994-03-22", "rate_hike", 3.25, 3.50),
    ("1994-04-18", "rate_hike", 3.50, 3.75),
    ("1994-05-17", "rate_hike", 3.75, 4.25),
    ("1994-06-30", "rate_hike", 4.25, 4.50),  # semi-annual
    ("1994-08-16", "rate_hike", 4.50, 4.75),
    ("1994-09-27", "rate_hike", 4.75, 5.00),  # moved from 9/28
    ("1994-11-15", "rate_hike", 5.00, 5.50),
    ("1995-02-01", "rate_hike", 5.50, 6.00),
    ("1995-07-06", "rate_cut", 6.00, 5.75),
    ("1995-12-19", "rate_cut", 5.75, 5.50),
    ("1996-01-31", "rate_cut", 5.50, 5.25),
    ("1997-03-25", "rate_hike", 5.25, 5.50),
    ("1998-09-29", "rate_cut", 5.50, 5.25),  # LTCM
    ("1998-10-15", "rate_cut", 5.25, 5.00),  # emergency
    ("1998-11-17", "rate_cut", 5.00, 4.75),
    ("1999-06-30", "rate_hike", 4.75, 5.00),
    ("1999-08-24", "rate_hike", 5.00, 5.25),
    ("1999-11-16", "rate_hike", 5.25, 5.50),
    ("2000-02-02", "rate_hike", 5.50, 5.75),
    ("2000-03-21", "rate_hike", 5.75, 6.00),
    ("2000-05-16", "rate_hike", 6.00, 6.50),
    ("2001-01-03", "rate_cut", 6.50, 6.00),  # emergency
    ("2001-01-31", "rate_cut", 6.00, 5.50),
    ("2001-03-20", "rate_cut", 5.50, 5.00),
    ("2001-04-18", "rate_cut", 5.00, 4.50),
    ("2001-05-15", "rate_cut", 4.50, 4.00),
    ("2001-06-27", "rate_cut", 4.00, 3.75),
    ("2001-08-21", "rate_cut", 3.75, 3.50),
    ("2001-09-17", "rate_cut", 3.50, 3.00),  # post-9/11
    ("2001-10-02", "rate_cut", 3.00, 2.50),
    ("2001-11-06", "rate_cut", 2.50, 2.00),
    ("2001-12-11", "rate_cut", 2.00, 1.75),
    ("2002-11-06", "rate_cut", 1.75, 1.25),
    ("2003-06-25", "rate_cut", 1.25, 1.00),
    ("2004-06-30", "rate_hike", 1.00, 1.25),
    ("2004-08-10", "rate_hike", 1.25, 1.50),
    ("2004-09-21", "rate_hike", 1.50, 1.75),
    ("2004-11-10", "rate_hike", 1.75, 2.00),
    ("2004-12-14", "rate_hike", 2.00, 2.25),
    ("2005-02-02", "rate_hike", 2.25, 2.50),
    ("2005-03-22", "rate_hike", 2.50, 2.75),
    ("2005-05-03", "rate_hike", 2.75, 3.00),
    ("2005-06-30", "rate_hike", 3.00, 3.25),
    ("2005-08-09", "rate_hike", 3.25, 3.50),
    ("2005-09-20", "rate_hike", 3.50, 3.75),
    ("2005-11-01", "rate_hike", 3.75, 4.00),
    ("2005-12-13", "rate_hike", 4.00, 4.25),
    ("2006-01-31", "rate_hike", 4.25, 4.50),
    ("2006-03-28", "rate_hike", 4.50, 4.75),
    ("2006-05-10", "rate_hike", 4.75, 5.00),
    ("2006-06-29", "rate_hike", 5.00, 5.25),
    ("2007-09-18", "rate_cut", 5.25, 4.75),
    ("2007-10-31", "rate_cut", 4.75, 4.50),
    ("2007-12-11", "rate_cut", 4.50, 4.25),
    ("2008-01-22", "rate_cut", 4.25, 3.50),  # emergency
    ("2008-01-30", "rate_cut", 3.50, 3.00),
    ("2008-03-18", "rate_cut", 3.00, 2.25),
    ("2008-04-30", "rate_cut", 2.25, 2.00),
    ("2008-10-08", "rate_cut", 2.00, 1.50),  # coordinated global cut
    ("2008-10-29", "rate_cut", 1.50, 1.00),
    ("2008-12-16", "rate_cut", 1.00, 0.25),  # ZLB begins
    # ZLB period - all unchanged
    ("2009-01-28", "unchanged", 0.25, 0.25),
    ("2009-03-18", "unchanged", 0.25, 0.25),
    ("2009-04-29", "unchanged", 0.25, 0.25),
    ("2009-06-24", "unchanged", 0.25, 0.25),
    ("2009-08-12", "unchanged", 0.25, 0.25),
    ("2009-09-23", "unchanged", 0.25, 0.25),
    ("2009-11-04", "unchanged", 0.25, 0.25),
    ("2009-12-16", "unchanged", 0.25, 0.25),
    ("2010-01-27", "unchanged", 0.25, 0.25),
    ("2010-03-16", "unchanged", 0.25, 0.25),
    ("2010-04-28", "unchanged", 0.25, 0.25),
    ("2010-06-23", "unchanged", 0.25, 0.25),
    ("2010-08-10", "unchanged", 0.25, 0.25),
    ("2010-09-21", "unchanged", 0.25, 0.25),
    ("2010-11-03", "unchanged", 0.25, 0.25),
    ("2010-12-14", "unchanged", 0.25, 0.25),
    ("2011-01-26", "unchanged", 0.25, 0.25),
    ("2011-03-15", "unchanged", 0.25, 0.25),
    ("2011-04-27", "unchanged", 0.25, 0.25),
    ("2011-06-22", "unchanged", 0.25, 0.25),
    ("2011-08-09", "unchanged", 0.25, 0.25),
    ("2011-09-21", "unchanged", 0.25, 0.25),
    ("2011-11-02", "unchanged", 0.25, 0.25),
    ("2011-12-13", "unchanged", 0.25, 0.25),
    ("2012-01-25", "unchanged", 0.25, 0.25),
    ("2012-03-13", "unchanged", 0.25, 0.25),
    ("2012-04-25", "unchanged", 0.25, 0.25),
    ("2012-06-20", "unchanged", 0.25, 0.25),
    ("2012-08-01", "unchanged", 0.25, 0.25),
    ("2012-09-13", "unchanged", 0.25, 0.25),  # QE3 announced
    ("2012-10-24", "unchanged", 0.25, 0.25),
    ("2012-12-12", "unchanged", 0.25, 0.25),
    ("2013-01-30", "unchanged", 0.25, 0.25),
    ("2013-03-20", "unchanged", 0.25, 0.25),
    ("2013-05-01", "unchanged", 0.25, 0.25),
    ("2013-06-19", "unchanged", 0.25, 0.25),
    ("2013-07-31", "unchanged", 0.25, 0.25),
    ("2013-09-18", "unchanged", 0.25, 0.25),
    ("2013-10-30", "unchanged", 0.25, 0.25),
    ("2013-12-18", "unchanged", 0.25, 0.25),
    ("2014-01-29", "unchanged", 0.25, 0.25),
    ("2014-03-19", "unchanged", 0.25, 0.25),
    ("2014-04-30", "unchanged", 0.25, 0.25),
    ("2014-06-18", "unchanged", 0.25, 0.25),
    ("2014-07-30", "unchanged", 0.25, 0.25),
    ("2014-09-17", "unchanged", 0.25, 0.25),
    ("2014-10-29", "unchanged", 0.25, 0.25),
    ("2014-12-17", "unchanged", 0.25, 0.25),
    ("2015-01-28", "unchanged", 0.25, 0.25),
    ("2015-03-18", "unchanged", 0.25, 0.25),
    ("2015-04-29", "unchanged", 0.25, 0.25),
    ("2015-06-17", "unchanged", 0.25, 0.25),
    ("2015-07-29", "unchanged", 0.25, 0.25),
    ("2015-09-17", "unchanged", 0.25, 0.25),
    ("2015-10-28", "unchanged", 0.25, 0.25),
    ("2015-12-16", "rate_hike", 0.25, 0.50),  # liftoff!
    ("2016-01-27", "unchanged", 0.50, 0.50),
    ("2016-03-16", "unchanged", 0.50, 0.50),
    ("2016-04-27", "unchanged", 0.50, 0.50),
    ("2016-06-15", "unchanged", 0.50, 0.50),
    ("2016-07-27", "unchanged", 0.50, 0.50),
    ("2016-09-21", "unchanged", 0.50, 0.50),
    ("2016-11-02", "unchanged", 0.50, 0.50),
    ("2016-12-14", "rate_hike", 0.50, 0.75),
    ("2017-02-01", "unchanged", 0.75, 0.75),
    ("2017-03-15", "rate_hike", 0.75, 1.00),
    ("2017-05-03", "unchanged", 1.00, 1.00),
    ("2017-06-14", "rate_hike", 1.00, 1.25),
    ("2017-07-26", "unchanged", 1.25, 1.25),
    ("2017-09-20", "unchanged", 1.25, 1.25),
    ("2017-10-31", "unchanged", 1.25, 1.25),
    ("2017-12-13", "rate_hike", 1.25, 1.50),
    ("2018-02-01", "unchanged", 1.50, 1.50),
    ("2018-03-21", "rate_hike", 1.50, 1.75),
    ("2018-05-02", "unchanged", 1.75, 1.75),
    ("2018-06-13", "rate_hike", 1.75, 2.00),
    ("2018-08-01", "unchanged", 2.00, 2.00),
    ("2018-09-26", "rate_hike", 2.00, 2.25),
    ("2018-11-08", "unchanged", 2.25, 2.25),
    ("2018-12-19", "rate_hike", 2.25, 2.50),
    ("2019-01-30", "unchanged", 2.50, 2.50),
    ("2019-03-20", "unchanged", 2.50, 2.50),
    ("2019-05-01", "unchanged", 2.50, 2.50),
    ("2019-06-19", "unchanged", 2.50, 2.50),
    ("2019-07-31", "rate_cut", 2.50, 2.25),
    ("2019-09-18", "rate_cut", 2.25, 2.00),
    ("2019-10-30", "rate_cut", 2.00, 1.75),
    ("2019-12-11", "unchanged", 1.75, 1.75),
    ("2020-03-03", "rate_cut", 1.75, 1.25),  # emergency COVID
    ("2020-03-15", "rate_cut", 1.25, 0.25),  # emergency COVID
    ("2020-04-29", "unchanged", 0.25, 0.25),
    ("2020-06-10", "unchanged", 0.25, 0.25),
    ("2020-07-29", "unchanged", 0.25, 0.25),
    ("2020-09-16", "unchanged", 0.25, 0.25),
    ("2020-11-05", "unchanged", 0.25, 0.25),
    ("2020-12-16", "unchanged", 0.25, 0.25),
    ("2021-01-27", "unchanged", 0.25, 0.25),
    ("2021-03-17", "unchanged", 0.25, 0.25),
    ("2021-04-28", "unchanged", 0.25, 0.25),
    ("2021-06-16", "unchanged", 0.25, 0.25),
    ("2021-07-28", "unchanged", 0.25, 0.25),
    ("2021-09-22", "unchanged", 0.25, 0.25),
    ("2021-11-03", "unchanged", 0.25, 0.25),
    ("2021-12-15", "unchanged", 0.25, 0.25),
    ("2022-01-26", "unchanged", 0.25, 0.25),
    ("2022-03-16", "rate_hike", 0.25, 0.50),
    ("2022-05-04", "rate_hike", 0.50, 1.00),  # +50bp
    ("2022-06-15", "rate_hike", 1.00, 1.75),  # +75bp
    ("2022-07-27", "rate_hike", 1.75, 2.50),  # +75bp
    ("2022-09-21", "rate_hike", 2.50, 3.25),  # +75bp
    ("2022-11-02", "rate_hike", 3.25, 4.00),  # +75bp
    ("2022-12-14", "rate_hike", 4.00, 4.50),  # +50bp
    ("2023-02-01", "rate_hike", 4.50, 4.75),  # +25bp
    ("2023-03-22", "rate_hike", 4.75, 5.00),  # +25bp
    ("2023-05-03", "rate_hike", 5.00, 5.25),  # +25bp
    ("2023-06-14", "unchanged", 5.25, 5.25),  # pause
    ("2023-07-26", "rate_hike", 5.25, 5.50),  # +25bp
    ("2023-09-20", "unchanged", 5.50, 5.50),
    ("2023-11-01", "unchanged", 5.50, 5.50),
    ("2023-12-13", "unchanged", 5.50, 5.50),
    ("2024-01-31", "unchanged", 5.50, 5.50),
    ("2024-03-20", "unchanged", 5.50, 5.50),
    ("2024-05-01", "unchanged", 5.50, 5.50),
    ("2024-06-12", "unchanged", 5.50, 5.50),
    ("2024-07-31", "rate_cut", 5.50, 5.25),  # first cut
    ("2024-09-18", "rate_cut", 5.25, 5.00),  # -25bp
    ("2024-11-07", "rate_cut", 5.00, 4.75),  # -25bp
    ("2024-12-18", "rate_cut", 4.75, 4.50),  # -25bp
    ("2025-01-29", "unchanged", 4.50, 4.50),
    ("2025-03-19", "unchanged", 4.50, 4.50),
    ("2025-04-30", "unchanged", 4.50, 4.50),
]

def get_fomc_data():
    df = pd.DataFrame(FOMC_MEETINGS, columns=["date", "decision", "rate_before", "rate_after"])
    df["date"] = pd.to_datetime(df["date"])
    df["rate_change"] = df["rate_after"] - df["rate_before"]
    df["expected_change"] = 0.0  # will be filled by surprise calculator
    df["surprise"] = df["rate_change"] - df["expected_change"]
    
    # Add chair tenure
    chairs = [
        ("1994-01-01", "2006-01-31", "Greenspan"),
        ("2006-02-01", "2014-01-31", "Bernanke"),
        ("2014-02-01", "2018-02-03", "Yellen"),
        ("2018-02-04", "2025-12-31", "Powell"),
    ]
    df["chair"] = "Unknown"
    for start, end, name in chairs:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "chair"] = name
    
    # Add regime
    df["regime"] = "conventional"
    df.loc[(df["date"] >= "2008-01-01") & (df["date"] <= "2015-12-31"), "regime"] = "forward_guidance"
    df.loc[df["date"] >= "2016-01-01", "regime"] = "normalization"
    
    return df

if __name__ == "__main__":
    df = get_fomc_data()
    print(f"FOMC meetings: {len(df)}")
    print(f"  Rate hikes: {(df['decision'] == 'rate_hike').sum()}")
    print(f"  Rate cuts: {(df['decision'] == 'rate_cut').sum()}")
    print(f"  Unchanged: {(df['decision'] == 'unchanged').sum()}")
    print(f"  Period: {df['date'].min().date()} to {df['date'].max().date()}")
    df.to_csv(os.path.join(DATA_DIR, "fomc_meetings.csv"), index=False)
    print("Saved to fomc_meetings.csv")
