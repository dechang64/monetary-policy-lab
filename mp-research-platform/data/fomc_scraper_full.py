"""
Complete FOMC Statement Scraper - All meetings 1994-2026
Uses known FOMC meeting dates and URL pattern to download statements.
"""
import urllib.request
import re
import os
import json
import time

BASE_URL = "https://www.federalreserve.gov"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}

# All scheduled FOMC meeting dates (1994-2026)
# Source: Federal Reserve FOMC calendars
FOMC_DATES = [
    # 1994
    "19940204","19940322","19940418","19940517","19940628","19940816","19940927",
    "19941115","19941220",
    # 1995
    "19950131","19950201","19950328","19950502","19950628","19950706","19950822",
    "19951003","19951115","19951219",
    # 1996
    "19960131","19960326","19960521","19960626","19960820","19960924","19961113",
    "19961217",
    # 1997
    "19970205","19970325","19970520","19970626","19970820","19970930","19971112",
    "19971216",
    # 1998
    "19980204","19980331","19980519","19980630","19980818","19980929","19981015",
    "19981117","19981222",
    # 1999
    "19990203","19990330","19990518","19990630","19990824","19991005","19991116",
    "19991221",
    # 2000
    "20000202","20000321","20000516","20000628","20000822","20001003","20001115",
    "20001219",
    # 2001
    "20010131","20010320","20010515","20010627","20010821","20010917","20011002",
    "20011106","20011211",
    # 2002
    "20020130","20020319","20020507","20020626","20020813","20020924","20021106",
    "20021210",
    # 2003
    "20030129","20030318","20030506","20030625","20030812","20030916","20031028",
    "20031209",
    # 2004
    "20040128","20040316","20040504","20040630","20040810","20040921","20041110",
    "20041214",
    # 2005
    "20050202","20050322","20050503","20050630","20050809","20050920","20051101",
    "20051213",
    # 2006
    "20060131","20060328","20060510","20060629","20060808","20060920","20061025",
    "20061212",
    # 2007
    "20070131","20070321","20070509","20070628","20070817","20070918","20071031",
    "20071211",
    # 2008
    "20080130","20080318","20080430","20080625","20080805","20080916","20081029",
    "20081216",
    # 2009
    "20090128","20090318","20090429","20090624","20090812","20090923","20091104",
    "20091216",
    # 2010
    "20100127","20100316","20100428","20100623","20100810","20100921","20101103",
    "20101214",
    # 2011
    "20110126","20110315","20110427","20110622","20110809","20110921","20111102",
    "20111213",
    # 2012
    "20120125","20120313","20120425","20120620","20120801","20120913","20121024",
    "20121212",
    # 2013
    "20130130","20130320","20130501","20130619","20130731","20130918","20131030",
    "20131218",
    # 2014
    "20140129","20140319","20140430","20140618","20140730","20140917","20141029",
    "20141217",
    # 2015
    "20150128","20150318","20150429","20150617","20150729","20150917","20151028",
    "20151216",
    # 2016
    "20160127","20160316","20160427","20160615","20160727","20160921","20161102",
    "20161214",
    # 2017
    "20170201","20170315","20170503","20170614","20170726","20170920","20171101",
    "20171213",
    # 2018
    "20180131","20180321","20180502","20180613","20180801","20180926","20181108",
    "20181219",
    # 2019
    "20190130","20190320","20190501","20190619","20190731","20190918","20191030",
    "20191211",
    # 2020
    "20200129","20200303","20200315","20200429","20200610","20200729","20200916",
    "20201105","20201216",
    # 2021
    "20210127","20210317","20210428","20210616","20210728","20210922","20211103",
    "20211215",
    # 2022
    "20220126","20220316","20220504","20220615","20220727","20220921","20221102",
    "20221214",
    # 2023
    "20230201","20230322","20230503","20230614","20230726","20230920","20231101",
    "20231213",
    # 2024
    "20240131","20240320","20240501","20240612","20240731","20240918","20241107",
    "20241218",
    # 2025
    "20250129","20250319","20250507","20250618","20250730","20250917","20251029",
    "20251210",
    # 2026
    "20260128","20260318",
]

def extract_text(html):
    """Extract statement text from HTML."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def download_statement(date_str):
    """Try to download FOMC statement for a given date."""
    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # Try URL patterns
    patterns = [
        f"/newsevents/pressreleases/monetary{date_str}a.htm",
        f"/newsevents/pressreleases/monetary{date_str}a1.htm",
        f"/newsevents/pressreleases/monetary{date_str}.htm",
    ]
    
    for pattern in patterns:
        try:
            url = BASE_URL + pattern
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=10)
            html = resp.read().decode('utf-8')
            text = extract_text(html)
            if len(text) > 100:
                return text
        except:
            continue
    return None

def main():
    # Load existing
    existing_path = os.path.join(DATA_DIR, "fomc_statements_all.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            statements = json.load(f)
    else:
        statements = {}
    
    print(f"Existing statements: {len(statements)}")
    print(f"Total FOMC dates to check: {len(FOMC_DATES)}")
    
    new_count = 0
    for date_str in FOMC_DATES:
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        if date_fmt in statements:
            continue
        
        text = download_statement(date_str)
        if text:
            statements[date_fmt] = text
            new_count += 1
            print(f"  ✅ {date_fmt} ({len(text)} chars)")
        else:
            print(f"  ❌ {date_fmt}")
        
        time.sleep(0.2)
    
    print(f"\nNew statements: {new_count}")
    print(f"Total statements: {len(statements)}")
    
    # Save
    with open(existing_path, "w") as f:
        json.dump(statements, f, indent=2)
    print(f"Saved to {existing_path}")
    
    # Stats
    years = {}
    for d in statements:
        yr = d[:4]
        years[yr] = years.get(yr, 0) + 1
    print("\nBy year:")
    for yr in sorted(years):
        print(f"  {yr}: {years[yr]}")

if __name__ == "__main__":
    main()
