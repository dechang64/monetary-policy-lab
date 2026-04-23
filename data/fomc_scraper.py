"""
FOMC Statement Scraper
======================
Fetch historical FOMC statements from the Federal Reserve website.

The Fed publishes statements at:
https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDD.htm

This module:
- Scrapes historical FOMC statements
- Caches them locally
- Provides them for NLP sentiment analysis
"""

import requests
from bs4 import BeautifulSoup
import re
import os
import json
from datetime import datetime
from typing import Optional


class FOMCScraper:
    """
    Scrape FOMC statements from the Federal Reserve website.
    """

    BASE_URL = "https://www.federalreserve.gov"
    STATEMENT_PATTERN = re.compile(r"monetary(\d{4})(\d{2})(\d{2})[a-z]?\.htm")

    # Known FOMC statement URLs (manually curated, 1994-2025)
    KNOWN_STATEMENTS = {
        # 2024
        "2024-12-18": "newsevents/pressreleases/monetary20241218a.htm",
        "2024-11-07": "newsevents/pressreleases/monetary20241107a.htm",
        "2024-09-18": "newsevents/pressreleases/monetary20240918a.htm",
        "2024-07-31": "newsevents/pressreleases/monetary20240731a.htm",
        "2024-06-12": "newsevents/pressreleases/monetary20240612a.htm",
        "2024-05-01": "newsevents/pressreleases/monetary20240501a.htm",
        "2024-03-20": "newsevents/pressreleases/monetary20240320a.htm",
        "2024-01-31": "newsevents/pressreleases/monetary20240131a.htm",
        # 2023
        "2023-12-13": "newsevents/pressreleases/monetary20231213a.htm",
        "2023-11-01": "newsevents/pressreleases/monetary20231101a.htm",
        "2023-09-20": "newsevents/pressreleases/monetary20230920a.htm",
        "2023-07-26": "newsevents/pressreleases/monetary20230726a.htm",
        "2023-05-03": "newsevents/pressreleases/monetary20230503a.htm",
        "2023-03-22": "newsevents/pressreleases/monetary20230322a.htm",
        "2023-02-01": "newsevents/pressreleases/monetary20230201a.htm",
        # 2022
        "2022-12-14": "newsevents/pressreleases/monetary20221214a.htm",
        "2022-11-02": "newsevents/pressreleases/monetary20221102a.htm",
        "2022-09-21": "newsevents/pressreleases/monetary20220921a.htm",
        "2022-07-27": "newsevents/pressreleases/monetary20220727a.htm",
        "2022-06-15": "newsevents/pressreleases/monetary20220615a.htm",
        "2022-05-04": "newsevents/pressreleases/monetary20220504a.htm",
        "2022-03-16": "newsevents/pressreleases/monetary20220316a.htm",
        "2022-01-26": "newsevents/pressreleases/monetary20220126a.htm",
        # 2021
        "2021-12-15": "newsevents/pressreleases/monetary20211215a.htm",
        "2021-11-03": "newsevents/pressreleases/monetary20211103a.htm",
        "2021-09-22": "newsevents/pressreleases/monetary20210922a.htm",
        "2021-07-28": "newsevents/pressreleases/monetary20210728a.htm",
        "2021-06-16": "newsevents/pressreleases/monetary20210616a.htm",
        "2021-04-28": "newsevents/pressreleases/monetary20210428a.htm",
        "2021-03-17": "newsevents/pressreleases/monetary20210317a.htm",
        "2021-01-27": "newsevents/pressreleases/monetary20210127a.htm",
        # 2020
        "2020-12-16": "newsevents/pressreleases/monetary20201216a.htm",
        "2020-11-05": "newsevents/pressreleases/monetary20201105a.htm",
        "2020-09-16": "newsevents/pressreleases/monetary20200916a.htm",
        "2020-07-29": "newsevents/pressreleases/monetary20200729a.htm",
        "2020-06-10": "newsevents/pressreleases/monetary20200610a.htm",
        "2020-04-29": "newsevents/pressreleases/monetary20200429a.htm",
        "2020-03-15": "newsevents/pressreleases/monetary20200315a.htm",
        "2020-03-03": "newsevents/pressreleases/monetary20200303a.htm",
        "2020-01-29": "newsevents/pressreleases/monetary20200129a.htm",
        # 2019
        "2019-12-11": "newsevents/pressreleases/monetary20191211a.htm",
        "2019-10-30": "newsevents/pressreleases/monetary20191030a.htm",
        "2019-09-18": "newsevents/pressreleases/monetary20190918a.htm",
        "2019-07-31": "newsevents/pressreleases/monetary20190731a.htm",
        "2019-06-19": "newsevents/pressreleases/monetary20190619a.htm",
        "2019-05-01": "newsevents/pressreleases/monetary20190501a.htm",
        "2019-03-20": "newsevents/pressreleases/monetary20190320a.htm",
        "2019-01-30": "newsevents/pressreleases/monetary20190130a.htm",
        # 2018
        "2018-12-19": "newsevents/pressreleases/monetary20181219a.htm",
        "2018-11-08": "newsevents/pressreleases/monetary20181108a.htm",
        "2018-09-26": "newsevents/pressreleases/monetary20180926a.htm",
        "2018-08-01": "newsevents/pressreleases/monetary20180801a.htm",
        "2018-06-13": "newsevents/pressreleases/monetary20180613a.htm",
        "2018-05-02": "newsevents/pressreleases/monetary20180502a.htm",
        "2018-03-21": "newsevents/pressreleases/monetary20180321a.htm",
        "2018-02-01": "newsevents/pressreleases/monetary20180201a.htm",
        # 2017
        "2017-12-13": "newsevents/pressreleases/monetary20171213a.htm",
        "2017-11-01": "newsevents/pressreleases/monetary20171101a.htm",
        "2017-09-20": "newsevents/pressreleases/monetary20170920a.htm",
        "2017-07-26": "newsevents/pressreleases/monetary20170726a.htm",
        "2017-06-14": "newsevents/pressreleases/monetary20170614a.htm",
        "2017-05-03": "newsevents/pressreleases/monetary20170503a.htm",
        "2017-03-15": "newsevents/pressreleases/monetary20170315a.htm",
        "2017-02-01": "newsevents/pressreleases/monetary20170201a.htm",
        # 2016
        "2016-12-14": "newsevents/pressreleases/monetary20161214a.htm",
        "2016-11-02": "newsevents/pressreleases/monetary20161102a.htm",
        "2016-09-21": "newsevents/pressreleases/monetary20160921a.htm",
        "2016-07-27": "newsevents/pressreleases/monetary20160727a.htm",
        "2016-06-15": "newsevents/pressreleases/monetary20160615a.htm",
        "2016-04-27": "newsevents/pressreleases/monetary20160427a.htm",
        "2016-03-16": "newsevents/pressreleases/monetary20160316a.htm",
        "2016-01-27": "newsevents/pressreleases/monetary20160127a.htm",
        # 2015
        "2015-12-16": "newsevents/pressreleases/monetary20151216a.htm",
        "2015-10-28": "newsevents/pressreleases/monetary20151028a.htm",
        "2015-09-17": "newsevents/pressreleases/monetary20150917a.htm",
        "2015-07-29": "newsevents/pressreleases/monetary20150729a.htm",
        "2015-06-17": "newsevents/pressreleases/monetary20150617a.htm",
        "2015-04-29": "newsevents/pressreleases/monetary20150429a.htm",
        "2015-03-18": "newsevents/pressreleases/monetary20150318a.htm",
        "2015-01-28": "newsevents/pressreleases/monetary20150128a.htm",
        # 2014
        "2014-12-17": "newsevents/pressreleases/monetary20141217a.htm",
        "2014-10-29": "newsevents/pressreleases/monetary20141029a.htm",
        "2014-09-17": "newsevents/pressreleases/monetary20140917a.htm",
        "2014-07-30": "newsevents/pressreleases/monetary20140730a.htm",
        "2014-06-18": "newsevents/pressreleases/monetary20140618a.htm",
        "2014-04-30": "newsevents/pressreleases/monetary20140430a.htm",
        "2014-03-19": "newsevents/pressreleases/monetary20140319a.htm",
        "2014-01-29": "newsevents/pressreleases/monetary20140129a.htm",
        # 2013
        "2013-12-18": "newsevents/pressreleases/monetary20131218a.htm",
        "2013-10-30": "newsevents/pressreleases/monetary20131030a.htm",
        "2013-09-18": "newsevents/pressreleases/monetary20130918a.htm",
        "2013-07-31": "newsevents/pressreleases/monetary20130731a.htm",
        "2013-06-19": "newsevents/pressreleases/monetary20130619a.htm",
        "2013-05-01": "newsevents/pressreleases/monetary20130501a.htm",
        "2013-03-20": "newsevents/pressreleases/monetary20130320a.htm",
        "2013-01-30": "newsevents/pressreleases/monetary20130130a.htm",
        # 2012
        "2012-12-12": "newsevents/pressreleases/monetary20121212a.htm",
        "2012-10-24": "newsevents/pressreleases/monetary20121024a.htm",
        "2012-09-13": "newsevents/pressreleases/monetary20120913a.htm",
        "2012-08-01": "newsevents/pressreleases/monetary20120801a.htm",
        "2012-06-20": "newsevents/pressreleases/monetary20120620a.htm",
        "2012-04-25": "newsevents/pressreleases/monetary20120425a.htm",
        "2012-03-13": "newsevents/pressreleases/monetary20120313a.htm",
        "2012-01-25": "newsevents/pressreleases/monetary20120125a.htm",
        # 2011
        "2011-12-13": "newsevents/pressreleases/monetary20111213a.htm",
        "2011-11-02": "newsevents/pressreleases/monetary20111102a.htm",
        "2011-09-21": "newsevents/pressreleases/monetary20110921a.htm",
        "2011-08-09": "newsevents/pressreleases/monetary20110809a.htm",
        "2011-06-22": "newsevents/pressreleases/monetary20110622a.htm",
        "2011-04-27": "newsevents/pressreleases/monetary20110427a.htm",
        "2011-03-15": "newsevents/pressreleases/monetary20110315a.htm",
        "2011-01-26": "newsevents/pressreleases/monetary20110126a.htm",
        # 2010
        "2010-12-14": "newsevents/pressreleases/monetary20101214a.htm",
        "2010-11-03": "newsevents/pressreleases/monetary20101103a.htm",
        "2010-09-21": "newsevents/pressreleases/monetary20100921a.htm",
        "2010-08-10": "newsevents/pressreleases/monetary20100810a.htm",
        "2010-06-23": "newsevents/pressreleases/monetary20100623a.htm",
        "2010-04-28": "newsevents/pressreleases/monetary20100428a.htm",
        "2010-03-16": "newsevents/pressreleases/monetary20100316a.htm",
        "2010-01-27": "newsevents/pressreleases/monetary20100127a.htm",
        # 2009
        "2009-12-16": "newsevents/pressreleases/monetary20091216a.htm",
        "2009-11-04": "newsevents/pressreleases/monetary20091104a.htm",
        "2009-09-23": "newsevents/pressreleases/monetary20090923a.htm",
        "2009-08-12": "newsevents/pressreleases/monetary20090812a.htm",
        "2009-06-24": "newsevents/pressreleases/monetary20090624a.htm",
        "2009-04-29": "newsevents/pressreleases/monetary20090429a.htm",
        "2009-03-18": "newsevents/pressreleases/monetary20090318a.htm",
        "2009-01-28": "newsevents/pressreleases/monetary20090128a.htm",
        # 2008
        "2008-12-16": "newsevents/pressreleases/monetary20081216a.htm",
        "2008-10-29": "newsevents/pressreleases/monetary20081029a.htm",
        "2008-09-16": "newsevents/pressreleases/monetary20080916a.htm",
        "2008-06-25": "newsevents/pressreleases/monetary20080625a.htm",
        "2008-04-30": "newsevents/pressreleases/monetary20080430a.htm",
        "2008-03-18": "newsevents/pressreleases/monetary20080318a.htm",
        "2008-01-30": "newsevents/pressreleases/monetary20080130a.htm",
        "2008-01-22": "newsevents/pressreleases/monetary20080122a.htm",
        # 2007
        "2007-12-11": "newsevents/pressreleases/monetary20071211a.htm",
        "2007-10-31": "newsevents/pressreleases/monetary20071031a.htm",
        "2007-09-18": "newsevents/pressreleases/monetary20070918a.htm",
        "2007-08-17": "newsevents/pressreleases/monetary20070817a.htm",
        "2007-06-28": "newsevents/pressreleases/monetary20070628a.htm",
        "2007-05-09": "newsevents/pressreleases/monetary20070509a.htm",
        "2007-03-21": "newsevents/pressreleases/monetary20070321a.htm",
        "2007-01-31": "newsevents/pressreleases/monetary20070131a.htm",
        # 2006
        "2006-12-12": "newsevents/pressreleases/monetary20061212.htm",
        "2006-10-25": "newsevents/pressreleases/monetary20061025.htm",
        "2006-09-20": "newsevents/pressreleases/monetary20060920.htm",
        "2006-08-08": "newsevents/pressreleases/monetary20060808.htm",
        "2006-06-29": "newsevents/pressreleases/monetary20060629.htm",
        "2006-05-10": "newsevents/pressreleases/monetary20060510.htm",
        "2006-03-28": "newsevents/pressreleases/monetary20060328.htm",
        "2006-01-31": "newsevents/pressreleases/monetary20060131.htm",
        # 2005
        "2005-12-13": "newsevents/pressreleases/monetary20051213.htm",
        "2005-11-01": "newsevents/pressreleases/monetary20051101.htm",
        "2005-09-20": "newsevents/pressreleases/monetary20050920.htm",
        "2005-08-09": "newsevents/pressreleases/monetary20050809.htm",
        "2005-06-30": "newsevents/pressreleases/monetary20050630.htm",
        "2005-05-03": "newsevents/pressreleases/monetary20050503.htm",
        "2005-03-22": "newsevents/pressreleases/monetary20050322.htm",
        "2005-02-02": "newsevents/pressreleases/monetary20050202.htm",
        # 2004
        "2004-12-14": "newsevents/pressreleases/monetary20041214.htm",
        "2004-11-10": "newsevents/pressreleases/monetary20041110.htm",
        "2004-09-21": "newsevents/pressreleases/monetary20040921.htm",
        "2004-08-10": "newsevents/pressreleases/monetary20040810.htm",
        "2004-06-30": "newsevents/pressreleases/monetary20040630.htm",
        "2004-05-04": "newsevents/pressreleases/monetary20040504.htm",
        "2004-03-16": "newsevents/pressreleases/monetary20040316.htm",
        "2004-01-28": "newsevents/pressreleases/monetary20040128.htm",
        # 2003
        "2003-12-09": "newsevents/pressreleases/monetary20031209.htm",
        "2003-09-16": "newsevents/pressreleases/monetary20030916.htm",
        "2003-06-25": "newsevents/pressreleases/monetary20030625.htm",
        "2003-05-06": "newsevents/pressreleases/monetary20030506.htm",
        "2003-03-18": "newsevents/pressreleases/monetary20030318.htm",
        "2003-01-29": "newsevents/pressreleases/monetary20030129.htm",
        # 2002
        "2002-11-06": "newsevents/pressreleases/monetary20021106.htm",
        "2002-09-24": "newsevents/pressreleases/monetary20020924.htm",
        "2002-08-13": "newsevents/pressreleases/monetary20020813.htm",
        "2002-06-26": "newsevents/pressreleases/monetary20020626.htm",
        "2002-05-07": "newsevents/pressreleases/monetary20020507.htm",
        "2002-03-19": "newsevents/pressreleases/monetary20020319.htm",
        "2002-01-30": "newsevents/pressreleases/monetary20020130.htm",
        # 2001
        "2001-12-11": "newsevents/pressreleases/monetary20011211.htm",
        "2001-11-06": "newsevents/pressreleases/monetary20011106.htm",
        "2001-10-02": "newsevents/pressreleases/monetary20011002.htm",
        "2001-08-21": "newsevents/pressreleases/monetary20010821.htm",
        "2001-06-27": "newsevents/pressreleases/monetary20010626.htm",
        "2001-05-15": "newsevents/pressreleases/monetary20010515.htm",
        "2001-03-20": "newsevents/pressreleases/monetary20010320.htm",
        "2001-01-31": "newsevents/pressreleases/monetary20010131.htm",
        # 2000
        "2000-12-19": "newsevents/pressreleases/monetary20001219.htm",
        "2000-11-15": "newsevents/pressreleases/monetary20001115.htm",
        "2000-10-03": "newsevents/pressreleases/monetary20001003.htm",
        "2000-08-22": "newsevents/pressreleases/monetary20000822.htm",
        "2000-06-28": "newsevents/pressreleases/monetary20000628.htm",
        "2000-05-16": "newsevents/pressreleases/monetary20000516.htm",
        "2000-03-21": "newsevents/pressreleases/monetary20000321.htm",
        "2000-02-02": "newsevents/pressreleases/monetary20000202.htm",
        # 1999
        "1999-12-21": "newsevents/pressreleases/monetary19991221.htm",
        "1999-11-16": "newsevents/pressreleases/monetary19991116.htm",
        "1999-10-05": "newsevents/pressreleases/monetary19991005.htm",
        "1999-08-24": "newsevents/pressreleases/monetary19990824.htm",
        "1999-06-30": "newsevents/pressreleases/monetary19990630.htm",
        "1999-05-18": "newsevents/pressreleases/monetary19990518.htm",
        "1999-03-30": "newsevents/pressreleases/monetary19990330.htm",
        "1999-02-03": "newsevents/pressreleases/monetary19990202.htm",
        # 1998
        "1998-12-22": "newsevents/pressreleases/monetary19981222.htm",
        "1998-11-17": "newsevents/pressreleases/monetary19981117.htm",
        "1998-10-06": "newsevents/pressreleases/monetary19981006.htm",
        "1998-09-29": "newsevents/pressreleases/monetary19980929.htm",
        "1998-08-18": "newsevents/pressreleases/monetary19980818.htm",
        "1998-07-01": "newsevents/pressreleases/monetary19980701.htm",
        "1998-05-19": "newsevents/pressreleases/monetary19980519.htm",
        "1998-03-31": "newsevents/pressreleases/monetary19980331.htm",
        # 1997
        "1997-12-16": "newsevents/pressreleases/monetary19971216.htm",
        "1997-11-12": "newsevents/pressreleases/monetary19971112.htm",
        "1997-10-07": "newsevents/pressreleases/monetary19971007.htm",
        "1997-09-30": "newsevents/pressreleases/monetary19970930.htm",
        "1997-08-19": "newsevents/pressreleases/monetary19970819.htm",
        "1997-07-02": "newsevents/pressreleases/monetary19970701.htm",
        "1997-05-20": "newsevents/pressreleases/monetary19970520.htm",
        "1997-03-25": "newsevents/pressreleases/monetary19970325.htm",
        # 1996
        "1996-12-17": "newsevents/pressreleases/monetary19961217.htm",
        "1996-11-13": "newsevents/pressreleases/monetary19961113.htm",
        "1996-10-01": "newsevents/pressreleases/monetary19961001.htm",
        "1996-09-24": "newsevents/pressreleases/monetary19960924.htm",
        "1996-08-20": "newsevents/pressreleases/monetary19960820.htm",
        "1996-07-02": "newsevents/pressreleases/monetary19960702.htm",
        "1996-05-21": "newsevents/pressreleases/monetary19960521.htm",
        "1996-03-26": "newsevents/pressreleases/monetary19960326.htm",
        # 1995
        "1995-12-19": "newsevents/pressreleases/monetary19951219.htm",
        "1995-11-15": "newsevents/pressreleases/monetary19951115.htm",
        "1995-10-03": "newsevents/pressreleases/monetary19951003.htm",
        "1995-08-22": "newsevents/pressreleases/monetary19950822.htm",
        "1995-07-06": "newsevents/pressreleases/monetary19950705.htm",
        "1995-05-23": "newsevents/pressreleases/monetary19950523.htm",
        "1995-03-28": "newsevents/pressreleases/monetary19950328.htm",
        "1995-02-01": "newsevents/pressreleases/monetary19950201.htm",
        # 1994
        "1994-12-20": "newsevents/pressreleases/monetary19941220.htm",
        "1994-11-15": "newsevents/pressreleases/monetary19941115.htm",
        "1994-09-27": "newsevents/pressreleases/monetary19940927.htm",
        "1994-08-16": "newsevents/pressreleases/monetary19940816.htm",
        "1994-07-06": "newsevents/pressreleases/monetary19940705.htm",
        "1994-05-17": "newsevents/pressreleases/monetary19940517.htm",
        "1994-03-22": "newsevents/pressreleases/monetary19940322.htm",
        "1994-02-04": "newsevents/pressreleases/monetary19940203.htm",
    }

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache", "fomc")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = {}

    def fetch_statement(self, date_str: str) -> Optional[str]:
        """
        Fetch a single FOMC statement.

        Args:
            date_str: FOMC date (YYYY-MM-DD)

        Returns:
            Statement text or None if not found
        """
        # Check cache
        cache_file = os.path.join(self.cache_dir, f"{date_str}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return f.read()

        # Check known URLs
        if date_str not in self.KNOWN_STATEMENTS:
            return None

        url = f"{self.BASE_URL}/{self.KNOWN_STATEMENTS[date_str]}"

        try:
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Research Bot)"
            })
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract the main statement text
            # FOMC statements are typically in a div with id="article"
            article = soup.find("div", id="article")
            if not article:
                article = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")

            if article:
                # Remove script and style elements
                for tag in article(["script", "style", "nav"]):
                    tag.decompose()

                text = article.get_text(separator="\n", strip=True)
                # Clean up whitespace
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = text.strip()

                # Cache it
                with open(cache_file, "w") as f:
                    f.write(text)

                return text

            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def fetch_multiple(self, date_list: list) -> dict:
        """
        Fetch multiple FOMC statements.

        Args:
            date_list: List of FOMC date strings

        Returns:
            Dict mapping date → statement text
        """
        statements = {}
        for date_str in date_list:
            text = self.fetch_statement(date_str)
            if text:
                statements[date_str] = text
        return statements

    def get_available_dates(self) -> list:
        """Return list of dates with known statement URLs."""
        return sorted(self.KNOWN_STATEMENTS.keys())

    def get_rate_decision(self, date_str: str) -> Optional[str]:
        """
        Extract the rate decision from a statement.

        Returns:
            "hike", "cut", "hold", or None
        """
        text = self.fetch_statement(date_str)
        if not text:
            return None

        text_lower = text.lower()

        if "raise" in text_lower or "increase" in text_lower:
            return "hike"
        elif "lower" in text_lower or "decrease" in text_lower or "reduce" in text_lower:
            return "cut"
        elif "maintain" in text_lower or "decided to keep" in text_lower:
            return "hold"

        return None
