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

    # Known FOMC statement URLs (manually curated for reliability)
    KNOWN_STATEMENTS = {
        "2024-12-18": "newsevents/pressreleases/monetary20241218a.htm",
        "2024-11-07": "newsevents/pressreleases/monetary20241107a.htm",
        "2024-09-18": "newsevents/pressreleases/monetary20240918a.htm",
        "2024-07-31": "newsevents/pressreleases/monetary20240731a.htm",
        "2024-06-12": "newsevents/pressreleases/monetary20240612a.htm",
        "2024-05-01": "newsevents/pressreleases/monetary20240501a.htm",
        "2024-03-20": "newsevents/pressreleases/monetary20240320a.htm",
        "2024-01-31": "newsevents/pressreleases/monetary20240131a.htm",
        "2023-12-13": "newsevents/pressreleases/monetary20231213a.htm",
        "2023-11-01": "newsevents/pressreleases/monetary20231101a.htm",
        "2023-09-20": "newsevents/pressreleases/monetary20230920a.htm",
        "2023-07-26": "newsevents/pressreleases/monetary20230726a.htm",
        "2023-05-03": "newsevents/pressreleases/monetary20230503a.htm",
        "2023-03-22": "newsevents/pressreleases/monetary20230322a.htm",
        "2023-02-01": "newsevents/pressreleases/monetary20230201a.htm",
        "2022-12-14": "newsevents/pressreleases/monetary20221214a.htm",
        "2022-11-02": "newsevents/pressreleases/monetary20221102a.htm",
        "2022-09-21": "newsevents/pressreleases/monetary20220921a.htm",
        "2022-07-27": "newsevents/pressreleases/monetary20220727a.htm",
        "2022-06-15": "newsevents/pressreleases/monetary20220615a.htm",
        "2022-05-04": "newsevents/pressreleases/monetary20220504a.htm",
        "2022-03-16": "newsevents/pressreleases/monetary20220316a.htm",
        "2022-01-26": "newsevents/pressreleases/monetary20220126a.htm",
        "2021-12-15": "newsevents/pressreleases/monetary20211215a.htm",
        "2021-11-03": "newsevents/pressreleases/monetary20211103a.htm",
        "2021-09-22": "newsevents/pressreleases/monetary20210922a.htm",
        "2021-07-28": "newsevents/pressreleases/monetary20210728a.htm",
        "2021-06-16": "newsevents/pressreleases/monetary20210616a.htm",
        "2021-04-28": "newsevents/pressreleases/monetary20210428a.htm",
        "2021-03-17": "newsevents/pressreleases/monetary20210317a.htm",
        "2021-01-27": "newsevents/pressreleases/monetary20210127a.htm",
        "2020-12-16": "newsevents/pressreleases/monetary20201216a.htm",
        "2020-11-05": "newsevents/pressreleases/monetary20201105a.htm",
        "2020-09-16": "newsevents/pressreleases/monetary20200916a.htm",
        "2020-07-29": "newsevents/pressreleases/monetary20200729a.htm",
        "2020-06-10": "newsevents/pressreleases/monetary20200610a.htm",
        "2020-04-29": "newsevents/pressreleases/monetary20200429a.htm",
        "2020-03-15": "newsevents/pressreleases/monetary20200315a.htm",
        "2020-03-03": "newsevents/pressreleases/monetary20200303a.htm",
        "2020-01-29": "newsevents/pressreleases/monetary20200129a.htm",
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
