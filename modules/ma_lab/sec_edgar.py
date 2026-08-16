"""
SEC EDGAR Scraper for M&A Filings
==================================
Fetches 8-K and DEFM14A filings from SEC EDGAR Full-Text Search API.

API: https://efts.sec.gov/LATEST/search-index?q=...&dateRange=custom...
Returns JSON with filings metadata; fetch full text separately.

Free, no API key required. Rate limit: 10 requests/sec.
"""

import urllib.request
import urllib.parse
import json
import time
from typing import List, Dict, Optional


EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

# IMPORTANT: SEC requires a User-Agent header
USER_AGENT = "Research Lab academic-user@example.com"


def search_filings(
    query: str,
    form_type: str = "8-K",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_results: int = 50,
) -> List[Dict]:
    """
    Search SEC EDGAR full-text for filings.
    
    Args:
        query: search terms, e.g., "merger agreement"
        form_type: "8-K", "DEFM14A", "10-K", "SC 13D"
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        max_results: number of results
    
    Returns:
        list of dicts with: title, filing_date, form, cik, company, url
    """
    params = {
        "q": f'"{query}"',
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "forms": form_type,
    }
    url = f"https://efts.sec.gov/LATEST/search-index?{urllib.parse.urlencode(params)}"
    
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return [{"error": str(e)}]
    
    hits = data.get("hits", {}).get("hits", [])
    results = []
    for hit in hits[:max_results]:
        src = hit.get("_source", {})
        results.append({
            "title": src.get("display_names", [""])[0] if src.get("display_names") else "",
            "filing_date": src.get("file_date", ""),
            "form": src.get("form", ""),
            "cik": src.get("cik", ""),
            "company": src.get("entity_name", ""),
            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={src.get('cik','')}&type={form_type}",
            "adsh": hit.get("_id", "").replace("-", ""),
        })
    return results


def fetch_filing_text(adsh: str, cik: str) -> Optional[str]:
    """
    Fetch the full text of a filing from EDGAR archive.
    adsh: accession number without dashes.
    """
    # Build URL: /Archives/edgar/data/{CIK}/{ADSH}/{ADSH}.txt
    cik_int = str(int(cik)).lstrip("0")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh}/{adsh}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def search_ma_announcements(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_results: int = 20,
) -> List[Dict]:
    """Search for 8-K filings mentioning 'merger agreement'."""
    return search_filings(
        query="merger agreement",
        form_type="8-K",
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )


def search_defm14a(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_results: int = 20,
) -> List[Dict]:
    """Search for definitive merger proxy statements (DEFM14A)."""
    return search_filings(
        query="merger consideration",
        form_type="DEFM14A",
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )
