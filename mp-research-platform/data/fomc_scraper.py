"""
FOMC Statement Scraper - Download all FOMC statements from Federal Reserve website
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

def get_page(url):
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=15)
    return resp.read().decode('utf-8')

def get_statement_links(year_page_url):
    """Get all monetary statement links from a year page."""
    html = get_page(year_page_url)
    links = re.findall(r'href="(/newsevents/pressreleases/monetary\d{8}[a-z]*\.htm)"', html)
    # Filter to only 'a' versions (main statements, not supplements)
    main_links = [l for l in links if l.endswith('a.htm')]
    return main_links

def extract_statement_text(html):
    """Extract the main statement text from FOMC HTML page."""
    # Remove HTML tags, scripts, styles
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Find the statement body - look for key phrases
    markers = [
        "The Federal Reserve Board",
        "The Federal Open Market Committee",
        "Information received since",
        "The Committee decided",
    ]
    
    best_start = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0 and pos < best_start:
            best_start = pos
    
    if best_start < len(text):
        text = text[best_start:]
    
    # Trim at footer
    footer_markers = ["Last Update:", "For media inquiries", "Return to text"]
    for marker in footer_markers:
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    
    return text.strip()

def scrape_all_statements():
    """Scrape all FOMC statements from 1994 to 2025."""
    all_statements = {}
    
    # Get year pages
    year_pages = get_page(f"{BASE_URL}/newsevents/pressreleases.htm")
    year_links = re.findall(r'href="(/newsevents/pressreleases/\d{4}-press-fomc\.htm)"', year_pages)
    
    print(f"Found {len(year_links)} year pages")
    
    for year_link in year_links:
        year = re.search(r'(\d{4})', year_link).group(1)
        if int(year) < 1994:
            continue
        
        full_url = BASE_URL + year_link
        print(f"\n  Processing {year}...", end=" ")
        
        try:
            statement_links = get_statement_links(full_url)
            print(f"{len(statement_links)} statements", end=" ")
            
            for link in statement_links:
                date_str = re.search(r'(\d{8})', link).group(1)
                date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                
                if date_formatted in all_statements:
                    continue
                
                try:
                    html = get_page(BASE_URL + link)
                    text = extract_statement_text(html)
                    
                    if len(text) > 100:  # minimum length check
                        all_statements[date_formatted] = text
                        print(".", end="", flush=True)
                    time.sleep(0.3)  # be polite
                except Exception as e:
                    print(f"x", end="", flush=True)
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.5)
    
    print(f"\n\nTotal statements collected: {len(all_statements)}")
    
    # Save
    output_path = os.path.join(DATA_DIR, "fomc_statements_all.json")
    with open(output_path, "w") as f:
        json.dump(all_statements, f, indent=2)
    print(f"Saved to {output_path}")
    
    return all_statements

if __name__ == "__main__":
    scrape_all_statements()
