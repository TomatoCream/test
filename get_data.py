#!/usr/bin/env python3

import requests
import json
import time
from typing import List, Dict, Any, Optional

def build_url(page: int, limit: int = 100) -> str:
    """Build the API URL with page and limit parameters."""
    return f"https://api.mycareersfuture.gov.sg/v2/search?limit={limit}&page={page}"

def fetch_page(page: int, limit: int = 100, delay: int = 3) -> Optional[Dict[str, Any]]:
    """Fetch data from a specific page with rate limiting."""
    url = build_url(page, limit)
    
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en;q=0.9',
        'content-type': 'application/json',
        'mcf-client': 'jobseeker',
        'origin': 'https://www.mycareersfuture.gov.sg',
        'priority': 'u=1, i',
        'referer': 'https://www.mycareersfuture.gov.sg/',
        'sec-ch-ua': '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
    }
    
    payload = {
        "sessionId": "",
        "postingCompany": []
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        # Try to parse as JSON to validate
        data = response.json()
        print(data)
        return data
    except json.JSONDecodeError:
        # If not valid JSON, return None
        return None
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        return None
    finally:
        # Always sleep to respect rate limits
        time.sleep(delay)

def fetch_all_data(delay: int = 3) -> List[Dict[str, Any]]:
    """Fetch all pages of data until reaching a page with invalid JSON."""
    all_results = []
    page = 0
    
    while True:
        print(f"Fetching page {page}...")
        data = fetch_page(page, delay=delay)
        
        if data is None:
            print(f"Reached end at page {page}")
            break
            
        if "results" in data:
            all_results.extend(data["results"])
            print(f"Retrieved {len(data['results'])} results from page {page}")
        else:
            print(f"Warning: No results field in response on page {page}")
            
        page += 1
    
    return all_results

def main():
    # Set delay between requests (in seconds)
    delay = 3
    
    print(f"Starting data collection with {delay} second delay between requests")
    results = fetch_all_data(delay)
    
    # Save results to file
    output_file = "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} job listings to {output_file}")

if __name__ == "__main__":
    main() 