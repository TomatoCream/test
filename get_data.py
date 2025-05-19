#!/usr/bin/env python3

import requests
import json
import time
from typing import List, Dict, Any, Optional
from enum import Enum

class JobCategory:
    def __init__(self, url_param_name: str, category_name: str):
        self.url_param_name = url_param_name
        self.category_name = category_name

JOB_CATEGORIES: List[JobCategory] = [
    JobCategory("accounting", "Accounting / Auditing / Taxation"),
    JobCategory("admin", "Admin / Secretarial"),
    JobCategory("advertising", "Advertising / Media"),
    JobCategory("architecture", "Architecture / Interior Design"),
    JobCategory("banking-finance", "Banking and Finance"),
    JobCategory("building-construction", "Building and Construction"),
    JobCategory("consulting", "Consulting"),
    JobCategory("customer-service", "Customer Service"),
    JobCategory("design", "Design"),
    JobCategory("education-training", "Education and Training"),
    JobCategory("engineering", "Engineering"),
    JobCategory("entertainment", "Entertainment"),
    JobCategory("environment", "Environment / Health"),
    JobCategory("events", "Events / Promotions"),
    JobCategory("food-and-beverage", "F&B"),
    JobCategory("general-management", "General Management"),
    JobCategory("general-work", "General Work"),
    JobCategory("healthcare", "Healthcare / Pharmaceutical"),
    JobCategory("hospitality", "Hospitality"),
    JobCategory("human-resources", "Human Resources"),
    JobCategory("information-technology", "Information Technology"),
    JobCategory("insurance", "Insurance"),
    JobCategory("legal", "Legal"),
    JobCategory("logistics", "Logistics / Supply Chain"),
    JobCategory("manufacturing", "Manufacturing"),
    JobCategory("marketing", "Marketing / Public Relations"),
    JobCategory("medical", "Medical / Therapy Services"),
    JobCategory("others", "Others"),
    JobCategory("personal-care", "Personal Care / Beauty"),
    JobCategory("precision-engineering", "Precision Engineering"),
    JobCategory("professional-services", "Professional Services"),
    JobCategory("public", "Public / Civil Service"),
    JobCategory("purchasing", "Purchasing / Merchandising"),
    JobCategory("real-estate", "Real Estate / Property Management"),
    JobCategory("repair-maintenance", "Repair and Maintenance"),
    JobCategory("risk-management", "Risk Management"),
    JobCategory("sales", "Sales / Retail"),
    JobCategory("sciences", "Sciences / Laboratory / R&D"),
    JobCategory("security", "Security and Investigation"),
    JobCategory("social-services", "Social Services"),
    JobCategory("telecommunications", "Telecommunications"),
    JobCategory("travel", "Travel / Tourism"),
    JobCategory("wholesale-trade", "Wholesale Trade"),
]

class SortBy(Enum):
    NEW_POSTING_DATE = "new_posting_date"
    RELEVANCY = "relevancy"
    SALARY = "salary"
    # Add other sort options if known

class PayloadBuilder:
    def __init__(self):
        self._payload: Dict[str, Any] = {
            "sessionId": "",
            "categories": [],
            "postingCompany": [],
            "sortBy": [SortBy.NEW_POSTING_DATE.value]  # Default sort
        }

    def session_id(self, session_id: str) -> 'PayloadBuilder':
        self._payload["sessionId"] = session_id
        return self

    def categories(self, categories: List[JobCategory]) -> 'PayloadBuilder':
        self._payload["categories"] = [cat.url_param_name for cat in categories]
        return self
    
    def add_category(self, category: JobCategory) -> 'PayloadBuilder':
        if category.url_param_name not in self._payload["categories"]:
            self._payload["categories"].append(category.url_param_name)
        return self

    def posting_companies(self, companies: List[str]) -> 'PayloadBuilder':
        self._payload["postingCompany"] = companies
        return self
    
    def add_posting_company(self, company: str) -> 'PayloadBuilder':
        if company not in self._payload["postingCompany"]:
            self._payload["postingCompany"].append(company)
        return self

    def sort_by(self, sort_options: List[SortBy]) -> 'PayloadBuilder':
        self._payload["sortBy"] = [option.value for option in sort_options]
        return self

    def build(self) -> Dict[str, Any]:
        return self._payload

def build_url(page: int, limit: int = 100) -> str:
    """Build the API URL with page and limit parameters."""
    return f"https://api.mycareersfuture.gov.sg/v2/search?limit={limit}&page={page}"

def fetch_page(page: int, payload: Dict[str, Any], limit: int = 100, delay: int = 3) -> Optional[Dict[str, Any]]:
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
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        # Try to parse as JSON to validate
        data = response.json()
        # print(data) # Commenting out to reduce noise during multiple page fetches
        return data
    except json.JSONDecodeError:
        # If not valid JSON, return None
        print(f"Warning: Could not decode JSON from page {page}. Response: {response.text[:200]}...") # Log part of the response
        return None
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        return None
    finally:
        # Always sleep to respect rate limits
        time.sleep(delay)

def fetch_all_data(initial_payload: Dict[str, Any], delay: int = 3) -> List[Dict[str, Any]]:
    """Fetch all pages of data until reaching a page with invalid JSON."""
    all_results = []
    page = 0
    
    while True:
        print(f"Fetching page {page}...")
        # For subsequent pages, we might not need to send the full payload if the API maintains session state
        # However, to be safe, we send it every time.
        # If the API has a concept of 'next page' token, that would be more efficient.
        current_payload = initial_payload.copy() # Ensure we don't modify the original
        
        data = fetch_page(page, payload=current_payload, delay=delay)
        
        if data is None:
            print(f"Reached end or encountered an error at page {page}")
            break
            
        if "results" in data and isinstance(data["results"], list):
            if not data["results"] and page > 0: # Check if results are empty and it's not the first page
                print(f"No more results found at page {page}. Total results: {len(all_results)}")
                break
            all_results.extend(data["results"])
            print(f"Retrieved {len(data['results'])} results from page {page}. Total so far: {len(all_results)}")
        else:
            print(f"Warning: No 'results' list in response on page {page}. Response keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
            if page > 0 : # Stop if no results after the first page to prevent infinite loops on bad responses
                break
            
        page += 1
        if page > 200: # Safety break for very long runs / potential issues
            print("Stopping after 200 pages to prevent excessive requests.")
            break
    
    return all_results

def main():
    # Set delay between requests (in seconds)
    delay = 1 # Adjusted for potentially faster testing, increase if rate limited
    
    print(f"Starting data collection with {delay} second delay between requests")

    # Example usage of the PayloadBuilder
    wholesale_trade_category = next((cat for cat in JOB_CATEGORIES if cat.category_name == "Wholesale Trade"), None)
    
    if not wholesale_trade_category:
        print("Error: 'Wholesale Trade' category not found.")
        return

    payload_builder = PayloadBuilder()
    payload_builder.add_category(wholesale_trade_category)
    payload_builder.sort_by([SortBy.NEW_POSTING_DATE])
    # You can add more configurations here:
    # payload_builder.session_id("your_session_id")
    # payload_builder.add_posting_company("Some Company")
    
    custom_payload = payload_builder.build()
    print(f"Generated payload: {json.dumps(custom_payload, indent=2)}")
    
    results = fetch_all_data(initial_payload=custom_payload, delay=delay)
    
    # Save results to file
    output_file = "results_custom_payload.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} job listings to {output_file}")

if __name__ == "__main__":
    main() 