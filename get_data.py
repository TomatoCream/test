#!/usr/bin/env python3

import requests
import json
import time
from typing import List, Dict, Any, Optional
from enum import Enum
import os
import glob
from datetime import datetime

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

# Maximum pages to fetch in a single run for a category if resuming/starting fresh.
# This is a safety net to prevent runaway requests if "no results" isn't detected.
MAX_PAGES_PER_CATEGORY_RUN = 200 

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
        self._payload["categories"] = [cat.category_name for cat in categories]
        return self
    
    def add_category(self, category: JobCategory) -> 'PayloadBuilder':
        if category.category_name not in self._payload["categories"]:
            self._payload["categories"].append(category.category_name)
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

def fetch_job_details(job_uuid: str, delay: int = 3) -> Optional[Dict[str, Any]]:
    """Fetch detailed data for a specific job UUID with rate limiting."""
    url = f"https://api.mycareersfuture.gov.sg/v2/jobs/{job_uuid}?updateApplicationCount=true"
    
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en;q=0.9',
        # 'content-type': 'application/json', # Not typically needed for GET requests without a body
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
        print(f"Fetching job details for UUID: {job_uuid} from {url}")
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Warning: Received status code {response.status_code} for job UUID {job_uuid}. Response: {response.text[:200]}...")
            return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON for job UUID {job_uuid}. Response: {response.text[:200]}...")
        return None
    except Exception as e:
        print(f"Error fetching job details for UUID {job_uuid}: {e}")
        return None
    finally:
        time.sleep(delay)

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
        if response.status_code == 200:
            # Try to parse as JSON to validate
            data = response.json()
            # print(data) # Commenting out to reduce noise during multiple page fetches
            return data
        else:
            print(f"Warning: Received status code {response.status_code} for page {page}. Response: {response.text[:200]}...")
            return None
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

def _fetch_and_save_category_data(
    job_category: JobCategory,
    date_str: str,
    target_dir_for_date: str, # This is output_dir_base/date_str
    delay: int = 3,
    session_id: Optional[str] = None
):
    """Fetch data for a single job category, save page by page, and resume if possible."""
    print(f"Processing category: {job_category.category_name} ({job_category.url_param_name})")
    print(f"Target directory for this category's data: {target_dir_for_date}")

    # Determine starting page by checking existing files for this category and date
    start_page = 0
    file_pattern = os.path.join(target_dir_for_date, f"{date_str}_{job_category.url_param_name}_*.json")
    existing_files = glob.glob(file_pattern)
    
    if existing_files:
        max_page_found = -1
        for f_path in existing_files:
            try:
                # Extract page number from filename: yyyymmdd_category_PAGENUMBER.json
                page_num_str = os.path.basename(f_path).split('_')[-1].replace('.json', '')
                # Accommodate new zero-padded format when parsing existing files
                page_num = int(page_num_str) 
                if page_num > max_page_found:
                    max_page_found = page_num
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse page number from file {f_path}: {e}")
        
        if max_page_found >= 0:
            start_page = max_page_found + 1
            print(f"Resuming from page {start_page} for category '{job_category.url_param_name}'.")
        else:
            print(f"No valid prior pages found for '{job_category.url_param_name}', starting from page 0.")
    else:
        print(f"No existing files found for category '{job_category.url_param_name}' and date '{date_str}'. Starting from page 0.")

    payload_builder = PayloadBuilder()
    if session_id:
        payload_builder.session_id(session_id)
    payload_builder.add_category(job_category) # Only this specific category
    payload_builder.sort_by([SortBy.NEW_POSTING_DATE]) # Default sort, can be made configurable
    
    current_payload = payload_builder.build()
    print(f"Using payload for {job_category.url_param_name}: {json.dumps(current_payload)}")

    page = start_page
    pages_fetched_this_run = 0
    total_results_for_category_this_run = 0

    while True:
        if pages_fetched_this_run >= MAX_PAGES_PER_CATEGORY_RUN:
            print(f"Reached max pages ({MAX_PAGES_PER_CATEGORY_RUN}) for this run for category '{job_category.url_param_name}'. Stopping.")
            break

        print(f"Fetching page {page} for category '{job_category.url_param_name}'...")
        
        data = fetch_page(page, payload=current_payload, delay=delay)
        
        if data is None:
            print(f"No data returned or error on page {page} for category '{job_category.url_param_name}'. Stopping this category.")
            break # Exit loop for this category if fetch failed
            
        # Only proceed if data is valid
        # Define file path for the current page's data
        file_name = f"{date_str}_{job_category.url_param_name}_{page:04d}.json" # Format page number
        file_path = os.path.join(target_dir_for_date, file_name)

        # Save the current page's data
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved page {page} data for '{job_category.url_param_name}' to {file_path}")
        except Exception as e:
            print(f"Error saving data for page {page}, category '{job_category.url_param_name}' to {file_path}: {e}")
            break 
            
        num_results_on_page = 0
        if "results" in data and isinstance(data["results"], list):
            num_results_on_page = len(data["results"])
            total_results_for_category_this_run += num_results_on_page
            if not data["results"] and page > 0: # Check if results list is empty (and not the first page attempt)
                print(f"No more results found at page {page} for '{job_category.url_param_name}'.")
                break
            print(f"Retrieved {num_results_on_page} results from page {page} for '{job_category.url_param_name}'.")
        else:
            # If 'results' key is missing or not a list, it's problematic.
            print(f"Warning: 'results' key missing or not a list in response on page {page} for '{job_category.url_param_name}'. Response keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
            if page > 0 : # Stop if this happens after the first page to prevent loops on bad but non-None responses
                print(f"Stopping category '{job_category.url_param_name}' due to unexpected response structure after page 0.")
                break
            # If it's page 0 and no results, it might just be an empty category.
            if num_results_on_page == 0:
                 print(f"No results found on initial page {page} for '{job_category.url_param_name}'. This category might be empty.")
                 break

        page += 1
        pages_fetched_this_run += 1
    
    print(f"Finished processing category: '{job_category.category_name}'. Fetched {pages_fetched_this_run} pages and {total_results_for_category_this_run} results in this run.")

def fetch_all_data(date_str: str, output_dir_base: str = "raw_data", delay: int = 3):
    """
    Fetches data for all job categories, saving each page to a separate JSON file
    organized by date and category.

    Args:
        date_str (str): The date in 'yyyymmdd' format, used for subdirectory naming.
        output_dir_base (str): The base directory to store data (e.g., 'raw_data').
        delay (int): Delay in seconds between API requests.
    """
    print(f"Starting data collection for date: {date_str}")
    print(f"Base output directory: {output_dir_base}")

    # Create the base output directory if it doesn't exist
    os.makedirs(output_dir_base, exist_ok=True)
    print(f"Ensured base directory exists: {output_dir_base}")

    # Create the date-specific subdirectory
    target_dir_for_date = os.path.join(output_dir_base, date_str)
    os.makedirs(target_dir_for_date, exist_ok=True)
    print(f"Ensured date directory exists: {target_dir_for_date}")
    
    # Get a session ID if needed - for now, not implemented, assuming API handles it or not required per category sequence
    # session_id = "some_session_id_if_api_needs_it" 
    session_id = None # Placeholder

    for category in JOB_CATEGORIES:
        print(f"\\n--- Processing Category: {category.category_name} ---")
        _fetch_and_save_category_data(
            job_category=category,
            date_str=date_str,
            target_dir_for_date=target_dir_for_date,
            delay=delay,
            session_id=session_id
        )
        print(f"--- Finished Category: {category.category_name} ---\n")
    
    print(f"All categories processed for date {date_str}.")


def main():
    # Set delay between requests (in seconds)
    delay = 1 # Adjusted for potentially faster testing, increase if rate limited
    
    # Get current date in yyyymmdd format
    current_date_str = datetime.now().strftime("%Y%m%d")
    
    output_directory = "raw_data" # Define the base directory for raw data

    print(f"Starting data collection process for {current_date_str} with {delay} second delay.")
    print(f"Data will be saved in subdirectories under ./{output_directory}/{current_date_str}/")

    fetch_all_data(
        date_str=current_date_str,
        output_dir_base=output_directory,
        delay=delay
    )
    
    print(f"Data collection finished. Check the '{output_directory}/{current_date_str}' directory.")

if __name__ == "__main__":
    main() 