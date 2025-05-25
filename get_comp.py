#!/usr/bin/env python3
import argparse
import requests
import orjson
import os
from datetime import datetime
import time
import threading
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C) gracefully"""
    global shutdown_requested
    print(f"\n\nReceived interrupt signal ({signal.Signals(signum).name}). Initiating graceful shutdown...")
    print("Waiting for current downloads to complete...")
    print("Press Ctrl+C again to force quit (may leave corrupted files)")
    shutdown_requested = True

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def get_api_url(data_type, limit=100, page=0):
    """Generate the API URL for the specified data type endpoint"""
    return f"https://api.mycareersfuture.gov.sg/v2/{data_type}?limit={limit}&page={page}"

def get_companies_url(limit=100, page=0):
    """Generate the API URL for companies endpoint (deprecated - use get_api_url)"""
    return get_api_url("companies", limit, page)

def get_headers():
    """Return the headers needed for the API request"""
    return {
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

def fetch_json_from_url(url, headers=None, max_retries=3, backoff_delay=1.0):
    """
    Fetch JSON data from a URL with retry logic and exponential backoff.
    
    Args:
        url (str): The URL to fetch data from
        headers (dict, optional): HTTP headers to include in the request
        max_retries (int): Maximum number of retry attempts (default: 3)
        backoff_delay (float): Initial delay in seconds between retries (default: 1.0)
    
    Returns:
        dict: JSON data if successful, None if failed after all retries
    """
    for attempt in range(max_retries + 1):  # +1 because first attempt is not a retry
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Check if response is empty
            if not response.content:
                print(f"Empty response from URL {url}")
                if attempt < max_retries:
                    delay = backoff_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    continue
                return None
                
            # Parse JSON
            data = orjson.loads(response.content)
            
            # Basic validation - ensure it's a dictionary
            if not isinstance(data, dict):
                print(f"Invalid response format from URL {url}: expected dict, got {type(data)}")
                if attempt < max_retries:
                    delay = backoff_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    continue
                return None
                
            # Success - return the data
            return data
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 'Unknown'
            print(f"HTTP error fetching URL {url}: {e} (Status: {status_code})")
            
            # Don't retry on client errors (4xx) except for rate limiting (429)
            if e.response and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                print(f"Client error {status_code} - not retrying")
                return None
                
            if attempt < max_retries:
                delay = backoff_delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                continue
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching URL {url}: {e}")
            if attempt < max_retries:
                delay = backoff_delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                continue
            return None
            
        except orjson.JSONDecodeError as e:
            print(f"JSON decode error from URL {url}: {e}")
            if attempt < max_retries:
                delay = backoff_delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                continue
            return None
            
        except Exception as e:
            print(f"Unexpected error fetching URL {url}: {e}")
            if attempt < max_retries:
                delay = backoff_delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                continue
            return None
    
    # This should never be reached, but just in case
    return None

def create_directory_structure(output_dir, data_type):
    """Create the directory structure for storing data"""
    today = datetime.now().strftime("%Y%m%d")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data type directory (e.g., companies)
    data_dir = os.path.join(output_dir, data_type)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create date directory
    date_dir = os.path.join(data_dir, today)
    os.makedirs(date_dir, exist_ok=True)
    
    # Create pages directory
    pages_dir = os.path.join(date_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    
    return pages_dir, today

def fetch_and_save_page(page, pages_dir, today, headers, data_type, delay=0.5):
    """
    Fetch and save a single page of data.
    
    Args:
        page (int): Page number to fetch
        pages_dir (str): Directory to save the file (pages directory)
        today (str): Today's date string for filename
        headers (dict): HTTP headers for the request
        data_type (str): Type of data to fetch (companies, jobs, etc.)
        delay (float): Delay after request to be respectful to API
    
    Returns:
        tuple: (page, success, num_results) where success is bool and num_results is int
    """
    global shutdown_requested
    thread_id = threading.current_thread().name
    
    # Check for shutdown request before starting
    if shutdown_requested:
        print(f"[{thread_id}] Shutdown requested, skipping page {page}")
        return page, False, 0
    
    # Check if file already exists
    filename = f"{today}_{data_type}_{page:04d}.json"
    filepath = os.path.join(pages_dir, filename)
    
    if os.path.exists(filepath):
        try:
            # Try to read existing file to get result count
            with open(filepath, 'rb') as f:
                existing_data = orjson.loads(f.read())
            num_results = len(existing_data.get('results', []))
            print(f"[{thread_id}] Page {page} already exists, skipping ({num_results} results)")
            return page, True, num_results
        except (orjson.JSONDecodeError, Exception) as e:
            print(f"[{thread_id}] Existing file {filename} is corrupted, re-downloading: {e}")
            # File exists but is corrupted, continue with download
    
    # Check for shutdown request before making API call
    if shutdown_requested:
        print(f"[{thread_id}] Shutdown requested, skipping page {page}")
        return page, False, 0
    
    print(f"[{thread_id}] Fetching page {page}...")
    
    # Make API request
    url = get_api_url(data_type, limit=100, page=page)
    data = fetch_json_from_url(url, headers)
    
    # Check if request failed
    if not data:
        print(f"[{thread_id}] Failed to fetch data for page {page}")
        return page, False, 0
    
    # Check for shutdown request before saving
    if shutdown_requested:
        print(f"[{thread_id}] Shutdown requested, not saving page {page}")
        return page, False, 0
    
    # Check if results are empty (end of pagination)
    if 'results' in data and len(data['results']) == 0:
        print(f"[{thread_id}] No results found for page {page}")
        return page, True, 0
    
    # Save the response to file
    try:
        with open(filepath, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        
        num_results = len(data.get('results', []))
        print(f"[{thread_id}] Saved page {page} to {filename} ({num_results} results)")
        
        # Add delay to be respectful to the API (but check for shutdown during delay)
        if delay > 0 and not shutdown_requested:
            time.sleep(delay)
        
        return page, True, num_results
        
    except Exception as e:
        print(f"[{thread_id}] Error saving page {page}: {e}")
        return page, False, 0

def fetch_data(data_type, output_dir, delay=0.5, num_threads=4, max_pages=None, start_page=0):
    """
    Fetch data from the API with pagination using multiple threads.
    
    Args:
        data_type (str): Type of data to fetch (companies, jobs, etc.)
        output_dir (str): Directory to save data
        delay (float): Delay between requests per thread
        num_threads (int): Number of worker threads
        max_pages (int, optional): Maximum number of pages to fetch (for testing)
        start_page (int): Page number to start fetching from (default: 0)
    """
    global shutdown_requested
    print(f"Fetching {data_type} data using {num_threads} threads starting from page {start_page}...")
    print("Press Ctrl+C to gracefully stop the program")
    
    # Create directory structure
    pages_dir, today = create_directory_structure(output_dir, data_type)
    headers = get_headers()
    
    # First, do a quick check to estimate total pages by fetching page 0
    print("Estimating total pages...")
    test_data = fetch_json_from_url(get_api_url(data_type, limit=100, page=0), headers)
    if not test_data:
        print("Failed to fetch initial page. Aborting.")
        return
    
    # Check for early shutdown
    if shutdown_requested:
        print("Shutdown requested during initialization. Exiting.")
        return
    
    # Estimate total pages (this is approximate)
    total_results = test_data.get('total', 0)
    estimated_pages = (total_results // 100) + 1 if total_results > 0 else 100
    
    # Adjust for start page
    if start_page >= estimated_pages:
        print(f"Start page {start_page} is beyond estimated total pages {estimated_pages}. Aborting.")
        return
    
    # Calculate actual pages to fetch
    pages_to_fetch = estimated_pages - start_page
    if max_pages:
        pages_to_fetch = min(pages_to_fetch, max_pages)
    
    print(f"Estimated total pages: {estimated_pages}")
    print(f"Starting from page: {start_page}")
    print(f"Pages to fetch: {pages_to_fetch}")
    
    # Create thread pool and submit jobs
    successful_pages = 0
    empty_pages = 0
    failed_pages = 0
    cancelled_pages = 0
    
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit jobs: start from start_page and fetch pages_to_fetch pages
            futures = {}
            for i in range(pages_to_fetch):
                if shutdown_requested:
                    print("Shutdown requested, not submitting more jobs")
                    break
                page = start_page + i
                future = executor.submit(fetch_and_save_page, page, pages_dir, today, headers, data_type, delay)
                futures[future] = page
            
            # Process completed jobs
            for future in as_completed(futures):
                page = futures[future]
                try:
                    page_num, success, num_results = future.result()
                    if success:
                        if num_results > 0:
                            successful_pages += 1
                        else:
                            empty_pages += 1
                            # If we hit multiple empty pages, we might be done
                            if empty_pages >= num_threads:
                                print("Multiple threads found empty pages. Likely reached end of data.")
                    else:
                        if shutdown_requested:
                            cancelled_pages += 1
                        else:
                            failed_pages += 1
                except Exception as e:
                    print(f"Error processing page {page}: {e}")
                    failed_pages += 1
                
                # Check if we should stop processing
                if shutdown_requested:
                    print("Shutdown requested, waiting for remaining threads to complete...")
                    break
    
    except KeyboardInterrupt:
        # This shouldn't happen due to signal handling, but just in case
        print("\nForced interruption detected!")
        shutdown_requested = True
    
    # Final status report
    total_processed = successful_pages + empty_pages + failed_pages + cancelled_pages
    
    if shutdown_requested:
        print(f"\n{'='*50}")
        print("PROGRAM INTERRUPTED")
        print(f"{'='*50}")
    else:
        print(f"\nCompleted fetching {data_type} data:")
    
    print(f"  Successful pages: {successful_pages}")
    print(f"  Empty pages: {empty_pages}")
    print(f"  Failed pages: {failed_pages}")
    if cancelled_pages > 0:
        print(f"  Cancelled pages: {cancelled_pages}")
    print(f"  Total pages processed: {total_processed}")
    
    if shutdown_requested:
        print(f"\nYou can resume from page {start_page + total_processed} using:")
        print(f"  python {sys.argv[0]} {data_type} --start-page {start_page + total_processed}")
        if max_pages:
            remaining_pages = max_pages - total_processed
            if remaining_pages > 0:
                print(f"  --max-pages {remaining_pages}")
        print(f"  --output-dir {output_dir}")
        if delay != 0.5:
            print(f"  --delay {delay}")
        if num_threads != 4:
            print(f"  --threads {num_threads}")
        print("\nAll completed downloads have been saved successfully.")

def fetch_companies_data(output_dir, delay=0.5, num_threads=4, max_pages=None, start_page=0):
    """
    Fetch companies data from the API with pagination using multiple threads.
    (Deprecated - use fetch_data instead)
    """
    return fetch_data("companies", output_dir, delay, num_threads, max_pages, start_page)

def main():
    parser = argparse.ArgumentParser(description='Fetch data from MyCareersFuture API')
    parser.add_argument('data_type', choices=['companies', 'jobs'], 
                       help='Type of data to fetch (companies or jobs)')
    parser.add_argument('--output-dir', default='raw_data',
                       help='Output directory for storing data (default: raw_data)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API requests in seconds per thread (default: 0.5)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    parser.add_argument('--max-pages', type=int, default=None,
                       help='Maximum number of pages to fetch (for testing)')
    parser.add_argument('--start-page', type=int, default=0,
                       help='Page number to start fetching from (default: 0)')
    
    args = parser.parse_args()
    
    # Use the generic fetch_data function for all data types
    fetch_data(args.data_type, args.output_dir, args.delay, args.threads, args.max_pages, args.start_page)

if __name__ == "__main__":
    setup_signal_handlers()
    main()