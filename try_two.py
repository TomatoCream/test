import argparse
import json
from pathlib import Path
# pandas import removed as it's not used in this version.

# Assuming schema.py is in the same directory or accessible via PYTHONPATH
from schema import JobSearchResponse, JobResult # Consolidated imports
from get_data import fetch_job_details # Import the function from get_data.py

def main():
    parser = argparse.ArgumentParser(
        description="Fetch a job detail from API using UUID from a local file and compare."
    )
    parser.add_argument(
        "date",
        type=str,
        help="Date in YYYYMMDD format, corresponding to the subdirectory in raw_data_root."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="raw_data",
        help="Root directory containing dated subdirectories of JSON files. Default: 'raw_data'."
    )
    # Output argument removed as it's not used for this task.
    args = parser.parse_args()

    data_path = Path(args.directory) / args.date

    if not data_path.is_dir():
        print(f"Error: Directory not found: {data_path}")
        return

    print(f"Searching for JSON files in: {data_path}")
    
    # Try to get the first JSON file in the directory
    try:
        first_json_file = next(data_path.glob("*.json"))
    except StopIteration:
        print(f"Error: No JSON files found in {data_path}")
        return

    print(f"Using data from the first JSON file found: {first_json_file.name}")

    try:
        with open(first_json_file, 'r', encoding='utf-8') as f:
            file_data_json = json.load(f)
        
        # Parse the entire file content as JobSearchResponse
        parsed_file_content = JobSearchResponse.model_validate(file_data_json)

        if not parsed_file_content.results:
            print(f"No job results found in the file: {first_json_file.name}")
            return

        # Take the first job result from the file
        job_from_file_obj = parsed_file_content.results[0]
        job_uuid_to_fetch = job_from_file_obj.uuid

        print(f"\n--- Original Job Data (from file: {first_json_file.name}) ---")
        print(f"UUID: {job_uuid_to_fetch}")
        print(f"Title: {job_from_file_obj.title}")
        print(f"Company: {job_from_file_obj.postedCompany.name}")
        # print(job_from_file_obj.model_dump_json(indent=2, exclude_none=True))


        # Fetch job details from API using the extracted UUID
        print(f"\nFetching job details from API for UUID: {job_uuid_to_fetch}...")
        # Using a small delay for this single test call, default is 3s in get_data
        api_response_data = fetch_job_details(job_uuid=job_uuid_to_fetch, delay=1) 

        if api_response_data:
            # Parse the API response. Assuming the API returns a single JobResult structure.
            job_detail_from_api_obj = JobResult.model_validate(api_response_data)
            
            print(f"\n--- Fetched Job Detail Data (from API) ---")
            print(f"UUID: {job_detail_from_api_obj.uuid}")
            print(f"Title: {job_detail_from_api_obj.title}")
            print(f"Company: {job_detail_from_api_obj.postedCompany.name}")
            print(job_detail_from_api_obj.model_dump_json(indent=2, exclude_none=True))
            
            # Basic comparison
            print("\n--- Comparison Highlights ---")
            print(f"{'Field':<25} | {'From File':<50} | {'From API':<50}")
            print("-" * 130)
            
            def safe_get_attr(obj, attr_path, default="N/A"):
                try:
                    current = obj
                    for part in attr_path.split('.'):
                        current = getattr(current, part)
                    return str(current) if current is not None else default
                except AttributeError:
                    return default

            fields_to_compare = {
                "UUID": "uuid",
                "Title": "title",
                "Company Name": "postedCompany.name",
                "New Posting Date": "metadata.newPostingDate",
                "Applications": "metadata.totalNumberJobApplication",
                "Updated At": "metadata.updatedAt",
                "Salary Min": "salary.minimum",
                "Salary Max": "salary.maximum",
                "Job Status": "status.jobStatus"
            }

            for display_name, attr_path in fields_to_compare.items():
                file_val = safe_get_attr(job_from_file_obj, attr_path)
                api_val = safe_get_attr(job_detail_from_api_obj, attr_path)
                # Truncate long values for display
                file_val_disp = (file_val[:47] + '...') if len(file_val) > 50 else file_val
                api_val_disp = (api_val[:47] + '...') if len(api_val) > 50 else api_val
                print(f"{display_name:<25} | {file_val_disp:<50} | {api_val_disp:<50}")

            if job_from_file_obj.metadata.totalNumberJobApplication != job_detail_from_api_obj.metadata.totalNumberJobApplication:
                print(f"\nNote: 'totalNumberJobApplication' differs: File={job_from_file_obj.metadata.totalNumberJobApplication}, API={job_detail_from_api_obj.metadata.totalNumberJobApplication}")
            if job_from_file_obj.metadata.updatedAt != job_detail_from_api_obj.metadata.updatedAt:
                 print(f"Note: 'updatedAt' differs: File={job_from_file_obj.metadata.updatedAt}, API={job_detail_from_api_obj.metadata.updatedAt}")
            
            # For a quick check if other parts are identical, can compare model dumps.
            # Excluding metadata as it's likely to change (updatedAt, totalApplicationCount)
            file_dump = job_from_file_obj.model_dump(exclude={'metadata'}, exclude_none=True)
            api_dump = job_detail_from_api_obj.model_dump(exclude={'metadata'}, exclude_none=True)
            if file_dump == api_dump:
                print("\nNote: Other parts of the job data (excluding metadata) appear identical.")
            else:
                print("\nNote: Other parts of the job data (excluding metadata) have differences.")

        else:
            print(f"Failed to fetch job details from API for UUID: {job_uuid_to_fetch}.")

    except FileNotFoundError:
        # This case should be caught by the earlier check, but good to have.
        print(f"Error: The file {data_path / '<filename>'} was not found.") # Corrected placeholder
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file: {first_json_file.name}") # Added filename
    except StopIteration: 
        # This should ideally be caught by the initial check for first_json_file
        print(f"Error: No JSON files found in {data_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
