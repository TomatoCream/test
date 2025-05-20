import argparse
import json
from pathlib import Path
from typing import Any, Dict

# Assuming schema.py is in the same directory or accessible via PYTHONPATH
from schema import (
    JobSearchResponse, Metadata, Districts, PositionLevel,
    PostedCompany, Skill, JobEmploymentType, Category, Status
)

# Global maps to store the first encountered instance of each item type
metadata_map: Dict[str, Metadata] = {}
districts_map: Dict[Any, Districts] = {}  # Key can be int or str based on schema
position_levels_map: Dict[int, PositionLevel] = {}
posted_company_map: Dict[str, PostedCompany] = {}
skills_map: Dict[str, Skill] = {}
employment_types_map: Dict[Any, JobEmploymentType] = {} # Key can be int or str
categories_map: Dict[int, Category] = {}
status_map: Dict[Any, Status] = {} # Key can be Union[str, int]

def process_item(item: Any, item_map: Dict[Any, Any], item_key_value: Any, item_name: str):
    """
    Processes an item, checks for duplicates, and records differences.
    Compares the current item against the first item encountered with the same key.
    """
    if item_key_value in item_map:
        existing_item = item_map[item_key_value]
        if item != existing_item:
            print(f"Difference found for {item_name} with key '{item_key_value}'.")
            # To see the actual differences, you could add:
            # print(f"  Existing: {existing_item.model_dump(mode='json')}")
            # print(f"  New:      {item.model_dump(mode='json')}")
    else:
        item_map[item_key_value] = item

def main():
    parser = argparse.ArgumentParser(
        description="Investigate JSON data from MyCareersFuture job postings for consistency and reusability."
    )
    parser.add_argument(
        "date",
        type=str,
        help="Date in YYYYMMDD format, corresponding to the subdirectory in the raw_data_root."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="raw_data",
        help="Root directory containing dated subdirectories of JSON files. Default: 'raw_data'."
    )
    args = parser.parse_args()

    data_path = Path(args.directory) / args.date

    if not data_path.is_dir():
        print(f"Error: Directory not found: {data_path}")
        return

    print(f"Processing JSON files in: {data_path}")
    file_count = 0

    for json_file_path in data_path.glob("*.json"):
        file_count += 1
        print(f"--- Processing file: {json_file_path.name} ---")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data_str = f.read()
            
            job_response = JobSearchResponse.model_validate_json(json_data_str)

            for job_result in job_response.results:
                # 1. Metadata
                if job_result.metadata:
                    process_item(job_result.metadata, metadata_map, job_result.metadata.jobPostId, "Metadata")

                # 2. Districts (list within Address)
                if job_result.address and job_result.address.districts:
                    for district in job_result.address.districts:
                        process_item(district, districts_map, district.id, "District")
                
                # 3. PositionLevels (list)
                if job_result.positionLevels:
                    for level in job_result.positionLevels:
                        process_item(level, position_levels_map, level.id, "PositionLevel")

                # 4. PostedCompany
                if job_result.postedCompany:
                    process_item(job_result.postedCompany, posted_company_map, job_result.postedCompany.uen, "PostedCompany")

                # 5. Skills (list)
                if job_result.skills:
                    for skill_item in job_result.skills:
                        process_item(skill_item, skills_map, skill_item.uuid, "Skill")

                # 6. EmploymentTypes (list)
                if job_result.employmentTypes:
                    for emp_type in job_result.employmentTypes:
                        process_item(emp_type, employment_types_map, emp_type.id, "JobEmploymentType")
                
                # 7. Categories (list)
                if job_result.categories:
                    for category_item in job_result.categories:
                        process_item(category_item, categories_map, category_item.id, "Category")

                # 8. Status
                if job_result.status:
                    process_item(job_result.status, status_map, job_result.status.id, "Status")

        except FileNotFoundError: # Should ideally not be reached if glob works
            print(f"Error: File disappeared during processing: {json_file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {json_file_path}")
        except Exception as e: # Catches Pydantic's ValidationError and other unexpected errors
            print(f"An error occurred while processing file '{json_file_path}': {e}")
            # For detailed debugging, uncomment the following:
            # import traceback
            # traceback.print_exc()
    
    if file_count == 0:
        print(f"No JSON files found in {data_path}.")

    print("\n--- Investigation Complete ---")
    print(f"Processed {file_count} JSON file(s).")
    print(f"Total unique Metadata entries stored: {len(metadata_map)}")
    print(f"Total unique District entries stored: {len(districts_map)}")
    print(f"Total unique PositionLevel entries stored: {len(position_levels_map)}")
    print(f"Total unique PostedCompany entries stored: {len(posted_company_map)}")
    print(f"Total unique Skill entries stored: {len(skills_map)}")
    print(f"Total unique JobEmploymentType entries stored: {len(employment_types_map)}")
    print(f"Total unique Category entries stored: {len(categories_map)}")
    print(f"Total unique Status entries stored: {len(status_map)}")

if __name__ == "__main__":
    main() 