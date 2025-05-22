import argparse
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from datetime import datetime
import numpy as np

# Assuming schema.py is in the same directory or accessible via PYTHONPATH
from schema import (
    JobSearchResponse, PostedCompany
)

def main():
    parser = argparse.ArgumentParser(
        description="Investigate JSON data from MyCareersFuture job postings for consistency and reusability."
    )
    parser.add_argument(
        "date",
        type=str,
        help="Date in YYYYMMDD format, corresponding to the subdirectory in the raw_data_root or for naming Parquet files."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="raw_data",
        help="Root directory containing dated subdirectories of JSON files. Default: 'raw_data'."
    )
    parser.add_argument(
        "--database_directory",
        type=str,
        default="parquet_data",
        help="Directory to load existing Parquet files from and save new ones to. Default: 'parquet_data'."
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version number for this crawl. Default: 1."
    )
    args = parser.parse_args()

    data_path = Path(args.directory) / args.date
    db_dir = Path(args.database_directory)
    posted_company_parquet_file = db_dir / f"{args.date}-posted_companies.parquet"
    
    # Convert date string to standard format for storage
    crawl_date = args.date
    version = args.version

    # Initialize posted_company_df
    company_columns = list(PostedCompany.model_fields.keys())
    # Add crawl_date and version columns
    if 'crawl_date' not in company_columns:
        company_columns.append('crawl_date')
    if 'version' not in company_columns:
        company_columns.append('version')
    
    try:
        print(f"Attempting to load existing data from {posted_company_parquet_file}")
        posted_company_df = pd.read_parquet(posted_company_parquet_file)
        
        # Check and add required columns if missing
        for col in ['uen', 'crawl_date', 'version']:
            if col not in posted_company_df.columns:
                if col == 'crawl_date':
                    posted_company_df[col] = crawl_date
                elif col == 'version':
                    posted_company_df[col] = version
                else:
                    print(f"Warning: '{col}' column not found in {posted_company_parquet_file}. Creating new DataFrame.")
                    posted_company_df = pd.DataFrame(columns=company_columns)
                    break
        
        # Set up multi-index
        if not posted_company_df.empty:
            posted_company_df = posted_company_df.set_index(['uen', 'crawl_date', 'version'])
        
        # Ensure all model columns exist, add if missing (e.g. schema evolution)
        for col in company_columns:
            if col not in ['uen', 'crawl_date', 'version'] and col not in posted_company_df.columns:
                posted_company_df[col] = None
        
        print(f"Successfully loaded {len(posted_company_df)} records.")
    except FileNotFoundError:
        print(f"No existing Parquet file found at {posted_company_parquet_file}. Creating a new DataFrame.")
        posted_company_df = pd.DataFrame(columns=company_columns)
        # Initialize with empty dataframe with MultiIndex
        posted_company_df = posted_company_df.set_index(['uen', 'crawl_date', 'version'])
    except Exception as e:
        print(f"Error loading Parquet file {posted_company_parquet_file}: {e}. Creating a new DataFrame.")
        posted_company_df = pd.DataFrame(columns=company_columns)
        # Initialize with empty dataframe with MultiIndex
        posted_company_df = posted_company_df.set_index(['uen', 'crawl_date', 'version'])

    if not data_path.is_dir():
        print(f"Error: Data source directory not found: {data_path}")
        # Allow script to continue if only using database_directory for loading/saving

    print(f"Processing JSON files in: {data_path}")
    file_count = 0
    json_files_exist = any(data_path.glob("*.json"))

    if not json_files_exist and not posted_company_parquet_file.exists():
        print(f"No JSON files found in {data_path} and no existing Parquet file to process.")
    elif not json_files_exist and posted_company_parquet_file.exists():
        print(f"No JSON files found in {data_path}. Will only save existing loaded data if modified (or just resave).")

    new_companies = {}  # Dictionary to collect companies by UEN during JSON processing

    for json_file_path in data_path.glob("*.json"):
        if file_count > 10:
            break
        file_count += 1
        print(f"--- Processing file: {json_file_path.name} ---")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data_str = f.read()
            
            job_response = JobSearchResponse.model_validate_json(json_data_str)

            for job_result in job_response.results:
                if job_result.postedCompany:
                    company = job_result.postedCompany
                    uen = company.uen
                    
                    # Store the company by UEN for later processing
                    if uen not in new_companies:
                        new_companies[uen] = company

        except FileNotFoundError: 
            print(f"Error: File disappeared during processing: {json_file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {json_file_path}")
        except Exception as e: 
            print(f"An error occurred while processing file '{json_file_path}': {e}")
            import traceback
            traceback.print_exc()
    
    if file_count == 0 and json_files_exist:
         print(f"No JSON files were processed despite being found initially in {data_path}.")
    elif file_count == 0 and not json_files_exist:
         print(f"No JSON files found in {data_path} to process.")

    # Process the collected companies
    processed_count = 0
    for uen, company in new_companies.items():
        processed_count += 1
        
        # Initialize next_version with the current crawl's version
        # This will be updated if existing versions are found
        next_version = version 

        try:
            # Attempt to get existing records for this UEN and crawl date
            # This will raise a KeyError if (uen, crawl_date) is not in the index
            uen_records = posted_company_df.loc[(uen, crawl_date)]

            # Convert to DataFrame if it's a Series (single record)
            if isinstance(uen_records, pd.Series):
                # Only one record exists, convert to DataFrame
                # The 'version' is in the name of the Series (from the MultiIndex)
                series_version = uen_records.name # This should be the version from the index
                uen_records_df = pd.DataFrame([uen_records])
                # uen_records_df.index = [series_version] # Set index to be the version
                # uen_records_df = uen_records_df.rename_axis('version')
                uen_records_df['version'] = series_version # Correctly assign version from Series name
                uen_records = uen_records_df.reset_index(drop=True) # Ensure 'version' is a column
            else:
                # Already a DataFrame, reset the MultiIndex to get 'version' as a column
                uen_records = uen_records.reset_index()
            
            # Ensure 'version' column exists and is of a suitable type if uen_records was not empty
            if not uen_records.empty and 'version' not in uen_records.columns:
                # This case should ideally not happen if indexing is correct
                # but as a safeguard, if 'version' is missing from a multi-record df, re-evaluate logic
                print(f"Warning: 'version' column missing for UEN {uen} after .loc. Re-fetching or assuming current version.")
                # Fallback: If version column is unexpectedly missing, could try to re-add it based on args.version
                # For now, let it proceed, sort_values might fail or behave unexpectedly.
                # Or, treat as if no valid existing records found, and create new one.
                # For this patch, we'll assume this state implies new record if sort fails.
                pass


            if not uen_records.empty:
                uen_records = uen_records.sort_values('version')
                
                # Get the latest version number from the sorted records
                latest_version = uen_records['version'].max()
                next_version = latest_version + 1 # Update next_version based on existing records
                
                # Build a complete record by merging all versions
                complete_record = {}
                for _, record_row in uen_records.iterrows(): # Renamed 'record' to 'record_row' to avoid conflict
                    for col in record_row.index:
                        # Check if the value is NA. pd.isna can return a boolean Series/array if record_row[col] is array-like.
                        # We consider a value non-NA if it's not a scalar NA.
                        # An empty array or an array with NAs is not itself a scalar NA.
                        is_na_check = pd.isna(record_row[col])
                        if col != 'version':
                            if isinstance(is_na_check, np.bool_): # Scalar boolean from pd.isna (e.g. for np.nan, None)
                                if not is_na_check: # If it's False (not NA)
                                    complete_record[col] = record_row[col]
                            else: # It's an array-like (e.g. np.array([]), or np.array([1, np.nan])) - not a scalar NA
                                complete_record[col] = record_row[col]
                
                complete_record['uen'] = uen
                try:
                    existing_company_model = PostedCompany(**complete_record)
                    
                    if company != existing_company_model:
                        print(f"Difference found for PostedCompany with UEN '{uen}', creating new version {next_version}.")
                        new_dump = company.model_dump()
                        old_dump = existing_company_model.model_dump()
                        diff_record = {col: None for col in company_columns if col not in ['uen', 'crawl_date', 'version']}
                        for key in new_dump.keys():
                            if key != 'uen' and new_dump.get(key) != old_dump.get(key):
                                diff_record[key] = new_dump.get(key)
                                print(f"  Difference in '{key}':")
                                print(f"    Previous: {old_dump.get(key)}")
                                print(f"    New:      {new_dump.get(key)}")
                        
                        idx = (uen, crawl_date, next_version)
                        posted_company_df.loc[idx] = pd.Series(diff_record)
                        print(f"Added new version {next_version} for UEN: {uen}, crawl_date: {crawl_date}")
                    else:
                        print(f"No differences found for UEN '{uen}', not creating new version.")
                except Exception as ex_pydantic:
                    print(f"Error reconstructing/comparing PostedCompany from DataFrame for UEN {uen}: {ex_pydantic}")
                    new_company_data = company.model_dump()
                    idx = (uen, crawl_date, next_version) # Use updated next_version
                    row_data = {k: v for k, v in new_company_data.items() if k != 'uen'}
                    posted_company_df.loc[idx] = pd.Series(row_data)
                    print(f"Added new version {next_version} for UEN: {uen} with complete data due to comparison error.")
            else: # uen_records is empty (either from .loc or after processing if it became empty)
                print(f"No existing valid versioned records found for UEN {uen}, crawl_date {crawl_date}. Adding as new with version {version}.")
                new_company_data = company.model_dump()
                idx = (uen, crawl_date, version) # Use current crawl's version
                row_data = {k: v for k, v in new_company_data.items() if k != 'uen'}
                posted_company_df.loc[idx] = pd.Series(row_data)

        except KeyError: # This means (uen, crawl_date) was not found in the index
            # No existing records for this UEN and crawl_date, add as new
            new_company_data = company.model_dump()
            idx = (uen, crawl_date, version) # Use current crawl's version
            row_data = {k: v for k, v in new_company_data.items() if k != 'uen'}
            posted_company_df.loc[idx] = pd.Series(row_data)
            print(f"New PostedCompany added (due to KeyError) with UEN: {uen}, crawl_date: {crawl_date}, version: {version}")
        
        except Exception as e:
            print(f"General error processing UEN {uen}, crawl_date {crawl_date}: {e}")
            # Fall back to adding a new version with all data, using the initial 'version' for this crawl
            import traceback
            traceback.print_exc()
            new_company_data = company.model_dump()
            idx = (uen, crawl_date, version) # Fallback to current crawl's version
            row_data = {k: v for k, v in new_company_data.items() if k != 'uen'}
            posted_company_df.loc[idx] = pd.Series(row_data)
            print(f"Added new version {version} for UEN: {uen} with complete data due to general processing error.")

    # Saving logic using args.database_directory
    print(f"\nSaving PostedCompany data as Parquet file in directory: {db_dir}")
    try:
        db_dir.mkdir(parents=True, exist_ok=True)

        if not posted_company_df.empty:
            df_to_save = posted_company_df.reset_index()  # Make 'uen', 'crawl_date', 'version' columns for saving
            output_parquet_path = db_dir / f"{args.date}-posted_companies.parquet"
            df_to_save.to_parquet(output_parquet_path, index=False)
            print(f"  Saved {output_parquet_path.name}")
        else:
            print(f"  Skipping {args.date}-posted_companies.parquet as no data was found/generated.")
        
        print(f"Successfully processed Parquet saving to {db_dir}")
    except Exception as e:
        print(f"Error saving Parquet data to '{db_dir}': {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Investigation Complete ---")
    if json_files_exist or file_count > 0:
        print(f"Processed {file_count} JSON file(s) from {data_path}.")
    print(f"Total unique PostedCompany entries stored: {len(posted_company_df)}")
    print(f"Processed {processed_count} companies from JSON files.")

if __name__ == "__main__":
    main()