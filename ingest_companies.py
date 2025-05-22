import argparse
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

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
    args = parser.parse_args()

    data_path = Path(args.directory) / args.date
    db_dir = Path(args.database_directory)
    posted_company_parquet_file = db_dir / f"{args.date}-posted_companies.parquet"

    # Initialize posted_company_df
    company_columns = list(PostedCompany.model_fields.keys())
    try:
        print(f"Attempting to load existing data from {posted_company_parquet_file}")
        posted_company_df = pd.read_parquet(posted_company_parquet_file)
        if 'uen' not in posted_company_df.columns:
            print(f"Warning: 'uen' column not found in {posted_company_parquet_file}. Creating new DataFrame.")
            posted_company_df = pd.DataFrame(columns=company_columns)
            posted_company_df = posted_company_df.set_index('uen')
        else:
            posted_company_df = posted_company_df.set_index('uen')
            # Ensure all model columns exist, add if missing (e.g. schema evolution)
            for col in company_columns:
                if col != 'uen' and col not in posted_company_df.columns:
                    posted_company_df[col] = None # Or pd.NA
            print(f"Successfully loaded {len(posted_company_df)} records.")
    except FileNotFoundError:
        print(f"No existing Parquet file found at {posted_company_parquet_file}. Creating a new DataFrame.")
        posted_company_df = pd.DataFrame(columns=company_columns)
        if 'uen' in posted_company_df.columns: # Should be true if company_columns from model
            posted_company_df = posted_company_df.set_index('uen')
        else: # Should not happen if PostedCompany has uen
            print("Error: 'uen' field not found in PostedCompany model for index.")
            return # Or raise error
    except Exception as e:
        print(f"Error loading Parquet file {posted_company_parquet_file}: {e}. Creating a new DataFrame.")
        posted_company_df = pd.DataFrame(columns=company_columns)
        if 'uen' in posted_company_df.columns:
            posted_company_df = posted_company_df.set_index('uen')
        else:
            print("Error: 'uen' field not found in PostedCompany model for index during error recovery.")
            return

    print(posted_company_df.describe())

    if not data_path.is_dir():
        print(f"Error: Data source directory not found: {data_path}")
        # Allow script to continue if only using database_directory for loading/saving
        # return 

    print(f"Processing JSON files in: {data_path}")
    file_count = 0
    json_files_exist = any(data_path.glob("*.json"))

    if not json_files_exist and not posted_company_parquet_file.exists():
        print(f"No JSON files found in {data_path} and no existing Parquet file to process.")
        # Potentially exit if there's nothing to do at all
        # return
    elif not json_files_exist and posted_company_parquet_file.exists():
        print(f"No JSON files found in {data_path}. Will only save existing loaded data if modified (or just resave).")


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
                    new_company_data = company.model_dump()

                    if uen in posted_company_df.index:
                        existing_series = posted_company_df.loc[uen]
                        # Convert series to dict, replacing NaNs with Nones for Pydantic model reconstruction
                        existing_data_dict = existing_series.where(pd.notna(existing_series), None).to_dict()
                        existing_data_dict['uen'] = uen # Add uen back as it's the index name

                        try:
                            existing_company_model = PostedCompany(**existing_data_dict)
                            if company != existing_company_model:
                                print(f"Difference found for PostedCompany with UEN '{uen}'.")
                                new_dump = company.model_dump()
                                old_dump = existing_company_model.model_dump()
                                all_keys = sorted(list(set(new_dump.keys()) | set(old_dump.keys())))
                                for key in all_keys:
                                    val_new = new_dump.get(key)
                                    val_old = old_dump.get(key)
                                    if val_new != val_old:
                                        print(f"  Mismatch in '{key}':")
                                        print(f"    DF (existing): {val_old}")
                                        print(f"    JSON (new):    {val_new}")
                                
                                # Simply append the new company data to the DataFrame
                                # We'll need to reset the index first to avoid uniqueness constraints
                                posted_company_df = posted_company_df.reset_index()
                                new_row = pd.DataFrame([new_dump])
                                posted_company_df = pd.concat([posted_company_df, new_row], ignore_index=True)
                                # Set back the UEN as index (non-unique)
                                posted_company_df = posted_company_df.set_index('uen')
                                print(f"Added new row for PostedCompany with UEN: {uen}")
                        except Exception as ex_pydantic:
                            print(f"Error reconstructing/comparing PostedCompany from DataFrame for UEN {uen}: {ex_pydantic}")
                            print(f"  Existing data from DF: {existing_data_dict}")
                            print(f"  New data from JSON: {new_company_data}")
                    else:
                        # Add new company to DataFrame
                        row_data = {k: v for k, v in new_company_data.items() if k != 'uen'}
                        posted_company_df.loc[uen] = pd.Series(row_data)
                        print(f"New PostedCompany added with UEN: {uen}")

        except FileNotFoundError: 
            print(f"Error: File disappeared during processing: {json_file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {json_file_path}")
        except Exception as e: 
            print(f"An error occurred while processing file '{json_file_path}': {e}")
            import traceback
            traceback.print_exc()
    
    if file_count == 0 and json_files_exist: # Should not happen if glob worked and then files disappeared
         print(f"No JSON files were processed despite being found initially in {data_path}.")
    elif file_count == 0 and not json_files_exist:
         print(f"No JSON files found in {data_path} to process.")


    # Saving logic using args.database_directory
    print(f"\nSaving PostedCompany data as Parquet file in directory: {db_dir}")
    try:
        db_dir.mkdir(parents=True, exist_ok=True)

        if not posted_company_df.empty:
            df_to_save = posted_company_df.reset_index() # Make 'uen' a column for saving
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
    if json_files_exist or file_count > 0 :
        print(f"Processed {file_count} JSON file(s) from {data_path}.")
    print(f"Total unique PostedCompany entries stored: {len(posted_company_df)}")

if __name__ == "__main__":
    main()