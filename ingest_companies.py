import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Type, Tuple
import pandas as pd
from datetime import datetime
import numpy as np
import traceback

# Assuming schema.py is in the same directory or accessible via PYTHONPATH
from schema import (
    JobSearchResponse, PostedCompany, BaseModel
)

def merge_df_rows_by_sort_key(
    df: pd.DataFrame, 
    sort_key_columns: list[str]
) -> Dict[str, Any]:
    """
    Sorts a DataFrame by specified key columns and merges rows into a single record.
    The merge strategy is to take the value from the "latest" row (after sorting 
    in ascending order) for each column, provided the value is not NA. 
    The sort key columns themselves are excluded from the merged record's fields.

    Args:
        df: DataFrame containing multiple rows to be merged.
        sort_key_columns: A list of column names to sort by.

    Returns:
        A dictionary representing the merged record. Returns an empty dict if df is empty.
    """
    if df.empty:
        return {}

    # Ensure sort_key_columns exist. If not, sort_values will raise KeyError.
    # Pandas sorts NaNs last by default with ascending=True, which is generally fine.
    sorted_df = df.sort_values(by=sort_key_columns, ascending=True)

    merged_data: Dict[str, Any] = {}
    for _idx, row in sorted_df.iterrows():
        for column_name, value in row.items():
            if column_name in sort_key_columns:  # Exclude the sort key columns
                continue
            
            is_na_check = pd.isna(value)
            # If 'value' is a scalar (e.g., number, string, bool, np.nan, None):
            if isinstance(is_na_check, (bool, np.bool_)): 
                if not is_na_check:  # Value is not NA
                    merged_data[column_name] = value
            # If 'value' is array-like (e.g., list, np.array), keep it as is.
            # This matches original logic where non-scalar NAs were assigned.
            else: 
                merged_data[column_name] = value
                
    return merged_data

def get_attribute_changes(
    old_attributes: Dict[str, Any],
    new_attributes: Dict[str, Any],
    ignore_keys: list[str] = None
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any], bool]:
    """
    Compares two dictionaries of attributes, ignoring specified keys.

    Args:
        old_attributes: Dictionary of old attribute values.
        new_attributes: Dictionary of new attribute values.
        ignore_keys: List of keys to ignore during comparison (e.g., primary identifiers).
                     These keys will not appear in the returned difference reports.

    Returns:
        A tuple containing:
        - detailed_diffs (Dict[str, Dict[str, Any]]): 
            {'attribute_name': {'previous': old_val, 'new': new_val}}
        - changed_attributes_new_values (Dict[str, Any]): 
            {'attribute_name': new_val} for changed attributes.
        - has_changes (bool): True if any differences were found.
    """
    if ignore_keys is None:
        ignore_keys = []
        
    detailed_diffs = {}
    changed_attributes_new_values = {}
    has_changes = False

    all_keys = set(old_attributes.keys()) | set(new_attributes.keys())

    for key in all_keys:
        if key in ignore_keys:
            continue

        old_val = old_attributes.get(key)
        new_val = new_attributes.get(key)

        # Note: This basic inequality might need adjustment for complex types 
        # or specific NaN handling if not already addressed by model_dump().
        if old_val != new_val:
            has_changes = True
            detailed_diffs[key] = {'previous': old_val, 'new': new_val}
            changed_attributes_new_values[key] = new_val
            
    return detailed_diffs, changed_attributes_new_values, has_changes

# New function to process item versioning
def _process_item_versioning(
    df: pd.DataFrame,
    key_name: str,
    key_value: Any,
    item_model_instance: BaseModel,
    pydantic_model: Type[BaseModel],
    all_df_columns: List[str],
    crawl_date: str,
    current_version: int,
    diff_summary_list: List[Dict[str, Any]],
    debug_mode: bool = False
) -> Tuple[int, int]:
    """
    Processes a single item for versioning, comparing it with existing records in the DataFrame.

    Args:
        df: The main DataFrame holding all versioned items. Its index is (key_name, 'crawl_date', 'version').
            The columns of df are the fields of the Pydantic model, excluding key_name, 'crawl_date', and 'version'.
        key_name: The name of the primary key column (e.g., 'uen').
        key_value: The value of the primary key for the current item.
        item_model_instance: The Pydantic model instance of the new item data.
        pydantic_model: The Pydantic model class (e.g., PostedCompany).
        all_df_columns: A list of all column names that *could* be in the DataFrame if it were fully populated
                        (including key_name, crawl_date, version, and all Pydantic model fields).
                        Used for reconstructing records and series.
        crawl_date: The current crawl date.
        current_version: The version number of the current crawl (used for new items or as a base).
        diff_summary_list: A list to append difference summaries if debug_mode is True.
        debug_mode: If True, enables detailed logging and difference summary collection.

    Returns:
        A tuple containing:
        - errors_encountered (int): Number of errors during processing this item.
        - new_items_added (int): 1 if the item was newly added, 0 otherwise.
    """
    errors_encountered = 0
    new_items_added = 0
    # next_version will be determined based on existing records or current_version for new items.

    try:
        # item_records_slice will contain data for the given key_value.
        # Its index will be ('crawl_date', 'version').
        # Its columns will be the data columns of the main df.
        item_records_slice = df.loc[(key_value)]

        # Convert to DataFrame and reset index to get 'crawl_date' and 'version' as columns
        if isinstance(item_records_slice, pd.Series):
            # If it's a Series, it means there's only one matching record for this key_value.
            # The Series' name will be a tuple (crawl_date_val, version_val)
            # Convert to a single-row DataFrame.
            temp_df = item_records_slice.to_frame().T
            # The index of temp_df is now a MultiIndex with one entry: ((crawl_date_val, version_val),)
            # We need to get crawl_date and version from its name and make them columns.
            crawl_date_val, version_val = item_records_slice.name
            temp_df['crawl_date'] = crawl_date_val
            temp_df['version'] = version_val
            temp_df = temp_df.reset_index(drop=True) # Remove the (crawl_date_val, version_val) index
            current_versions_df = temp_df
        else: # Already a DataFrame
            current_versions_df = item_records_slice.reset_index()


        if not current_versions_df.empty:
            # Ensure 'version' column is numeric for max() and sorting
            if 'version' in current_versions_df.columns:
                current_versions_df['version'] = pd.to_numeric(current_versions_df['version'], errors='coerce')
            else: # Should not happen if index was ('key_name', 'crawl_date', 'version')
                if debug_mode: print(f"Warning: 'version' column unexpectedly missing for {key_name} {key_value} after .loc and reset_index.")
                current_versions_df['version'] = np.nan # Add it as nan to avoid errors, but this is problematic

            # Filter out rows where version is NaN after conversion, as they cannot be sorted reliably or used for max()
            valid_versions_df = current_versions_df.dropna(subset=['version'])

            if not valid_versions_df.empty:
                valid_versions_df = valid_versions_df.sort_values(['version'])
                latest_version = int(valid_versions_df['version'].max())
                next_version = latest_version + 1

                # Reconstruct the full record for Pydantic model instantiation.
                # merge_df_rows_by_sort_key expects all data columns + sort key columns.
                # Here, sort key is 'version'. The key_name (e.g., uen) is not in current_versions_df columns yet.
                # Add key_name to the DataFrame passed to merge_df_rows_by_sort_key temporarily if needed,
                # or ensure merge_df_rows_by_sort_key can handle its absence and it's added later.
                
                # The `merge_df_rows_by_sort_key` expects the sort_key_columns to be present in the input df.
                # `current_versions_df` now has 'version' as a column.
                # It also has all the data columns from the original `df`.
                complete_record_dict = merge_df_rows_by_sort_key(df=valid_versions_df, sort_key_columns=['version'])
                complete_record_dict[key_name] = key_value # Add the main key back

                try:
                    existing_item_model = pydantic_model(**complete_record_dict)

                    if item_model_instance != existing_item_model:
                        if debug_mode:
                            print(f"Difference found for {pydantic_model.__name__} with {key_name} '{key_value}', creating new version {next_version}.")

                        old_attrs = existing_item_model.model_dump()
                        new_attrs = item_model_instance.model_dump()

                        detailed_diffs, changed_attrs_new_vals, actual_attr_changes_found = get_attribute_changes(
                            old_attributes=old_attrs,
                            new_attributes=new_attrs,
                            ignore_keys=[key_name, 'crawl_date', 'version'] # Also ignore versioning fields
                        )

                        if actual_attr_changes_found:
                            # Prepare data for the new row in the main DataFrame `df`
                            # The columns of `df` are the Pydantic fields (excluding key_name, crawl_date, version)
                            series_data = {}
                            for pydantic_field in pydantic_model.model_fields.keys():
                                if pydantic_field in [key_name, 'crawl_date', 'version']:
                                    continue
                                if pydantic_field in changed_attrs_new_vals:
                                    series_data[pydantic_field] = changed_attrs_new_vals[pydantic_field]
                                else:
                                    # If not changed, take from existing model, or new model if preferred (original takes from existing via None)
                                    series_data[pydantic_field] = old_attrs.get(pydantic_field) # Or new_attrs.get(pydantic_field)

                            if debug_mode:
                                summary_entry = {
                                    key_name: key_value,
                                    'crawl_date': crawl_date,
                                    'new_version': next_version,
                                    'differences': detailed_diffs
                                }
                                diff_summary_list.append(summary_entry)
                                for field, changes in detailed_diffs.items():
                                    print(f"  Difference in '{field}': Previous: {changes['previous']}, New: {changes['new']}")
                            
                            idx = (key_value, crawl_date, next_version)
                            df.loc[idx] = pd.Series(series_data, index=df.columns) # Ensure Series index matches df columns
                            if debug_mode:
                                print(f"Added new version {next_version} for {key_name}: {key_value}, crawl_date: {crawl_date}")

                        elif item_model_instance != existing_item_model and not actual_attr_changes_found:
                            if debug_mode:
                                print(f"Warning: Models for {key_name} '{key_value}' (version {next_version}) unequal by Pydantic, but no attribute diffs. Storing full new model data.")
                            
                            full_new_data_for_series = {}
                            model_dump_new = item_model_instance.model_dump()
                            for pydantic_field in pydantic_model.model_fields.keys():
                                if pydantic_field not in [key_name, 'crawl_date', 'version']:
                                    full_new_data_for_series[pydantic_field] = model_dump_new.get(pydantic_field)

                            if debug_mode:
                                summary_entry = {
                                    key_name: key_value,
                                    'crawl_date': crawl_date,
                                    'new_version': next_version,
                                    'differences': {'_comment': 'Pydantic models unequal, but no attribute diffs. Full new model stored.'}
                                }
                                diff_summary_list.append(summary_entry)

                            idx = (key_value, crawl_date, next_version)
                            df.loc[idx] = pd.Series(full_new_data_for_series, index=df.columns)
                            if debug_mode:
                                print(f"Added new version {next_version} for {key_name}: {key_value} with full new data (Pydantic inequality fallback).")
                        else: # No differences found
                            if debug_mode: print(f"No differences found for {key_name} '{key_value}', not creating new version.")
                    else: # item_model_instance == existing_item_model
                        if debug_mode: print(f"No differences found for {key_name} '{key_value}', not creating new version.")

                except Exception as ex_pydantic:
                    if debug_mode:
                        print(f"Error reconstructing/comparing {pydantic_model.__name__} from DataFrame for {key_name} {key_value}: {ex_pydantic}")
                    errors_encountered += 1
                    # Fallback: store the new item with a new version if comparison fails
                    new_item_data_dump = item_model_instance.model_dump()
                    series_data = {k: v for k, v in new_item_data_dump.items() if k not in [key_name, 'crawl_date', 'version']}
                    # Use determined next_version if available, otherwise fallback for new version
                    idx = (key_value, crawl_date, next_version if 'next_version' in locals() else current_version +1 if valid_versions_df.empty else latest_version +1 ) 
                    df.loc[idx] = pd.Series(series_data, index=df.columns)
                    if debug_mode: print(f"Added new version for {key_name}: {key_value} with complete data due to comparison error. Used version {idx[2]}.")
            else: # valid_versions_df is empty (no previous versions or versions were NaN)
                if debug_mode: print(f"No valid existing versions found for {key_name} {key_value}. Adding as new with version {current_version}.")
                new_item_data_dump = item_model_instance.model_dump()
                series_data = {k: v for k, v in new_item_data_dump.items() if k not in [key_name, 'crawl_date', 'version']}
                idx = (key_value, crawl_date, current_version)
                df.loc[idx] = pd.Series(series_data, index=df.columns)
                new_items_added += 1
        
        else: # current_versions_df is empty (no prior records for this key_value at all)
            if debug_mode: print(f"No existing versioned records found for {key_name} {key_value}. Adding as new with version {current_version}.")
            new_item_data_dump = item_model_instance.model_dump()
            series_data = {k: v for k, v in new_item_data_dump.items() if k not in [key_name, 'crawl_date', 'version']}
            idx = (key_value, crawl_date, current_version)
            df.loc[idx] = pd.Series(series_data, index=df.columns)
            new_items_added += 1

    except KeyError: # This means (key_value) was not found by df.loc[(key_value)] -> new item
        if debug_mode: print(f"New {pydantic_model.__name__} added (KeyError on initial lookup) with {key_name}: {key_value}, version: {current_version}")
        new_item_data_dump = item_model_instance.model_dump()
        series_data = {k: v for k, v in new_item_data_dump.items() if k not in [key_name, 'crawl_date', 'version']}
        idx = (key_value, crawl_date, current_version)
        df.loc[idx] = pd.Series(series_data, index=df.columns) # Ensure Series index matches df columns
        new_items_added += 1
    
    except Exception as e:
        if debug_mode:
            print(f"General error processing {key_name} {key_value}, crawl_date {crawl_date}: {e}")
            traceback.print_exc()
        errors_encountered += 1
        # Fallback: attempt to save the current item with a new version
        new_item_data_dump = item_model_instance.model_dump()
        series_data = {k: v for k, v in new_item_data_dump.items() if k not in [key_name, 'crawl_date', 'version']}
         # Fallback versioning: use current_version, or try to increment if context allows
        fallback_version = current_version
        try: # Attempt to get a next version if some context exists
            item_records_slice_fallback = df.loc[(key_value)]
            if not item_records_slice_fallback.empty:
                 temp_df_fallback = item_records_slice_fallback.reset_index()
                 if 'version' in temp_df_fallback.columns:
                    temp_df_fallback['version'] = pd.to_numeric(temp_df_fallback['version'], errors='coerce')
                    valid_fallback_versions = temp_df_fallback.dropna(subset=['version'])
                    if not valid_fallback_versions.empty:
                        fallback_version = int(valid_fallback_versions['version'].max()) + 1
        except: # Ignore errors in fallback version calculation
            pass

        idx = (key_value, crawl_date, fallback_version)
        df.loc[idx] = pd.Series(series_data, index=df.columns)
        if debug_mode: print(f"Added new version {fallback_version} for {key_name}: {key_value} with complete data due to general processing error.")

    return errors_encountered, new_items_added

def main():
    parser = argparse.ArgumentParser(
        description="Investigate JSON data from MyCareersFuture job postings for consistency and reusability."
    )
    parser.add_argument(
        "previous_date",
        type=str,
        help="Date in YYYYMMDD format, for loading existing Parquet data."
    )
    parser.add_argument(
        "next_date",
        type=str,
        help="Date in YYYYMMDD format, for processing new JSON files and naming the output Parquet file."
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
        default="db_data",
        help="Directory to load existing Parquet files from and save new ones to. Default: 'db_data'."
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version number for this crawl. Default: 1."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose logging and difference summaries. Default: False."
    )
    args = parser.parse_args()

    data_path = Path(args.directory) / args.next_date
    db_dir = Path(args.database_directory)
    posted_company_parquet_file = db_dir / args.previous_date / "posted_companies.parquet"
    
    # Convert date string to standard format for storage
    crawl_date = args.next_date
    version = args.version

    # Initialize posted_company_df
    company_columns = list(PostedCompany.model_fields.keys())
    # Add crawl_date and version columns
    if 'crawl_date' not in company_columns:
        company_columns.append('crawl_date')
    if 'version' not in company_columns:
        company_columns.append('version')

    error_count = 0
    diff_companies_summary = []
    newly_added_companies_count = 0 # Initialize counter for newly added companies
    
    # Define key name for companies
    company_key_name = 'uen'

    try:
        print(f"Attempting to load existing data from {posted_company_parquet_file}")
        posted_company_df = pd.read_parquet(posted_company_parquet_file)
        
        # Check and add required columns if missing from Parquet (e.g. older schema)
        # The index columns 'uen', 'crawl_date', 'version' are critical.
        # If they were not saved as regular columns but as index, reset_index() later will handle it.
        # For now, ensure they exist if it was saved with index=False or with different names.
        
        # Ensure all model columns (plus crawl_date, version) exist, add if missing
        # This is important if the Parquet file is from an older schema.
        for col in company_columns: # company_columns now includes crawl_date, version
            if col not in posted_company_df.columns and col not in [company_key_name, 'crawl_date', 'version']:
                 # Only add if it's a data column, not an index column (which might be handled by set_index)
                posted_company_df[col] = None
        
        # Standardize 'uen', 'crawl_date', 'version' if they were part of the index or need creation
        # If loaded with index, reset it to make them columns before set_index
        if isinstance(posted_company_df.index, pd.MultiIndex) or posted_company_df.index.name is not None:
            posted_company_df = posted_company_df.reset_index()

        for col in [company_key_name, 'crawl_date', 'version']:
            if col not in posted_company_df.columns:
                if col == 'crawl_date':
                    # This is a tricky fill; previous_date might not be right for all historical records.
                    # Best if crawl_date was always present. Forcing it might be incorrect.
                    # Original code assigned args.previous_date.
                    print(f"Warning: '{col}' column missing. Attempting to fill with {args.previous_date}.")
                    posted_company_df[col] = args.previous_date 
                elif col == 'version':
                     # Similar issue for version.
                    print(f"Warning: '{col}' column missing. Attempting to fill with {version}.")
                    posted_company_df[col] = version 
                elif col == company_key_name:
                    # This is critical. If uen is missing, we have a big problem.
                    print(f"CRITICAL Error: '{col}' (key column) not found in {posted_company_parquet_file}. Cannot proceed with versioning.")
                    # Forcing creation of empty DF as fallback
                    posted_company_df = pd.DataFrame(columns=company_columns)
                    posted_company_df = posted_company_df.set_index([company_key_name, 'crawl_date', 'version'], drop=True)
                    posted_company_df = posted_company_df.sort_index()
                    break 
                else: # Should not be reached if col is one of the three
                    posted_company_df[col] = None

        # Ensure all model fields (non-index) are present
        for model_field in PostedCompany.model_fields.keys():
            if model_field not in [company_key_name, 'crawl_date', 'version'] and model_field not in posted_company_df.columns:
                posted_company_df[model_field] = None


        # Set up multi-index using the key, crawl_date, and version
        # We need to ensure these columns are suitable for indexing (e.g., not all NaNs)
        if not posted_company_df.empty:
            # Before setting index, ensure the columns exist and have valid data
            # For example, if 'version' was just added and is all None, indexing might be problematic.
            # The original code assumed 'version' would be populated or handled.
            # If 'version' or 'crawl_date' were missing and filled with a single value, that's fine for indexing.
            # If company_key_name (e.g. 'uen') is missing, the critical error above should handle it.
            
            # Convert to appropriate types before setting index, if necessary
            # Example: posted_company_df['version'] = pd.to_numeric(posted_company_df['version'], errors='coerce').fillna(0).astype(int)

            posted_company_df = posted_company_df.set_index([company_key_name, 'crawl_date', 'version'], drop=True)
            posted_company_df = posted_company_df.sort_index()
        
        print(f"Successfully loaded {len(posted_company_df)} records.")

    except FileNotFoundError:
        print(f"No existing Parquet file found at {posted_company_parquet_file}. Creating a new DataFrame.")
        posted_company_df = pd.DataFrame(columns=company_columns)
        posted_company_df = posted_company_df.set_index([company_key_name, 'crawl_date', 'version'], drop=True)
    except Exception as e:
        print(f"Error loading Parquet file {posted_company_parquet_file}: {e}. Creating a new DataFrame.")
        traceback.print_exc()
        posted_company_df = pd.DataFrame(columns=company_columns)
        posted_company_df = posted_company_df.set_index([company_key_name, 'crawl_date', 'version'], drop=True)

    # Ensure the DataFrame has all columns required by the Pydantic model + versioning columns, even if empty
    # The index columns are already handled by set_index(drop=False)
    for col in PostedCompany.model_fields.keys():
        if col not in posted_company_df.columns and col not in posted_company_df.index.names:
            posted_company_df[col] = None
    # Ensure 'crawl_date' and 'version' columns exist if not part of index (though they should be)
    if 'crawl_date' not in posted_company_df.columns and 'crawl_date' not in posted_company_df.index.names:
        posted_company_df['crawl_date'] = None # Or a default like args.previous_date
    if 'version' not in posted_company_df.columns and 'version' not in posted_company_df.index.names:
         posted_company_df['version'] = None # Or a default like 1


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
            error_count += 1
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {json_file_path}")
            error_count += 1
        except Exception as e: 
            print(f"An error occurred while processing file '{json_file_path}': {e}")
            error_count += 1
            traceback.print_exc()
    
    if file_count == 0 and json_files_exist:
         print(f"No JSON files were processed despite being found initially in {data_path}.")
    elif file_count == 0 and not json_files_exist:
         print(f"No JSON files found in {data_path} to process.")

    # Process the collected companies
    processed_count = 0
    for uen, company in new_companies.items():
        processed_count += 1
        
        item_errors, item_newly_added = _process_item_versioning(
            df=posted_company_df,
            key_name=company_key_name,
            key_value=uen,
            item_model_instance=company,
            pydantic_model=PostedCompany,
            all_df_columns=company_columns, # These are all potential columns for the DataFrame
            crawl_date=crawl_date,
            current_version=version,
            diff_summary_list=diff_companies_summary,
            debug_mode=args.debug
        )
        error_count += item_errors
        newly_added_companies_count += item_newly_added

    # Saving logic using args.database_directory
    output_dir_for_parquet = db_dir / args.next_date
    print(f"\nSaving PostedCompany data in directory: {output_dir_for_parquet}")
    try:
        output_dir_for_parquet.mkdir(parents=True, exist_ok=True)
        if not posted_company_df.empty:
            df_to_save = posted_company_df.reset_index()  # Make 'uen', 'crawl_date', 'version' columns for saving
            output_parquet_path = output_dir_for_parquet / "posted_companies.parquet" # Use next_date for output subdirectory
            df_to_save.to_parquet(output_parquet_path, index=False)
            print(f"  Saved {output_parquet_path}")
        else:
            # Construct the expected path for the print statement
            skipped_output_parquet_path = output_dir_for_parquet / "posted_companies.parquet"
            print(f"  Skipping {skipped_output_parquet_path} as no data was found/generated.")
        
        print(f"Successfully processed Parquet saving to {output_dir_for_parquet}")
    except Exception as e:
        print(f"Error saving Parquet data to '{output_dir_for_parquet}': {e}")
        error_count += 1
        traceback.print_exc()

    print("\n--- Investigation Complete ---")
    if json_files_exist or file_count > 0:
        print(f"Processed {file_count} JSON file(s) from {data_path}.")
    print(f"Total unique PostedCompany entries stored: {len(posted_company_df)}")
    print(f"Processed {processed_count} companies from JSON files.")
    print(f"Newly added companies: {newly_added_companies_count}") # Print the count of newly added companies
    print(f"Total errors encountered: {error_count}")

    if args.debug and diff_companies_summary: # Only print if debug and list is not empty
        print("\n--- Summary of Differences ---")
        for entry in diff_companies_summary:
            print(f"UEN: {entry['uen']}, Crawl Date: {entry['crawl_date']}, New Version: {entry['new_version']}")
            # Check if 'differences' key exists and is a dictionary
            if isinstance(entry.get('differences'), dict):
                for field, changes in entry['differences'].items():
                    # Handle cases where changes might not be the expected dict
                    if isinstance(changes, dict) and 'previous' in changes and 'new' in changes:
                        print(f"  Field '{field}':")
                        print(f"    Previous: {changes['previous']}")
                        print(f"    New:      {changes['new']}")
                    elif field == '_comment': # Handle special comment for Pydantic inequality
                        print(f"  Comment: {changes}")
                    else:
                        print(f"  Field '{field}': {changes}") # Fallback for unexpected format
            else:
                print(f"  Differences: {entry.get('differences')}") # Fallback if 'differences' is not a dict
            print("---")

if __name__ == "__main__":
    main()