import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Type, Tuple, Callable
import pandas as pd
from datetime import datetime
import numpy as np
import traceback
from pydantic import BaseModel, Field, field_validator

# Assuming schema.py is in the same directory or accessible via PYTHONPATH
from schema import (
    JobSearchResponse, JobResult, PostedCompany, Districts, PositionLevel, Skill, JobEmploymentType, Category, Status
)

# Configuration for different data types
DATA_TYPE_CONFIG: Dict[str, Dict[str, Any]] = {
    "JobResult": {
        "model": JobResult,
        "key_name": "uuid",
        "parquet_filename": "job_results.parquet",
        "json_path_getter": lambda job_result: [job_result] if job_result else [],
        "description": "Job results"
    },
    "PostedCompany": {
        "model": PostedCompany,
        "key_name": "uen",
        "parquet_filename": "posted_companies.parquet",
        "json_path_getter": lambda job_result: [job_result.postedCompany] if job_result.postedCompany else [],
        "description": "Company posting the job"
    },
    "Districts": {
        "model": Districts,
        "key_name": "id",
        "parquet_filename": "districts.parquet",
        "json_path_getter": lambda job_result: job_result.address.districts if job_result.address and job_result.address.districts else [],
        "description": "Job location districts"
    },
    "PositionLevel": {
        "model": PositionLevel,
        "key_name": "id",
        "parquet_filename": "position_levels.parquet",
        "json_path_getter": lambda job_result: job_result.positionLevels if job_result.positionLevels else [],
        "description": "Job position levels"
    },
    "Skill": {
        "model": Skill,
        "key_name": "uuid", # Assuming 'uuid' for Skill based on investigate_data_consistency.py
        "parquet_filename": "skills.parquet",
        "json_path_getter": lambda job_result: job_result.skills if job_result.skills else [],
        "description": "Job skills"
    },
    "JobEmploymentType": {
        "model": JobEmploymentType,
        "key_name": "id",
        "parquet_filename": "employment_types.parquet",
        "json_path_getter": lambda job_result: job_result.employmentTypes if job_result.employmentTypes else [],
        "description": "Job employment types"
    },
    "Category": {
        "model": Category,
        "key_name": "id",
        "parquet_filename": "categories.parquet",
        "json_path_getter": lambda job_result: job_result.categories if job_result.categories else [],
        "description": "Job categories"
    },
    "Status": {
        "model": Status,
        "key_name": "id",
        "parquet_filename": "statuses.parquet",
        "json_path_getter": lambda job_result: [job_result.status] if job_result.status else [],
        "description": "Job statuses"
    },
    # Add other data types here as needed
}

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
        key_name: The name of the primary key column (e.g., 'uen', 'id').
        key_value: The value of the primary key for the current item.
        item_model_instance: The Pydantic model instance of the new item data.
        pydantic_model: The Pydantic model class (e.g., PostedCompany, District).
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
        "--data-type",
        type=str,
        nargs='+',  # Allow multiple data types
        default=["Districts","PositionLevel","Skill","JobEmploymentType","Category","Status","PostedCompany","JobResult"],
        choices=list(DATA_TYPE_CONFIG.keys()),
        help="Specify one or more data types to process."
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
    
    # Convert date string to standard format for storage
    crawl_date = args.next_date # This is common for all data types processed in this run
    version = args.version # Common version for this crawl

    # --- Read all JSON files once ---
    json_responses: List[JobSearchResponse] = []
    if data_path.is_dir():
        print(f"Reading JSON files from: {data_path}")
        json_file_count = 0
        for json_file_path in data_path.glob("*.json"):
            json_file_count += 1
            print(f"--- Reading file: {json_file_path.name} ---")
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    json_data_str = f.read()
                job_response = JobSearchResponse.model_validate_json(json_data_str)
                json_responses.append(job_response)
            except FileNotFoundError:
                print(f"Error: File disappeared during reading: {json_file_path}")
                # Potentially log this error more formally or add to a run summary
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file: {json_file_path}")
            except Exception as e:
                print(f"An error occurred while reading file '{json_file_path}': {e}")
                traceback.print_exc()
        if json_file_count == 0:
            print(f"No JSON files found in {data_path}.")
    else:
        print(f"Warning: Data source directory not found: {data_path}. Proceeding without JSON data.")

    # --- Loop through each selected data type ---
    selected_data_types = args.data_type
    for selected_data_type in selected_data_types:
        config = DATA_TYPE_CONFIG[selected_data_type]
        pydantic_model = config["model"]
        item_key_name = config["key_name"]
        parquet_filename = config["parquet_filename"]
        json_path_getter = config["json_path_getter"]
        item_description = config["description"]
        
        print(f"\n=== Processing data type: {item_description} ({selected_data_type}) ===")
        
        item_parquet_file = db_dir / args.previous_date / parquet_filename
        
        # Initialize DataFrame for the current data type
        item_columns = list(pydantic_model.model_fields.keys())
        if 'crawl_date' not in item_columns:
            item_columns.append('crawl_date')
        if 'version' not in item_columns:
            item_columns.append('version')

        error_count_dt = 0 # Error count for this data type
        diff_items_summary_dt: List[Dict[str, Any]] = [] # Diff summary for this data type
        newly_added_items_count_dt = 0 # Newly added items for this data type
        
        try:
            print(f"Attempting to load existing data from {item_parquet_file}")
            item_df = pd.read_parquet(item_parquet_file)
            
            if isinstance(item_df.index, pd.MultiIndex) or item_df.index.name is not None:
                item_df = item_df.reset_index()

            for col in item_columns:
                if col not in item_df.columns and col not in [item_key_name, 'crawl_date', 'version']:
                    item_df[col] = None
            
            for col in [item_key_name, 'crawl_date', 'version']:
                if col not in item_df.columns:
                    fill_value = None
                    if col == 'crawl_date': fill_value = args.previous_date
                    elif col == 'version': fill_value = version # Default version if missing
                    
                    if col == item_key_name and fill_value is None: # Should not happen if key_name is always there
                         print(f"CRITICAL Error: '{col}' (key column) not found in {item_parquet_file} and no default. Cannot proceed for {selected_data_type}.")
                         item_df = pd.DataFrame(columns=item_columns).set_index([item_key_name, 'crawl_date', 'version'], drop=True).sort_index()
                         # Skip further processing for this data type or handle error appropriately
                         continue # Or raise an error
                    else:
                        print(f"Warning: Column '{col}' missing in {item_parquet_file} for {selected_data_type}. Attempting to fill.")
                        item_df[col] = fill_value

            for model_field in pydantic_model.model_fields.keys():
                if model_field not in [item_key_name, 'crawl_date', 'version'] and model_field not in item_df.columns:
                    item_df[model_field] = None
            
            # Set index after ensuring columns exist
            item_df = item_df.set_index([item_key_name, 'crawl_date', 'version'], drop=True).sort_index()
            print(f"Successfully loaded {len(item_df)} records for {selected_data_type}.")

        except FileNotFoundError:
            print(f"No existing Parquet file found at {item_parquet_file}. Creating a new DataFrame for {selected_data_type}.")
            data_cols = [f for f in pydantic_model.model_fields.keys() if f not in [item_key_name, 'crawl_date', 'version']]
            index_names = [item_key_name, 'crawl_date', 'version']
            item_df = pd.DataFrame(columns=data_cols,
                                   index=pd.MultiIndex.from_tuples([], names=index_names))
        except Exception as e:
            print(f"Error loading Parquet file {item_parquet_file} for {selected_data_type}: {e}. Creating a new DataFrame.")
            traceback.print_exc()
            data_cols = [f for f in pydantic_model.model_fields.keys() if f not in [item_key_name, 'crawl_date', 'version']]
            index_names = [item_key_name, 'crawl_date', 'version']
            item_df = pd.DataFrame(columns=data_cols,
                                   index=pd.MultiIndex.from_tuples([], names=index_names))
            error_count_dt += 1

        # Ensure all necessary columns (Pydantic fields not in index) exist in the DataFrame
        current_df_cols = list(item_df.columns)
        current_df_index_names = list(item_df.index.names)
        for col_model in pydantic_model.model_fields.keys():
            if col_model not in current_df_cols and col_model not in current_df_index_names:
                item_df[col_model] = None
        
        # If DataFrame is empty and uninitialized (e.g. error during load created a shell)
        if item_df.empty and not item_df.columns.any() and (not item_df.index.names or not item_df.index.names[0]):
            print(f"Re-initializing empty DataFrame for {selected_data_type} with defined columns and index.")
            data_columns_for_empty_df = [
                f for f in pydantic_model.model_fields.keys() 
                if f not in [item_key_name, 'crawl_date', 'version']
            ]
            item_df = pd.DataFrame(columns=data_columns_for_empty_df)
            item_df = item_df.set_index([item_key_name, 'crawl_date', 'version'], drop=True)


        # --- Extract items of current data type from pre-read JSON responses ---
        new_items_map = {} # Items for the current data_type
        if json_responses: # Only process if JSON files were successfully read
            print(f"Extracting {item_description} from JSON data...")
            for job_response in json_responses:
                for job_result in job_response.results:
                    extracted_instances = json_path_getter(job_result)
                    for instance in extracted_instances:
                        if instance:
                            key_val = getattr(instance, item_key_name, None)
                            if key_val is not None:
                                if key_val not in new_items_map: # Keep only the first encountered for a given key from JSONs
                                    new_items_map[key_val] = instance
                            elif args.debug:
                                print(f"Debug: Instance of {selected_data_type} found without key '{item_key_name}': {instance}")
        else:
            print(f"No JSON data to process for {selected_data_type}.")


        # --- Process the collected items for the current data type ---
        processed_count_dt = 0
        if new_items_map:
            print(f"Processing {len(new_items_map)} unique {item_description.lower()} items for versioning...")
            for key_value, item_instance in new_items_map.items():
                processed_count_dt += 1
                item_errors, item_newly_added = _process_item_versioning(
                    df=item_df, # This is the DataFrame for the current selected_data_type
                    key_name=item_key_name,
                    key_value=key_value,
                    item_model_instance=item_instance,
                    pydantic_model=pydantic_model,
                    crawl_date=crawl_date, # Common crawl_date
                    current_version=version, # Common version
                    diff_summary_list=diff_items_summary_dt,
                    debug_mode=True
                )
                error_count_dt += item_errors
                newly_added_items_count_dt += item_newly_added
        else:
            print(f"No new items from JSON to process for {selected_data_type}.")


        # --- Saving logic for the current data type ---
        output_dir_for_parquet = db_dir / args.next_date
        print(f"\nSaving {selected_data_type} data in directory: {output_dir_for_parquet}")
        try:
            output_dir_for_parquet.mkdir(parents=True, exist_ok=True)
            if not item_df.empty:
                # Ensure index is as expected before saving
                if not (isinstance(item_df.index, pd.MultiIndex) and \
                        item_df.index.names == [item_key_name, 'crawl_date', 'version']):
                    print(f"Warning: Index for {selected_data_type} is not as expected before saving. Resetting and setting index.")
                    temp_df_to_save = item_df.reset_index()
                     # Check if all key_name, crawl_date, version columns exist after reset
                    for col_idx in [item_key_name, 'crawl_date', 'version']:
                         if col_idx not in temp_df_to_save.columns:
                              print(f"Critical error: Index column '{col_idx}' missing before saving {selected_data_type}.")
                              # Fallback or error
                    df_to_save = temp_df_to_save
                else:
                    df_to_save = item_df.reset_index()

                output_parquet_path = output_dir_for_parquet / parquet_filename
                df_to_save.to_parquet(output_parquet_path, index=True)
                print(f"  Saved {output_parquet_path}")
            else:
                skipped_output_parquet_path = output_dir_for_parquet / parquet_filename
                print(f"  Skipping save for {skipped_output_parquet_path} as DataFrame is empty for {selected_data_type}.")
            
            print(f"Successfully processed Parquet saving to {output_dir_for_parquet} for {selected_data_type}")
        except Exception as e:
            print(f"Error saving Parquet data for {selected_data_type} to '{output_dir_for_parquet}': {e}")
            error_count_dt += 1
            traceback.print_exc()

        # --- Print summary for the current data type ---
        print(f"\n--- {selected_data_type} Processing Complete ---")
        if json_responses : # check if json_responses list is populated
             # File count for JSON reading is done once globally
             pass # Global count will be printed at the end if needed.
        print(f"Total unique {selected_data_type} entries stored/updated: {len(item_df)}")
        print(f"Processed {processed_count_dt} unique {item_description.lower()} items from JSON files for {selected_data_type}.")
        print(f"Newly added {selected_data_type.lower()} items: {newly_added_items_count_dt}")
        print(f"Total errors encountered for {selected_data_type}: {error_count_dt}")

        if args.debug and diff_items_summary_dt:
            print(f"\n--- Summary of Differences for {selected_data_type} ---")
            for entry in diff_items_summary_dt:
                print(f"{item_key_name.upper()}: {entry.get(item_key_name)}, Crawl Date: {entry.get('crawl_date')}, New Version: {entry.get('new_version')}")
                if isinstance(entry.get('differences'), dict):
                    for field, changes in entry['differences'].items():
                        if isinstance(changes, dict) and 'previous' in changes and 'new' in changes:
                            print(f"  Field '{field}':")
                            print(f"    Previous: {changes['previous']}")
                            print(f"    New:      {changes['new']}")
                        elif field == '_comment':
                            print(f"  Comment: {changes}")
                        else:
                            print(f"  Field '{field}': {changes}")
                else:
                    print(f"  Differences: {entry.get('differences')}")
                print("---")
        # End of loop for selected_data_type

    print("\n=== All Processing Complete ===")
    # Optionally, print global summary here if needed, e.g., total JSON files read.
    # The individual error counts, etc., are per data type.

if __name__ == "__main__":
    main()