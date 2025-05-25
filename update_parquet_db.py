#!/usr/bin/env python3
import argparse
import orjson
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, cast
import logging
from schema_v2 import Company, Job, Skills, CombinedResultsContainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(db_data_dir: str, next_date: str) -> str:
    """Create necessary directories for the database."""
    next_date_dir = os.path.join(db_data_dir, next_date)
    os.makedirs(next_date_dir, exist_ok=True)
    return next_date_dir

def read_previous_parquet(db_data_dir: str, previous_date: str, data_type: str) -> Optional[pd.DataFrame]:
    """Read previous day's parquet file if it exists."""
    previous_parquet_path = os.path.join(db_data_dir, previous_date, f"{previous_date}_{data_type}.parquet")
    
    if os.path.exists(previous_parquet_path):
        logger.info(f"Reading previous {data_type} parquet file: {previous_parquet_path}")
        try:
            df = pd.read_parquet(previous_parquet_path)
            logger.info(f"Loaded {len(df)} {data_type} records from previous data")
            return df
        except Exception as e:
            logger.error(f"Error reading previous {data_type} parquet file: {e}")
            return None
    else:
        logger.info(f"No previous {data_type} parquet file found at {previous_parquet_path}, starting fresh")
        return None

def read_raw_json_data(raw_data_dir: str, next_date: str, data_type: str) -> Optional[bytes]:
    """Read raw JSON data for a given data type (jobs or companies) and return as bytes."""
    file_path = os.path.join(raw_data_dir, data_type, next_date, "combine", f"{next_date}_{data_type}_combine.json")
    
    if not os.path.exists(file_path):
        logger.error(f"{data_type.capitalize()} file not found: {file_path}")
        return None
    
    logger.info(f"Reading {data_type} data from: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            json_bytes = f.read()
        
        logger.info(f"Successfully read {data_type} data as bytes")
        return json_bytes
        
    except Exception as e:
        logger.error(f"Error reading {data_type} data: {e}")
        return None

def read_raw_companies_data(raw_data_dir: str, next_date: str) -> Optional[List[Company]]:
    """Read and parse the raw companies JSON data."""
    json_bytes = read_raw_json_data(raw_data_dir, next_date, "companies")

    if json_bytes is None:
        return None

    try:
        # Validate and parse using Pydantic schema
        companies_container = CombinedResultsContainer.model_validate_json(json_bytes)
        logger.info(f"Successfully parsed {len(companies_container.results)} companies")
        return cast(List[Company], companies_container.results)

    except Exception as e:
        logger.error(f"Error parsing companies data: {e}")
        return None

def read_raw_jobs_data(raw_data_dir: str, next_date: str) -> Optional[List[Job]]:
    """Read and parse the raw jobs JSON data."""
    json_bytes = read_raw_json_data(raw_data_dir, next_date, "jobs")

    if json_bytes is None:
        return None

    try:
        # Validate and parse using Pydantic schema
        jobs_container = CombinedResultsContainer.model_validate_json(json_bytes)
        logger.info(f"Successfully parsed {len(jobs_container.results)} jobs")
        return cast(List[Job], jobs_container.results)

    except Exception as e:
        logger.error(f"Error parsing jobs data: {e}")
        return None

def process_skills_data(jobs: List[Job], previous_skills_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process skills data from jobs, update skills DataFrame, and replace job skills with IDs."""

    new_skills_added_count = 0
    updated_skills_count = 0

    if previous_skills_df is not None and not previous_skills_df.empty:
        if previous_skills_df.index.names != [None]:
            skills_df = previous_skills_df.reset_index()
        else:
            skills_df = previous_skills_df.copy()

        if 'uuid' not in skills_df.columns or 'id' not in skills_df.columns:
            logger.warning("Previous skills DataFrame is missing 'uuid' or 'id' columns. Starting fresh.")
            skills_df = pd.DataFrame(columns=['id', 'uuid', 'skill', 'confidence'])
            next_id = 0
        else:
            # Ensure 'id' is integer type for proper max() calculation and lookups
            skills_df['id'] = skills_df['id'].astype(int)
            next_id = (skills_df['id'].max() + 1) if not skills_df.empty else 0
    else:
        skills_df = pd.DataFrame(columns=['id', 'uuid', 'skill', 'confidence'])
        next_id = 0

    # Collect all skills from jobs into a single DataFrame
    all_skills_data = []
    for job in jobs:
        if job.skills:
            for skill_obj in job.skills:
                skill_data = skill_obj.model_dump()
                all_skills_data.append(skill_data)

    if not all_skills_data:
        logger.info("No skills found in jobs data")
        return skills_df

    # Create DataFrame from all skills
    new_skills_df = pd.DataFrame(all_skills_data)
    
    if skills_df.empty:
        # No previous data, assign IDs to all skills
        new_skills_df['id'] = range(len(new_skills_df))
        new_skills_added_count = len(new_skills_df)
        skills_df = new_skills_df
    else:
        # Merge with existing skills data
        # First, identify which skills are new vs existing
        existing_uuids = set(skills_df['uuid'])
        new_skills_mask = ~new_skills_df['uuid'].isin(existing_uuids)
        
        # Handle existing skills - update them
        existing_skills = new_skills_df[~new_skills_mask]
        if not existing_skills.empty:
            # Update existing skills by merging on uuid
            skills_df = skills_df.set_index('uuid')
            existing_skills = existing_skills.set_index('uuid')
            
            # Update existing records
            skills_df.update(existing_skills)
            skills_df = skills_df.reset_index()
            updated_skills_count = len(existing_skills)
        
        # Handle new skills - assign new IDs
        new_skills = new_skills_df[new_skills_mask]
        if not new_skills.empty:
            new_skills = new_skills.copy()
            new_skills['id'] = range(next_id, next_id + len(new_skills))
            skills_df = pd.concat([skills_df, new_skills], ignore_index=True)
            new_skills_added_count = len(new_skills)

    logger.info(f"Processed skills: {len(skills_df)} total skills ({new_skills_added_count} new, {updated_skills_count} updated)")

    if not skills_df.empty:
        if skills_df['id'].duplicated().any():
            logger.warning("Duplicate IDs found in skills data. Rectifying...")
            skills_df = skills_df.drop_duplicates(subset=['id'], keep='last')

        if skills_df['uuid'].duplicated().any():
            logger.warning("Duplicate UUIDs found in skills data. Rectifying...")
            skills_df = skills_df.drop_duplicates(subset=['uuid'], keep='last')

        skills_df = skills_df.sort_values(by=['id', 'uuid'])
        skills_df = skills_df.set_index(['id', 'uuid'])

    # Now map skills in jobs to skill IDs to save space
    # for job in jobs:
    #     if job.skills:
    #         skill_ids = []
    #         for skill_obj in job.skills:
    #             if skill_obj.uuid in skills_lookup:
    #                 skill_ids.append(skills_lookup[skill_obj.uuid])
    #         # Replace the skills objects with just the list of IDs
    #         job.skills = skill_ids

    return skills_df

def update_companies_data_with_jobs_data(jobs: List[Job], companies_df: pd.DataFrame) -> None:
    """Update companies data with company information extracted from jobs data."""
    
    if companies_df.empty:
        logger.warning("Companies DataFrame is empty, cannot update with jobs data")
        return
    
    # Reset index if it's set to work with the DataFrame directly
    if companies_df.index.names != [None]:
        companies_df_work = companies_df.reset_index()
    else:
        companies_df_work = companies_df.copy()
    
    # Extract all unique companies from jobs into a list of dicts
    companies_data = []
    companies_seen = set()
    
    for job in jobs:
        companies_to_process = []
        
        # Add hiring company if it exists
        if job.hiring_company and job.hiring_company.uen:
            companies_to_process.append(job.hiring_company)
            
        # Add posted company if it exists and different from hiring company
        if (job.posted_company and job.posted_company.uen and 
            job.posted_company.uen != (job.hiring_company.uen if job.hiring_company else None)):
            companies_to_process.append(job.posted_company)
        
        # Process each company found in this job
        for company in companies_to_process:
            if company.uen not in companies_seen:
                companies_seen.add(company.uen)
                company_data = company.model_dump(by_alias=True, exclude_none=True)
                companies_data.append(company_data)
    
    if not companies_data:
        logger.info("No companies found in jobs data to update")
        return
    
    # Create DataFrame from companies data
    companies_from_jobs_df = pd.DataFrame(companies_data)
    
    # Merge with existing companies data, updating only null/empty values
    # First, set UEN as index for both DataFrames for efficient merging
    companies_df_work = companies_df_work.set_index('uen')
    companies_from_jobs_df = companies_from_jobs_df.set_index('uen')
    
    # Find companies that exist in both DataFrames
    common_uens = companies_df_work.index.intersection(companies_from_jobs_df.index)
    
    updated_companies_count = 0
    
    if len(common_uens) > 0:
        # For each column in the jobs companies data, update null/empty values in the main DataFrame
        for col in companies_from_jobs_df.columns:
            if col in companies_df_work.columns:
                # Create mask for rows that need updating (null, None, or empty string)
                mask = (
                    companies_df_work.loc[common_uens, col].isna() | 
                    (companies_df_work.loc[common_uens, col] == '') |
                    companies_df_work.loc[common_uens, col].isnull()
                )
                
                # Update only the rows that need updating
                companies_df_work.loc[common_uens[mask], col] = companies_from_jobs_df.loc[common_uens[mask], col]
        
        updated_companies_count = len(common_uens)
    
    # Reset index back to original state
    companies_df_work = companies_df_work.reset_index()
    
    logger.info(f"Updated companies with jobs data: {updated_companies_count} companies updated from {len(companies_seen)} unique companies found in jobs")
    
    # Update the original DataFrame with the changes
    if companies_df.index.names != [None]:
        # If original had multi-index, restore it
        companies_df_work = companies_df_work.set_index(companies_df.index.names)
        companies_df.update(companies_df_work)
    else:
        companies_df.update(companies_df_work)

def process_companies_data_from_new_file(companies: List[Company], previous_companies_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process companies data from companies and create/update companies DataFrame."""
    
    # Initialize companies tracking
    if previous_companies_df is not None and not previous_companies_df.empty:
        # Reset index if it's set to work with the DataFrame directly
        if previous_companies_df.index.names != [None]:
            companies_df = previous_companies_df.reset_index()
        else:
            companies_df = previous_companies_df.copy()
        
        # Create a lookup dictionary for existing companies
        companies_lookup = {row['uen']: row['id'] for _, row in companies_df.iterrows()}
        next_id = companies_df['id'].max() + 1
        
        new_companies_count = 0
        updated_companies_count = 0
        
        # Process each company: update existing or create new
        for company in companies:
            if company.uen in companies_lookup:
                # Update existing company data directly in DataFrame
                existing_id = companies_lookup[company.uen]
                mask = companies_df['uen'] == company.uen
                
                # Convert company to dict and update the row
                updated_data = company.model_dump(by_alias=True)
                updated_data['id'] = existing_id
                
                # Update all columns for this row
                for col, value in updated_data.items():
                    if col in companies_df.columns:
                        companies_df.loc[mask, col] = value
                
                updated_companies_count += 1
            else:
                # Create new company
                new_company_record = company.model_dump(by_alias=True)
                new_company_record['id'] = next_id
                
                # Add new row to DataFrame
                new_row_df = pd.DataFrame([new_company_record])
                companies_df = pd.concat([companies_df, new_row_df], ignore_index=True)
                
                companies_lookup[company.uen] = next_id
                next_id += 1
                new_companies_count += 1
        
        logger.info(f"Processed companies: {len(companies_df)} total companies ({new_companies_count} new, {updated_companies_count} updated)")
    else:
        # Create new DataFrame from scratch
        companies_records = []
        
        # Add all companies as new
        for i, company in enumerate(companies):
            new_company_record = company.model_dump(by_alias=True)
            new_company_record['id'] = i
            companies_records.append(new_company_record)
        
        companies_df = pd.DataFrame(companies_records)
        logger.info(f"Processed companies: {len(companies_df)} total companies ({len(companies_df)} new)")
    
    # Verify uniqueness and sort for consistent ordering
    if not companies_df.empty:
        # Verify that id and uen are unique
        if companies_df['id'].duplicated().any():
            logger.warning("Duplicate IDs found in companies data")
        if companies_df['uen'].duplicated().any():
            logger.warning("Duplicate UENs found in companies data")
        
        # Sort by both id and uen for consistent ordering (more performant to sort before indexing)
        companies_df = companies_df.sort_values(['id', 'uen'])
        companies_df = companies_df.set_index(['id', 'uen'])
    
    return companies_df

def save_parquet_file(df: pd.DataFrame, output_path: str, table_name: str):
    """Save DataFrame to parquet file."""
    try:
        df.to_parquet(output_path, index=True, engine='pyarrow')
        file_size = os.path.getsize(output_path)
        logger.info(f"Saved {table_name} data to {output_path} ({file_size:,} bytes, {len(df)} records)")
    except Exception as e:
        logger.error(f"Error saving {table_name} parquet file: {e}")
        raise

def update_databases(previous_date: str, next_date: str, raw_data_dir: str, db_data_dir: str):
    """Main function to update the parquet database with new job data including skills, companies, districts, position levels, employment types, status, flexible work arrangements, and categories."""
    logger.info(f"Starting database update from {previous_date} to {next_date}")
    
    # Setup directories
    next_date_dir = setup_directories(db_data_dir, next_date)
    
    # Read raw jobs data
    jobs = read_raw_jobs_data(raw_data_dir, next_date)
    if not jobs:
        logger.error("Failed to read jobs data, aborting")
        return

    # Process Companies Data
    logger.info("Processing companies data...")
    companies = read_raw_companies_data(raw_data_dir, next_date)
    previous_companies_df = read_previous_parquet(db_data_dir, previous_date, "companies")
    companies_df = process_companies_data_from_new_file(companies, previous_companies_df)

    # process companies data combine with jobs data
    update_companies_data_with_jobs_data(jobs, companies_df)
    
    # Save companies data
    companies_output_path = os.path.join(next_date_dir, f"{next_date}_companies.parquet")
    save_parquet_file(companies_df, companies_output_path, "companies")
    
    # Process Skills Data
    logger.info("Processing skills data...")
    previous_skills_df = read_previous_parquet(db_data_dir, previous_date, "skills")
    skills_df = process_skills_data(jobs, previous_skills_df)
    
    # Save skills data
    skills_output_path = os.path.join(next_date_dir, f"{next_date}_skills.parquet")
    save_parquet_file(skills_df, skills_output_path, "skills")
    
    # Configuration for lookup tables processing
    lookup_tables_config = [
        {
            'name': 'districts',
            'extractor': lambda job: job.address.districts if job.address and job.address.districts else None,
            'columns': ['id', 'sectors', 'region_id', 'location', 'region']
        },
        {
            'name': 'position_levels',
            'extractor': lambda job: job.position_levels if job.position_levels else None,
            'columns': ['id', 'position']
        },
        {
            'name': 'employment_types',
            'extractor': lambda job: job.employment_types if job.employment_types else None,
            'columns': ['id', 'employment_type']
        },
        {
            'name': 'status',
            'extractor': lambda job: job.status if job.status else None,
            'columns': ['id', 'job_status']
        },
        {
            'name': 'flexible_work_arrangements',
            'extractor': lambda job: job.flexible_work_arrangements if job.flexible_work_arrangements else None,
            'columns': ['id', 'flexible_work_arrangement']
        },
        {
            'name': 'categories',
            'extractor': lambda job: job.categories if job.categories else None,
            'columns': ['id', 'category']
        }
    ]
    
    # Process all lookup tables using the configuration
    for config in lookup_tables_config:
        table_name = config['name']
        extractor_func = config['extractor']
        columns = config['columns']
        
        logger.info(f"Processing {table_name} data...")
        previous_df = read_previous_parquet(db_data_dir, previous_date, table_name)
        processed_df = process_lookup_table_data(
            jobs=jobs,
            previous_df=previous_df,
            data_type_name=table_name,
            extractor_func=extractor_func,
            columns=columns
        )
        
        # Save data
        output_path = os.path.join(next_date_dir, f"{next_date}_{table_name}.parquet")
        save_parquet_file(processed_df, output_path, table_name)
    
    logger.info("Database update completed successfully!")

def process_lookup_table_data(
    jobs: List[Job], 
    previous_df: Optional[pd.DataFrame], 
    data_type_name: str,
    extractor_func: callable,
    columns: List[str],
    id_column: str = 'id'
) -> pd.DataFrame:
    """
    Generic function to process lookup table data from jobs.
    
    Args:
        jobs: List of job objects
        previous_df: Previous DataFrame or None
        data_type_name: Name for logging (e.g., "districts", "employment_types")
        extractor_func: Function that takes a job and returns list of objects or None
        columns: List of column names for the DataFrame
        id_column: Name of the ID column (default: 'id')
    
    Returns:
        Processed DataFrame
    """
    new_added_count = 0
    updated_count = 0

    if previous_df is not None and not previous_df.empty:
        if previous_df.index.names != [None]:
            df = previous_df.reset_index()
        else:
            df = previous_df.copy()

        if id_column not in df.columns:
            logger.warning(f"Previous {data_type_name} DataFrame is missing '{id_column}' column. Starting fresh.")
            df = pd.DataFrame(columns=columns)
        else:
            # Convert ID column to appropriate type for proper lookups
            if data_type_name != 'status':  # Status ID might be string
                df[id_column] = df[id_column].astype(int)
    else:
        df = pd.DataFrame(columns=columns)

    # Collect all data from jobs into a single DataFrame
    all_data = []
    for job in jobs:
        extracted_data = extractor_func(job)
        if extracted_data:
            if isinstance(extracted_data, list):
                for item in extracted_data:
                    all_data.append(item.model_dump())
            else:
                # Single object
                all_data.append(extracted_data.model_dump())

    if not all_data:
        logger.info(f"No {data_type_name} found in jobs data")
        return df

    # Create DataFrame from all extracted data
    new_df = pd.DataFrame(all_data)
    
    if df.empty:
        # No previous data, use all new data
        new_added_count = len(new_df)
        df = new_df
    else:
        # Merge with existing data
        # First, identify which items are new vs existing based on id
        existing_ids = set(df[id_column])
        new_mask = ~new_df[id_column].isin(existing_ids)
        
        # Handle existing items - update them
        existing_items = new_df[~new_mask]
        if not existing_items.empty:
            # Update existing items by merging on id
            df = df.set_index(id_column)
            existing_items = existing_items.set_index(id_column)
            
            # Update existing records
            df.update(existing_items)
            df = df.reset_index()
            updated_count = len(existing_items)
        
        # Handle new items - add them directly
        new_items = new_df[new_mask]
        if not new_items.empty:
            df = pd.concat([df, new_items], ignore_index=True)
            new_added_count = len(new_items)

    logger.info(f"Processed {data_type_name}: {len(df)} total {data_type_name} ({new_added_count} new, {updated_count} updated)")

    if not df.empty:
        if df[id_column].duplicated().any():
            logger.warning(f"Duplicate {id_column}s found in {data_type_name} data. Rectifying...")
            df = df.drop_duplicates(subset=[id_column], keep='last')

        df = df.sort_values(by=[id_column])
        df = df.set_index(id_column)

    return df

def main():
    parser = argparse.ArgumentParser(description='Update parquet database with new job data including skills, companies, districts, position levels, employment types, status, flexible work arrangements, and categories')
    parser.add_argument('previous_date', type=str, help='Previous date in YYYYMMDD format')
    parser.add_argument('next_date', type=str, help='Next date in YYYYMMDD format')
    parser.add_argument('raw_data_dir', type=str, default='raw_data', help='Directory containing raw data')
    parser.add_argument('db_data_dir', type=str, default='db_data', help='Directory for parquet database files')
    
    args = parser.parse_args()
    
    # Validate date formats
    try:
        datetime.strptime(args.previous_date, "%Y%m%d")
        datetime.strptime(args.next_date, "%Y%m%d")
    except ValueError:
        logger.error("Dates must be in YYYYMMDD format")
        return
    
    # Validate directories
    if not os.path.exists(args.raw_data_dir):
        logger.error(f"Raw data directory does not exist: {args.raw_data_dir}")
        return
    
    # Create db_data_dir if it doesn't exist
    os.makedirs(args.db_data_dir, exist_ok=True)
    
    # Run the update
    update_databases(args.previous_date, args.next_date, args.raw_data_dir, args.db_data_dir)

if __name__ == "__main__":
    main() 