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
            skills_lookup = {}
            next_id = 0
        else:
            # Ensure 'id' is integer type for proper max() calculation and lookups
            skills_df['id'] = skills_df['id'].astype(int)
            skills_lookup = {row['uuid']: int(row['id']) for _, row in skills_df.iterrows()}
            next_id = (skills_df['id'].max() + 1) if not skills_df.empty else 0
    else:
        skills_df = pd.DataFrame(columns=['id', 'uuid', 'skill', 'confidence'])
        skills_lookup = {}
        next_id = 0

    new_skill_records = []

    for job_idx, job in enumerate(jobs): # Iterate over jobs to process skills and collect IDs
        if job.skills:
            # Iterate over a copy or the original list of skill objects.
            # job.skills itself will not be modified in this first loop.
            current_job_skill_objects = job.skills 

            for skill_obj in current_job_skill_objects:  # skill_obj is of type schema_v2.Skills
                skill_data = skill_obj.model_dump()

                if skill_obj.uuid in skills_lookup:
                    # Existing skill, update it in skills_df
                    existing_id = skills_lookup[skill_obj.uuid]
                    
                    mask = skills_df['id'] == existing_id
                    
                    for col, value in skill_data.items():
                        if col in skills_df.columns:
                             skills_df.loc[mask, col] = value
                    
                    updated_skills_count +=1
                else:
                    # New skill
                    skill_data['id'] = next_id
                    new_skill_records.append(skill_data)
                    skills_lookup[skill_obj.uuid] = next_id
                    next_id += 1
                    new_skills_added_count += 1

    if new_skill_records:
        new_skills_df = pd.DataFrame(new_skill_records)
        skills_df = pd.concat([skills_df, new_skills_df], ignore_index=True)

    logger.info(f"Processed skills: {len(skills_df)} total skills ({new_skills_added_count} new, {updated_skills_count} updated)")

    if not skills_df.empty:
        if skills_df['id'].duplicated().any():
            logger.warning("Duplicate IDs found in skills data. Rectifying...")
            # Potentially complex to fix if actual duplicates are generated.
            # For now, we assume IDs should be unique due to next_id logic.
            # If duplicates arise from previous_df not having unique IDs, that's a deeper issue.
            skills_df = skills_df.drop_duplicates(subset=['id'], keep='last')


        if skills_df['uuid'].duplicated().any():
            logger.warning("Duplicate UUIDs found in skills data. Rectifying...")
            # This implies a skill UUID appeared multiple times in the input, or was already in skills_df with a different ID.
            # The logic should handle this by updating the first instance.
            skills_df = skills_df.drop_duplicates(subset=['uuid'], keep='last')


        skills_df = skills_df.sort_values(by=['id', 'uuid'])
        skills_df = skills_df.set_index(['id', 'uuid'])

    # Now map skills in jobs to skill IDs to save space
    for job in jobs:
        if job.skills:
            skill_ids = []
            for skill_obj in job.skills:
                if skill_obj.uuid in skills_lookup:
                    skill_ids.append(skills_lookup[skill_obj.uuid])
            # Replace the skills objects with just the list of IDs
            job.skills = skill_ids

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

def update_skills_database(previous_date: str, next_date: str, raw_data_dir: str, db_data_dir: str):
    """Main function to update the skills parquet database."""
    logger.info(f"Starting skills database update from {previous_date} to {next_date}")
    
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
    
    logger.info("Skills database update completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Update skills parquet database with new job data')
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
    update_skills_database(args.previous_date, args.next_date, args.raw_data_dir, args.db_data_dir)

if __name__ == "__main__":
    main() 