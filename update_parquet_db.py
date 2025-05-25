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

def update_skills_database(previous_date: str, next_date: str, raw_data_dir: str, db_data_dir: str):
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
    
    # Process Districts Data
    logger.info("Processing districts data...")
    previous_districts_df = read_previous_parquet(db_data_dir, previous_date, "districts")
    districts_df = process_districts_data(jobs, previous_districts_df)
    
    # Save districts data
    districts_output_path = os.path.join(next_date_dir, f"{next_date}_districts.parquet")
    save_parquet_file(districts_df, districts_output_path, "districts")
    
    # Process Position Levels Data
    logger.info("Processing position levels data...")
    previous_position_levels_df = read_previous_parquet(db_data_dir, previous_date, "position_levels")
    position_levels_df = process_position_levels_data(jobs, previous_position_levels_df)
    
    # Save position levels data
    position_levels_output_path = os.path.join(next_date_dir, f"{next_date}_position_levels.parquet")
    save_parquet_file(position_levels_df, position_levels_output_path, "position_levels")
    
    # Process Employment Types Data
    logger.info("Processing employment types data...")
    previous_employment_types_df = read_previous_parquet(db_data_dir, previous_date, "employment_types")
    employment_types_df = process_employment_types_data(jobs, previous_employment_types_df)
    
    # Save employment types data
    employment_types_output_path = os.path.join(next_date_dir, f"{next_date}_employment_types.parquet")
    save_parquet_file(employment_types_df, employment_types_output_path, "employment_types")
    
    # Process Status Data
    logger.info("Processing status data...")
    previous_status_df = read_previous_parquet(db_data_dir, previous_date, "status")
    status_df = process_status_data(jobs, previous_status_df)
    
    # Save status data
    status_output_path = os.path.join(next_date_dir, f"{next_date}_status.parquet")
    save_parquet_file(status_df, status_output_path, "status")
    
    # Process Flexible Work Arrangements Data
    logger.info("Processing flexible work arrangements data...")
    previous_flexible_work_arrangements_df = read_previous_parquet(db_data_dir, previous_date, "flexible_work_arrangements")
    flexible_work_arrangements_df = process_flexible_work_arrangements_data(jobs, previous_flexible_work_arrangements_df)
    
    # Save flexible work arrangements data
    flexible_work_arrangements_output_path = os.path.join(next_date_dir, f"{next_date}_flexible_work_arrangements.parquet")
    save_parquet_file(flexible_work_arrangements_df, flexible_work_arrangements_output_path, "flexible_work_arrangements")
    
    # Process Categories Data
    logger.info("Processing categories data...")
    previous_categories_df = read_previous_parquet(db_data_dir, previous_date, "categories")
    categories_df = process_categories_data(jobs, previous_categories_df)
    
    # Save categories data
    categories_output_path = os.path.join(next_date_dir, f"{next_date}_categories.parquet")
    save_parquet_file(categories_df, categories_output_path, "categories")
    
    logger.info("Database update completed successfully!")

def process_districts_data(jobs: List[Job], previous_districts_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process districts data from jobs, update districts DataFrame using the original district ID."""

    new_districts_added_count = 0
    updated_districts_count = 0

    if previous_districts_df is not None and not previous_districts_df.empty:
        if previous_districts_df.index.names != [None]:
            districts_df = previous_districts_df.reset_index()
        else:
            districts_df = previous_districts_df.copy()

        if 'id' not in districts_df.columns:
            logger.warning("Previous districts DataFrame is missing 'id' column. Starting fresh.")
            districts_df = pd.DataFrame(columns=['id', 'sectors', 'region_id', 'location', 'region'])
        else:
            # Ensure 'id' is integer type for proper lookups
            districts_df['id'] = districts_df['id'].astype(int)
    else:
        districts_df = pd.DataFrame(columns=['id', 'sectors', 'region_id', 'location', 'region'])

    # Collect all districts from jobs into a single DataFrame
    all_districts_data = []
    for job in jobs:
        if job.address and job.address.districts:
            for district_obj in job.address.districts:
                district_data = district_obj.model_dump()
                all_districts_data.append(district_data)

    if not all_districts_data:
        logger.info("No districts found in jobs data")
        return districts_df

    # Create DataFrame from all districts
    new_districts_df = pd.DataFrame(all_districts_data)
    
    if districts_df.empty:
        # No previous data, use all new districts
        new_districts_added_count = len(new_districts_df)
        districts_df = new_districts_df
    else:
        # Merge with existing districts data
        # First, identify which districts are new vs existing based on id
        existing_ids = set(districts_df['id'])
        new_districts_mask = ~new_districts_df['id'].isin(existing_ids)
        
        # Handle existing districts - update them
        existing_districts = new_districts_df[~new_districts_mask]
        if not existing_districts.empty:
            # Update existing districts by merging on id
            districts_df = districts_df.set_index('id')
            existing_districts = existing_districts.set_index('id')
            
            # Update existing records
            districts_df.update(existing_districts)
            districts_df = districts_df.reset_index()
            updated_districts_count = len(existing_districts)
        
        # Handle new districts - add them directly
        new_districts = new_districts_df[new_districts_mask]
        if not new_districts.empty:
            districts_df = pd.concat([districts_df, new_districts], ignore_index=True)
            new_districts_added_count = len(new_districts)

    logger.info(f"Processed districts: {len(districts_df)} total districts ({new_districts_added_count} new, {updated_districts_count} updated)")

    if not districts_df.empty:
        if districts_df['id'].duplicated().any():
            logger.warning("Duplicate ids found in districts data. Rectifying...")
            districts_df = districts_df.drop_duplicates(subset=['id'], keep='last')

        districts_df = districts_df.sort_values(by=['id'])
        districts_df = districts_df.set_index('id')

    return districts_df

def process_position_levels_data(jobs: List[Job], previous_position_levels_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process position levels data from jobs, update position levels DataFrame using the original ID."""

    new_position_levels_added_count = 0
    updated_position_levels_count = 0

    if previous_position_levels_df is not None and not previous_position_levels_df.empty:
        if previous_position_levels_df.index.names != [None]:
            position_levels_df = previous_position_levels_df.reset_index()
        else:
            position_levels_df = previous_position_levels_df.copy()

        if 'id' not in position_levels_df.columns:
            logger.warning("Previous position levels DataFrame is missing 'id' column. Starting fresh.")
            position_levels_df = pd.DataFrame(columns=['id', 'position'])
        else:
            # Ensure 'id' is integer type for proper lookups
            position_levels_df['id'] = position_levels_df['id'].astype(int)
    else:
        position_levels_df = pd.DataFrame(columns=['id', 'position'])

    # Collect all position levels from jobs into a single DataFrame
    all_position_levels_data = []
    for job in jobs:
        if job.position_levels:
            for position_level_obj in job.position_levels:
                position_level_data = position_level_obj.model_dump()
                all_position_levels_data.append(position_level_data)

    if not all_position_levels_data:
        logger.info("No position levels found in jobs data")
        return position_levels_df

    # Create DataFrame from all position levels
    new_position_levels_df = pd.DataFrame(all_position_levels_data)
    
    if position_levels_df.empty:
        # No previous data, use all new position levels
        new_position_levels_added_count = len(new_position_levels_df)
        position_levels_df = new_position_levels_df
    else:
        # Merge with existing position levels data
        # First, identify which position levels are new vs existing based on id
        existing_ids = set(position_levels_df['id'])
        new_position_levels_mask = ~new_position_levels_df['id'].isin(existing_ids)
        
        # Handle existing position levels - update them
        existing_position_levels = new_position_levels_df[~new_position_levels_mask]
        if not existing_position_levels.empty:
            # Update existing position levels by merging on id
            position_levels_df = position_levels_df.set_index('id')
            existing_position_levels = existing_position_levels.set_index('id')
            
            # Update existing records
            position_levels_df.update(existing_position_levels)
            position_levels_df = position_levels_df.reset_index()
            updated_position_levels_count = len(existing_position_levels)
        
        # Handle new position levels - add them directly
        new_position_levels = new_position_levels_df[new_position_levels_mask]
        if not new_position_levels.empty:
            position_levels_df = pd.concat([position_levels_df, new_position_levels], ignore_index=True)
            new_position_levels_added_count = len(new_position_levels)

    logger.info(f"Processed position levels: {len(position_levels_df)} total position levels ({new_position_levels_added_count} new, {updated_position_levels_count} updated)")

    if not position_levels_df.empty:
        if position_levels_df['id'].duplicated().any():
            logger.warning("Duplicate ids found in position levels data. Rectifying...")
            position_levels_df = position_levels_df.drop_duplicates(subset=['id'], keep='last')

        position_levels_df = position_levels_df.sort_values(by=['id'])
        position_levels_df = position_levels_df.set_index('id')

    return position_levels_df

def process_employment_types_data(jobs: List[Job], previous_employment_types_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process employment types data from jobs, update employment types DataFrame using the original ID."""

    new_employment_types_added_count = 0
    updated_employment_types_count = 0

    if previous_employment_types_df is not None and not previous_employment_types_df.empty:
        if previous_employment_types_df.index.names != [None]:
            employment_types_df = previous_employment_types_df.reset_index()
        else:
            employment_types_df = previous_employment_types_df.copy()

        if 'id' not in employment_types_df.columns:
            logger.warning("Previous employment types DataFrame is missing 'id' column. Starting fresh.")
            employment_types_df = pd.DataFrame(columns=['id', 'employment_type'])
        else:
            # Ensure 'id' is integer type for proper lookups
            employment_types_df['id'] = employment_types_df['id'].astype(int)
    else:
        employment_types_df = pd.DataFrame(columns=['id', 'employment_type'])

    # Collect all employment types from jobs into a single DataFrame
    all_employment_types_data = []
    for job in jobs:
        if job.employment_types:
            for employment_type_obj in job.employment_types:
                employment_type_data = employment_type_obj.model_dump()
                all_employment_types_data.append(employment_type_data)

    if not all_employment_types_data:
        logger.info("No employment types found in jobs data")
        return employment_types_df

    # Create DataFrame from all employment types
    new_employment_types_df = pd.DataFrame(all_employment_types_data)
    
    if employment_types_df.empty:
        # No previous data, use all new employment types
        new_employment_types_added_count = len(new_employment_types_df)
        employment_types_df = new_employment_types_df
    else:
        # Merge with existing employment types data
        # First, identify which employment types are new vs existing based on id
        existing_ids = set(employment_types_df['id'])
        new_employment_types_mask = ~new_employment_types_df['id'].isin(existing_ids)
        
        # Handle existing employment types - update them
        existing_employment_types = new_employment_types_df[~new_employment_types_mask]
        if not existing_employment_types.empty:
            # Update existing employment types by merging on id
            employment_types_df = employment_types_df.set_index('id')
            existing_employment_types = existing_employment_types.set_index('id')
            
            # Update existing records
            employment_types_df.update(existing_employment_types)
            employment_types_df = employment_types_df.reset_index()
            updated_employment_types_count = len(existing_employment_types)
        
        # Handle new employment types - add them directly
        new_employment_types = new_employment_types_df[new_employment_types_mask]
        if not new_employment_types.empty:
            employment_types_df = pd.concat([employment_types_df, new_employment_types], ignore_index=True)
            new_employment_types_added_count = len(new_employment_types)

    logger.info(f"Processed employment types: {len(employment_types_df)} total employment types ({new_employment_types_added_count} new, {updated_employment_types_count} updated)")

    if not employment_types_df.empty:
        if employment_types_df['id'].duplicated().any():
            logger.warning("Duplicate ids found in employment types data. Rectifying...")
            employment_types_df = employment_types_df.drop_duplicates(subset=['id'], keep='last')

        employment_types_df = employment_types_df.sort_values(by=['id'])
        employment_types_df = employment_types_df.set_index('id')

    return employment_types_df

def process_status_data(jobs: List[Job], previous_status_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process status data from jobs, update status DataFrame using the original ID."""

    new_status_added_count = 0
    updated_status_count = 0

    if previous_status_df is not None and not previous_status_df.empty:
        if previous_status_df.index.names != [None]:
            status_df = previous_status_df.reset_index()
        else:
            status_df = previous_status_df.copy()

        if 'id' not in status_df.columns:
            logger.warning("Previous status DataFrame is missing 'id' column. Starting fresh.")
            status_df = pd.DataFrame(columns=['id', 'job_status'])
        else:
            # Ensure 'id' is integer type for proper lookups (note: status id might be string)
            pass  # Don't convert status id as it might be string
    else:
        status_df = pd.DataFrame(columns=['id', 'job_status'])

    # Collect all status from jobs into a single DataFrame
    all_status_data = []
    for job in jobs:
        if job.status:
            status_data = job.status.model_dump()
            all_status_data.append(status_data)

    if not all_status_data:
        logger.info("No status found in jobs data")
        return status_df

    # Create DataFrame from all status
    new_status_df = pd.DataFrame(all_status_data)
    
    if status_df.empty:
        # No previous data, use all new status
        new_status_added_count = len(new_status_df)
        status_df = new_status_df
    else:
        # Merge with existing status data
        # First, identify which status are new vs existing based on id
        existing_ids = set(status_df['id'])
        new_status_mask = ~new_status_df['id'].isin(existing_ids)
        
        # Handle existing status - update them
        existing_status = new_status_df[~new_status_mask]
        if not existing_status.empty:
            # Update existing status by merging on id
            status_df = status_df.set_index('id')
            existing_status = existing_status.set_index('id')
            
            # Update existing records
            status_df.update(existing_status)
            status_df = status_df.reset_index()
            updated_status_count = len(existing_status)
        
        # Handle new status - add them directly
        new_status = new_status_df[new_status_mask]
        if not new_status.empty:
            status_df = pd.concat([status_df, new_status], ignore_index=True)
            new_status_added_count = len(new_status)

    logger.info(f"Processed status: {len(status_df)} total status ({new_status_added_count} new, {updated_status_count} updated)")

    if not status_df.empty:
        if status_df['id'].duplicated().any():
            logger.warning("Duplicate ids found in status data. Rectifying...")
            status_df = status_df.drop_duplicates(subset=['id'], keep='last')

        status_df = status_df.sort_values(by=['id'])
        status_df = status_df.set_index('id')

    return status_df

def process_flexible_work_arrangements_data(jobs: List[Job], previous_flexible_work_arrangements_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process flexible work arrangements data from jobs, update flexible work arrangements DataFrame using the original ID."""

    new_flexible_work_arrangements_added_count = 0
    updated_flexible_work_arrangements_count = 0

    if previous_flexible_work_arrangements_df is not None and not previous_flexible_work_arrangements_df.empty:
        if previous_flexible_work_arrangements_df.index.names != [None]:
            flexible_work_arrangements_df = previous_flexible_work_arrangements_df.reset_index()
        else:
            flexible_work_arrangements_df = previous_flexible_work_arrangements_df.copy()

        if 'id' not in flexible_work_arrangements_df.columns:
            logger.warning("Previous flexible work arrangements DataFrame is missing 'id' column. Starting fresh.")
            flexible_work_arrangements_df = pd.DataFrame(columns=['id', 'flexible_work_arrangement'])
        else:
            # Ensure 'id' is integer type for proper lookups
            flexible_work_arrangements_df['id'] = flexible_work_arrangements_df['id'].astype(int)
    else:
        flexible_work_arrangements_df = pd.DataFrame(columns=['id', 'flexible_work_arrangement'])

    # Collect all flexible work arrangements from jobs into a single DataFrame
    all_flexible_work_arrangements_data = []
    for job in jobs:
        if job.flexible_work_arrangements:
            for flexible_work_arrangement_obj in job.flexible_work_arrangements:
                flexible_work_arrangement_data = flexible_work_arrangement_obj.model_dump()
                all_flexible_work_arrangements_data.append(flexible_work_arrangement_data)

    if not all_flexible_work_arrangements_data:
        logger.info("No flexible work arrangements found in jobs data")
        return flexible_work_arrangements_df

    # Create DataFrame from all flexible work arrangements
    new_flexible_work_arrangements_df = pd.DataFrame(all_flexible_work_arrangements_data)
    
    if flexible_work_arrangements_df.empty:
        # No previous data, use all new flexible work arrangements
        new_flexible_work_arrangements_added_count = len(new_flexible_work_arrangements_df)
        flexible_work_arrangements_df = new_flexible_work_arrangements_df
    else:
        # Merge with existing flexible work arrangements data
        # First, identify which flexible work arrangements are new vs existing based on id
        existing_ids = set(flexible_work_arrangements_df['id'])
        new_flexible_work_arrangements_mask = ~new_flexible_work_arrangements_df['id'].isin(existing_ids)
        
        # Handle existing flexible work arrangements - update them
        existing_flexible_work_arrangements = new_flexible_work_arrangements_df[~new_flexible_work_arrangements_mask]
        if not existing_flexible_work_arrangements.empty:
            # Update existing flexible work arrangements by merging on id
            flexible_work_arrangements_df = flexible_work_arrangements_df.set_index('id')
            existing_flexible_work_arrangements = existing_flexible_work_arrangements.set_index('id')
            
            # Update existing records
            flexible_work_arrangements_df.update(existing_flexible_work_arrangements)
            flexible_work_arrangements_df = flexible_work_arrangements_df.reset_index()
            updated_flexible_work_arrangements_count = len(existing_flexible_work_arrangements)
        
        # Handle new flexible work arrangements - add them directly
        new_flexible_work_arrangements = new_flexible_work_arrangements_df[new_flexible_work_arrangements_mask]
        if not new_flexible_work_arrangements.empty:
            flexible_work_arrangements_df = pd.concat([flexible_work_arrangements_df, new_flexible_work_arrangements], ignore_index=True)
            new_flexible_work_arrangements_added_count = len(new_flexible_work_arrangements)

    logger.info(f"Processed flexible work arrangements: {len(flexible_work_arrangements_df)} total flexible work arrangements ({new_flexible_work_arrangements_added_count} new, {updated_flexible_work_arrangements_count} updated)")

    if not flexible_work_arrangements_df.empty:
        if flexible_work_arrangements_df['id'].duplicated().any():
            logger.warning("Duplicate ids found in flexible work arrangements data. Rectifying...")
            flexible_work_arrangements_df = flexible_work_arrangements_df.drop_duplicates(subset=['id'], keep='last')

        flexible_work_arrangements_df = flexible_work_arrangements_df.sort_values(by=['id'])
        flexible_work_arrangements_df = flexible_work_arrangements_df.set_index('id')

    return flexible_work_arrangements_df

def process_categories_data(jobs: List[Job], previous_categories_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process categories data from jobs, update categories DataFrame using the original ID."""

    new_categories_added_count = 0
    updated_categories_count = 0

    if previous_categories_df is not None and not previous_categories_df.empty:
        if previous_categories_df.index.names != [None]:
            categories_df = previous_categories_df.reset_index()
        else:
            categories_df = previous_categories_df.copy()

        if 'id' not in categories_df.columns:
            logger.warning("Previous categories DataFrame is missing 'id' column. Starting fresh.")
            categories_df = pd.DataFrame(columns=['id', 'category'])
        else:
            # Ensure 'id' is integer type for proper lookups
            categories_df['id'] = categories_df['id'].astype(int)
    else:
        categories_df = pd.DataFrame(columns=['id', 'category'])

    # Collect all categories from jobs into a single DataFrame
    all_categories_data = []
    for job in jobs:
        if job.categories:
            for category_obj in job.categories:
                category_data = category_obj.model_dump()
                all_categories_data.append(category_data)

    if not all_categories_data:
        logger.info("No categories found in jobs data")
        return categories_df

    # Create DataFrame from all categories
    new_categories_df = pd.DataFrame(all_categories_data)
    
    if categories_df.empty:
        # No previous data, use all new categories
        new_categories_added_count = len(new_categories_df)
        categories_df = new_categories_df
    else:
        # Merge with existing categories data
        # First, identify which categories are new vs existing based on id
        existing_ids = set(categories_df['id'])
        new_categories_mask = ~new_categories_df['id'].isin(existing_ids)
        
        # Handle existing categories - update them
        existing_categories = new_categories_df[~new_categories_mask]
        if not existing_categories.empty:
            # Update existing categories by merging on id
            categories_df = categories_df.set_index('id')
            existing_categories = existing_categories.set_index('id')
            
            # Update existing records
            categories_df.update(existing_categories)
            categories_df = categories_df.reset_index()
            updated_categories_count = len(existing_categories)
        
        # Handle new categories - add them directly
        new_categories = new_categories_df[new_categories_mask]
        if not new_categories.empty:
            categories_df = pd.concat([categories_df, new_categories], ignore_index=True)
            new_categories_added_count = len(new_categories)

    logger.info(f"Processed categories: {len(categories_df)} total categories ({new_categories_added_count} new, {updated_categories_count} updated)")

    if not categories_df.empty:
        if categories_df['id'].duplicated().any():
            logger.warning("Duplicate ids found in categories data. Rectifying...")
            categories_df = categories_df.drop_duplicates(subset=['id'], keep='last')

        categories_df = categories_df.sort_values(by=['id'])
        categories_df = categories_df.set_index('id')

    return categories_df

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
    update_skills_database(args.previous_date, args.next_date, args.raw_data_dir, args.db_data_dir)

if __name__ == "__main__":
    main() 