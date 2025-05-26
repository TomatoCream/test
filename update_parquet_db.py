#!/usr/bin/env python3
import argparse
import orjson
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, cast
import logging
from schema_v2 import Company, CombinedResultsContainer

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

def read_raw_jobs_data(raw_data_dir: str, next_date: str) -> Optional[pd.DataFrame]:
    """Read and parse the raw jobs JSON data directly into a DataFrame."""
    json_bytes = read_raw_json_data(raw_data_dir, next_date, "jobs")

    if json_bytes is None:
        return None

    try:
        # Parse JSON directly without Pydantic validation
        jobs_data = orjson.loads(json_bytes)
        
        # Extract the results array
        if 'results' in jobs_data:
            jobs_list = jobs_data['results']
        else:
            jobs_list = jobs_data
        
        # Convert to DataFrame
        jobs_df = pd.DataFrame(jobs_list)
        logger.info(f"Successfully parsed {len(jobs_df)} jobs into DataFrame")
        return jobs_df

    except Exception as e:
        logger.error(f"Error parsing jobs data: {e}")
        return None

def process_skills_data(jobs_df: pd.DataFrame, previous_skills_df: Optional[pd.DataFrame], verbose_duplicates: bool = False) -> pd.DataFrame:
    """Process skills data from jobs DataFrame, update skills DataFrame, and replace job skills with IDs."""

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

    # Collect all unique skills from jobs DataFrame (deduplicate by UUID during extraction)
    all_skills_data = []
    seen_uuids = set()
    
    for _, job_row in jobs_df.iterrows():
        if 'skills' in job_row and job_row['skills'] is not None:
            skills_list = job_row['skills']
            if isinstance(skills_list, list):
                for skill_dict in skills_list:
                    if isinstance(skill_dict, dict) and 'uuid' in skill_dict:
                        skill_uuid = skill_dict['uuid']
                        if skill_uuid not in seen_uuids:
                            all_skills_data.append(skill_dict)
                            seen_uuids.add(skill_uuid)

    if not all_skills_data:
        logger.info("No skills found in jobs data")
        return skills_df

    # Create DataFrame from unique skills
    new_skills_df = pd.DataFrame(all_skills_data)
    
    if skills_df.empty:
        # No previous data, assign IDs to all skills
        new_skills_df['id'] = range(len(new_skills_df))
        new_skills_added_count = len(new_skills_df)
        skills_df = new_skills_df
    else:
        # Process each skill individually to check for actual differences
        existing_uuids = set(skills_df['uuid'])
        
        for _, new_skill_row in new_skills_df.iterrows():
            skill_uuid = new_skill_row['uuid']
            
            if skill_uuid in existing_uuids:
                # Check if the existing skill is actually different
                existing_skill_row = skills_df[skills_df['uuid'] == skill_uuid].iloc[0]
                
                # Compare all columns to see if there are actual differences
                has_differences = False
                for col in new_skill_row.index:
                    if col == 'id':
                        continue  # Ignore ID for comparison as requested
                    if col in existing_skill_row.index:
                        if pd.isna(new_skill_row[col]) and pd.isna(existing_skill_row[col]):
                            continue  # Both are NaN, no difference
                        elif new_skill_row[col] != existing_skill_row[col]:
                            has_differences = True
                            break
                    else:
                        # New column exists in new data but not in existing
                        has_differences = True
                        break
                
                # Only update if there are actual differences
                if has_differences:
                    # Update the existing skill (keep the original ID)
                    mask = skills_df['uuid'] == skill_uuid
                    for col in new_skill_row.index:
                        if col != 'id' and col in skills_df.columns:  # Don't update ID
                            skills_df.loc[mask, col] = new_skill_row[col]
                    updated_skills_count += 1
            else:
                # Truly new skill, assign new ID and add it
                new_skill_data = new_skill_row.to_dict()
                new_skill_data['id'] = next_id
                new_skill_df = pd.DataFrame([new_skill_data])
                skills_df = pd.concat([skills_df, new_skill_df], ignore_index=True)
                next_id += 1
                new_skills_added_count += 1

    logger.info(f"Processed skills: {len(skills_df)} total skills ({new_skills_added_count} new, {updated_skills_count} updated)")

    if not skills_df.empty:
        # Final check for any remaining duplicates (shouldn't happen with new logic)
        if skills_df['id'].duplicated().any():
            print_duplicate_data(skills_df.reset_index(), 'id', 'skills', verbose=verbose_duplicates)
            skills_df = skills_df.drop_duplicates(subset=['id'], keep='last')

        if skills_df['uuid'].duplicated().any():
            print_duplicate_data(skills_df.reset_index(), 'uuid', 'skills', verbose=verbose_duplicates)
            skills_df = skills_df.drop_duplicates(subset=['uuid'], keep='last')

        skills_df = skills_df.sort_values(by=['id', 'uuid'])
        skills_df = skills_df.set_index(['id', 'uuid'])

    return skills_df

def update_companies_data_with_jobs_data(jobs_df: pd.DataFrame, companies_df: pd.DataFrame) -> None:
    """Update companies data with company information extracted from jobs DataFrame."""
    
    if companies_df.empty:
        logger.warning("Companies DataFrame is empty, cannot update with jobs data")
        return
    
    # Reset index if it's set to work with the DataFrame directly
    if companies_df.index.names != [None]:
        companies_df_work = companies_df.reset_index()
    else:
        companies_df_work = companies_df.copy()
    
    # Extract all unique companies from jobs DataFrame into a list of dicts
    companies_data = []
    companies_seen = set()
    
    for _, job_row in jobs_df.iterrows():
        companies_to_process = []
        
        # Add hiring company if it exists
        if 'hiringCompany' in job_row and job_row['hiringCompany'] is not None:
            hiring_company = job_row['hiringCompany']
            if isinstance(hiring_company, dict) and 'uen' in hiring_company and hiring_company['uen']:
                companies_to_process.append(hiring_company)
            
        # Add posted company if it exists and different from hiring company
        if 'postedCompany' in job_row and job_row['postedCompany'] is not None:
            posted_company = job_row['postedCompany']
            if isinstance(posted_company, dict) and 'uen' in posted_company and posted_company['uen']:
                hiring_uen = None
                if 'hiringCompany' in job_row and job_row['hiringCompany'] is not None:
                    hiring_company = job_row['hiringCompany']
                    if isinstance(hiring_company, dict) and 'uen' in hiring_company:
                        hiring_uen = hiring_company['uen']
                
                if posted_company['uen'] != hiring_uen:
                    companies_to_process.append(posted_company)
        
        # Process each company found in this job
        for company_dict in companies_to_process:
            if company_dict['uen'] not in companies_seen:
                companies_seen.add(company_dict['uen'])
                companies_data.append(company_dict)
    
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

def process_companies_data_from_new_file(companies: List[Company], previous_companies_df: Optional[pd.DataFrame], verbose_duplicates: bool = False) -> pd.DataFrame:
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
            print_duplicate_data(companies_df, 'id', 'companies', verbose=verbose_duplicates)
        if companies_df['uen'].duplicated().any():
            print_duplicate_data(companies_df, 'uen', 'companies', verbose=verbose_duplicates)
        
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

def process_jobs_data(
    jobs_df: pd.DataFrame, 
    previous_jobs_df: Optional[pd.DataFrame],
    companies_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    districts_df: pd.DataFrame,
    position_levels_df: pd.DataFrame,
    employment_types_df: pd.DataFrame,
    status_df: pd.DataFrame,
    flexible_work_arrangements_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    verbose_duplicates: bool = False
) -> pd.DataFrame:
    """
    Process jobs DataFrame, replacing nested objects with their IDs and create/update jobs DataFrame.
    
    Args:
        jobs_df: Jobs DataFrame
        previous_jobs_df: Previous jobs DataFrame or None
        companies_df: Companies DataFrame with ID mappings
        skills_df: Skills DataFrame with ID mappings
        districts_df: Districts DataFrame with ID mappings
        position_levels_df: Position levels DataFrame with ID mappings
        employment_types_df: Employment types DataFrame with ID mappings
        status_df: Status DataFrame with ID mappings
        flexible_work_arrangements_df: Flexible work arrangements DataFrame with ID mappings
        categories_df: Categories DataFrame with ID mappings
        verbose_duplicates: Enable verbose duplicate data inspection output
    
    Returns:
        Processed jobs DataFrame
    """
    new_jobs_count = 0
    updated_jobs_count = 0

    # Create lookup dictionaries for ID mappings
    companies_lookup = {}
    if not companies_df.empty:
        companies_df_reset = companies_df.reset_index() if companies_df.index.names != [None] else companies_df
        companies_lookup = {row['uen']: row['id'] for _, row in companies_df_reset.iterrows()}

    skills_lookup = {}
    if not skills_df.empty:
        skills_df_reset = skills_df.reset_index() if skills_df.index.names != [None] else skills_df
        skills_lookup = {row['uuid']: row['id'] for _, row in skills_df_reset.iterrows()}

    districts_lookup = {}
    if not districts_df.empty:
        districts_df_reset = districts_df.reset_index() if districts_df.index.names != [None] else districts_df
        districts_lookup = {row['id']: row['id'] for _, row in districts_df_reset.iterrows()}

    position_levels_lookup = {}
    if not position_levels_df.empty:
        position_levels_df_reset = position_levels_df.reset_index() if position_levels_df.index.names != [None] else position_levels_df
        position_levels_lookup = {row['id']: row['id'] for _, row in position_levels_df_reset.iterrows()}

    employment_types_lookup = {}
    if not employment_types_df.empty:
        employment_types_df_reset = employment_types_df.reset_index() if employment_types_df.index.names != [None] else employment_types_df
        employment_types_lookup = {row['id']: row['id'] for _, row in employment_types_df_reset.iterrows()}

    status_lookup = {}
    if not status_df.empty:
        status_df_reset = status_df.reset_index() if status_df.index.names != [None] else status_df
        status_lookup = {row['id']: row['id'] for _, row in status_df_reset.iterrows()}

    flexible_work_arrangements_lookup = {}
    if not flexible_work_arrangements_df.empty:
        fwa_df_reset = flexible_work_arrangements_df.reset_index() if flexible_work_arrangements_df.index.names != [None] else flexible_work_arrangements_df
        flexible_work_arrangements_lookup = {row['id']: row['id'] for _, row in fwa_df_reset.iterrows()}

    categories_lookup = {}
    if not categories_df.empty:
        categories_df_reset = categories_df.reset_index() if categories_df.index.names != [None] else categories_df
        categories_lookup = {row['id']: row['id'] for _, row in categories_df_reset.iterrows()}

    # Initialize or load previous jobs data
    if previous_jobs_df is not None and not previous_jobs_df.empty:
        if previous_jobs_df.index.names != [None]:
            final_jobs_df = previous_jobs_df.reset_index()
        else:
            final_jobs_df = previous_jobs_df.copy()
        
        if 'uuid' not in final_jobs_df.columns:
            logger.warning("Previous jobs DataFrame is missing 'uuid' column. Starting fresh.")
            final_jobs_df = pd.DataFrame()
        else:
            # Create lookup for existing jobs
            existing_job_uuids = set(final_jobs_df['uuid'])
    else:
        final_jobs_df = pd.DataFrame()
        existing_job_uuids = set()

    # Process each job
    processed_jobs = []
    for _, job_row in jobs_df.iterrows():
        # Convert job row to dict
        job_data = job_row.to_dict()
        
        # Replace nested objects with IDs
        
        # Replace hiring company with ID
        if 'hiringCompany' in job_data and job_data['hiringCompany'] is not None:
            hiring_company = job_data['hiringCompany']
            if isinstance(hiring_company, dict) and 'uen' in hiring_company and hiring_company['uen'] in companies_lookup:
                job_data['hiring_company_id'] = companies_lookup[hiring_company['uen']]
            else:
                job_data['hiring_company_id'] = None
        else:
            job_data['hiring_company_id'] = None
        job_data.pop('hiringCompany', None)
        
        # Replace posted company with ID
        if 'postedCompany' in job_data and job_data['postedCompany'] is not None:
            posted_company = job_data['postedCompany']
            if isinstance(posted_company, dict) and 'uen' in posted_company and posted_company['uen'] in companies_lookup:
                job_data['posted_company_id'] = companies_lookup[posted_company['uen']]
            else:
                job_data['posted_company_id'] = None
        else:
            job_data['posted_company_id'] = None
        job_data.pop('postedCompany', None)
        
        # Replace skills with IDs
        if 'skills' in job_data and job_data['skills'] is not None:
            skills_list = job_data['skills']
            skill_ids = []
            if isinstance(skills_list, list):
                for skill_dict in skills_list:
                    if isinstance(skill_dict, dict) and 'uuid' in skill_dict and skill_dict['uuid'] in skills_lookup:
                        skill_ids.append(skills_lookup[skill_dict['uuid']])
            job_data['skill_ids'] = skill_ids
        else:
            job_data['skill_ids'] = []
        job_data.pop('skills', None)
        
        # Replace districts with IDs
        if 'address' in job_data and job_data['address'] is not None:
            address = job_data['address']
            if isinstance(address, dict) and 'districts' in address and address['districts'] is not None:
                districts_list = address['districts']
                district_ids = []
                if isinstance(districts_list, list):
                    for district_dict in districts_list:
                        if isinstance(district_dict, dict) and 'id' in district_dict and district_dict['id'] in districts_lookup:
                            district_ids.append(districts_lookup[district_dict['id']])
                job_data['district_ids'] = district_ids
                # Keep the rest of address but remove districts
                if isinstance(job_data['address'], dict):
                    job_data['address'] = {k: v for k, v in job_data['address'].items() if k != 'districts'}
            else:
                job_data['district_ids'] = []
        else:
            job_data['district_ids'] = []
        
        # Replace position levels with IDs
        if 'positionLevels' in job_data and job_data['positionLevels'] is not None:
            position_levels_list = job_data['positionLevels']
            position_level_ids = []
            if isinstance(position_levels_list, list):
                for position_level_dict in position_levels_list:
                    if isinstance(position_level_dict, dict) and 'id' in position_level_dict and position_level_dict['id'] in position_levels_lookup:
                        position_level_ids.append(position_levels_lookup[position_level_dict['id']])
            job_data['position_level_ids'] = position_level_ids
        else:
            job_data['position_level_ids'] = []
        job_data.pop('positionLevels', None)
        
        # Replace employment types with IDs
        if 'employmentTypes' in job_data and job_data['employmentTypes'] is not None:
            employment_types_list = job_data['employmentTypes']
            employment_type_ids = []
            if isinstance(employment_types_list, list):
                for employment_type_dict in employment_types_list:
                    if isinstance(employment_type_dict, dict) and 'id' in employment_type_dict and employment_type_dict['id'] in employment_types_lookup:
                        employment_type_ids.append(employment_types_lookup[employment_type_dict['id']])
            job_data['employment_type_ids'] = employment_type_ids
        else:
            job_data['employment_type_ids'] = []
        job_data.pop('employmentTypes', None)
        
        # Replace status with ID
        if 'status' in job_data and job_data['status'] is not None:
            status = job_data['status']
            if isinstance(status, dict) and 'id' in status and status['id'] in status_lookup:
                job_data['status_id'] = status_lookup[status['id']]
            else:
                job_data['status_id'] = None
        else:
            job_data['status_id'] = None
        job_data.pop('status', None)
        
        # Replace flexible work arrangements with IDs
        if 'flexibleWorkArrangements' in job_data and job_data['flexibleWorkArrangements'] is not None:
            fwa_list = job_data['flexibleWorkArrangements']
            fwa_ids = []
            if isinstance(fwa_list, list):
                for fwa_dict in fwa_list:
                    if isinstance(fwa_dict, dict) and 'id' in fwa_dict and fwa_dict['id'] in flexible_work_arrangements_lookup:
                        fwa_ids.append(flexible_work_arrangements_lookup[fwa_dict['id']])
            job_data['flexible_work_arrangement_ids'] = fwa_ids
        else:
            job_data['flexible_work_arrangement_ids'] = []
        job_data.pop('flexibleWorkArrangements', None)
        
        # Replace categories with IDs
        if 'categories' in job_data and job_data['categories'] is not None:
            categories_list = job_data['categories']
            category_ids = []
            if isinstance(categories_list, list):
                for category_dict in categories_list:
                    if isinstance(category_dict, dict) and 'id' in category_dict and category_dict['id'] in categories_lookup:
                        category_ids.append(categories_lookup[category_dict['id']])
            job_data['category_ids'] = category_ids
        else:
            job_data['category_ids'] = []
        job_data.pop('categories', None)
        
        processed_jobs.append(job_data)
        
        # Track if this is a new or updated job
        if 'uuid' in job_data and job_data['uuid'] in existing_job_uuids:
            updated_jobs_count += 1
        else:
            new_jobs_count += 1

    # Create DataFrame from processed jobs
    new_jobs_df = pd.DataFrame(processed_jobs)
    
    if final_jobs_df.empty:
        # No previous data, use all new data
        final_jobs_df = new_jobs_df
    else:
        # Merge with existing data
        # Remove existing jobs that are being updated
        final_jobs_df = final_jobs_df[~final_jobs_df['uuid'].isin(new_jobs_df['uuid'])]
        # Add all new/updated jobs
        final_jobs_df = pd.concat([final_jobs_df, new_jobs_df], ignore_index=True)

    logger.info(f"Processed jobs: {len(final_jobs_df)} total jobs ({new_jobs_count} new, {updated_jobs_count} updated)")

    if not final_jobs_df.empty:
        # Verify uniqueness and sort
        if final_jobs_df['uuid'].duplicated().any():
            print_duplicate_data(final_jobs_df.reset_index(), 'uuid', 'jobs', verbose=verbose_duplicates)
            final_jobs_df = final_jobs_df.drop_duplicates(subset=['uuid'], keep='last')

        final_jobs_df = final_jobs_df.sort_values(by=['uuid'])
        final_jobs_df = final_jobs_df.set_index('uuid')

    return final_jobs_df

def update_databases(previous_date: str, next_date: str, raw_data_dir: str, db_data_dir: str, verbose_duplicates: bool):
    """Main function to update the parquet database with new job data including skills, companies, districts, position levels, employment types, status, flexible work arrangements, and categories."""
    logger.info(f"Starting database update from {previous_date} to {next_date}")
    
    # Setup directories
    next_date_dir = setup_directories(db_data_dir, next_date)
    
    # Read raw jobs data as DataFrame
    jobs_df = read_raw_jobs_data(raw_data_dir, next_date)
    if jobs_df is None or jobs_df.empty:
        logger.error("Failed to read jobs data, aborting")
        return

    # Inspect raw jobs data for duplicates
    inspect_raw_jobs_duplicates(jobs_df, verbose_duplicates)

    # Process Companies Data
    logger.info("Processing companies data...")
    companies = read_raw_companies_data(raw_data_dir, next_date)
    previous_companies_df = read_previous_parquet(db_data_dir, previous_date, "companies")
    companies_df = process_companies_data_from_new_file(companies, previous_companies_df, verbose_duplicates)

    # process companies data combine with jobs data
    update_companies_data_with_jobs_data(jobs_df, companies_df)
    
    # Save companies data
    companies_output_path = os.path.join(next_date_dir, f"{next_date}_companies.parquet")
    save_parquet_file(companies_df, companies_output_path, "companies")
    
    # Process Skills Data
    logger.info("Processing skills data...")
    previous_skills_df = read_previous_parquet(db_data_dir, previous_date, "skills")
    skills_df = process_skills_data(jobs_df, previous_skills_df, verbose_duplicates)
    
    # Save skills data
    skills_output_path = os.path.join(next_date_dir, f"{next_date}_skills.parquet")
    save_parquet_file(skills_df, skills_output_path, "skills")
    
    # Configuration for lookup tables processing
    lookup_tables_config = [
        {
            'name': 'districts',
            'extractor': lambda job_row: job_row.get('address', {}).get('districts') if job_row.get('address') and job_row.get('address', {}).get('districts') else None,
            'columns': ['id', 'sectors', 'region_id', 'location', 'region']
        },
        {
            'name': 'position_levels',
            'extractor': lambda job_row: job_row.get('positionLevels') if job_row.get('positionLevels') else None,
            'columns': ['id', 'position']
        },
        {
            'name': 'employment_types',
            'extractor': lambda job_row: job_row.get('employmentTypes') if job_row.get('employmentTypes') else None,
            'columns': ['id', 'employment_type']
        },
        {
            'name': 'status',
            'extractor': lambda job_row: job_row.get('status') if job_row.get('status') else None,
            'columns': ['id', 'job_status']
        },
        {
            'name': 'flexible_work_arrangements',
            'extractor': lambda job_row: job_row.get('flexibleWorkArrangements') if job_row.get('flexibleWorkArrangements') else None,
            'columns': ['id', 'flexible_work_arrangement']
        },
        {
            'name': 'categories',
            'extractor': lambda job_row: job_row.get('categories') if job_row.get('categories') else None,
            'columns': ['id', 'category']
        }
    ]
    
    # Process all lookup tables using the configuration and store results
    lookup_tables = {}
    for config in lookup_tables_config:
        table_name = config['name']
        extractor_func = config['extractor']
        columns = config['columns']
        
        logger.info(f"Processing {table_name} data...")
        previous_df = read_previous_parquet(db_data_dir, previous_date, table_name)
        processed_df = process_lookup_table_data(
            jobs_df=jobs_df,
            previous_df=previous_df,
            data_type_name=table_name,
            extractor_func=extractor_func,
            columns=columns,
            verbose_duplicates=verbose_duplicates
        )
        
        # Store the processed DataFrame for later use in jobs processing
        lookup_tables[table_name] = processed_df
        
        # Save data
        output_path = os.path.join(next_date_dir, f"{next_date}_{table_name}.parquet")
        save_parquet_file(processed_df, output_path, table_name)
    
    # Process Jobs Data (replace nested objects with IDs)
    logger.info("Processing jobs data...")
    previous_jobs_df = read_previous_parquet(db_data_dir, previous_date, "jobs")
    jobs_df = process_jobs_data(
        jobs_df=jobs_df,
        previous_jobs_df=previous_jobs_df,
        companies_df=companies_df,
        skills_df=skills_df,
        districts_df=lookup_tables['districts'],
        position_levels_df=lookup_tables['position_levels'],
        employment_types_df=lookup_tables['employment_types'],
        status_df=lookup_tables['status'],
        flexible_work_arrangements_df=lookup_tables['flexible_work_arrangements'],
        categories_df=lookup_tables['categories'],
        verbose_duplicates=verbose_duplicates
    )
    
    # Save jobs data
    jobs_output_path = os.path.join(next_date_dir, f"{next_date}_jobs.parquet")
    save_parquet_file(jobs_df, jobs_output_path, "jobs")
    
    logger.info("Database update completed successfully!")

def process_lookup_table_data(
    jobs_df: pd.DataFrame, 
    previous_df: Optional[pd.DataFrame], 
    data_type_name: str,
    extractor_func: callable,
    columns: List[str],
    id_column: str = 'id',
    verbose_duplicates: bool = False
) -> pd.DataFrame:
    """
    Generic function to process lookup table data from jobs DataFrame.
    
    Args:
        jobs_df: Jobs DataFrame
        previous_df: Previous DataFrame or None
        data_type_name: Name for logging (e.g., "districts", "employment_types")
        extractor_func: Function that takes a job row (pd.Series) and returns list of dicts or None
        columns: List of column names for the DataFrame
        id_column: Name of the ID column (default: 'id')
        verbose_duplicates: Enable verbose duplicate data inspection output
    
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

    # Collect all unique data from jobs DataFrame
    all_data = []
    seen_ids = set()
    
    for _, job_row in jobs_df.iterrows():
        extracted_data = extractor_func(job_row)
        if extracted_data:
            if isinstance(extracted_data, list):
                for item in extracted_data:
                    if isinstance(item, dict) and id_column in item:
                        item_id = item[id_column]
                        if item_id not in seen_ids:
                            all_data.append(item)
                            seen_ids.add(item_id)
            elif isinstance(extracted_data, dict) and id_column in extracted_data:
                item_id = extracted_data[id_column]
                if item_id not in seen_ids:
                    all_data.append(extracted_data)
                    seen_ids.add(item_id)

    if not all_data:
        logger.info(f"No {data_type_name} found in jobs data")
        return df

    # Create DataFrame from unique extracted data
    new_df = pd.DataFrame(all_data)
    
    if df.empty:
        # No previous data, use all new data
        new_added_count = len(new_df)
        df = new_df
    else:
        # Process each item individually to check for actual differences
        existing_ids = set(df[id_column])
        
        for _, new_row in new_df.iterrows():
            item_id = new_row[id_column]
            
            if item_id in existing_ids:
                # Check if the existing item is actually different
                existing_row = df[df[id_column] == item_id].iloc[0]
                
                # Compare all columns to see if there are actual differences
                has_differences = False
                for col in new_row.index:
                    if col in existing_row.index:
                        if pd.isna(new_row[col]) and pd.isna(existing_row[col]):
                            continue  # Both are NaN, no difference
                        elif new_row[col] != existing_row[col]:
                            has_differences = True
                            break
                    else:
                        # New column exists in new data but not in existing
                        has_differences = True
                        break
                
                # Only update if there are actual differences
                if has_differences:
                    # Update the existing row
                    mask = df[id_column] == item_id
                    for col in new_row.index:
                        if col in df.columns:
                            df.loc[mask, col] = new_row[col]
                    updated_count += 1
            else:
                # Truly new item, add it
                new_row_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_row_df], ignore_index=True)
                new_added_count += 1

    logger.info(f"Processed {data_type_name}: {len(df)} total {data_type_name} ({new_added_count} new, {updated_count} updated)")

    if not df.empty:
        # Final check for any remaining duplicates (shouldn't happen with new logic)
        if df[id_column].duplicated().any():
            print_duplicate_data(df.reset_index(), id_column, data_type_name, verbose=verbose_duplicates)
            df = df.drop_duplicates(subset=[id_column], keep='last')

        df = df.sort_values(by=[id_column])
        df = df.set_index(id_column)

    return df

def print_duplicate_data(df: pd.DataFrame, column: str, data_type: str, max_examples: int = 5, verbose: bool = True):
    """Print detailed information about duplicate data for inspection."""
    if df[column].duplicated().any():
        duplicates = df[df[column].duplicated(keep=False)]
        duplicate_values = duplicates[column].value_counts()
        
        logger.warning(f"Found {len(duplicate_values)} unique {column} values with duplicates in {data_type} data:")
        logger.warning(f"Total duplicate records: {len(duplicates)}")
        
        if not verbose:
            return True
        
        print(f"\n=== DUPLICATE {data_type.upper()} DATA INSPECTION ===")
        print(f"Column: {column}")
        print(f"Unique duplicate values: {len(duplicate_values)}")
        print(f"Total duplicate records: {len(duplicates)}")
        
        # Show top duplicate values by count
        print(f"\nTop duplicate {column} values by count:")
        for value, count in duplicate_values.head(10).items():
            print(f"  {column} '{value}': {count} occurrences")
        
        # Show detailed examples of duplicate records
        print(f"\nDetailed examples (showing up to {max_examples} duplicate groups):")
        shown_examples = 0
        for value in duplicate_values.head(max_examples).index:
            duplicate_records = duplicates[duplicates[column] == value]
            print(f"\n--- Duplicate group for {column} = '{value}' ({len(duplicate_records)} records) ---")
            
            # Reset index to show original row numbers if available
            if hasattr(duplicate_records, 'reset_index'):
                display_df = duplicate_records.reset_index()
            else:
                display_df = duplicate_records
            
            # Show all columns for first few records, then just key columns for the rest
            if len(duplicate_records) <= 3:
                print(display_df.to_string(max_cols=None, max_colwidth=50))
            else:
                # Show first 2 records in full
                print("First 2 records (full details):")
                print(display_df.head(2).to_string(max_cols=None, max_colwidth=50))
                
                # Show remaining records with key columns only
                key_columns = [column]
                if 'uuid' in display_df.columns and column != 'uuid':
                    key_columns.append('uuid')
                if 'uen' in display_df.columns and column != 'uen':
                    key_columns.append('uen')
                if 'id' in display_df.columns and column != 'id':
                    key_columns.append('id')
                
                print(f"\nRemaining {len(duplicate_records) - 2} records (key columns only):")
                # remaining_df = display_df.iloc[2:][key_columns]
                # print(remaining_df.to_string())
            
            shown_examples += 1
        
        print(f"\n=== END DUPLICATE {data_type.upper()} DATA INSPECTION ===\n")
        return True
    return False

def inspect_raw_jobs_duplicates(jobs_df: pd.DataFrame, verbose_duplicates: bool = False):
    """Inspect duplicates in raw jobs data before processing."""
    if jobs_df.empty:
        return
    
    print("\n=== RAW JOBS DATA DUPLICATE INSPECTION ===")
    
    # Check for duplicate UUIDs in raw data
    if 'uuid' in jobs_df.columns:
        if jobs_df['uuid'].duplicated().any():
            print_duplicate_data(jobs_df, 'uuid', 'raw jobs', verbose=verbose_duplicates)
        else:
            print("No duplicate UUIDs found in raw jobs data.")
    
    # Check for duplicate job post IDs if available
    if 'metadata' in jobs_df.columns:
        # Extract jobPostId from metadata if it's a dict
        job_post_ids = []
        for _, row in jobs_df.iterrows():
            if isinstance(row['metadata'], dict) and 'jobPostId' in row['metadata']:
                job_post_ids.append(row['metadata']['jobPostId'])
            else:
                job_post_ids.append(None)
        
        if job_post_ids:
            temp_df = jobs_df.copy()
            temp_df['jobPostId'] = job_post_ids
            temp_df = temp_df[temp_df['jobPostId'].notna()]
            
            if not temp_df.empty and temp_df['jobPostId'].duplicated().any():
                print_duplicate_data(temp_df, 'jobPostId', 'raw jobs (by jobPostId)', verbose=verbose_duplicates)
            else:
                print("No duplicate jobPostIds found in raw jobs data.")
    
    print("=== END RAW JOBS DATA DUPLICATE INSPECTION ===\n")

def main():
    parser = argparse.ArgumentParser(description='Update parquet database with new job data including skills, companies, districts, position levels, employment types, status, flexible work arrangements, and categories')
    parser.add_argument('previous_date', type=str, help='Previous date in YYYYMMDD format')
    parser.add_argument('next_date', type=str, help='Next date in YYYYMMDD format')
    parser.add_argument('raw_data_dir', type=str, default='raw_data', help='Directory containing raw data')
    parser.add_argument('db_data_dir', type=str, default='db_data', help='Directory for parquet database files')
    parser.add_argument('--verbose-duplicates', action='store_true', help='Enable verbose duplicate data inspection output')
    
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
    update_databases(args.previous_date, args.next_date, args.raw_data_dir, args.db_data_dir, args.verbose_duplicates)

if __name__ == "__main__":
    main() 