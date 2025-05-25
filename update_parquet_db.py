#!/usr/bin/env python3
import argparse
import orjson
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from schema_v2 import JobResultsContainer, Job, Skills, Company, CombinedResultsContainer

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
        return companies_container.results

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
        jobs_container = JobResultsContainer.model_validate_json(json_bytes)
        logger.info(f"Successfully parsed {len(jobs_container.results)} jobs")
        return jobs_container.results

    except Exception as e:
        logger.error(f"Error parsing jobs data: {e}")
        return None

def process_skills_data(jobs: List[Job], previous_skills_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process skills data from jobs and create/update skills DataFrame."""
    
    # Initialize skills tracking
    if previous_skills_df is not None and not previous_skills_df.empty:
        # Create a lookup dictionary for existing skills
        skills_lookup = {row['uuid']: row['id'] for _, row in previous_skills_df.iterrows()}
        next_id = previous_skills_df['id'].max() + 1
        skills_records = previous_skills_df.to_dict('records')
    else:
        skills_lookup = {}
        next_id = 0
        skills_records = []
    
    # Process skills from all jobs
    for job in jobs:
        if job.skills:
            for skill in job.skills:
                if skill.uuid not in skills_lookup:
                    # Add new skill
                    new_skill_record = {
                        'id': next_id,
                        'uuid': skill.uuid,
                        'skill': skill.skill,
                        'confidence': skill.confidence,
                    }
                    skills_records.append(new_skill_record)
                    skills_lookup[skill.uuid] = next_id
                    next_id += 1
    
    logger.info(f"Processed skills: {len(skills_records)} total skills ({len(skills_records) - len(previous_skills_df) if previous_skills_df is not None else len(skills_records)} new)")
    return pd.DataFrame(skills_records)

def process_companies_data(companies: List[Company], previous_companies_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Process companies data from companies and create/update companies DataFrame."""
    
    # Initialize companies tracking
    if previous_companies_df is not None and not previous_companies_df.empty:
        # Create a lookup dictionary for existing companies
        companies_lookup = {row['uen']: row['id'] for _, row in previous_companies_df.iterrows()}
        next_id = previous_companies_df['id'].max() + 1
        companies_records = previous_companies_df.to_dict('records')
        
        new_companies_count = 0
        updated_companies_count = 0
        
        # Process each company: update existing or create new
        for company in companies:
            if company.uen in companies_lookup:
                # Update existing company data
                existing_id = companies_lookup[company.uen]
                # Find and update the existing record
                for i, record in enumerate(companies_records):
                    if record['uen'] == company.uen:
                        # Convert company to dict and preserve the existing ID
                        updated_record = company.model_dump(by_alias=True)
                        updated_record['id'] = existing_id
                        companies_records[i] = updated_record
                        updated_companies_count += 1
                        break
            else:
                # Create new company
                new_company_record = company.model_dump(by_alias=True)
                new_company_record['id'] = next_id
                companies_records.append(new_company_record)
                companies_lookup[company.uen] = next_id
                next_id += 1
                new_companies_count += 1
        
        logger.info(f"Processed companies: {len(companies_records)} total companies ({new_companies_count} new, {updated_companies_count} updated)")
    else:
        companies_lookup = {}
        next_id = 0
        companies_records = []
        
        # Add all companies as new
        for company in companies:
            new_company_record = company.model_dump(by_alias=True)
            new_company_record['id'] = next_id
            companies_records.append(new_company_record)
            companies_lookup[company.uen] = next_id
            next_id += 1
        
        logger.info(f"Processed companies: {len(companies_records)} total companies ({len(companies_records)} new)")
    
    df = pd.DataFrame(companies_records)
    # Set index to both id and uen for efficient lookups
    if not df.empty:
        df = df.set_index(['id', 'uen'])
    return df

def save_parquet_file(df: pd.DataFrame, output_path: str, table_name: str):
    """Save DataFrame to parquet file."""
    try:
        df.to_parquet(output_path, index=False, engine='pyarrow')
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

    # Process Companies Data
    logger.info("Processing companies data...")
    companies = read_raw_companies_data(raw_data_dir, next_date)
    previous_companies_df = read_previous_parquet(db_data_dir, previous_date, "companies")
    companies_df = process_companies_data(companies, previous_companies_df)
    
    # Save companies data
    companies_output_path = os.path.join(next_date_dir, "companies.parquet")
    save_parquet_file(companies_df, companies_output_path, "companies")
    
    # Read raw jobs data
    jobs = read_raw_jobs_data(raw_data_dir, next_date)
    if not jobs:
        logger.error("Failed to read jobs data, aborting")
        return
    
    # Process Skills Data
    logger.info("Processing skills data...")
    previous_skills_df = read_previous_parquet(db_data_dir, previous_date, "skills")
    skills_df = process_skills_data(jobs, previous_skills_df)
    
    # Save skills data
    skills_output_path = os.path.join(next_date_dir, "skills.parquet")
    save_parquet_file(skills_df, skills_output_path, "skills")
    
    logger.info("Skills database update completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Update skills parquet database with new job data')
    parser.add_argument('previous_date', type=str, help='Previous date in YYYYMMDD format')
    parser.add_argument('next_date', type=str, help='Next date in YYYYMMDD format')
    parser.add_argument('raw_data_dir', type=str, help='Directory containing raw data')
    parser.add_argument('db_data_dir', type=str, help='Directory for parquet database files')
    
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