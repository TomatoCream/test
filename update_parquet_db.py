#!/usr/bin/env python3
import argparse
import orjson
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from schema_v2 import JobResultsContainer, Job, Skills

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(db_data_dir: str, next_date: str) -> str:
    """Create necessary directories for the database."""
    next_date_dir = os.path.join(db_data_dir, next_date)
    os.makedirs(next_date_dir, exist_ok=True)
    return next_date_dir

def read_previous_parquet(db_data_dir: str, previous_date: str, table_name: str) -> Optional[pd.DataFrame]:
    """Read previous day's parquet file if it exists."""
    previous_parquet_path = os.path.join(db_data_dir, previous_date, f"{previous_date}_{table_name}.parquet")
    
    if os.path.exists(previous_parquet_path):
        logger.info(f"Reading previous parquet file: {previous_parquet_path}")
        try:
            df = pd.read_parquet(previous_parquet_path)
            logger.info(f"Loaded {len(df)} records from previous {table_name} data")
            return df
        except Exception as e:
            logger.error(f"Error reading previous parquet file: {e}")
            return None
    else:
        logger.info(f"No previous parquet file found at {previous_parquet_path}, starting fresh")
        return None

def read_raw_jobs_data(raw_data_dir: str, next_date: str) -> Optional[List[Job]]:
    """Read and parse the raw jobs JSON data."""
    jobs_file_path = os.path.join(raw_data_dir, "jobs", next_date, "combine", f"{next_date}_jobs_combine.json")
    
    if not os.path.exists(jobs_file_path):
        logger.error(f"Jobs file not found: {jobs_file_path}")
        return None
    
    logger.info(f"Reading jobs data from: {jobs_file_path}")
    
    try:
        with open(jobs_file_path, 'rb') as f:
            raw_data = orjson.loads(f.read())
        
        # Validate and parse using Pydantic schema
        jobs_container = JobResultsContainer(**raw_data)
        logger.info(f"Successfully parsed {len(jobs_container.results)} jobs")
        return jobs_container.results
        
    except Exception as e:
        logger.error(f"Error reading/parsing jobs data: {e}")
        return None

def extract_skills_data(jobs: List[Job], next_date: str) -> pd.DataFrame:
    """Extract skills data from jobs and create a DataFrame."""
    skills_records = []
    
    for job in jobs:
        if job.skills:
            for skill in job.skills:
                # TODO: implement
                pass
    
    logger.info(f"Extracted {len(skills_records)} skill records from {len(jobs)} jobs")
    return pd.DataFrame(skills_records)

def extract_jobs_data(jobs: List[Job], next_date: str) -> pd.DataFrame:
    """Extract core job data and create a DataFrame."""
    jobs_records = []
    
    for job in jobs:
        # TODO: implement
        pass
    
    logger.info(f"Extracted {len(jobs_records)} job records")
    return pd.DataFrame(jobs_records)

def extract_companies_data(jobs: List[Job], next_date: str) -> pd.DataFrame:
    """Extract unique company data from jobs."""
    companies_dict = {}
    
    for job in jobs:
        # Process hiring company
        pass

    logger.info(f"Extracted {len(companies_records)} unique companies")
    return pd.DataFrame(companies_records)

def combine_and_deduplicate_data(previous_df: Optional[pd.DataFrame], new_df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
    """Combine previous and new data, removing duplicates based on key columns."""
    if previous_df is None or previous_df.empty:
        logger.info("No previous data to combine, using only new data")
        return new_df
    
    # Combine dataframes
    combined_df = pd.concat([previous_df, new_df], ignore_index=True)
    
    # Remove duplicates based on key columns, keeping the latest (last occurrence)
    combined_df = combined_df.drop_duplicates(subset=key_columns, keep='last')
    
    logger.info(f"Combined data: {len(previous_df)} previous + {len(new_df)} new = {len(combined_df)} total (after deduplication)")
    return combined_df

def save_parquet_file(df: pd.DataFrame, output_path: str, table_name: str):
    """Save DataFrame to parquet file."""
    try:
        df.to_parquet(output_path, index=False, engine='pyarrow')
        file_size = os.path.getsize(output_path)
        logger.info(f"Saved {table_name} data to {output_path} ({file_size:,} bytes, {len(df)} records)")
    except Exception as e:
        logger.error(f"Error saving {table_name} parquet file: {e}")
        raise

def update_parquet_database(previous_date: str, next_date: str, raw_data_dir: str, db_data_dir: str):
    """Main function to update the parquet database."""
    logger.info(f"Starting parquet database update from {previous_date} to {next_date}")
    
    # Setup directories
    next_date_dir = setup_directories(db_data_dir, next_date)
    
    # Read raw jobs data
    jobs = read_raw_jobs_data(raw_data_dir, next_date)
    if not jobs:
        logger.error("Failed to read jobs data, aborting")
        return
    
    # Process Skills Data
    logger.info("Processing skills data...")
    previous_skills_df = read_previous_parquet(db_data_dir, previous_date, "skills")
    new_skills_df = extract_skills_data(jobs, next_date)
    
    if not new_skills_df.empty:
        combined_skills_df = combine_and_deduplicate_data(
            previous_skills_df, 
            new_skills_df, 
            ['job_uuid', 'skill_uuid']
        )
        skills_output_path = os.path.join(next_date_dir, f"{next_date}_skills.parquet")
        save_parquet_file(combined_skills_df, skills_output_path, "skills")
    
    # Process Jobs Data
    logger.info("Processing jobs data...")
    previous_jobs_df = read_previous_parquet(db_data_dir, previous_date, "jobs")
    new_jobs_df = extract_jobs_data(jobs, next_date)
    
    if not new_jobs_df.empty:
        combined_jobs_df = combine_and_deduplicate_data(
            previous_jobs_df, 
            new_jobs_df, 
            ['job_uuid']
        )
        jobs_output_path = os.path.join(next_date_dir, f"{next_date}_jobs.parquet")
        save_parquet_file(combined_jobs_df, jobs_output_path, "jobs")
    
    # Process Companies Data
    logger.info("Processing companies data...")
    previous_companies_df = read_previous_parquet(db_data_dir, previous_date, "companies")
    new_companies_df = extract_companies_data(jobs, next_date)
    
    if not new_companies_df.empty:
        combined_companies_df = combine_and_deduplicate_data(
            previous_companies_df, 
            new_companies_df, 
            ['company_uen']
        )
        companies_output_path = os.path.join(next_date_dir, f"{next_date}_companies.parquet")
        save_parquet_file(combined_companies_df, companies_output_path, "companies")
    
    logger.info("Parquet database update completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Update parquet database with new job data')
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
    update_parquet_database(args.previous_date, args.next_date, args.raw_data_dir, args.db_data_dir)

if __name__ == "__main__":
    main() 