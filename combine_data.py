#!/usr/bin/env python3
import argparse
import orjson
import os
from datetime import datetime
import glob
from pathlib import Path

def get_data_files(output_dir, data_type, date_str=None):
    """
    Get all data files for a specific data type and date.
    
    Args:
        output_dir (str): Base output directory
        data_type (str): Type of data (jobs, companies, etc.)
        date_str (str, optional): Date string in YYYYMMDD format. If None, uses today.
    
    Returns:
        list: List of file paths
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
    # Construct the path to the pages directory
    pages_dir = os.path.join(output_dir, data_type, date_str, "pages")
    
    if not os.path.exists(pages_dir):
        print(f"Directory not found: {pages_dir}")
        return []
    
    # Find all JSON files for this data type and date
    pattern = os.path.join(pages_dir, f"{date_str}_{data_type}_*.json")
    files = glob.glob(pattern)
    
    # Sort files by page number (extract page number from filename)
    def extract_page_number(filepath):
        filename = os.path.basename(filepath)
        # Format: YYYYMMDD_datatype_NNNN.json
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            try:
                return int(parts[-1])  # Last part should be the page number
            except ValueError:
                return 0
        return 0
    
    files.sort(key=extract_page_number)
    return files

def combine_data_files(files, data_type):
    """
    Combine multiple data files into a single data structure.
    
    Args:
        files (list): List of file paths to combine
        data_type (str): Type of data being combined
    
    Returns:
        dict: Combined data structure
    """
    combined_results = []
    total_files = len(files)
    successful_files = 0
    failed_files = 0
    
    print(f"Combining {total_files} files...")
    
    for i, filepath in enumerate(files, 1):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = orjson.loads(f.read())
            
            # Extract results from this file
            if 'results' in data and isinstance(data['results'], list):
                results = data['results']
                combined_results.extend(results)
                print(f"[{i}/{total_files}] Processed {os.path.basename(filepath)}: {len(results)} items")
                successful_files += 1
            else:
                print(f"[{i}/{total_files}] Warning: No 'results' found in {os.path.basename(filepath)}")
                successful_files += 1
                
        except orjson.JSONDecodeError as e:
            print(f"[{i}/{total_files}] Error: Invalid JSON in {os.path.basename(filepath)}: {e}")
            failed_files += 1
        except Exception as e:
            print(f"[{i}/{total_files}] Error reading {os.path.basename(filepath)}: {e}")
            failed_files += 1
    
    print(f"\nCombining complete:")
    print(f"  Successful files: {successful_files}")
    print(f"  Failed files: {failed_files}")
    print(f"  Total items combined: {len(combined_results)}")
    
    # Create the combined data structure
    combined_data = {
        "metadata": {
            "data_type": data_type,
            "combined_at": datetime.now().isoformat(),
            "total_files_processed": successful_files,
            "failed_files": failed_files,
            "total_items": len(combined_results)
        },
        "results": combined_results
    }
    
    return combined_data

def create_combine_directory(output_dir, data_type, date_str):
    """Create the combine directory if it doesn't exist"""
    combine_dir = os.path.join(output_dir, data_type, date_str, "combine")
    os.makedirs(combine_dir, exist_ok=True)
    return combine_dir

def combine_data(data_type, output_dir, date_str=None):
    """
    Combine data files for a specific data type and date.
    
    Args:
        data_type (str): Type of data to combine (jobs, companies, etc.)
        output_dir (str): Base output directory
        date_str (str, optional): Date string in YYYYMMDD format. If None, uses today.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
    print(f"Combining {data_type} data for date: {date_str}")
    
    # Get all data files for this type and date
    files = get_data_files(output_dir, data_type, date_str)
    
    if not files:
        print(f"No {data_type} files found for date {date_str} in {output_dir}")
        print(f"Expected directory: {os.path.join(output_dir, data_type, date_str, 'pages')}")
        return
    
    print(f"Found {len(files)} files to combine")
    
    # Combine the data
    combined_data = combine_data_files(files, data_type)
    
    # Create combine directory
    combine_dir = create_combine_directory(output_dir, data_type, date_str)
    
    # Create output filename
    output_filename = f"{date_str}_{data_type}_combine.json"
    output_filepath = os.path.join(combine_dir, output_filename)
    
    # Write combined data to file with pretty printing
    try:
        with open(output_filepath, 'wb') as f:
            f.write(orjson.dumps(combined_data, option=orjson.OPT_INDENT_2))
        
        print(f"\nCombined data saved to: {output_filepath}")
        print(f"File size: {os.path.getsize(output_filepath):,} bytes")
        
    except Exception as e:
        print(f"Error saving combined data: {e}")

def list_available_dates(output_dir, data_type):
    """
    List available dates for a specific data type.
    
    Args:
        output_dir (str): Base output directory
        data_type (str): Type of data
    """
    data_dir = os.path.join(output_dir, data_type)
    
    if not os.path.exists(data_dir):
        print(f"No data directory found for {data_type}: {data_dir}")
        return
    
    # Find all date directories
    date_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and len(item) == 8 and item.isdigit():
            # Check if pages directory exists and has files
            pages_dir = os.path.join(item_path, "pages")
            if os.path.exists(pages_dir):
                files = glob.glob(os.path.join(pages_dir, f"{item}_{data_type}_*.json"))
                if files:
                    date_dirs.append((item, len(files)))
    
    if not date_dirs:
        print(f"No data found for {data_type} in {output_dir}")
        return
    
    print(f"Available dates for {data_type}:")
    for date_str, file_count in sorted(date_dirs):
        print(f"  {date_str}: {file_count} files")

def main():
    parser = argparse.ArgumentParser(description='Combine data files from MyCareersFuture API')
    parser.add_argument('data_type', choices=['jobs', 'companies'], 
                       help='Type of data to combine (jobs or companies)')
    parser.add_argument('--output-dir', default='raw_data',
                       help='Output directory where data is stored (default: raw_data)')
    parser.add_argument('--date', type=str, default=None,
                       help='Date to combine in YYYYMMDD format (default: today)')
    parser.add_argument('--list-dates', action='store_true',
                       help='List available dates for the specified data type')
    
    args = parser.parse_args()
    
    # Validate date format if provided
    if args.date:
        try:
            datetime.strptime(args.date, "%Y%m%d")
        except ValueError:
            print("Error: Date must be in YYYYMMDD format (e.g., 20241201)")
            return
    
    # List available dates if requested
    if args.list_dates:
        list_available_dates(args.output_dir, args.data_type)
        return
    
    # Combine the data
    combine_data(args.data_type, args.output_dir, args.date)

if __name__ == "__main__":
    main() 