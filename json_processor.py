import json
import sys
from typing import Any, Dict, List, Union

def simplify(data: Any) -> Any:
    """
    Simplify the input data structure.
    - For None or primitive types, return as is
    - For dictionaries, apply simplify to all values
    - For lists, apply simplify to all elements and fold with mappend
    """
    # Handle None and primitive types
    if data is None or isinstance(data, (int, float, str, bool)):
        return data
    
    # Handle dictionaries
    if isinstance(data, dict):
        return {k: simplify(v) for k, v in data.items()}
    
    # Handle lists
    if isinstance(data, list):
        if not data:  # Empty list
            return []
        
        # Simplify all elements in the list
        simplified_list = [simplify(item) for item in data]
        
        # Fold the list using mappend
        result = simplified_list[0]
        for item in simplified_list[1:]:
            result = mappend(result, item)
        
        return [result]
    
    # If it's some other type, convert to string
    return str(data)

def mappend(a: Any, b: Any) -> Any:
    """
    Merge two values according to the specified rules.
    """
    # Both None
    if a is None and b is None:
        return None
    
    # One is None
    if a is None:
        return b
    if b is None:
        return a
    
    # Both are numbers
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(a, b)
    
    # Both are strings
    if isinstance(a, str) and isinstance(b, str):
        return a if len(a) >= len(b) else b
    
    # Both are dictionaries
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        # Create a superset of keys
        all_keys = set(a.keys()) | set(b.keys())
        
        for key in all_keys:
            a_val = simplify(a.get(key))
            b_val = simplify(b.get(key))
            
            # If the key exists in both dictionaries, merge the values
            if key in a and key in b:
                result[key] = mappend(a_val, b_val)
            # If the key only exists in one dictionary, use that value
            elif key in a:
                result[key] = a_val
            else:
                result[key] = b_val
        
        return result
    
    # Both are lists
    if isinstance(a, list) and isinstance(b, list):
        combined = a + b
        if not combined:
            return []
        
        result = combined[0]
        for item in combined[1:]:
            result = mappend(result, item)
        
        return [results]
    
    # Different types - return the non-None value
    # This is a fallback; the spec doesn't explicitly handle mixed types
    return a

def process_json_file(input_file: str, output_file: str) -> None:
    """
    Process a JSON file by simplifying its content and writing the result.
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Apply the simplify function
        simplified_data = simplify(data)
        
        # Write the result to the output file
        with open(output_file, 'w') as f:
            json.dump(simplified_data, f, indent=4)
        
        print(f"Successfully processed {input_file} and wrote result to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not valid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_processor.py <input_json_file> <output_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_json_file(input_file, output_file)
