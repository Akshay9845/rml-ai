#!/usr/bin/env python3
"""
Fast RML Combiner - Optimized for large-scale combination of RML components
"""

import json
import os
import re
import sys
from collections import defaultdict
from tqdm import tqdm
import ijson  # For streaming JSON parsing

# Configuration
INPUT_DIRS = [
    "/Users/elite/R-LLM/data/rml_extracted/part2_previous",
    "/Users/elite/R-LLM/data/cr_simple"
]
OUTPUT_FILE = "/Users/elite/R-LLM/data/combined_rml_final.jsonl"
TEMP_DIR = "/tmp/rml_combiner"

# RML fields to combine
RML_FIELDS = [
    'concept', 'emotion', 'entity', 'event', 
    'intent', 'reasoning', 'summary', 'tag', 
    'triple', 'vector'
]

def normalize_uri(uri):
    """Normalize URI for better matching"""
    if not uri:
        return ""
    try:
        # Basic normalization
        uri = str(uri).strip()
        # Remove common tracking parameters and fragments
        uri = re.sub(r'[?#&](utm_|ref=|source=)[^&]*', '', uri)
        # Remove trailing slashes and normalize case
        uri = uri.rstrip('/').lower()
        return uri
    except Exception as e:
        print(f"Error normalizing URI: {e}")
        return str(uri).strip()

def process_file(file_path):
    """Process a single file and return entries"""
    entries = defaultdict(lambda: {field: set() for field in RML_FIELDS})
    field_name = os.path.splitext(os.path.basename(file_path))[0].rstrip('s')
    
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            # Use ijson for streaming JSON parsing
            for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}", 
                           total=file_size/1024, unit='KB'):
                try:
                    data = json.loads(line)
                    uri = data.get('uri')
                    if not uri:
                        continue
                        
                    norm_uri = normalize_uri(uri)
                    value = data.get(field_name) or data.get(field_name + 's')
                    
                    if value:
                        if isinstance(value, (list, set)):
                            entries[norm_uri][field_name].update(v for v in value if v)
                        elif value:  # Skip empty strings
                            entries[norm_uri][field_name].add(value)
                            
                except (json.JSONDecodeError, TypeError) as e:
                    continue
                    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return entries

def combine_all_rml():
    """Combine all RML components into a single file"""
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = []
    for dir_path in INPUT_DIRS:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.jsonl') and 'combined' not in file:
                    jsonl_files.append(os.path.join(root, file))
    
    print(f"Found {len(jsonl_files)} RML component files")
    
    # Process files one by one (memory efficient)
    all_entries = defaultdict(lambda: {field: set() for field in RML_FIELDS})
    
    for file_path in tqdm(jsonl_files, desc="Processing files"):
        try:
            entries = process_file(file_path)
            # Merge into all_entries
            for uri, fields in entries.items():
                for field, values in fields.items():
                    all_entries[uri][field].update(values)
            
            # Periodically write to disk to save memory
            if sys.getsizeof(all_entries) > 500 * 1024 * 1024:  # 500MB
                write_partial_results(all_entries, TEMP_DIR)
                all_entries = defaultdict(lambda: {field: set() for field in RML_FIELDS})
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write remaining entries
    if all_entries:
        write_partial_results(all_entries, TEMP_DIR)
    
    # Combine all partial results
    combine_partial_results(TEMP_DIR, OUTPUT_FILE)
    
    print(f"✅ Success! Combined data written to {OUTPUT_FILE}")

def write_partial_results(entries, temp_dir):
    """Write a portion of results to a temporary file"""
    if not entries:
        return
        
    temp_file = os.path.join(temp_dir, f"partial_{len(os.listdir(temp_dir))}.jsonl")
    with open(temp_file, 'w', encoding='utf-8') as f:
        for uri, fields in entries.items():
            entry = {'uri': uri}
            for field, values in fields.items():
                if values:
                    if field == 'vector':
                        entry[field] = [float(v) for v in values if is_number(v)]
                    else:
                        entry[field] = list(values)
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def is_number(s):
    """Check if a string can be converted to a number"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def combine_partial_results(temp_dir, output_file):
    """Combine all partial result files into the final output"""
    print("\nCombining partial results...")
    partial_files = [f for f in os.listdir(temp_dir) if f.startswith('partial_')]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for pfile in tqdm(partial_files, desc="Combining partial files"):
            try:
                with open(os.path.join(temp_dir, pfile), 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                # Remove the partial file after processing
                os.remove(os.path.join(temp_dir, pfile))
            except Exception as e:
                print(f"Error combining {pfile}: {e}")
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

if __name__ == "__main__":
    try:
        combine_all_rml()
    except KeyboardInterrupt:
        print("\nProcess interrupted. Cleaning up...")
        sys.exit(1)
