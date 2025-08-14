#!/usr/bin/env python3
"""
Aggregate and Prepare RML Training Data
Scans /Users/elite/R-LLM/data for all .jsonl files, merges, shuffles, and outputs training/validation sets.
"""

import os
import json
import random
from glob import glob

DATA_DIR = "/Users/elite/R-LLM/data"
OUTPUT_TRAIN = "data/all_rml_training_data.jsonl"
OUTPUT_VAL = "data/all_rml_validation_data.jsonl"
VAL_RATIO = 0.1  # 10% for validation
MAX_RECORDS = None  # Set to None for all, or an int for limit


def find_jsonl_files(data_dir):
    """Recursively find all .jsonl files in data_dir"""
    jsonl_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files


def load_all_records(jsonl_files, max_records=None):
    """Load all records from a list of JSONL files"""
    records = []
    total_files = len(jsonl_files)
    
    print(f"📁 Processing {total_files} files...")
    
    for i, file in enumerate(jsonl_files):
        print(f"📂 Processing file {i+1}/{total_files}: {os.path.basename(file)}")
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                file_records = 0
                for line_num, line in enumerate(f):
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                        file_records += 1
                        
                        if max_records and len(records) >= max_records:
                            print(f"✅ Reached max records limit: {max_records}")
                            return records
                            
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON at line {line_num+1} in {file}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error processing line {line_num+1} in {file}: {e}")
                        continue
                
                print(f"📊 Loaded {file_records} records from {os.path.basename(file)}")
                
        except Exception as e:
            print(f"❌ Error reading file {file}: {e}")
            continue
    
    return records


def main():
    print("🚀 Starting RML Data Aggregation")
    print("="*50)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    
    print(f"🔍 Scanning {DATA_DIR} for .jsonl files...")
    jsonl_files = find_jsonl_files(DATA_DIR)
    
    if not jsonl_files:
        print("❌ No .jsonl files found!")
        return
    
    print(f"✅ Found {len(jsonl_files)} .jsonl files:")
    for file in jsonl_files[:5]:  # Show first 5 files
        print(f"   - {file}")
    if len(jsonl_files) > 5:
        print(f"   ... and {len(jsonl_files) - 5} more files")

    print(f"\n📥 Loading all records...")
    records = load_all_records(jsonl_files, MAX_RECORDS)
    
    if not records:
        print("❌ No valid records found!")
        return
    
    print(f"✅ Loaded {len(records)} total records.")

    print(f"🔀 Shuffling records...")
    random.shuffle(records)

    # Split into train/val
    val_size = int(len(records) * VAL_RATIO)
    val_records = records[:val_size]
    train_records = records[val_size:]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)

    print(f"💾 Writing {len(train_records)} training records to {OUTPUT_TRAIN}")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for rec in train_records:
            f.write(json.dumps(rec) + '\n')

    print(f"💾 Writing {len(val_records)} validation records to {OUTPUT_VAL}")
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        for rec in val_records:
            f.write(json.dumps(rec) + '\n')

    print("\n✅ Data aggregation complete!")
    print(f"📊 Summary:")
    print(f"   - Total records: {len(records)}")
    print(f"   - Training records: {len(train_records)}")
    print(f"   - Validation records: {len(val_records)}")
    print(f"   - Training file: {OUTPUT_TRAIN}")
    print(f"   - Validation file: {OUTPUT_VAL}")

if __name__ == "__main__":
    main()