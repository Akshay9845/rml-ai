#!/usr/bin/env python3
"""
Fast RML Training Data Preparation
Processes a subset of RML data files efficiently to avoid memory issues
"""

import os
import json
import random
import glob

# Use existing working data files we know about
KNOWN_DATA_FILES = [
    "data/converted_rml/complete_rml/rml_data.jsonl",
    "data/python_c4_final_backup_20250731_043743/concepts_batch_1753912494.jsonl",
    "data/python_c4_final_backup_20250731_043743/intents_batch_1753913673.jsonl",
    "data/python_c4_final_backup_20250731_043743/events_batch_1753916240.jsonl"
]

OUTPUT_TRAIN = "data/all_rml_training_data.jsonl"
OUTPUT_VAL = "data/all_rml_validation_data.jsonl"
VAL_RATIO = 0.1
MAX_RECORDS = 10000  # Limit to avoid memory issues


def load_records_from_files(file_list, max_records=None):
    """Load records from a list of files"""
    records = []
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"📂 Loading: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
                        continue
                    except Exception as e:
                        continue
                
                print(f"📊 Loaded {file_records} records from {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue
    
    return records


def main():
    print("🚀 Fast RML Data Preparation")
    print("="*40)
    
    # Use known working files
    print("📁 Using known working RML data files...")
    
    records = load_records_from_files(KNOWN_DATA_FILES, MAX_RECORDS)
    
    if not records:
        print("❌ No records loaded!")
        return
    
    print(f"✅ Loaded {len(records)} total records")
    
    # Shuffle
    print("🔀 Shuffling records...")
    random.shuffle(records)
    
    # Split
    val_size = int(len(records) * VAL_RATIO)
    val_records = records[:val_size]
    train_records = records[val_size:]
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    
    # Save
    print(f"💾 Writing {len(train_records)} training records...")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for rec in train_records:
            f.write(json.dumps(rec) + '\n')
    
    print(f"💾 Writing {len(val_records)} validation records...")
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        for rec in val_records:
            f.write(json.dumps(rec) + '\n')
    
    print("\n✅ Data preparation complete!")
    print(f"📊 Summary:")
    print(f"   - Total records: {len(records)}")
    print(f"   - Training: {len(train_records)}")
    print(f"   - Validation: {len(val_records)}")
    print(f"   - Training file: {OUTPUT_TRAIN}")
    print(f"   - Validation file: {OUTPUT_VAL}")


if __name__ == "__main__":
    main() 