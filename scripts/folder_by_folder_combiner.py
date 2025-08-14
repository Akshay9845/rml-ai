#!/usr/bin/env python3
"""
Folder-by-Folder RML Combiner
Processes each folder separately and combines RML components within each folder
"""

import json
import os
import glob
import shutil
from collections import defaultdict
from pathlib import Path

def find_rml_components_in_folder(folder_path):
    """Find all RML component files in a folder"""
    components = ['concepts', 'emotions', 'entities', 'events', 'intents', 
                 'reasoning', 'summaries', 'tags', 'triples', 'vectors']
    
    found_components = {}
    
    for component in components:
        # Look for files containing the component name
        pattern = os.path.join(folder_path, f"*{component}*.jsonl")
        files = glob.glob(pattern)
        
        if files:
            found_components[component] = files
            print(f"  📁 Found {component}: {len(files)} files")
    
    return found_components

def extract_id_from_record(data):
    """Extract ID from record - try different possible ID fields"""
    # Try different possible ID fields
    for id_field in ['doc_id', 'record_id', 'id', 'chunk_id']:
        if id_field in data:
            return data[id_field]
    
    # If no ID found, return None
    return None

def combine_folder_components(folder_path, output_file):
    """Combine RML components within a single folder"""
    print(f"\n🔍 Processing folder: {folder_path}")
    
    # Find all RML components in this folder
    components = find_rml_components_in_folder(folder_path)
    
    if not components:
        print(f"  ⚠️ No RML components found in {folder_path}")
        return 0
    
    # Store all components by ID
    all_records = defaultdict(dict)
    total_files = 0
    total_records = 0
    
    # Process each component type
    for component_name, files in components.items():
        print(f"  📄 Processing {component_name}...")
        
        for file_path in files:
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                record_id = extract_id_from_record(data)
                                
                                if record_id is not None:
                                    # Store the component data
                                    all_records[record_id][component_name] = data.get('data', [])
                                    total_records += 1
                                    
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                print(f"    ❌ Error reading {file_path}: {e}")
                continue
    
    print(f"  📊 Found {len(all_records)} unique records")
    print(f"  📊 Processed {total_files} files, {total_records} records")
    
    # Create complete RML records
    complete_records = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record_id, components_data in all_records.items():
            # Create complete RML record with all 10 components
            complete_record = {
                'record_id': record_id,
                'concepts': components_data.get('concepts', []),
                'emotions': components_data.get('emotions', []),
                'entities': components_data.get('entities', []),
                'events': components_data.get('events', []),
                'intents': components_data.get('intents', []),
                'reasoning': components_data.get('reasoning', []),
                'summaries': components_data.get('summaries', []),
                'tags': components_data.get('tags', []),
                'triples': components_data.get('triples', []),
                'vectors': components_data.get('vectors', [])
            }
            
            # Write the complete record
            out_f.write(json.dumps(complete_record, ensure_ascii=False) + '\n')
            complete_records += 1
    
    print(f"  ✅ Created {complete_records} complete RML records")
    print(f"  📁 Saved to: {output_file}")
    
    return complete_records

def process_all_folders():
    """Process all folders in the data directory"""
    data_dir = "/Users/elite/R-LLM/data"
    output_dir = "/Users/elite/R-LLM/data/combined_rml"
    
    print("🚀 Folder-by-Folder RML Combiner")
    print("="*60)
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories in data
    folders = [f for f in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, f)) 
               and not f.startswith('.')]
    
    print(f"\n📊 Found {len(folders)} folders to process")
    
    total_combined = 0
    
    for i, folder_name in enumerate(folders, 1):
        folder_path = os.path.join(data_dir, folder_name)
        output_file = os.path.join(output_dir, f"{folder_name}_combined.jsonl")
        
        print(f"\n[{i}/{len(folders)}] Processing: {folder_name}")
        
        try:
            records = combine_folder_components(folder_path, output_file)
            total_combined += records
            
            if records > 0:
                print(f"  ✅ Success: {records} records combined")
            else:
                print(f"  ⚠️ No records combined")
                
        except Exception as e:
            print(f"  ❌ Error processing {folder_name}: {e}")
            continue
    
    print(f"\n🎉 COMPLETED!")
    print(f"📊 Total combined records: {total_combined}")
    print(f"📁 All combined files saved in: {output_dir}")

if __name__ == "__main__":
    process_all_folders() 