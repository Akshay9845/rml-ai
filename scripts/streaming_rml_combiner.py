#!/usr/bin/env python3
"""
Streaming RML Combiner
Combines RML components with minimal memory usage
"""

import json
import os
import glob
from collections import defaultdict
import time

def streaming_combine_rml(data_dir, output_dir, max_records=1000):
    """Streaming combine RML components by doc_id"""
    print(f"🚀 Streaming RML combiner from: {data_dir}")
    print(f"📁 Output to: {output_dir}")
    print(f"📊 Max records: {max_records}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # RML components to look for
    components = ['concepts', 'emotions', 'entities', 'events', 'intents', 
                 'reasoning', 'summaries', 'tags', 'triples', 'vectors']
    
    # Store components by doc_id (limited memory)
    all_components = defaultdict(dict)
    total_files = 0
    total_records = 0
    complete_records = 0
    
    output_file = os.path.join(output_dir, "complete_rml_records.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as output_f:
        # Process each component type
        for component in components:
            pattern = os.path.join(data_dir, f"*{component}*.jsonl")
            files = glob.glob(pattern)
            
            if not files:
                print(f"⚠️ No files found for {component}")
                continue
                
            print(f"📁 Processing {component}: {len(files)} files")
            
            for file_path in files:
                total_files += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if line.strip():
                                try:
                                    data = json.loads(line.strip())
                                    doc_id = data.get('doc_id')
                                    
                                    if doc_id is not None:
                                        all_components[doc_id][component] = data.get('data', [])
                                        total_records += 1
                                        
                                        # Check if we have enough components for this doc_id
                                        if len(all_components[doc_id]) >= 3:
                                            # Create complete record
                                            complete_record = {
                                                'doc_id': doc_id,
                                                'concepts': all_components[doc_id].get('concepts', []),
                                                'emotions': all_components[doc_id].get('emotions', []),
                                                'entities': all_components[doc_id].get('entities', []),
                                                'events': all_components[doc_id].get('events', []),
                                                'intents': all_components[doc_id].get('intents', []),
                                                'reasoning': all_components[doc_id].get('reasoning', []),
                                                'summaries': all_components[doc_id].get('summaries', []),
                                                'tags': all_components[doc_id].get('tags', []),
                                                'triples': all_components[doc_id].get('triples', []),
                                                'vectors': all_components[doc_id].get('vectors', [])
                                            }
                                            
                                            output_f.write(json.dumps(complete_record, ensure_ascii=False) + '\n')
                                            complete_records += 1
                                            
                                            # Remove from memory to save space
                                            del all_components[doc_id]
                                            
                                            # Check if we've reached max records
                                            if complete_records >= max_records:
                                                print(f"✅ Reached max records: {max_records}")
                                                return complete_records
                                        
                                except json.JSONDecodeError:
                                    continue
                                    
                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
                    continue
    
    print(f"📊 Found {len(all_components)} remaining doc_ids")
    print(f"📊 Processed {total_files} files, {total_records} records")
    print(f"✅ Created {complete_records} complete RML records")
    
    # Show file size
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
    
    return complete_records

def main():
    """Main function"""
    print("🚀 Streaming RML Component Combiner")
    print("="*50)
    
    # Test with a small dataset first
    test_dir = "data/cpp_rml_output_v5"
    output_dir = "output/streaming_combiner"
    
    if os.path.exists(test_dir):
        print(f"🧪 Testing with: {test_dir}")
        
        # Start with 100 records to test
        records = streaming_combine_rml(test_dir, output_dir, max_records=100)
        
        if records > 0:
            print(f"✅ SUCCESS! Created {records} complete records")
            
            # Show sample
            output_file = os.path.join(output_dir, "complete_rml_records.jsonl")
            if os.path.exists(output_file):
                print("\n📄 Sample complete record:")
                with open(output_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 1:  # Show first record
                            data = json.loads(line)
                            print(json.dumps(data, indent=2, ensure_ascii=False))
                            break
                            
            # Check disk space
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            print(f"\n💾 Available disk space: {free_gb:.1f} GB")
            
            if free_gb > 1.0:  # If we have more than 1GB free
                print("🚀 Processing more records...")
                records = streaming_combine_rml(test_dir, output_dir, max_records=1000)
                print(f"✅ Total records created: {records}")
        else:
            print("❌ No complete records created")
    else:
        print(f"❌ Test directory not found: {test_dir}")

if __name__ == "__main__":
    main() 