#!/usr/bin/env python3
"""
Simple RML Combiner
Combines scattered RML components into complete records
"""

import json
import os
import glob
from collections import defaultdict
import time

def combine_rml_components(data_dir, output_dir):
    """Combine RML components by doc_id"""
    print(f"🚀 Combining RML components from: {data_dir}")
    print(f"📁 Output to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # RML components to look for
    components = ['concepts', 'emotions', 'entities', 'events', 'intents', 
                 'reasoning', 'summaries', 'tags', 'triples', 'vectors']
    
    # Store all components by doc_id
    all_components = defaultdict(dict)
    total_files = 0
    total_records = 0
    
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
                                    
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue
    
    print(f"📊 Found {len(all_components)} unique doc_ids")
    print(f"📊 Processed {total_files} files, {total_records} records")
    
    # Create complete records
    complete_records = 0
    output_file = os.path.join(output_dir, "complete_rml_records.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, components_data in all_components.items():
            # Check if we have at least some components
            if len(components_data) >= 3:  # At least 3 components
                complete_record = {
                    'doc_id': doc_id,
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
                
                f.write(json.dumps(complete_record, ensure_ascii=False) + '\n')
                complete_records += 1
    
    print(f"✅ Created {complete_records} complete RML records")
    print(f"📁 Output file: {output_file}")
    
    # Show file size
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📊 File size: {size_mb:.2f} MB")
    
    return complete_records

def main():
    """Main function"""
    print("🚀 Simple RML Component Combiner")
    print("="*50)
    
    # Test with a small dataset first
    test_dir = "data/cpp_rml_output_v5"
    output_dir = "output/simple_combiner"
    
    if os.path.exists(test_dir):
        print(f"🧪 Testing with: {test_dir}")
        records = combine_rml_components(test_dir, output_dir)
        
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
        else:
            print("❌ No complete records created")
    else:
        print(f"❌ Test directory not found: {test_dir}")

if __name__ == "__main__":
    main() 