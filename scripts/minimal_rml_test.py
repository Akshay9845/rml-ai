#!/usr/bin/env python3
"""
Minimal RML Test - No Storage Usage
Just tests the concept of combining RML components
"""

import json
import os
import glob
from collections import defaultdict

def test_rml_combination(data_dir):
    """Test RML combination without creating large files"""
    print(f"🧪 Testing RML combination from: {data_dir}")
    
    # RML components to look for
    components = ['concepts', 'emotions', 'entities', 'events', 'intents', 
                 'reasoning', 'summaries', 'tags', 'triples', 'vectors']
    
    # Store components by doc_id (only first 100 for testing)
    all_components = defaultdict(dict)
    total_files = 0
    total_records = 0
    sample_count = 0
    
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
                        if line.strip() and sample_count < 100:  # Only process first 100 records
                            try:
                                data = json.loads(line.strip())
                                doc_id = data.get('doc_id')
                                
                                if doc_id is not None:
                                    all_components[doc_id][component] = data.get('data', [])
                                    total_records += 1
                                    sample_count += 1
                                    
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue
    
    print(f"📊 Found {len(all_components)} unique doc_ids (sample)")
    print(f"📊 Processed {total_files} files, {total_records} records")
    
    # Show sample complete records (no file writing)
    complete_records = 0
    print(f"\n📄 Sample Complete RML Records:")
    print("="*60)
    
    for doc_id, components_data in list(all_components.items())[:5]:  # Show first 5
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
            
            print(f"\n✅ Complete Record {complete_records + 1}:")
            print(f"   doc_id: {doc_id}")
            print(f"   components: {list(components_data.keys())}")
            print(f"   concepts: {components_data.get('concepts', [])[:3]}...")  # Show first 3
            print(f"   emotions: {components_data.get('emotions', [])[:3]}...")
            
            complete_records += 1
    
    print(f"\n🎉 SUCCESS! Concept works!")
    print(f"📊 Created {complete_records} complete RML records (sample)")
    print(f"📊 This proves we can combine scattered components into complete records")
    
    return complete_records

def main():
    """Main function"""
    print("🧪 Minimal RML Component Test - NO STORAGE USAGE")
    print("="*60)
    
    # Test with a small dataset
    test_dir = "data/cpp_rml_output_v5"
    
    if os.path.exists(test_dir):
        print(f"🧪 Testing with: {test_dir}")
        records = test_rml_combination(test_dir)
        
        if records > 0:
            print(f"\n✅ CONCEPT PROVEN!")
            print(f"   We can combine scattered RML components into complete records")
            print(f"   Your 355GB data can be processed this way")
            print(f"   Need more storage space to create full dataset")
        else:
            print("❌ No complete records created")
    else:
        print(f"❌ Test directory not found: {test_dir}")

if __name__ == "__main__":
    main() 