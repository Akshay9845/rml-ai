#!/usr/bin/env python3
"""
Process RML Data with Lightweight Processor
Uses MEGA volume for storage and processes large datasets efficiently
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_lightweight_final import RMLProcessor, RMLConfig

def process_rml_file(input_file: str, output_file: str, processor: RMLProcessor, max_items: int = 1000):
    """Process a single RML file"""
    
    print(f"📄 Processing: {input_file}")
    print(f"📄 Output: {output_file}")
    
    processed_count = 0
    start_time = time.time()
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            if line.strip() and processed_count < max_items:
                try:
                    rml_data = json.loads(line.strip())
                    
                    # Process the RML data
                    result = processor.process_rml_data(rml_data, "What are the main concepts?")
                    
                    # Write result
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        print(f"   📊 Processed {processed_count} items ({rate:.1f} items/sec)")
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"   ❌ Error on line {line_num}: {e}")
                    continue
    
    elapsed = time.time() - start_time
    print(f"✅ Completed: {processed_count} items in {elapsed:.1f}s")
    
    return processed_count

def main():
    """Main function to process RML data"""
    
    # Configuration
    config = RMLConfig(
        output_dir="/Volumes/MEGA/R-LLM-output",
        save_memory_graphs=True,
        log_level="INFO"
    )
    
    # Initialize processor
    processor = RMLProcessor(config)
    
    print("🚀 RML Data Processing with Lightweight Processor")
    print("="*60)
    
    # Find actual RML data files
    data_files = [
        "data/python_c4_final_backup_20250731_043743/concepts_batch_1753912494.jsonl",
        "data/python_c4_final_backup_20250731_043743/intents_batch_1753913673.jsonl",
        "data/python_c4_final_backup_20250731_043743/events_batch_1753916240.jsonl"
    ]
    
    total_processed = 0
    
    for input_file in data_files:
        if os.path.exists(input_file):
            # Create output file path on MEGA volume
            input_path = Path(input_file)
            output_file = f"/Volumes/MEGA/R-LLM-output/processed_{input_path.name}"
            
            # Process file
            processed = process_rml_file(input_file, output_file, processor, max_items=200)
            total_processed += processed
            
            print(f"📊 File completed: {processed} items")
        else:
            print(f"⚠️ File not found: {input_file}")
    
    print(f"\n🎉 Total processed: {total_processed} items")
    print(f"📁 Output saved to: /Volumes/MEGA/R-LLM-output/")

if __name__ == "__main__":
    main() 