#!/usr/bin/env python3
"""
Enhanced RML Data Processing - Production Script
Processes large datasets with enhanced features and MEGA storage
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_enhanced import EnhancedRMLProcessor, RMLConfig

def process_rml_file_enhanced(input_file: str, output_file: str, processor: EnhancedRMLProcessor, max_items: int = 1000):
    """Process a single RML file with enhanced features"""
    
    print(f"📄 Processing: {input_file}")
    print(f"📄 Output: {output_file}")
    
    processed_count = 0
    start_time = time.time()
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            if line.strip() and processed_count < max_items:
                try:
                    rml_data = json.loads(line.strip())
                    
                    # Process the RML data with enhanced features
                    result = processor.process_rml_data(rml_data, "What are the main concepts and their relationships?")
                    
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
    """Main function to process RML data with enhanced features"""
    
    # Configuration
    config = RMLConfig(
        output_dir="/Volumes/MEGA/R-LLM-enhanced-output",
        save_memory_graphs=True,
        log_level="INFO",
        use_models=False,  # Start with lightweight mode for speed
        model_path="models/Phi-3-mini-4k-instruct-q4_0.gguf",  # Will be used if available
        max_new_tokens=128,
        temperature=0.7
    )
    
    # Initialize enhanced processor
    processor = EnhancedRMLProcessor(config)
    
    print("🚀 Enhanced RML Data Processing")
    print("="*60)
    print(f"📁 Output directory: {config.output_dir}")
    print(f"🤖 Model integration: {config.use_models}")
    print(f"🧠 Memory graphs: {config.save_memory_graphs}")
    
    # Process your existing data files
    data_files = [
        "data/python_c4_final_backup_20250731_043743/concepts_batch_1753912494.jsonl",
        "data/python_c4_final_backup_20250731_043743/intents_batch_1753913673.jsonl",
        "data/python_c4_final_backup_20250731_043743/events_batch_1753916240.jsonl"
    ]
    
    total_processed = 0
    total_start_time = time.time()
    
    for input_file in data_files:
        if os.path.exists(input_file):
            # Create output file path on MEGA volume
            input_path = Path(input_file)
            output_file = f"/Volumes/MEGA/R-LLM-enhanced-output/enhanced_{input_path.name}"
            
            # Process file with enhanced features
            processed = process_rml_file_enhanced(input_file, output_file, processor, max_items=300)
            total_processed += processed
            
            print(f"📊 File completed: {processed} items")
        else:
            print(f"⚠️ File not found: {input_file}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n🎉 Total processed: {total_processed} items in {total_elapsed:.1f}s")
    print(f"📁 Output saved to: /Volumes/MEGA/R-LLM-enhanced-output/")
    
    # Show memory statistics
    memory_size = len(processor.memory.concept_graph)
    print(f"🧠 Final memory size: {memory_size} concepts")
    
    # Save final memory state
    final_memory_path = "/Volumes/MEGA/R-LLM-enhanced-output/final_memory.json"
    processor.memory.save_memory(final_memory_path)
    print(f"💾 Final memory saved to: {final_memory_path}")

if __name__ == "__main__":
    main() 