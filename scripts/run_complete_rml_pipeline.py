#!/usr/bin/env python3
"""
Production Script for Complete RML Pipeline
Processes large RML datasets using E5-Mistral encoder and Phi-3 decoder
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_complete_pipeline import RMLCompletePipeline, RMLPipelineConfig

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rml_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_rml_data(file_path: str, max_items: int = None) -> list:
    """Load RML data from JSONL file"""
    data = []
    count = 0
    
    logger = logging.getLogger(__name__)
    logger.info(f"📂 Loading RML data from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_items and count >= max_items:
                    break
                
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Skipping invalid JSON line: {e}")
                    continue
        
        logger.info(f"✅ Loaded {len(data)} items from {file_path}")
        return data
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading file {file_path}: {e}")
        return []

def save_results(results: list, output_dir: str, batch_name: str):
    """Save processing results"""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results
    results_file = os.path.join(output_dir, f"{batch_name}_complete_results.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{batch_name}_summary.json")
    summary = {
        'total_processed': len(results),
        'successful': len([r for r in results if 'error' not in r]),
        'errors': len([r for r in results if 'error' in r]),
        'total_concepts': sum(len(r.get('concepts', [])) for r in results if 'error' not in r),
        'avg_response_length': sum(len(r.get('response', '')) for r in results if 'error' not in r) / max(1, len([r for r in results if 'error' not in r]))
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"💾 Saved results to {output_dir}")
    logger.info(f"📊 Summary: {summary['successful']}/{summary['total_processed']} successful")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Complete RML Pipeline")
    parser.add_argument("--input-file", required=True, help="Input RML data file (JSONL)")
    parser.add_argument("--output-dir", default="/Volumes/MEGA/R-LLM-complete-output", help="Output directory")
    parser.add_argument("--max-items", type=int, default=100, help="Maximum items to process")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--use-quantization", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("🚀 Starting Complete RML Pipeline")
    logger.info(f"📂 Input: {args.input_file}")
    logger.info(f"📁 Output: {args.output_dir}")
    logger.info(f"📊 Max items: {args.max_items}")
    
    # Load RML data
    rml_data = load_rml_data(args.input_file, args.max_items)
    
    if not rml_data:
        logger.error("❌ No data loaded. Exiting.")
        return
    
    # Configuration
    config = RMLPipelineConfig(
        encoder_model_name="intfloat/e5-mistral-7b-instruct",
        decoder_model_name="microsoft/phi-3-mini-4k-instruct",
        use_quantization=args.use_quantization,
        quantization_bits=4,
        device_map="auto",
        max_input_length=1024,
        max_new_tokens=128,
        temperature=0.7,
        output_dir=args.output_dir,
        save_embeddings=True,
        save_responses=True,
        log_level=args.log_level
    )
    
    # Initialize pipeline
    pipeline = None
    try:
        logger.info("🔧 Initializing RML pipeline...")
        pipeline = RMLCompletePipeline(config)
        
        # Process in batches
        results = []
        batch_name = Path(args.input_file).stem
        
        for i in range(0, len(rml_data), args.batch_size):
            batch = rml_data[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(rml_data) + args.batch_size - 1) // args.batch_size
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            try:
                batch_results = pipeline.process_batch(batch)
                results.extend(batch_results)
                
                # Save intermediate results
                if batch_num % 5 == 0:  # Save every 5 batches
                    intermediate_dir = os.path.join(args.output_dir, "intermediate")
                    save_results(results, intermediate_dir, f"{batch_name}_batch_{batch_num}")
                
                logger.info(f"✅ Batch {batch_num} completed")
                
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_num}: {e}")
                # Add error results
                for item in batch:
                    results.append({'error': str(e), 'rml_data': item})
        
        # Save final results
        logger.info("💾 Saving final results...")
        save_results(results, args.output_dir, batch_name)
        
        logger.info("🎉 RML pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise
        
    finally:
        if pipeline:
            logger.info("🧹 Cleaning up...")
            pipeline.cleanup()

if __name__ == "__main__":
    main() 