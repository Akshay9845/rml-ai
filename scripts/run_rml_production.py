#!/usr/bin/env python3
"""
RML Production Pipeline Runner
Process large RML datasets efficiently using E5-Mistral → RML Memory → Phi-3
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_encoder_decoder import RMLConfig
from rml_streaming_processor import RMLStreamingProcessor

def setup_logging(log_level: str, log_file: str = None):
    """Setup logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def find_rml_files(data_dir: str, pattern: str = "*.jsonl") -> list:
    """Find all RML files in the data directory"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all matching files
    files = list(data_path.rglob(pattern))
    
    # Filter out files that are too small (likely empty or corrupted)
    valid_files = []
    for file_path in files:
        try:
            if file_path.stat().st_size > 1024:  # At least 1KB
                valid_files.append(str(file_path))
        except OSError:
            continue
    
    return sorted(valid_files)

def estimate_processing_time(file_size_mb: float, rate_items_per_sec: float = 10.0) -> str:
    """Estimate processing time for a file"""
    
    # Rough estimate: 1MB ≈ 1000 JSONL lines
    estimated_items = int(file_size_mb * 1000)
    estimated_seconds = estimated_items / rate_items_per_sec
    
    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f}s"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f}m"
    else:
        return f"{estimated_seconds/3600:.1f}h"

def process_single_file(input_file: str, output_dir: str, config: RMLConfig, 
                       chunk_size: int = 100, use_multiprocessing: bool = False) -> dict:
    """Process a single RML file"""
    
    logger = logging.getLogger(__name__)
    
    # Create output file path
    input_path = Path(input_file)
    output_file = Path(output_dir) / f"processed_{input_path.name}"
    
    # Get file size for estimation
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    estimated_time = estimate_processing_time(file_size_mb)
    
    logger.info(f"📄 Processing: {input_path.name}")
    logger.info(f"   Size: {file_size_mb:.1f}MB")
    logger.info(f"   Estimated time: {estimated_time}")
    logger.info(f"   Output: {output_file}")
    
    # Initialize processor
    processor = RMLStreamingProcessor(config, max_workers=4 if use_multiprocessing else 1)
    
    # Process file
    start_time = datetime.now()
    
    if use_multiprocessing:
        result = processor.process_with_multiprocessing(
            input_file=input_file,
            output_file=str(output_file),
            chunk_size=chunk_size
        )
    else:
        result = processor.process_file(
            input_file=input_file,
            output_file=str(output_file),
            chunk_size=chunk_size
        )
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Add metadata
    result['file_size_mb'] = file_size_mb
    result['estimated_time'] = estimated_time
    result['actual_time'] = processing_time
    result['processing_start'] = start_time.isoformat()
    result['processing_end'] = end_time.isoformat()
    
    logger.info(f"✅ Completed: {input_path.name}")
    logger.info(f"   Processed: {result['total_processed']} items")
    logger.info(f"   Results: {result['total_results']} items")
    logger.info(f"   Time: {processing_time:.1f}s")
    logger.info(f"   Rate: {result['rate']:.1f} items/sec")
    
    return result

def process_directory(input_dir: str, output_dir: str, config: RMLConfig, 
                     file_pattern: str = "*.jsonl", chunk_size: int = 100,
                     use_multiprocessing: bool = False, max_files: int = None) -> list:
    """Process all RML files in a directory"""
    
    logger = logging.getLogger(__name__)
    
    # Find all RML files
    logger.info(f"🔍 Scanning directory: {input_dir}")
    rml_files = find_rml_files(input_dir, file_pattern)
    
    if not rml_files:
        logger.warning(f"No RML files found in {input_dir} matching pattern: {file_pattern}")
        return []
    
    logger.info(f"📊 Found {len(rml_files)} RML files")
    
    # Limit number of files if specified
    if max_files:
        rml_files = rml_files[:max_files]
        logger.info(f"📊 Processing first {len(rml_files)} files (limited by max_files)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    results = []
    total_start_time = datetime.now()
    
    for i, input_file in enumerate(rml_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📄 File {i}/{len(rml_files)}: {Path(input_file).name}")
        logger.info(f"{'='*60}")
        
        try:
            result = process_single_file(
                input_file=input_file,
                output_dir=output_dir,
                config=config,
                chunk_size=chunk_size,
                use_multiprocessing=use_multiprocessing
            )
            results.append(result)
            
            # Save progress
            progress_file = os.path.join(output_dir, "processing_progress.json")
            with open(progress_file, 'w') as f:
                json.dump({
                    'total_files': len(rml_files),
                    'processed_files': len(results),
                    'current_file': i,
                    'results': results
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Error processing {input_file}: {e}")
            results.append({
                'input_file': input_file,
                'error': str(e),
                'status': 'failed'
            })
            continue
    
    # Final summary
    total_end_time = datetime.now()
    total_time = (total_end_time - total_start_time).total_seconds()
    
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {len(rml_files)}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(failed_results)}")
    logger.info(f"Total time: {total_time:.1f}s")
    
    if successful_results:
        total_processed = sum(r.get('total_processed', 0) for r in successful_results)
        total_results = sum(r.get('total_results', 0) for r in successful_results)
        avg_rate = total_processed / total_time if total_time > 0 else 0
        
        logger.info(f"Total items processed: {total_processed}")
        logger.info(f"Total results generated: {total_results}")
        logger.info(f"Average rate: {avg_rate:.1f} items/sec")
    
    # Save final results
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'processing_start': total_start_time.isoformat(),
            'processing_end': total_end_time.isoformat(),
            'total_time_seconds': total_time,
            'total_files': len(rml_files),
            'successful_files': len(successful_results),
            'failed_files': len(failed_results),
            'results': results
        }, f, indent=2)
    
    logger.info(f"📄 Summary saved to: {summary_file}")
    
    return results

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="RML Production Pipeline Runner")
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                       help="Input directory or file path")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory path")
    
    # Processing options
    parser.add_argument("--chunk-size", "-c", type=int, default=100,
                       help="Chunk size for processing (default: 100)")
    parser.add_argument("--max-files", "-m", type=int,
                       help="Maximum number of files to process")
    parser.add_argument("--file-pattern", "-p", default="*.jsonl",
                       help="File pattern to match (default: *.jsonl)")
    
    # Model configuration
    parser.add_argument("--encoder-model", default="intfloat/e5-mistral-7b-instruct",
                       help="E5-Mistral model path")
    parser.add_argument("--decoder-model", default="microsoft/phi-3-medium-128k-instruct",
                       help="Phi-3 model path")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--max-concepts", type=int, default=30,
                       help="Maximum concepts per input")
    
    # Performance options
    parser.add_argument("--multiprocessing", action="store_true",
                       help="Use multiprocessing for better performance")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file",
                       help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Create configuration
    config = RMLConfig(
        encoder_model=args.encoder_model,
        decoder_model=args.decoder_model,
        batch_size=args.batch_size,
        max_concepts_per_input=args.max_concepts,
        device=args.device,
        output_dir=args.output,
        save_memory_graphs=True,
        log_level=args.log_level
    )
    
    logger.info("🚀 RML Production Pipeline Starting")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Multiprocessing: {args.multiprocessing}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Check if input is file or directory
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Process single file
            logger.info("📄 Processing single file")
            result = process_single_file(
                input_file=args.input,
                output_dir=args.output,
                config=config,
                chunk_size=args.chunk_size,
                use_multiprocessing=args.multiprocessing
            )
            results = [result]
            
        elif input_path.is_dir():
            # Process directory
            logger.info("📁 Processing directory")
            results = process_directory(
                input_dir=args.input,
                output_dir=args.output,
                config=config,
                file_pattern=args.file_pattern,
                chunk_size=args.chunk_size,
                use_multiprocessing=args.multiprocessing,
                max_files=args.max_files
            )
            
        else:
            logger.error(f"Input path does not exist: {args.input}")
            return 1
        
        # Final status
        successful = len([r for r in results if 'error' not in r])
        total = len(results)
        
        logger.info(f"\n🎉 Processing completed!")
        logger.info(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
        
        return 0 if successful == total else 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 