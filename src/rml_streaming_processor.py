#!/usr/bin/env python3
"""
RML Streaming Processor
Handles large datasets efficiently by streaming through data in chunks
and integrating with the E5-Mistral → RML Memory → Phi-3 pipeline
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Iterator, Any
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import gc

from rml_encoder_decoder import RMLPipeline, RMLConfig

class RMLStreamingProcessor:
    """Streaming processor for large RML datasets"""
    
    def __init__(self, config: RMLConfig, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.pipeline = RMLPipeline(config)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self.processed_count = 0
        self.total_count = 0
        self.start_time = None
        
        # Thread-safe counters
        self._lock = threading.Lock()
        
    def count_lines(self, filepath: str) -> int:
        """Count total lines in a file (for progress tracking)"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            self.logger.warning(f"Could not count lines in {filepath}: {e}")
            return 0
    
    def stream_jsonl(self, filepath: str, chunk_size: int = 100) -> Iterator[List[Dict[str, Any]]]:
        """Stream JSONL file in chunks"""
        
        chunk = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        chunk.append(data)
                        
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON at line {line_num}")
                        continue
            
            # Yield remaining chunk
            if chunk:
                yield chunk
    
    def process_chunk(self, chunk: List[Dict[str, Any]], chunk_id: int) -> List[Dict[str, Any]]:
        """Process a chunk of RML data"""
        
        results = []
        
        for i, rml_data in enumerate(chunk):
            try:
                # Extract text from RML data
                text = self._extract_text_from_rml(rml_data)
                
                if text and len(text.strip()) > 50:  # Minimum text length
                    # Process through RML pipeline
                    result = self.pipeline.process_text(text)
                    result['rml_data'] = rml_data
                    result['chunk_id'] = chunk_id
                    result['item_id'] = i
                    results.append(result)
                
                # Update progress
                with self._lock:
                    self.processed_count += 1
                    if self.processed_count % 100 == 0:
                        self._log_progress()
                        
            except Exception as e:
                self.logger.error(f"Error processing item {i} in chunk {chunk_id}: {e}")
                continue
        
        return results
    
    def _extract_text_from_rml(self, rml_data: Dict[str, Any]) -> str:
        """Extract text content from RML data structure"""
        
        # Try different possible text fields
        text_fields = ['text', 'content', 'summary', 'description', 'input']
        
        for field in text_fields:
            if field in rml_data and rml_data[field]:
                return str(rml_data[field])
        
        # If no direct text field, try to reconstruct from concepts
        if 'concepts' in rml_data and rml_data['concepts']:
            concepts = rml_data['concepts']
            if isinstance(concepts, list):
                return " ".join([str(c) for c in concepts[:10]])
            elif isinstance(concepts, dict):
                return " ".join([str(v) for v in concepts.values()][:10])
        
        return ""
    
    def _log_progress(self):
        """Log processing progress"""
        if self.total_count > 0:
            progress = (self.processed_count / self.total_count) * 100
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed if elapsed > 0 else 0
            
            self.logger.info(
                f"📊 Progress: {self.processed_count}/{self.total_count} "
                f"({progress:.1f}%) - Rate: {rate:.1f} items/sec"
            )
    
    def process_file(self, input_file: str, output_file: str, chunk_size: int = 100) -> Dict[str, Any]:
        """Process a single RML file with streaming"""
        
        self.logger.info(f"🚀 Starting streaming processing of: {input_file}")
        
        # Count total lines for progress tracking
        self.total_count = self.count_lines(input_file)
        self.processed_count = 0
        self.start_time = time.time()
        
        self.logger.info(f"📊 Total items to process: {self.total_count}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process chunks
        chunk_id = 0
        total_results = []
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for chunk in self.stream_jsonl(input_file, chunk_size):
                chunk_id += 1
                self.logger.info(f"📦 Processing chunk {chunk_id} ({len(chunk)} items)")
                
                # Process chunk
                results = self.process_chunk(chunk, chunk_id)
                
                # Write results
                for result in results:
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    total_results.append(result)
                
                # Memory cleanup
                if chunk_id % 10 == 0:
                    gc.collect()
        
        # Final progress log
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"✅ Completed processing: {len(total_results)} results in {elapsed:.1f}s"
        )
        
        return {
            'input_file': input_file,
            'output_file': output_file,
            'total_processed': self.processed_count,
            'total_results': len(total_results),
            'processing_time': elapsed,
            'rate': self.processed_count / elapsed if elapsed > 0 else 0
        }
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "*.jsonl", chunk_size: int = 100) -> List[Dict[str, Any]]:
        """Process all matching files in a directory"""
        
        self.logger.info(f"📁 Processing directory: {input_dir}")
        
        # Find all matching files
        input_path = Path(input_dir)
        files = list(input_path.glob(file_pattern))
        
        if not files:
            self.logger.warning(f"No files found matching pattern: {file_pattern}")
            return []
        
        self.logger.info(f"📊 Found {len(files)} files to process")
        
        results = []
        
        for i, file_path in enumerate(files, 1):
            self.logger.info(f"📄 Processing file {i}/{len(files)}: {file_path.name}")
            
            # Create output file path
            output_file = Path(output_dir) / f"processed_{file_path.name}"
            
            # Process file
            result = self.process_file(str(file_path), str(output_file), chunk_size)
            results.append(result)
            
            # Save memory state periodically
            if i % 5 == 0:
                memory_path = os.path.join(output_dir, f"rml_memory_checkpoint_{i}.json")
                self.pipeline.memory.save_memory(memory_path)
                self.logger.info(f"💾 Saved memory checkpoint: {memory_path}")
        
        return results
    
    def process_with_multiprocessing(self, input_file: str, output_file: str, 
                                   chunk_size: int = 100) -> Dict[str, Any]:
        """Process file using multiple processes for better performance"""
        
        self.logger.info(f"🚀 Starting multiprocessing of: {input_file}")
        
        # Count total lines
        self.total_count = self.count_lines(input_file)
        self.processed_count = 0
        self.start_time = time.time()
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process chunks with ThreadPoolExecutor
        chunk_id = 0
        total_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit chunks for processing
            for chunk in self.stream_jsonl(input_file, chunk_size):
                chunk_id += 1
                future = executor.submit(self.process_chunk, chunk, chunk_id)
                futures.append((future, chunk_id))
            
            # Collect results
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for future, chunk_id in futures:
                    try:
                        results = future.result()
                        for result in results:
                            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                            total_results.append(result)
                        
                        self.logger.info(f"✅ Completed chunk {chunk_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error in chunk {chunk_id}: {e}")
        
        # Final progress log
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"✅ Completed multiprocessing: {len(total_results)} results in {elapsed:.1f}s"
        )
        
        return {
            'input_file': input_file,
            'output_file': output_file,
            'total_processed': self.processed_count,
            'total_results': len(total_results),
            'processing_time': elapsed,
            'rate': self.processed_count / elapsed if elapsed > 0 else 0,
            'workers_used': self.max_workers
        }

def main():
    """Main function for testing the streaming processor"""
    
    # Configuration
    config = RMLConfig(
        output_dir="output/rml_streaming",
        save_memory_graphs=True,
        log_level="INFO",
        batch_size=4,  # Smaller batch size for streaming
        max_concepts_per_input=30  # Limit concepts for efficiency
    )
    
    # Initialize processor
    processor = RMLStreamingProcessor(config, max_workers=2)
    
    # Test with a small file first
    test_file = "data/cpp_rml_output_v5/concepts.jsonl"
    
    if os.path.exists(test_file):
        print("🧪 Testing RML Streaming Processor")
        print("="*50)
        
        # Process with streaming
        result = processor.process_file(
            input_file=test_file,
            output_file="output/rml_streaming/test_output.jsonl",
            chunk_size=50
        )
        
        print(f"📊 Processing Results:")
        print(f"   Input: {result['input_file']}")
        print(f"   Output: {result['output_file']}")
        print(f"   Processed: {result['total_processed']} items")
        print(f"   Results: {result['total_results']} items")
        print(f"   Time: {result['processing_time']:.1f}s")
        print(f"   Rate: {result['rate']:.1f} items/sec")
        
    else:
        print(f"❌ Test file not found: {test_file}")
        print("💡 Try running with an existing RML data file")

if __name__ == "__main__":
    main() 