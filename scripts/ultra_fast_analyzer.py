#!/usr/bin/env python3
"""
Ultra-Fast RML Data Analyzer and Converter
First analyzes data structure, then processes efficiently in batches
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict, Counter
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraFastAnalyzer:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.data_structure = defaultdict(list)
        self.file_types = Counter()
        self.total_files = 0
        self.processed_files = 0
        
    def quick_analyze_structure(self):
        """Quick analysis of data structure - just sample files"""
        logger.info("🔍 QUICK DATA STRUCTURE ANALYSIS...")
        
        # Find all JSONL files
        jsonl_files = glob.glob(os.path.join(self.data_dir, "**/*.jsonl"), recursive=True)
        self.total_files = len(jsonl_files)
        
        logger.info(f"📁 Found {self.total_files:,} JSONL files")
        
        # Sample first 100 files to understand structure
        sample_files = jsonl_files[:100]
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read first line only
                    line = f.readline().strip()
                    if line:
                        data = json.loads(line)
                        keys = list(data.keys())
                        
                        # Categorize file type
                        if len(keys) >= 8:  # Likely complete RML
                            self.file_types['complete_rml'] += 1
                        elif len(keys) <= 3:  # Likely component file
                            self.file_types['component_file'] += 1
                        else:
                            self.file_types['mixed_data'] += 1
                        
                        # Store key patterns
                        key_pattern = tuple(sorted(keys))
                        self.data_structure[key_pattern].append(file_path)
                        
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        # Print analysis results
        logger.info("📊 QUICK ANALYSIS RESULTS:")
        logger.info(f"   Complete RML files: {self.file_types['complete_rml']}")
        logger.info(f"   Component files: {self.file_types['component_file']}")
        logger.info(f"   Mixed data files: {self.file_types['mixed_data']}")
        logger.info(f"   Unique key patterns: {len(self.data_structure)}")
        
        # Show most common patterns
        logger.info("🔍 Most common data patterns:")
        for pattern, files in sorted(self.data_structure.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            logger.info(f"   Pattern {pattern}: {len(files)} files")
        
        return self.data_structure
    
    def estimate_processing_time(self):
        """Estimate how long processing will take"""
        logger.info("⏱️ ESTIMATING PROCESSING TIME...")
        
        # Based on sample analysis, estimate
        total_size_gb = 372  # Your total data size
        estimated_time_minutes = (self.total_files / 1000) * 2  # Rough estimate
        
        logger.info(f"📊 ESTIMATES:")
        logger.info(f"   Total files: {self.total_files:,}")
        logger.info(f"   Total data: {total_size_gb} GB")
        logger.info(f"   Estimated time: {estimated_time_minutes:.1f} minutes")
        
        return estimated_time_minutes
    
    def show_processing_plan(self):
        """Show the processing plan"""
        logger.info("📋 PROCESSING PLAN:")
        logger.info("1. 🚀 Process complete RML files first (fastest)")
        logger.info("2. 🔧 Process component files by type")
        logger.info("3. 🔄 Process mixed data files last")
        logger.info("4. 📁 Organize by converting in place")
        logger.info("5. 💾 Monitor storage throughout")
        
        # Ask user if they want to proceed
        response = input("\n❓ Do you want to proceed with this plan? (y/n): ")
        return response.lower() == 'y'

def main():
    analyzer = UltraFastAnalyzer()
    
    # Quick analysis
    structure = analyzer.quick_analyze_structure()
    
    # Estimate time
    estimated_time = analyzer.estimate_processing_time()
    
    # Show plan
    if analyzer.show_processing_plan():
        logger.info("🚀 Starting ultra-fast processing...")
        # Here we would start the actual processing
    else:
        logger.info("⏹️ Processing cancelled by user")

if __name__ == "__main__":
    main() 