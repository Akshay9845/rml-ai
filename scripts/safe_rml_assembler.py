#!/usr/bin/env python3
"""
Safe RML Assembler - Monitored and Crash-Proof
Processes one directory at a time with constant monitoring
"""

import os
import json
import glob
import psutil
import time
from pathlib import Path
from collections import defaultdict
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafeRMLAssembler:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.rml_components = [
            'concepts', 'emotions', 'entities', 'events', 'intents',
            'reasoning', 'summaries', 'tags', 'triples', 'vectors'
        ]
        
        # Memory limits (in MB)
        self.max_memory_mb = 2000  # 2GB limit
        self.max_storage_gb = 100  # 100GB limit
        
    def get_system_stats(self):
        """Get current system statistics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'memory_used_mb': memory.used / (1024 * 1024),
            'memory_percent': memory.percent,
            'disk_used_gb': disk.used / (1024 * 1024 * 1024),
            'disk_percent': (disk.used / disk.total) * 100
        }
    
    def check_system_health(self):
        """Check if system is healthy for processing"""
        stats = self.get_system_stats()
        
        # Memory check
        if stats['memory_used_mb'] > self.max_memory_mb:
            logger.warning(f"⚠️ Memory usage high: {stats['memory_used_mb']:.1f}MB")
            return False
        
        # Storage check
        if stats['disk_used_gb'] > self.max_storage_gb:
            logger.warning(f"⚠️ Storage usage high: {stats['disk_used_gb']:.1f}GB")
            return False
        
        return True
    
    def print_system_status(self):
        """Print current system status"""
        stats = self.get_system_stats()
        logger.info(f"💻 RAM: {stats['memory_used_mb']:.1f}MB ({stats['memory_percent']:.1f}%) | "
                   f"💾 Storage: {stats['disk_used_gb']:.1f}GB ({stats['disk_percent']:.1f}%)")
    
    def find_component_files(self, directory):
        """Find component files in a directory"""
        logger.info(f"🔍 Scanning: {directory}")
        
        component_files = {}
        for component in self.rml_components:
            pattern = f"*{component}*.jsonl"
            files = glob.glob(os.path.join(directory, pattern))
            if files:
                component_files[component] = files
                logger.info(f"  📁 {component}: {len(files)} files")
        
        return component_files
    
    def process_single_directory(self, directory):
        """Process a single directory safely"""
        logger.info(f"🚀 Processing: {directory}")
        self.print_system_status()
        
        # Find component files
        component_files = self.find_component_files(directory)
        
        if not component_files:
            logger.warning(f"  ⚠️ No component files found")
            return 0
        
        # Store component data by record_id
        component_data = defaultdict(dict)
        complete_samples = []
        
        # Process each component
        for component, files in component_files.items():
            logger.info(f"  🔄 Processing {component}...")
            
            for file_path in files:
                if not self.check_system_health():
                    logger.warning("  ⚠️ System resources low, pausing...")
                    time.sleep(5)
                    gc.collect()
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line_count = 0
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                record_id = data.get('record_id')
                                chunk = data.get('chunk', 1)
                                
                                if record_id is not None:
                                    key = f"{record_id}_{chunk}"
                                    component_data[key][component] = data
                                
                                line_count += 1
                                
                                # Progress update every 1000 lines
                                if line_count % 1000 == 0:
                                    logger.info(f"    📊 {line_count} lines processed from {os.path.basename(file_path)}")
                                    self.print_system_status()
                                
                                # Memory cleanup every 10000 lines
                                if line_count % 10000 == 0:
                                    gc.collect()
                                    
                            except json.JSONDecodeError:
                                continue
                            
                except Exception as e:
                    logger.error(f"  ❌ Error reading {file_path}: {e}")
                    continue
        
        logger.info(f"  ✅ Loaded {len(component_data)} unique records")
        
        # Assemble complete RML samples
        logger.info("  🔧 Assembling complete RML samples...")
        assembled_count = 0
        
        for record_key, components in component_data.items():
            if len(components) >= 3:  # Minimum 3 components
                complete_sample = self.create_complete_rml_sample(record_key, components)
                complete_samples.append(complete_sample)
                assembled_count += 1
                
                # Progress update every 1000 samples
                if assembled_count % 1000 == 0:
                    logger.info(f"    📊 Assembled {assembled_count} samples")
                    self.print_system_status()
        
        logger.info(f"  ✅ Assembled {assembled_count} complete RML samples")
        
        # Save results
        if assembled_count > 0:
            output_file = f"data/complete_rml_{os.path.basename(directory)}.jsonl"
            self.save_complete_rml(complete_samples, output_file)
        
        # Cleanup
        del component_data
        del complete_samples
        gc.collect()
        
        return assembled_count
    
    def create_complete_rml_sample(self, record_key, components):
        """Create a complete RML sample with all 10 components"""
        complete_sample = {
            'record_id': None,
            'chunk': 1,
            'concepts': [],
            'emotions': '',
            'entities': {},
            'events': [],
            'intents': '',
            'reasoning': '',
            'summaries': '',
            'tags': '',
            'triples': {},
            'vectors': {}
        }
        
        # Extract record_id and chunk from key
        if '_' in record_key:
            record_id, chunk = record_key.split('_', 1)
            complete_sample['record_id'] = int(record_id)
            complete_sample['chunk'] = int(chunk)
        
        # Fill in available components
        for component, data in components.items():
            if component == 'concepts' and 'concepts' in data:
                complete_sample['concepts'] = data['concepts']
            elif component == 'emotions' and 'tone' in data:
                complete_sample['emotions'] = data['tone']
            elif component == 'entities' and 'has_numbers' in data:
                complete_sample['entities'] = {'has_numbers': data['has_numbers']}
            elif component == 'events' and 'has_events' in data:
                complete_sample['events'] = [data['has_events']]
            elif component == 'intents' and 'is_informative' in data:
                complete_sample['intents'] = str(data['is_informative'])
            elif component == 'reasoning' and 'has_logic' in data:
                complete_sample['reasoning'] = str(data['has_logic'])
            elif component == 'summaries' and 'summary' in data:
                complete_sample['summaries'] = data['summary']
            elif component == 'tags' and 'category' in data:
                complete_sample['tags'] = data['category']
            elif component == 'triples' and all(k in data for k in ['subject', 'relation', 'object']):
                complete_sample['triples'] = {
                    'subject': data['subject'],
                    'relation': data['relation'],
                    'object': data['object']
                }
            elif component == 'vectors' and 'text_hash' in data:
                complete_sample['vectors'] = {'text_hash': data['text_hash']}
        
        return complete_sample
    
    def save_complete_rml(self, samples, output_file):
        """Save complete RML samples to file"""
        logger.info(f"  💾 Saving to: {output_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        size_mb = file_size / (1024 * 1024)
        logger.info(f"  📁 File size: {size_mb:.2f} MB")
    
    def find_rml_directories(self):
        """Find directories with RML component files"""
        logger.info("🔍 Finding RML directories...")
        
        rml_directories = []
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if any(component in file for component in self.rml_components) and file.endswith('.jsonl'):
                    rml_directories.append(root)
                    break
        
        # Remove duplicates and sort
        rml_directories = sorted(list(set(rml_directories)))
        
        logger.info(f"📁 Found {len(rml_directories)} directories with RML data")
        return rml_directories
    
    def run(self):
        """Main execution with monitoring"""
        logger.info("🚀 Starting Safe RML Assembly...")
        self.print_system_status()
        
        # Find all RML directories
        rml_directories = self.find_rml_directories()
        
        if not rml_directories:
            logger.warning("❌ No RML directories found!")
            return
        
        total_assembled = 0
        
        for i, directory in enumerate(rml_directories, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📁 Processing directory {i}/{len(rml_directories)}: {directory}")
            logger.info(f"{'='*60}")
            
            try:
                assembled = self.process_single_directory(directory)
                total_assembled += assembled
                
                logger.info(f"✅ Directory {i} completed: {assembled} samples")
                logger.info(f"📊 Total so far: {total_assembled} samples")
                self.print_system_status()
                
                # Small pause between directories
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing {directory}: {e}")
                continue
        
        logger.info(f"\n🎉 Complete RML assembly finished!")
        logger.info(f"📊 Total assembled: {total_assembled} complete RML samples")
        self.print_system_status()

def main():
    assembler = SafeRMLAssembler()
    assembler.run()

if __name__ == "__main__":
    main() 