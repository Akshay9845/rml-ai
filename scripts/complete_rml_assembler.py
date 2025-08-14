#!/usr/bin/env python3
"""
Complete RML Assembler
Reads separated component files and assembles them into complete RML objects
Every line will contain all 10 RML components
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteRMLAssembler:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.rml_components = [
            'concepts', 'emotions', 'entities', 'events', 'intents',
            'reasoning', 'summaries', 'tags', 'triples', 'vectors'
        ]
        
        # Store component data by record_id
        self.component_data = defaultdict(dict)
        self.complete_rml_samples = []
        
    def find_component_files(self, directory):
        """Find all component files in a directory"""
        logger.info(f"🔍 Finding component files in: {directory}")
        
        component_files = {}
        for component in self.rml_components:
            pattern = f"*{component}*.jsonl"
            files = glob.glob(os.path.join(directory, pattern))
            if files:
                component_files[component] = files
                logger.info(f"  📁 {component}: {len(files)} files")
        
        return component_files
    
    def load_component_data(self, component_files):
        """Load all component data into memory"""
        logger.info("📥 Loading component data...")
        
        for component, files in component_files.items():
            logger.info(f"  🔄 Loading {component} data...")
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                record_id = data.get('record_id')
                                chunk = data.get('chunk', 1)
                                
                                if record_id is not None:
                                    key = f"{record_id}_{chunk}"
                                    self.component_data[key][component] = data
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"    Line {line_num} in {file_path}: Invalid JSON")
                                continue
                            
                            # Progress update every 10000 lines
                            if line_num % 10000 == 0:
                                logger.info(f"    Processed {line_num} lines from {os.path.basename(file_path)}")
                                
                except Exception as e:
                    logger.error(f"  ❌ Error reading {file_path}: {e}")
        
        logger.info(f"✅ Loaded data for {len(self.component_data)} unique records")
    
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
    
    def assemble_complete_rml(self):
        """Assemble complete RML samples from component data"""
        logger.info("🔧 Assembling complete RML samples...")
        
        assembled_count = 0
        missing_components = defaultdict(int)
        
        for record_key, components in self.component_data.items():
            # Check if we have at least 3 components (minimum viable RML)
            if len(components) >= 3:
                complete_sample = self.create_complete_rml_sample(record_key, components)
                self.complete_rml_samples.append(complete_sample)
                assembled_count += 1
                
                # Track missing components
                missing = set(self.rml_components) - set(components.keys())
                for missing_comp in missing:
                    missing_components[missing_comp] += 1
            else:
                logger.debug(f"  ⚠️ Record {record_key} has only {len(components)} components, skipping")
        
        logger.info(f"✅ Assembled {assembled_count} complete RML samples")
        logger.info("📊 Missing components analysis:")
        for component, count in sorted(missing_components.items()):
            percentage = (count / assembled_count) * 100 if assembled_count > 0 else 0
            logger.info(f"  {component}: {count} samples missing ({percentage:.1f}%)")
        
        return assembled_count
    
    def save_complete_rml(self, output_file="data/complete_rml_data.jsonl"):
        """Save complete RML samples to file"""
        logger.info(f"💾 Saving complete RML data to: {output_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.complete_rml_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Saved {len(self.complete_rml_samples)} complete RML samples")
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        size_mb = file_size / (1024 * 1024)
        logger.info(f"📁 File size: {size_mb:.2f} MB")
    
    def process_directory(self, directory):
        """Process a single directory"""
        logger.info(f"🚀 Processing directory: {directory}")
        
        # Find component files
        component_files = self.find_component_files(directory)
        
        if not component_files:
            logger.warning(f"  ⚠️ No component files found in {directory}")
            return 0
        
        # Load component data
        self.load_component_data(component_files)
        
        # Assemble complete RML
        assembled_count = self.assemble_complete_rml()
        
        # Save results
        output_file = f"data/complete_rml_{os.path.basename(directory)}.jsonl"
        self.save_complete_rml(output_file)
        
        return assembled_count
    
    def process_all_directories(self):
        """Process all directories with RML data"""
        logger.info("🚀 Starting complete RML assembly for all directories...")
        
        # Find all directories that might contain RML data
        rml_directories = []
        
        # Look for directories with component files
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if any(component in file for component in self.rml_components) and file.endswith('.jsonl'):
                    rml_directories.append(root)
                    break
        
        # Remove duplicates and sort
        rml_directories = sorted(list(set(rml_directories)))
        
        logger.info(f"📁 Found {len(rml_directories)} directories with RML data")
        
        total_assembled = 0
        
        for directory in rml_directories:
            try:
                assembled = self.process_directory(directory)
                total_assembled += assembled
                
                # Clear component data for next directory
                self.component_data.clear()
                self.complete_rml_samples.clear()
                
            except Exception as e:
                logger.error(f"❌ Error processing {directory}: {e}")
        
        logger.info(f"🎉 Total assembled: {total_assembled} complete RML samples")
        return total_assembled

def main():
    assembler = CompleteRMLAssembler()
    
    # Process all directories
    total_assembled = assembler.process_all_directories()
    
    print(f"\n✅ Complete RML assembly finished!")
    print(f"📊 Total complete RML samples: {total_assembled}")
    print(f"📁 Check data/complete_rml_*.jsonl files for results")

if __name__ == "__main__":
    main() 