#!/usr/bin/env python3
"""
Smart RML Data Consolidator
Assembles scattered RML components from different files into complete training-ready format
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartRMLConsolidator:
    def __init__(self, data_dir="data/", output_dir="data/training_ready/"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.complete_rml_files = []  # Files with all 10 components
        self.component_files = defaultdict(list)  # Files with individual components
        self.record_mapping = defaultdict(dict)  # Maps record_id/doc_id to components
        self.processed_samples = 0
        self.duplicates_removed = 0
        
        # RML components we need
        self.rml_components = [
            'concepts', 'triples', 'entities', 'emotions', 
            'reasoning', 'intents', 'summaries', 'events', 'vectors', 'tags'
        ]
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/train", exist_ok=True)
        os.makedirs(f"{output_dir}/validation", exist_ok=True)
        os.makedirs(f"{output_dir}/test", exist_ok=True)
        
        # Output files
        self.train_file = open(f"{output_dir}/train/train.jsonl", "w")
        self.val_file = open(f"{output_dir}/validation/validation.jsonl", "w")
        self.test_file = open(f"{output_dir}/test/test.jsonl", "w")
        
        # Deduplication
        self.seen_hashes = set()
        
    def __del__(self):
        """Cleanup output files"""
        if hasattr(self, 'train_file'):
            self.train_file.close()
        if hasattr(self, 'val_file'):
            self.val_file.close()
        if hasattr(self, 'test_file'):
            self.test_file.close()
    
    def categorize_files(self):
        """Categorize all JSONL files by their content type"""
        logger.info("🔍 Categorizing RML files...")
        
        # Find all JSONL files
        jsonl_files = glob.glob(os.path.join(self.data_dir, "**/*.jsonl"), recursive=True)
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        for file_path in jsonl_files:
            try:
                # Check first few lines to understand format
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    
                    try:
                        data = json.loads(first_line)
                        keys = list(data.keys())
                        
                        # Check if this is a complete RML file
                        if all(comp in keys for comp in self.rml_components):
                            self.complete_rml_files.append(file_path)
                            logger.info(f"✅ Complete RML file: {os.path.basename(file_path)}")
                        
                        # Check for component-specific files
                        elif len(keys) <= 3:
                            # Look for component indicators in keys or values
                            component_found = None
                            for comp in self.rml_components:
                                if comp in keys or any(comp in str(v).lower() for v in data.values()):
                                    component_found = comp
                                    break
                            
                            if component_found:
                                self.component_files[component_found].append(file_path)
                                logger.info(f"📁 {component_found} file: {os.path.basename(file_path)}")
                        
                    except json.JSONDecodeError:
                        continue
                        
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        logger.info(f"📊 Categorization complete:")
        logger.info(f"   Complete RML files: {len(self.complete_rml_files)}")
        for comp, files in self.component_files.items():
            logger.info(f"   {comp} files: {len(files)}")
    
    def process_complete_rml_files(self):
        """Process files that already have complete RML data"""
        logger.info("🚀 Processing complete RML files...")
        
        for file_path in self.complete_rml_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            # Format for training
                            training_text = self.format_for_training(data, "complete_rml")
                            
                            # Check for duplicates
                            if self.is_duplicate(training_text):
                                self.duplicates_removed += 1
                                continue
                            
                            # Write to appropriate split
                            self.write_sample(training_text, "complete_rml")
                            self.processed_samples += 1
                            
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"✅ Processed {len(self.complete_rml_files)} complete RML files")
        logger.info(f"   Samples: {self.processed_samples}")
        logger.info(f"   Duplicates removed: {self.duplicates_removed}")
    
    def process_component_files(self):
        """Process scattered component files and assemble them"""
        logger.info("🔧 Processing scattered component files...")
        
        # Group files by their base identifier (record_id, doc_id, etc.)
        component_groups = defaultdict(lambda: defaultdict(list))
        
        for component, files in self.component_files.items():
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                
                                # Extract identifier
                                record_id = self.extract_record_id(data)
                                if record_id:
                                    component_groups[record_id][component].append(data)
                                    
                            except json.JSONDecodeError:
                                continue
                                
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        logger.info(f"📊 Found {len(component_groups)} record groups")
        
        # Assemble complete RML objects
        assembled_count = 0
        for record_id, components in component_groups.items():
            if len(components) >= 3:  # At least 3 components to be useful
                assembled_rml = self.assemble_rml_components(components, record_id)
                if assembled_rml:
                    training_text = self.format_for_training(assembled_rml, "assembled")
                    
                    if not self.is_duplicate(training_text):
                        self.write_sample(training_text, "assembled")
                        assembled_count += 1
                    else:
                        self.duplicates_removed += 1
        
        logger.info(f"✅ Assembled {assembled_count} complete RML objects from components")
        self.processed_samples += assembled_count
    
    def extract_record_id(self, data):
        """Extract record identifier from data"""
        # Try different possible ID fields
        for id_field in ['record_id', 'doc_id', 'document_id']:
            if id_field in data:
                return str(data[id_field])
        
        # If no ID field, create hash from content
        content_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def assemble_rml_components(self, components, record_id):
        """Assemble scattered components into complete RML object"""
        assembled = {}
        
        for component_name, component_data_list in components.items():
            if component_data_list:
                # Take the first (or best) component data
                component_data = component_data_list[0]
                
                # Extract the actual component value
                if component_name in component_data:
                    assembled[component_name] = component_data[component_name]
                elif 'data' in component_data:
                    assembled[component_name] = component_data['data']
                elif component_name in ['concepts', 'entities', 'emotions', 'intents', 'tags']:
                    # Handle single values
                    for key, value in component_data.items():
                        if key != 'record_id' and key != 'doc_id':
                            assembled[component_name] = [value] if not isinstance(value, list) else value
                            break
                elif component_name == 'triples':
                    # Handle triple format
                    if 'triple' in component_data:
                        assembled[component_name] = [{'subject': 'unknown', 'predicate': 'is', 'object': component_data['triple']}]
                    elif all(k in component_data for k in ['subject', 'relation', 'object']):
                        assembled[component_name] = [{'subject': component_data['subject'], 'predicate': component_data['relation'], 'object': component_data['object']}]
                elif component_name == 'summaries':
                    if 'summary' in component_data:
                        assembled[component_name] = [component_data['summary']]
                elif component_name == 'events':
                    if 'event' in component_data:
                        assembled[component_name] = [component_data['event']]
                elif component_name == 'vectors':
                    if 'vector' in component_data:
                        assembled[component_name] = [component_data['vector']]
                elif component_name == 'reasoning':
                    if 'reasoning' in component_data:
                        assembled[component_name] = [component_data['reasoning']]
        
        # Fill missing components with defaults
        for component in self.rml_components:
            if component not in assembled:
                if component in ['concepts', 'entities', 'emotions', 'intents', 'tags']:
                    assembled[component] = []
                elif component in ['triples', 'summaries', 'events', 'vectors', 'reasoning']:
                    assembled[component] = []
        
        return assembled if len(assembled) >= 3 else None
    
    def format_for_training(self, data, source_type):
        """Format RML data for training"""
        lines = []
        
        # Add source type tag
        lines.append(f"<SOURCE>{source_type}</SOURCE>")
        
        # Add components in order
        for component in self.rml_components:
            if component in data and data[component]:
                if component == 'triples':
                    # Format triples specially
                    triple_strs = []
                    for triple in data[component]:
                        if isinstance(triple, dict):
                            triple_strs.append(f"{triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}")
                        else:
                            triple_strs.append(str(triple))
                    lines.append(f"<{component.upper()}>{' | '.join(triple_strs)}</{component.upper()}>")
                else:
                    # Format other components
                    if isinstance(data[component], list):
                        content = ' '.join(str(item) for item in data[component])
                    else:
                        content = str(data[component])
                    lines.append(f"<{component.upper()}>{content}</{component.upper()}>")
        
        return '\n'.join(lines)
    
    def is_duplicate(self, text):
        """Check if text is duplicate"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False
    
    def write_sample(self, training_text, source_type):
        """Write sample to appropriate split"""
        import random
        
        # Create sample
        sample = {
            "text": training_text,
            "source_type": source_type
        }
        
        # Random split (80/10/10)
        rand_val = random.random()
        
        if rand_val < 0.8:
            self.train_file.write(json.dumps(sample) + '\n')
        elif rand_val < 0.9:
            self.val_file.write(json.dumps(sample) + '\n')
        else:
            self.test_file.write(json.dumps(sample) + '\n')
    
    def create_metadata(self):
        """Create metadata file"""
        metadata = {
            "dataset_info": {
                "name": "Smart RML Training Dataset",
                "version": "1.0",
                "description": "Consolidated RML data from multiple sources and formats"
            },
            "statistics": {
                "total_samples": self.processed_samples,
                "duplicates_removed": self.duplicates_removed,
                "complete_rml_files": len(self.complete_rml_files),
                "component_files_processed": sum(len(files) for files in self.component_files.values())
            },
            "processing": {
                "strategy": "smart_assembly",
                "components_assembled": list(self.component_files.keys())
            }
        }
        
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("💾 Metadata saved")
    
    def consolidate(self):
        """Run complete consolidation process"""
        logger.info("🚀 Starting SMART RML consolidation...")
        
        start_time = time.time()
        
        # Step 1: Categorize files
        self.categorize_files()
        
        # Step 2: Process complete RML files
        self.process_complete_rml_files()
        
        # Step 3: Process and assemble component files
        self.process_component_files()
        
        # Step 4: Create metadata
        self.create_metadata()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\n🎉 SMART RML CONSOLIDATION COMPLETE!")
        logger.info(f"📊 Total samples: {self.processed_samples}")
        logger.info(f"📊 Duplicates removed: {self.duplicates_removed}")
        logger.info(f"⏱️  Time taken: {duration/60:.2f} minutes")
        logger.info(f"📁 Output: {self.output_dir}")
        
        return self.processed_samples

def main():
    consolidator = SmartRMLConsolidator()
    samples = consolidator.consolidate()
    
    print(f"\n✅ Successfully consolidated {samples} RML samples!")
    print("📁 Check the training_ready/ directory for the output files.")

if __name__ == "__main__":
    main() 