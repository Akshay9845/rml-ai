#!/usr/bin/env python3
"""
Rule-Based RML Assembler
Checks for RML components within the same folder only, using record_id matching
"""

import os
import json
import glob
import psutil
import time
from pathlib import Path
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RuleBasedRMLAssembler:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.rml_components = [
            'concepts', 'emotions', 'entities', 'events', 'intents',
            'reasoning', 'summaries', 'tags', 'triples', 'vectors'
        ]
        
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
    
    def print_system_status(self):
        """Print current system status"""
        stats = self.get_system_stats()
        logger.info(f"💻 RAM: {stats['memory_used_mb']:.1f}MB ({stats['memory_percent']:.1f}%) | "
                   f"💾 Storage: {stats['disk_used_gb']:.1f}GB ({stats['disk_percent']:.1f}%)")
    
    def find_folders_with_concepts(self):
        """Find all folders that contain concepts files"""
        logger.info("🔍 Finding folders with concepts files...")
        
        folders_with_concepts = []
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if 'concepts' in file and file.endswith('.jsonl'):
                    folders_with_concepts.append(root)
                    logger.info(f"  📁 Found concepts in: {root}")
                    break
        
        # Remove duplicates and sort
        folders_with_concepts = sorted(list(set(folders_with_concepts)))
        
        logger.info(f"📊 Found {len(folders_with_concepts)} folders with concepts files")
        return folders_with_concepts
    
    def find_component_files_in_folder(self, folder_path):
        """Find all RML component files in a specific folder"""
        logger.info(f"🔍 Scanning folder: {folder_path}")
        
        component_files = defaultdict(list)
        
        for component in self.rml_components:
            pattern = f"*{component}*.jsonl"
            files = glob.glob(os.path.join(folder_path, pattern))
            if files:
                component_files[component] = files
                logger.info(f"  📁 {component}: {len(files)} files")
        
        return component_files
    
    def read_concepts_from_folder(self, concepts_files, max_records=10):
        """Read concepts records from a folder"""
        logger.info(f"📖 Reading concepts from {len(concepts_files)} files...")
        
        concepts_records = []
        
        for concepts_file in concepts_files:
            try:
                with open(concepts_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if len(concepts_records) >= max_records:
                            break
                            
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            record_id = data.get('record_id')
                            chunk = data.get('chunk', 1)
                            
                            if record_id is not None:
                                concepts_records.append({
                                    'record_id': record_id,
                                    'chunk': chunk,
                                    'concepts': data.get('concepts', []),
                                    'source_file': os.path.basename(concepts_file)
                                })
                                
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.error(f"  ❌ Error reading {concepts_file}: {e}")
                continue
        
        logger.info(f"  ✅ Found {len(concepts_records)} concepts records")
        return concepts_records
    
    def find_matching_components(self, folder_path, component_files, concepts_records):
        """Find matching RML components for each concepts record in the same folder"""
        logger.info(f"🔍 Finding matching components in folder: {folder_path}")
        
        complete_records = []
        
        for concept_record in concepts_records:
            record_id = concept_record['record_id']
            chunk = concept_record['chunk']
            
            logger.info(f"  🔍 Looking for record_id: {record_id}, chunk: {chunk}")
            
            found_components = {'concepts': concept_record}
            missing_components = []
            
            # Check each component type in the same folder
            for component in self.rml_components:
                if component == 'concepts':
                    continue  # Already have this
                
                if component in component_files:
                    # Search in all files of this component type
                    found = False
                    for file_path in component_files[component]:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    try:
                                        data = json.loads(line)
                                        if (data.get('record_id') == record_id and 
                                            data.get('chunk', 1) == chunk):
                                            found_components[component] = data
                                            found = True
                                            logger.info(f"    ✅ {component}: Found in {os.path.basename(file_path)}")
                                            break
                                    except json.JSONDecodeError:
                                        continue
                            
                            if found:
                                break
                                
                        except Exception as e:
                            logger.error(f"    ❌ Error reading {file_path}: {e}")
                            continue
                    
                    if not found:
                        missing_components.append(component)
                        logger.warning(f"    ❌ {component}: Missing")
                else:
                    missing_components.append(component)
                    logger.warning(f"    ❌ {component}: No files found")
            
            # Create complete record
            complete_record = self.create_complete_rml_record(
                record_id, chunk, found_components, missing_components, concept_record
            )
            
            complete_records.append(complete_record)
            
            logger.info(f"  📊 Record {record_id}: {len(found_components)}/10 components found")
        
        return complete_records
    
    def generate_missing_component(self, component, record_id, chunk, concepts_data):
        """Generate a missing RML component"""
        logger.info(f"🔧 Generating missing {component} for record_id: {record_id}")
        
        generated_data = {
            'record_id': record_id,
            'chunk': chunk
        }
        
        # Get concepts for generation
        concepts = concepts_data.get('concepts', []) if concepts_data else []
        
        # Generate based on component type
        if component == 'emotions':
            # Analyze concepts for emotion
            if concepts:
                # Simple emotion detection based on concepts
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'success']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disaster', 'disease', 'infection']
                
                text = ' '.join(concepts).lower()
                if any(word in text for word in positive_words):
                    generated_data['tone'] = 'positive'
                elif any(word in text for word in negative_words):
                    generated_data['tone'] = 'negative'
                else:
                    generated_data['tone'] = 'neutral'
            else:
                generated_data['tone'] = 'neutral'
                
        elif component == 'entities':
            # Check if concepts contain numbers or named entities
            has_numbers = any(any(char.isdigit() for char in str(concept)) for concept in concepts)
            generated_data['has_numbers'] = has_numbers
            
        elif component == 'events':
            # Check if concepts suggest events/actions
            action_words = ['run', 'jump', 'play', 'work', 'study', 'eat', 'sleep', 'infect', 'spread']
            text = ' '.join(concepts).lower()
            has_events = any(word in text for word in action_words)
            generated_data['has_events'] = has_events
            
        elif component == 'intents':
            # Determine intent based on concepts
            generated_data['is_informative'] = len(concepts) > 0
                
        elif component == 'reasoning':
            # Check for logical structures
            logic_words = ['because', 'therefore', 'since', 'thus', 'hence', 'if', 'then']
            text = ' '.join(concepts).lower()
            has_logic = any(word in text for word in logic_words)
            generated_data['has_logic'] = has_logic
            
        elif component == 'summaries':
            # Create summary from concepts
            if concepts:
                summary = f"Content contains concepts: {', '.join(concepts[:5])}"
                if len(concepts) > 5:
                    summary += f" and {len(concepts) - 5} more"
            else:
                summary = "No concepts available"
            generated_data['summary'] = summary
            
        elif component == 'tags':
            # Generate category tag based on folder name
            folder_name = os.path.basename(os.path.dirname(concepts_data.get('source_file', '')))
            generated_data['category'] = folder_name if folder_name else 'unknown'
            
        elif component == 'triples':
            # Generate basic triples from concepts
            if len(concepts) >= 3:
                generated_data['subject'] = [concepts[0]]
                generated_data['relation'] = 'contains'
                generated_data['object'] = concepts[1:3]
            else:
                generated_data['subject'] = ['content']
                generated_data['relation'] = 'has'
                generated_data['object'] = concepts if concepts else ['information']
                
        elif component == 'vectors':
            # Generate text hash
            text = ' '.join(concepts)
            text_hash = hash(text) % 1000000  # Simple hash
            generated_data['text_hash'] = text_hash
        
        logger.info(f"  ✅ Generated {component}: {generated_data}")
        return generated_data
    
    def create_complete_rml_record(self, record_id, chunk, found_components, missing_components, concepts_data):
        """Create a complete RML record with all 10 components"""
        logger.info(f"🔧 Creating complete RML record for record_id: {record_id}")
        
        complete_record = {
            'record_id': record_id,
            'chunk': chunk,
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
        
        # Fill in found components
        for component, data in found_components.items():
            if component == 'concepts' and 'concepts' in data:
                complete_record['concepts'] = data['concepts']
            elif component == 'emotions' and 'tone' in data:
                complete_record['emotions'] = data['tone']
            elif component == 'entities' and 'has_numbers' in data:
                complete_record['entities'] = {'has_numbers': data['has_numbers']}
            elif component == 'events' and 'has_events' in data:
                complete_record['events'] = [data['has_events']]
            elif component == 'intents' and 'is_informative' in data:
                complete_record['intents'] = str(data['is_informative'])
            elif component == 'reasoning' and 'has_logic' in data:
                complete_record['reasoning'] = str(data['has_logic'])
            elif component == 'summaries' and 'summary' in data:
                complete_record['summaries'] = data['summary']
            elif component == 'tags' and 'category' in data:
                complete_record['tags'] = data['category']
            elif component == 'triples' and all(k in data for k in ['subject', 'relation', 'object']):
                complete_record['triples'] = {
                    'subject': data['subject'],
                    'relation': data['relation'],
                    'object': data['object']
                }
            elif component == 'vectors' and 'text_hash' in data:
                complete_record['vectors'] = {'text_hash': data['text_hash']}
        
        # Generate missing components
        for component in missing_components:
            generated_data = self.generate_missing_component(component, record_id, chunk, concepts_data)
            
            # Add to complete record
            if component == 'emotions':
                complete_record['emotions'] = generated_data['tone']
            elif component == 'entities':
                complete_record['entities'] = {'has_numbers': generated_data['has_numbers']}
            elif component == 'events':
                complete_record['events'] = [generated_data['has_events']]
            elif component == 'intents':
                complete_record['intents'] = str(generated_data['is_informative'])
            elif component == 'reasoning':
                complete_record['reasoning'] = str(generated_data['has_logic'])
            elif component == 'summaries':
                complete_record['summaries'] = generated_data['summary']
            elif component == 'tags':
                complete_record['tags'] = generated_data['category']
            elif component == 'triples':
                complete_record['triples'] = {
                    'subject': generated_data['subject'],
                    'relation': generated_data['relation'],
                    'object': generated_data['object']
                }
            elif component == 'vectors':
                complete_record['vectors'] = {'text_hash': generated_data['text_hash']}
        
        logger.info(f"✅ Complete RML record created with all 10 components")
        return complete_record
    
    def save_complete_records(self, complete_records, folder_name, output_dir="data/complete_rml"):
        """Save complete RML records"""
        if not complete_records:
            logger.warning("  ⚠️ No complete records to save")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"complete_rml_{folder_name}.jsonl")
        
        logger.info(f"  💾 Saving {len(complete_records)} records to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in complete_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        size_mb = file_size / (1024 * 1024)
        logger.info(f"  📁 File size: {size_mb:.2f} MB")
    
    def process_folder(self, folder_path, max_records=10):
        """Process a single folder using rule-based approach"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing folder: {folder_path}")
        logger.info(f"{'='*60}")
        
        self.print_system_status()
        
        # Find component files in this folder
        component_files = self.find_component_files_in_folder(folder_path)
        
        if 'concepts' not in component_files:
            logger.warning(f"  ⚠️ No concepts files found in {folder_path}")
            return 0
        
        # Read concepts records
        concepts_records = self.read_concepts_from_folder(component_files['concepts'], max_records)
        
        if not concepts_records:
            logger.warning(f"  ⚠️ No concepts records found in {folder_path}")
            return 0
        
        # Find matching components
        complete_records = self.find_matching_components(folder_path, component_files, concepts_records)
        
        # Save results
        folder_name = os.path.basename(folder_path)
        self.save_complete_records(complete_records, folder_name)
        
        logger.info(f"✅ Folder {folder_name} completed: {len(complete_records)} complete records")
        
        return len(complete_records)
    
    def run(self, max_records_per_folder=10):
        """Main execution with rule-based approach"""
        logger.info("🚀 Starting Rule-Based RML Assembly...")
        self.print_system_status()
        
        # Find folders with concepts files
        folders_with_concepts = self.find_folders_with_concepts()
        
        if not folders_with_concepts:
            logger.warning("❌ No folders with concepts files found!")
            return
        
        total_assembled = 0
        
        for i, folder_path in enumerate(folders_with_concepts, 1):
            try:
                assembled = self.process_folder(folder_path, max_records_per_folder)
                total_assembled += assembled
                
                logger.info(f"📊 Progress: {i}/{len(folders_with_concepts)} folders processed")
                logger.info(f"📊 Total assembled so far: {total_assembled} records")
                
                # Small pause between folders
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing {folder_path}: {e}")
                continue
        
        logger.info(f"\n🎉 Rule-based RML assembly finished!")
        logger.info(f"📊 Total assembled: {total_assembled} complete RML records")
        logger.info(f"📁 Check data/complete_rml/ for results")
        self.print_system_status()

def main():
    assembler = RuleBasedRMLAssembler()
    assembler.run(max_records_per_folder=5)  # Process 5 records per folder for testing

if __name__ == "__main__":
    main() 