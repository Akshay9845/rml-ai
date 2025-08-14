#!/usr/bin/env python3
"""
RML Completeness Checker
Reads one line from concepts, checks all 10 RML components, generates missing ones
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

class RMLCompletenessChecker:
    def __init__(self, data_dir="data/pile_rml_final"):
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
    
    def find_component_files(self):
        """Find all component files in the directory"""
        logger.info(f"🔍 Finding component files in: {self.data_dir}")
        
        component_files = {}
        for component in self.rml_components:
            pattern = f"*{component}*.jsonl"
            files = glob.glob(os.path.join(self.data_dir, pattern))
            if files:
                component_files[component] = files[0]  # Take first file
                logger.info(f"  📁 {component}: {os.path.basename(files[0])}")
        
        return component_files
    
    def read_concepts_line(self, concepts_file, line_number=1):
        """Read a specific line from concepts file"""
        logger.info(f"📖 Reading line {line_number} from concepts file...")
        
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if i == line_number:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            logger.info(f"  ✅ Read concepts data: {data}")
                            return data
                        else:
                            logger.warning(f"  ⚠️ Line {line_number} is empty")
                            return None
        except Exception as e:
            logger.error(f"  ❌ Error reading concepts file: {e}")
            return None
    
    def find_record_in_component(self, component_file, record_id, chunk=1):
        """Find a specific record in a component file"""
        try:
            with open(component_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    if data.get('record_id') == record_id and data.get('chunk') == chunk:
                        return data
        except Exception as e:
            logger.error(f"  ❌ Error reading {component_file}: {e}")
        
        return None
    
    def check_record_completeness(self, record_id, chunk=1):
        """Check if a record has all 10 RML components"""
        logger.info(f"🔍 Checking completeness for record_id: {record_id}, chunk: {chunk}")
        
        component_files = self.find_component_files()
        found_components = {}
        missing_components = []
        
        # Check each component
        for component in self.rml_components:
            if component in component_files:
                data = self.find_record_in_component(component_files[component], record_id, chunk)
                if data:
                    found_components[component] = data
                    logger.info(f"  ✅ {component}: Found")
                else:
                    missing_components.append(component)
                    logger.warning(f"  ❌ {component}: Missing")
            else:
                missing_components.append(component)
                logger.warning(f"  ❌ {component}: File not found")
        
        logger.info(f"📊 Completeness: {len(found_components)}/10 components found")
        logger.info(f"📋 Missing: {missing_components}")
        
        return found_components, missing_components
    
    def generate_missing_component(self, component, record_id, chunk, concepts_data):
        """Generate a missing RML component"""
        logger.info(f"🔧 Generating missing {component} for record_id: {record_id}")
        
        generated_data = {
            'record_id': record_id,
            'chunk': chunk
        }
        
        # Generate based on component type
        if component == 'emotions':
            # Analyze concepts for emotion
            concepts = concepts_data.get('concepts', [])
            if concepts:
                # Simple emotion detection based on concepts
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disaster']
                
                if any(word in ' '.join(concepts).lower() for word in positive_words):
                    generated_data['tone'] = 'positive'
                elif any(word in ' '.join(concepts).lower() for word in negative_words):
                    generated_data['tone'] = 'negative'
                else:
                    generated_data['tone'] = 'neutral'
            else:
                generated_data['tone'] = 'neutral'
                
        elif component == 'entities':
            # Check if concepts contain numbers or named entities
            concepts = concepts_data.get('concepts', [])
            has_numbers = any(any(char.isdigit() for char in str(concept)) for concept in concepts)
            generated_data['has_numbers'] = has_numbers
            
        elif component == 'events':
            # Check if concepts suggest events/actions
            concepts = concepts_data.get('concepts', [])
            action_words = ['run', 'jump', 'play', 'work', 'study', 'eat', 'sleep']
            has_events = any(word in ' '.join(concepts).lower() for word in action_words)
            generated_data['has_events'] = has_events
            
        elif component == 'intents':
            # Determine intent based on concepts
            concepts = concepts_data.get('concepts', [])
            if concepts:
                generated_data['is_informative'] = True
            else:
                generated_data['is_informative'] = False
                
        elif component == 'reasoning':
            # Check for logical structures
            concepts = concepts_data.get('concepts', [])
            logic_words = ['because', 'therefore', 'since', 'thus', 'hence']
            has_logic = any(word in ' '.join(concepts).lower() for word in logic_words)
            generated_data['has_logic'] = has_logic
            
        elif component == 'summaries':
            # Create summary from concepts
            concepts = concepts_data.get('concepts', [])
            if concepts:
                summary = f"Content contains concepts: {', '.join(concepts[:5])}"
                if len(concepts) > 5:
                    summary += f" and {len(concepts) - 5} more"
            else:
                summary = "No concepts available"
            generated_data['summary'] = summary
            
        elif component == 'tags':
            # Generate category tag
            generated_data['category'] = 'pile_data'
            
        elif component == 'triples':
            # Generate basic triples from concepts
            concepts = concepts_data.get('concepts', [])
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
            concepts = concepts_data.get('concepts', [])
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
    
    def save_complete_record(self, complete_record, output_file="data/complete_rml_sample.jsonl"):
        """Save the complete RML record"""
        logger.info(f"💾 Saving complete record to: {output_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(complete_record, ensure_ascii=False, indent=2) + '\n')
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        size_kb = file_size / 1024
        logger.info(f"📁 File size: {size_kb:.2f} KB")
    
    def run_single_record_check(self, line_number=1):
        """Run completeness check for a single record"""
        logger.info("🚀 Starting RML Completeness Check...")
        self.print_system_status()
        
        # Find component files
        component_files = self.find_component_files()
        
        if 'concepts' not in component_files:
            logger.error("❌ Concepts file not found!")
            return
        
        # Read concepts line
        concepts_data = self.read_concepts_line(component_files['concepts'], line_number)
        
        if not concepts_data:
            logger.error("❌ Could not read concepts data!")
            return
        
        record_id = concepts_data.get('record_id')
        chunk = concepts_data.get('chunk', 1)
        
        if record_id is None:
            logger.error("❌ No record_id found in concepts data!")
            return
        
        # Check completeness
        found_components, missing_components = self.check_record_completeness(record_id, chunk)
        
        # Create complete record
        complete_record = self.create_complete_rml_record(
            record_id, chunk, found_components, missing_components, concepts_data
        )
        
        # Save complete record
        self.save_complete_record(complete_record)
        
        # Print summary
        logger.info(f"\n🎉 COMPLETENESS CHECK SUMMARY:")
        logger.info(f"📊 Record ID: {record_id}")
        logger.info(f"📊 Chunk: {chunk}")
        logger.info(f"📊 Found components: {len(found_components)}/10")
        logger.info(f"📊 Missing components: {len(missing_components)}")
        logger.info(f"📊 Generated components: {missing_components}")
        logger.info(f"📁 Complete record saved to: data/complete_rml_sample.jsonl")
        
        return complete_record

def main():
    checker = RMLCompletenessChecker()
    checker.run_single_record_check(line_number=1)

if __name__ == "__main__":
    main() 