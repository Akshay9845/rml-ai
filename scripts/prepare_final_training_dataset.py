#!/usr/bin/env python3
"""
Final Training Dataset Preparation
Converts 355GB of RML data into GPT-like training format
"""

import json
import os
import glob
import psutil
import time
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Any, Generator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalTrainingDatasetPreparer:
    def __init__(self, data_dir="data/", output_dir="data/final_training_dataset"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.rml_components = [
            'concepts', 'emotions', 'entities', 'events', 'intents',
            'reasoning', 'summaries', 'tags', 'triples', 'vectors'
        ]
        
        # Priority datasets (largest and highest quality)
        self.priority_datasets = [
            "pile_rml_final",  # 157GB - largest
            "consolidated_rml",  # 27GB
            "rml_extracted",  # 23GB
            "streaming_rml_output",  # 19GB
            "extracted RML DATA",  # 19GB
            "cpp_rml_output_v4",  # 10GB - high quality
            "cpp_rml_output_v5",  # 4.5GB - perfect quality
            "enhanced_rml_output",  # 804MB - perfect quality
            "ultimate_rml_output",  # 301MB - perfect quality
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
    
    def convert_rml_to_gpt_format(self, rml_record: Dict[str, Any]) -> str:
        """Convert RML record to GPT-like training format"""
        try:
            # Extract key components
            concepts = rml_record.get('concepts', [])
            summaries = rml_record.get('summaries', '')
            reasoning = rml_record.get('reasoning', '')
            entities = rml_record.get('entities', {})
            emotions = rml_record.get('emotions', '')
            events = rml_record.get('events', [])
            intents = rml_record.get('intents', '')
            tags = rml_record.get('tags', '')
            triples = rml_record.get('triples', {})
            
            # Create comprehensive text representation
            text_parts = []
            
            # Add summary if available
            if summaries:
                text_parts.append(f"Summary: {summaries}")
            
            # Add concepts
            if concepts and isinstance(concepts, list):
                concepts_text = ', '.join([str(c) for c in concepts[:10]])  # Limit to 10 concepts
                text_parts.append(f"Key Concepts: {concepts_text}")
            
            # Add reasoning
            if reasoning:
                text_parts.append(f"Reasoning: {reasoning}")
            
            # Add entities
            if entities and isinstance(entities, dict):
                entity_text = ', '.join([f"{k}: {v}" for k, v in list(entities.items())[:5]])
                if entity_text:
                    text_parts.append(f"Entities: {entity_text}")
            
            # Add emotions
            if emotions:
                text_parts.append(f"Emotional Context: {emotions}")
            
            # Add events
            if events and isinstance(events, list):
                events_text = ', '.join([str(e) for e in events[:5]])
                if events_text:
                    text_parts.append(f"Events: {events_text}")
            
            # Add intents
            if intents:
                text_parts.append(f"Intent: {intents}")
            
            # Add tags
            if tags:
                text_parts.append(f"Tags: {tags}")
            
            # Add triples
            if triples and isinstance(triples, dict):
                subject = triples.get('subject', [])
                relation = triples.get('relation', '')
                obj = triples.get('object', [])
                
                if subject and relation and obj:
                    subject_text = ', '.join(subject[:3]) if isinstance(subject, list) else str(subject)
                    obj_text = ', '.join(obj[:3]) if isinstance(obj, list) else str(obj)
                    text_parts.append(f"Relationships: {subject_text} {relation} {obj_text}")
            
            # Combine all parts
            if text_parts:
                combined_text = ' | '.join(text_parts)
                return combined_text
            else:
                return "No meaningful content extracted"
                
        except Exception as e:
            logger.error(f"Error converting RML to GPT format: {e}")
            return "Error in data conversion"
    
    def process_rml_file(self, filepath: str, max_records: int = 1000) -> Generator[str, None, None]:
        """Process RML file and yield GPT-formatted text"""
        logger.info(f"📖 Processing: {os.path.basename(filepath)}")
        
        records_processed = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if records_processed >= max_records:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Convert to GPT format
                        gpt_text = self.convert_rml_to_gpt_format(data)
                        
                        if gpt_text and gpt_text != "No meaningful content extracted":
                            records_processed += 1
                            yield gpt_text
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    def process_dataset(self, dataset_name: str, max_files: int = 10) -> int:
        """Process a specific dataset"""
        dataset_path = os.path.join(self.data_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}")
            return 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing Dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        self.print_system_status()
        
        # Find JSONL files
        jsonl_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
        
        if not jsonl_files:
            logger.warning(f"No JSONL files found in {dataset_name}")
            return 0
        
        # Limit files for processing
        jsonl_files = jsonl_files[:max_files]
        
        # Create output file for this dataset
        output_file = os.path.join(self.output_dir, f"{dataset_name}_gpt_format.jsonl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        total_records = 0
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for filepath in jsonl_files:
                try:
                    for gpt_text in self.process_rml_file(filepath, max_records=1000):
                        # Create GPT-like training record
                        training_record = {
                            "text": gpt_text,
                            "source_dataset": dataset_name,
                            "source_file": os.path.basename(filepath),
                            "length": len(gpt_text),
                            "timestamp": time.time()
                        }
                        
                        out_f.write(json.dumps(training_record, ensure_ascii=False) + '\n')
                        total_records += 1
                        
                        # Progress update
                        if total_records % 1000 == 0:
                            logger.info(f"  📊 Processed {total_records} records...")
                            self.print_system_status()
                            
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {e}")
                    continue
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        size_mb = file_size / (1024 * 1024)
        
        logger.info(f"✅ Dataset {dataset_name} completed:")
        logger.info(f"  📁 Output: {output_file}")
        logger.info(f"  📊 Records: {total_records:,}")
        logger.info(f"  💾 Size: {size_mb:.2f} MB")
        
        return total_records
    
    def create_training_splits(self):
        """Create train/validation/test splits"""
        logger.info("\n📊 Creating training splits...")
        
        # Combine all GPT format files
        gpt_files = glob.glob(os.path.join(self.output_dir, "*_gpt_format.jsonl"))
        
        if not gpt_files:
            logger.warning("No GPT format files found!")
            return
        
        all_records = []
        
        for gpt_file in gpt_files:
            logger.info(f"📖 Reading: {os.path.basename(gpt_file)}")
            
            with open(gpt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        all_records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        # Shuffle records
        import random
        random.shuffle(all_records)
        
        # Create splits (80/10/10)
        total_records = len(all_records)
        train_size = int(total_records * 0.8)
        val_size = int(total_records * 0.1)
        
        train_records = all_records[:train_size]
        val_records = all_records[train_size:train_size + val_size]
        test_records = all_records[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': train_records,
            'validation': val_records,
            'test': test_records
        }
        
        for split_name, records in splits.items():
            output_file = os.path.join(self.output_dir, f"{split_name}.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            file_size = os.path.getsize(output_file)
            size_mb = file_size / (1024 * 1024)
            
            logger.info(f"✅ {split_name.capitalize()} split:")
            logger.info(f"  📁 File: {output_file}")
            logger.info(f"  📊 Records: {len(records):,}")
            logger.info(f"  💾 Size: {size_mb:.2f} MB")
    
    def run(self, max_datasets: int = 5):
        """Main execution"""
        logger.info("🚀 Starting Final Training Dataset Preparation...")
        logger.info(f"📁 Output Directory: {self.output_dir}")
        self.print_system_status()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        total_records = 0
        
        # Process priority datasets
        for i, dataset_name in enumerate(self.priority_datasets[:max_datasets], 1):
            try:
                records = self.process_dataset(dataset_name, max_files=5)
                total_records += records
                
                logger.info(f"📊 Progress: {i}/{min(max_datasets, len(self.priority_datasets))} datasets processed")
                logger.info(f"📊 Total records so far: {total_records:,}")
                
                # Small pause between datasets
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing {dataset_name}: {e}")
                continue
        
        # Create training splits
        self.create_training_splits()
        
        logger.info(f"\n🎉 Final Training Dataset Preparation completed!")
        logger.info(f"📊 Total records processed: {total_records:,}")
        logger.info(f"📁 Check {self.output_dir} for results")
        self.print_system_status()

def main():
    """Main function"""
    preparer = FinalTrainingDatasetPreparer()
    preparer.run(max_datasets=3)  # Start with 3 datasets for testing

if __name__ == "__main__":
    main() 