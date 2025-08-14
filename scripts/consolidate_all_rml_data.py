#!/usr/bin/env python3
"""
Consolidate All RML Data
Consolidates RML extractions from different datasets (Pile, C4, CommonCrawl, RedPajama) into unified format
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMLDataConsolidator:
    def __init__(self, data_dir: str, output_dir: str, train_split: float = 0.8, 
                 val_split: float = 0.1, test_split: float = 0.1):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 0.001:
            raise ValueError("Train/val/test splits must sum to 1.0")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "validation").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        
        # Dataset sources to consolidate
        self.dataset_sources = [
            'pile_rml_final',
            'consolidated_rml', 
            'rml_extracted',
            'c4_backup_20250731_040122',
            'python_c4_backup_20250731_043302',
            'python_c4_final_backup_20250731_043743',
            'commoncrawl',
            'RedPajama-Data',
            'streaming_rml_output',
            'continuous_rml_output',
            'cpp_rml_output',
            'cpp_rml_output_v4',
            'cpp_rml_output_v5',
            'cr_production',
            'cr_simple',
            'cr_simple_fast',
            'cr_stable_fast',
            'extracted RML DATA'
        ]
        
        # RML component types
        self.rml_components = [
            'concepts', 'triples', 'entities', 'emotions', 
            'reasoning', 'intents', 'summaries', 'events', 'vectors'
        ]
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'datasets_processed': {},
            'components_found': set(),
            'total_size_mb': 0,
            'duplicates_removed': 0
        }
        
        # Deduplication set
        self.seen_hashes = set()
    
    def find_rml_files(self) -> List[Tuple[Path, str]]:
        """Find all RML JSONL files across all datasets"""
        logger.info("🔍 Finding RML files across all datasets...")
        
        rml_files = []
        for dataset_name in self.dataset_sources:
            dataset_path = self.data_dir / dataset_name
            if not dataset_path.exists():
                logger.warning(f"⚠️ Dataset not found: {dataset_name}")
                continue
            
            logger.info(f"📁 Scanning {dataset_name}...")
            dataset_files = 0
            
            for pattern in ['*.jsonl', '*.json']:
                for file_path in dataset_path.rglob(pattern):
                    rml_files.append((file_path, dataset_name))
                    dataset_files += 1
            
            logger.info(f"  Found {dataset_files} files in {dataset_name}")
        
        logger.info(f"📁 Total: {len(rml_files)} RML files found")
        return rml_files
    
    def parse_rml_line(self, line: str) -> Dict:
        """Parse a single RML JSONL line"""
        try:
            data = json.loads(line.strip())
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"❌ Invalid JSON line: {e}")
            return None
    
    def extract_rml_components(self, data: Dict) -> Dict:
        """Extract RML components from data"""
        components = {}
        
        # Extract different RML component types
        for component_type in self.rml_components:
            if component_type in data:
                components[component_type] = data[component_type]
            elif f'rml_{component_type}' in data:
                components[component_type] = data[f'rml_{component_type}']
        
        return components
    
    def format_for_training(self, data: Dict, components: Dict, dataset_name: str) -> str:
        """Format RML data for training with dataset source"""
        # Create training text with RML tags
        training_text = f"<DATASET>{dataset_name}</DATASET>\n"
        
        # Add concepts
        if 'concepts' in components and components['concepts']:
            training_text += f"<CONCEPT>{components['concepts']}</CONCEPT>\n"
        
        # Add triples
        if 'triples' in components and components['triples']:
            training_text += f"<TRIPLE>{components['triples']}</TRIPLE>\n"
        
        # Add entities
        if 'entities' in components and components['entities']:
            training_text += f"<ENTITY>{components['entities']}</ENTITY>\n"
        
        # Add emotions
        if 'emotions' in components and components['emotions']:
            training_text += f"<EMOTION>{components['emotions']}</EMOTION>\n"
        
        # Add reasoning
        if 'reasoning' in components and components['reasoning']:
            training_text += f"<REASONING>{components['reasoning']}</REASONING>\n"
        
        # Add intents
        if 'intents' in components and components['intents']:
            training_text += f"<INTENT>{components['intents']}</INTENT>\n"
        
        # Add summaries
        if 'summaries' in components and components['summaries']:
            training_text += f"<SUMMARY>{components['summaries']}</SUMMARY>\n"
        
        # Add events
        if 'events' in components and components['events']:
            training_text += f"<EVENT>{components['events']}</EVENT>\n"
        
        # Add vectors (if present)
        if 'vectors' in components and components['vectors']:
            training_text += f"<VECTOR>{components['vectors']}</VECTOR>\n"
        
        # Add original text if available
        if 'text' in data:
            training_text += f"<TEXT>{data['text']}</TEXT>\n"
        
        return training_text.strip()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate using hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False
    
    def process_file(self, file_path: Path, dataset_name: str) -> List[Dict]:
        """Process a single RML file"""
        logger.info(f"📄 Processing {file_path.name} from {dataset_name}")
        
        samples = []
        file_size = file_path.stat().st_size
        self.stats['total_size_mb'] += file_size / (1024 * 1024)
        
        # Initialize dataset stats
        if dataset_name not in self.stats['datasets_processed']:
            self.stats['datasets_processed'][dataset_name] = {
                'files': 0,
                'samples': 0,
                'size_mb': 0
            }
        
        self.stats['datasets_processed'][dataset_name]['files'] += 1
        self.stats['datasets_processed'][dataset_name]['size_mb'] += file_size / (1024 * 1024)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Processing {file_path.name}")):
                    if not line.strip():
                        continue
                    
                    # Parse RML data
                    data = self.parse_rml_line(line)
                    if not data:
                        continue
                    
                    # Extract RML components
                    components = self.extract_rml_components(data)
                    if not components:
                        continue
                    
                    # Format for training
                    training_text = self.format_for_training(data, components, dataset_name)
                    if not training_text.strip():
                        continue
                    
                    # Check for duplicates
                    if self.is_duplicate(training_text):
                        self.stats['duplicates_removed'] += 1
                        continue
                    
                    # Create training sample
                    sample = {
                        'text': training_text,
                        'rml_components': components,
                        'dataset_source': dataset_name,
                        'source_file': str(file_path),
                        'line_number': line_num + 1
                    }
                    
                    samples.append(sample)
                    self.stats['total_samples'] += 1
                    self.stats['datasets_processed'][dataset_name]['samples'] += 1
                    
                    # Track found components
                    for component_type in components:
                        self.stats['components_found'].add(component_type)
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
        
        return samples
    
    def split_dataset(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split samples into train/val/test"""
        logger.info("📊 Splitting consolidated dataset...")
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Calculate split indices
        total = len(samples)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        self.stats['train_samples'] = len(train_samples)
        self.stats['val_samples'] = len(val_samples)
        self.stats['test_samples'] = len(test_samples)
        
        logger.info(f"📈 Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return train_samples, val_samples, test_samples
    
    def save_split(self, samples: List[Dict], split_name: str):
        """Save samples to JSONL file"""
        output_file = self.output_dir / split_name / f"{split_name}.jsonl"
        
        logger.info(f"💾 Saving {len(samples)} samples to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(samples, desc=f"Saving {split_name}"):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    def create_metadata(self):
        """Create metadata file with dataset statistics"""
        metadata = {
            'dataset_info': {
                'name': 'Consolidated RML Training Dataset',
                'version': '1.0',
                'description': 'Unified RML dataset from multiple sources (Pile, C4, CommonCrawl, RedPajama)',
                'total_datasets': len(self.stats['datasets_processed'])
            },
            'statistics': {
                'total_samples': self.stats['total_samples'],
                'train_samples': self.stats['train_samples'],
                'validation_samples': self.stats['val_samples'],
                'test_samples': self.stats['test_samples'],
                'total_size_mb': round(self.stats['total_size_mb'], 2),
                'duplicates_removed': self.stats['duplicates_removed']
            },
            'splits': {
                'train': self.train_split,
                'validation': self.val_split,
                'test': self.test_split
            },
            'datasets_processed': self.stats['datasets_processed'],
            'rml_components': list(self.stats['components_found']),
            'format': {
                'type': 'jsonl',
                'encoding': 'utf-8',
                'rml_tags': True,
                'dataset_source_tag': True
            }
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 Metadata saved to {metadata_file}")
    
    def consolidate_data(self):
        """Main method to consolidate all RML data"""
        logger.info("🚀 Starting RML data consolidation...")
        
        # Find all RML files
        rml_files = self.find_rml_files()
        
        if not rml_files:
            logger.error("❌ No RML files found!")
            return
        
        # Process all files
        all_samples = []
        for file_path, dataset_name in rml_files:
            samples = self.process_file(file_path, dataset_name)
            all_samples.extend(samples)
        
        if not all_samples:
            logger.error("❌ No valid samples found!")
            return
        
        # Split dataset
        train_samples, val_samples, test_samples = self.split_dataset(all_samples)
        
        # Save splits
        self.save_split(train_samples, "train")
        self.save_split(val_samples, "validation")
        self.save_split(test_samples, "test")
        
        # Create metadata
        self.create_metadata()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print consolidation summary"""
        print("\n" + "="*80)
        print("🎉 RML DATA CONSOLIDATION COMPLETE!")
        print("="*80)
        
        print(f"\n📊 CONSOLIDATION STATISTICS:")
        print(f"  • Total samples: {self.stats['total_samples']:,}")
        print(f"  • Train samples: {self.stats['train_samples']:,}")
        print(f"  • Validation samples: {self.stats['val_samples']:,}")
        print(f"  • Test samples: {self.stats['test_samples']:,}")
        print(f"  • Total size: {self.stats['total_size_mb']:.2f} MB")
        print(f"  • Duplicates removed: {self.stats['duplicates_removed']:,}")
        
        print(f"\n🧠 RML COMPONENTS FOUND:")
        for component in sorted(self.stats['components_found']):
            print(f"  • {component}")
        
        print(f"\n📁 DATASETS PROCESSED:")
        for dataset_name, stats in self.stats['datasets_processed'].items():
            print(f"  • {dataset_name}: {stats['samples']:,} samples, {stats['files']} files, {stats['size_mb']:.2f} MB")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • Train: {self.output_dir}/train/train.jsonl")
        print(f"  • Validation: {self.output_dir}/validation/validation.jsonl")
        print(f"  • Test: {self.output_dir}/test/test.jsonl")
        print(f"  • Metadata: {self.output_dir}/metadata.json")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Review metadata.json for detailed statistics")
        print(f"  2. Use training_config.yaml to configure training")
        print(f"  3. Run tokenizer training on the consolidated data")
        print(f"  4. Start model training with DeepSpeed")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Consolidate RML data from multiple datasets")
    parser.add_argument("--data-dir", required=True, help="Data directory containing all RML extractions")
    parser.add_argument("--output-dir", required=True, help="Output directory for consolidated training data")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    
    args = parser.parse_args()
    
    # Create consolidator and run
    consolidator = RMLDataConsolidator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    consolidator.consolidate_data()

if __name__ == "__main__":
    main() 