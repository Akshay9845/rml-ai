#!/usr/bin/env python3
"""
Simple RML Data Consolidation
Single-threaded, reliable consolidation for 372GB dataset
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

class SimpleRMLConsolidator:
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
        
        # Priority datasets (largest first)
        self.priority_datasets = [
            'pile_rml_final',           # 157GB - Process first
            'consolidated_rml',         # 27GB
            'rml_extracted',            # 23GB
            'streaming_rml_output',     # 19GB
            'extracted RML DATA',       # 19GB
            'rml_extraction_part2_fixed', # 18GB
            'cpp_rml_output_v4',        # 10GB
            'cr_simple',                # 9.2GB
            'cr_production',            # 5.7GB
            'cpp_rml_output_v5',        # 4.5GB
            'real_redpajama',           # 4.5GB
            'continuous_rml_output',    # 2.4GB
            'commoncrawl',              # 1.1GB
            'c4_backup_20250731_040122', # 7.4MB
            'RedPajama-Data',           # 3.6MB
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
            'duplicates_removed': 0,
            'files_processed': 0
        }
        
        # Deduplication set
        self.seen_hashes = set()
    
    def find_rml_files(self) -> List[Tuple[Path, str]]:
        """Find RML files from priority datasets only"""
        logger.info("🔍 Finding RML files from priority datasets...")
        
        rml_files = []
        for dataset_name in self.priority_datasets:
            dataset_path = self.data_dir / dataset_name
            if not dataset_path.exists():
                logger.warning(f"⚠️ Dataset not found: {dataset_name}")
                continue
            
            logger.info(f"📁 Scanning {dataset_name}...")
            dataset_files = 0
            
            for pattern in ['*.jsonl', '*.json']:
                for file_path in dataset_path.rglob(pattern):
                    # Skip very small files
                    if file_path.stat().st_size < 1024:  # Less than 1KB
                        continue
                    rml_files.append((file_path, dataset_name))
                    dataset_files += 1
            
            logger.info(f"  Found {dataset_files} files in {dataset_name}")
        
        logger.info(f"📁 Total: {len(rml_files)} RML files found")
        return rml_files
    
    def process_file(self, file_path: Path, dataset_name: str) -> List[Dict]:
        """Process a single RML file"""
        samples = []
        
        try:
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
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    # Parse RML data
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    # Extract RML components
                    components = {}
                    for component_type in self.rml_components:
                        if component_type in data:
                            components[component_type] = data[component_type]
                        elif f'rml_{component_type}' in data:
                            components[component_type] = data[f'rml_{component_type}']
                    
                    if not components:
                        continue
                    
                    # Format for training
                    training_text = self.format_for_training(data, components, dataset_name)
                    if not training_text.strip():
                        continue
                    
                    # Check for duplicates using hash
                    text_hash = hashlib.md5(training_text.encode()).hexdigest()
                    if text_hash in self.seen_hashes:
                        self.stats['duplicates_removed'] += 1
                        continue
                    
                    self.seen_hashes.add(text_hash)
                    
                    # Create sample
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
    
    def format_for_training(self, data: Dict, components: Dict, dataset_name: str) -> str:
        """Format RML data for training"""
        training_text = f"<DATASET>{dataset_name}</DATASET>\n"
        
        # Add components in order
        for component_type in self.rml_components:
            if component_type in components and components[component_type]:
                training_text += f"<{component_type.upper()}>{components[component_type]}</{component_type.upper()}>\n"
        
        # Add original text if available
        if 'text' in data:
            training_text += f"<TEXT>{data['text']}</TEXT>\n"
        
        return training_text.strip()
    
    def process_all_files(self, rml_files: List[Tuple[Path, str]]) -> List[Dict]:
        """Process all files sequentially"""
        logger.info(f"🚀 Processing {len(rml_files)} files sequentially...")
        
        all_samples = []
        
        for file_path, dataset_name in tqdm(rml_files, desc="Processing files"):
            samples = self.process_file(file_path, dataset_name)
            all_samples.extend(samples)
            self.stats['files_processed'] += 1
            
            # Progress update every 100 files
            if self.stats['files_processed'] % 100 == 0:
                logger.info(f"📊 Progress: {self.stats['files_processed']}/{len(rml_files)} files, {self.stats['total_samples']:,} samples")
        
        return all_samples
    
    def split_dataset(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split samples into train/val/test"""
        logger.info("📊 Splitting dataset...")
        
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
        """Create metadata file"""
        metadata = {
            'dataset_info': {
                'name': 'Simple Consolidated RML Training Dataset',
                'version': '1.0',
                'description': 'Reliable consolidation of 372GB RML data from multiple datasets',
                'total_datasets': len(self.stats['datasets_processed'])
            },
            'statistics': {
                'total_samples': self.stats['total_samples'],
                'train_samples': self.stats['train_samples'],
                'validation_samples': self.stats['val_samples'],
                'test_samples': self.stats['test_samples'],
                'total_size_mb': round(self.stats['total_size_mb'], 2),
                'duplicates_removed': self.stats['duplicates_removed'],
                'files_processed': self.stats['files_processed']
            },
            'splits': {
                'train': self.train_split,
                'validation': self.val_split,
                'test': self.test_split
            },
            'datasets_processed': self.stats['datasets_processed'],
            'rml_components': list(self.stats['components_found']),
            'optimization': {
                'processing': 'single-threaded',
                'reliability': 'high'
            }
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 Metadata saved to {metadata_file}")
    
    def consolidate_data(self):
        """Main consolidation method"""
        logger.info("🚀 Starting SIMPLE RML data consolidation...")
        
        # Find RML files
        rml_files = self.find_rml_files()
        
        if not rml_files:
            logger.error("❌ No RML files found!")
            return
        
        # Process all files
        all_samples = self.process_all_files(rml_files)
        
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
        print("🎉 SIMPLE RML DATA CONSOLIDATION COMPLETE!")
        print("="*80)
        
        print(f"\n📊 CONSOLIDATION STATISTICS:")
        print(f"  • Total samples: {self.stats['total_samples']:,}")
        print(f"  • Train samples: {self.stats['train_samples']:,}")
        print(f"  • Validation samples: {self.stats['val_samples']:,}")
        print(f"  • Test samples: {self.stats['test_samples']:,}")
        print(f"  • Total size: {self.stats['total_size_mb']:.2f} MB")
        print(f"  • Duplicates removed: {self.stats['duplicates_removed']:,}")
        print(f"  • Files processed: {self.stats['files_processed']}")
        
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
        
        print(f"\n⚡ OPTIMIZATION:")
        print(f"  • Processing: Single-threaded (reliable)")
        print(f"  • Memory: Efficient deduplication")
        print(f"  • Progress: Real-time monitoring")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Simple RML data consolidation")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    
    args = parser.parse_args()
    
    # Create consolidator and run
    consolidator = SimpleRMLConsolidator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    consolidator.consolidate_data()

if __name__ == "__main__":
    main() 