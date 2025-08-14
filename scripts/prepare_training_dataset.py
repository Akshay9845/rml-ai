#!/usr/bin/env python3
"""
Prepare Training Dataset
Transforms 345GB of RML data into training-ready format for model training
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

class RMLTrainingDatasetPreparer:
    def __init__(self, input_dir: str, output_dir: str, train_split: float = 0.8, 
                 val_split: float = 0.1, test_split: float = 0.1):
        self.input_dir = Path(input_dir)
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
            'components_found': set(),
            'total_size_mb': 0,
            'duplicates_removed': 0
        }
        
        # Deduplication set
        self.seen_hashes = set()
    
    def find_rml_files(self) -> List[Path]:
        """Find all RML JSONL files in input directory"""
        logger.info("🔍 Finding RML files...")
        
        rml_files = []
        for pattern in ['*.jsonl', '*.json']:
            rml_files.extend(self.input_dir.rglob(pattern))
        
        logger.info(f"📁 Found {len(rml_files)} RML files")
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
    
    def format_for_training(self, data: Dict, components: Dict) -> str:
        """Format RML data for training"""
        # Create training text with RML tags
        training_text = ""
        
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
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a single RML file"""
        logger.info(f"📄 Processing {file_path.name}")
        
        samples = []
        file_size = file_path.stat().st_size
        self.stats['total_size_mb'] += file_size / (1024 * 1024)
        
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
                    training_text = self.format_for_training(data, components)
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
                        'source_file': str(file_path),
                        'line_number': line_num + 1
                    }
                    
                    samples.append(sample)
                    self.stats['total_samples'] += 1
                    
                    # Track found components
                    for component_type in components:
                        self.stats['components_found'].add(component_type)
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
        
        return samples
    
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
        """Create metadata file with dataset statistics"""
        metadata = {
            'dataset_info': {
                'name': 'RML Training Dataset',
                'version': '1.0',
                'description': 'Training-ready RML dataset prepared from 345GB extracted data'
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
            'rml_components': list(self.stats['components_found']),
            'format': {
                'type': 'jsonl',
                'encoding': 'utf-8',
                'rml_tags': True
            }
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 Metadata saved to {metadata_file}")
    
    def prepare_dataset(self):
        """Main method to prepare the training dataset"""
        logger.info("🚀 Starting RML training dataset preparation...")
        
        # Find all RML files
        rml_files = self.find_rml_files()
        
        if not rml_files:
            logger.error("❌ No RML files found!")
            return
        
        # Process all files
        all_samples = []
        for file_path in rml_files:
            samples = self.process_file(file_path)
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
        """Print dataset preparation summary"""
        print("\n" + "="*80)
        print("🎉 RML TRAINING DATASET PREPARATION COMPLETE!")
        print("="*80)
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"  • Total samples: {self.stats['total_samples']:,}")
        print(f"  • Train samples: {self.stats['train_samples']:,}")
        print(f"  • Validation samples: {self.stats['val_samples']:,}")
        print(f"  • Test samples: {self.stats['test_samples']:,}")
        print(f"  • Total size: {self.stats['total_size_mb']:.2f} MB")
        print(f"  • Duplicates removed: {self.stats['duplicates_removed']:,}")
        
        print(f"\n🧠 RML COMPONENTS FOUND:")
        for component in sorted(self.stats['components_found']):
            print(f"  • {component}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • Train: {self.output_dir}/train/train.jsonl")
        print(f"  • Validation: {self.output_dir}/validation/validation.jsonl")
        print(f"  • Test: {self.output_dir}/test/test.jsonl")
        print(f"  • Metadata: {self.output_dir}/metadata.json")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Review metadata.json for dataset statistics")
        print(f"  2. Use training_config.yaml to configure training")
        print(f"  3. Run tokenizer training on the prepared data")
        print(f"  4. Start model training with DeepSpeed")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Prepare RML training dataset")
    parser.add_argument("--input-dir", required=True, help="Input directory with RML data")
    parser.add_argument("--output-dir", required=True, help="Output directory for training data")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    
    args = parser.parse_args()
    
    # Create preparer and run
    preparer = RMLTrainingDatasetPreparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    preparer.prepare_dataset()

if __name__ == "__main__":
    main() 