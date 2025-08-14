#!/usr/bin/env python3
"""
Organize RML Components
Organizes 345GB of RML data by component type for modular access
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMLComponentOrganizer:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # RML component types
        self.rml_components = {
            'concepts': 'concepts',
            'triples': 'triples', 
            'entities': 'entities',
            'emotions': 'emotions',
            'reasoning': 'reasoning',
            'intents': 'intents',
            'summaries': 'summaries',
            'events': 'events',
            'vectors': 'vectors'
        }
        
        # Statistics
        self.stats = {
            'total_files_processed': 0,
            'total_samples': 0,
            'components_organized': {comp: 0 for comp in self.rml_components.keys()},
            'total_size_mb': 0,
            'errors': 0
        }
        
        # Create target directory structure
        self.setup_target_structure()
    
    def setup_target_structure(self):
        """Create target directory structure"""
        logger.info("📁 Setting up target directory structure...")
        
        # Create main target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create component directories
        for component_name in self.rml_components.keys():
            component_dir = self.target_dir / component_name
            component_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for organization
            (component_dir / "raw").mkdir(exist_ok=True)
            (component_dir / "processed").mkdir(exist_ok=True)
            (component_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info("✅ Target directory structure created")
    
    def find_rml_files(self) -> List[Path]:
        """Find all RML files in source directory"""
        logger.info("🔍 Finding RML files...")
        
        rml_files = []
        for pattern in ['*.jsonl', '*.json']:
            rml_files.extend(self.source_dir.rglob(pattern))
        
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
    
    def extract_component_data(self, data: Dict) -> Dict[str, List]:
        """Extract component data from RML data"""
        component_data = {comp: [] for comp in self.rml_components.keys()}
        
        # Extract different RML component types
        for component_key, component_name in self.rml_components.items():
            # Check for direct component key
            if component_key in data and data[component_key]:
                if isinstance(data[component_key], list):
                    component_data[component_key].extend(data[component_key])
                else:
                    component_data[component_key].append(data[component_key])
            
            # Check for rml_ prefixed key
            elif f'rml_{component_key}' in data and data[f'rml_{component_key}']:
                if isinstance(data[f'rml_{component_key}'], list):
                    component_data[component_key].extend(data[f'rml_{component_key}'])
                else:
                    component_data[component_key].append(data[f'rml_{component_key}'])
        
        return component_data
    
    def save_component_data(self, component_name: str, data: List, file_index: int):
        """Save component data to organized files"""
        if not data:
            return
        
        component_dir = self.target_dir / component_name
        
        # Save raw data
        raw_file = component_dir / "raw" / f"{component_name}_{file_index:06d}.jsonl"
        with open(raw_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Update statistics
        self.stats['components_organized'][component_name] += len(data)
    
    def process_file(self, file_path: Path, file_index: int):
        """Process a single RML file and organize components"""
        logger.info(f"📄 Processing {file_path.name}")
        
        file_size = file_path.stat().st_size
        self.stats['total_size_mb'] += file_size / (1024 * 1024)
        self.stats['total_files_processed'] += 1
        
        # Initialize component data for this file
        file_components = {comp: [] for comp in self.rml_components.keys()}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Processing {file_path.name}")):
                    if not line.strip():
                        continue
                    
                    # Parse RML data
                    data = self.parse_rml_line(line)
                    if not data:
                        continue
                    
                    # Extract component data
                    component_data = self.extract_component_data(data)
                    
                    # Add to file components
                    for component_name, component_items in component_data.items():
                        file_components[component_name].extend(component_items)
                    
                    self.stats['total_samples'] += 1
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            self.stats['errors'] += 1
        
        # Save component data for this file
        for component_name, component_data in file_components.items():
            self.save_component_data(component_name, component_data, file_index)
    
    def create_component_metadata(self):
        """Create metadata for each component"""
        logger.info("📄 Creating component metadata...")
        
        for component_name in self.rml_components.keys():
            component_dir = self.target_dir / component_name
            metadata_file = component_dir / "metadata" / "component_info.json"
            
            # Count files and samples
            raw_dir = component_dir / "raw"
            total_files = len(list(raw_dir.glob("*.jsonl")))
            total_samples = self.stats['components_organized'][component_name]
            
            metadata = {
                'component_name': component_name,
                'description': f'RML {component_name} component data',
                'total_files': total_files,
                'total_samples': total_samples,
                'organization_date': str(Path().cwd()),
                'source_directory': str(self.source_dir),
                'file_pattern': f'{component_name}_*.jsonl'
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def create_summary_metadata(self):
        """Create overall summary metadata"""
        logger.info("📄 Creating summary metadata...")
        
        summary_file = self.target_dir / "organization_summary.json"
        
        summary = {
            'organization_info': {
                'source_directory': str(self.source_dir),
                'target_directory': str(self.target_dir),
                'organization_date': str(Path().cwd()),
                'total_files_processed': self.stats['total_files_processed'],
                'total_samples': self.stats['total_samples'],
                'total_size_mb': round(self.stats['total_size_mb'], 2),
                'errors': self.stats['errors']
            },
            'component_statistics': self.stats['components_organized'],
            'directory_structure': {
                'components': list(self.rml_components.keys()),
                'subdirectories': ['raw', 'processed', 'metadata']
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Summary metadata saved to {summary_file}")
    
    def organize_components(self):
        """Main method to organize RML components"""
        logger.info("🚀 Starting RML component organization...")
        
        # Find all RML files
        rml_files = self.find_rml_files()
        
        if not rml_files:
            logger.error("❌ No RML files found!")
            return
        
        # Process all files
        for file_index, file_path in enumerate(rml_files):
            self.process_file(file_path, file_index)
        
        # Create metadata
        self.create_component_metadata()
        self.create_summary_metadata()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print organization summary"""
        print("\n" + "="*80)
        print("🎉 RML COMPONENT ORGANIZATION COMPLETE!")
        print("="*80)
        
        print(f"\n📊 ORGANIZATION STATISTICS:")
        print(f"  • Total files processed: {self.stats['total_files_processed']:,}")
        print(f"  • Total samples: {self.stats['total_samples']:,}")
        print(f"  • Total size: {self.stats['total_size_mb']:.2f} MB")
        print(f"  • Errors: {self.stats['errors']}")
        
        print(f"\n🧠 COMPONENT BREAKDOWN:")
        for component_name, count in self.stats['components_organized'].items():
            print(f"  • {component_name}: {count:,} samples")
        
        print(f"\n📁 ORGANIZED STRUCTURE:")
        print(f"  • Target directory: {self.target_dir}")
        for component_name in self.rml_components.keys():
            component_dir = self.target_dir / component_name
            print(f"    - {component_name}/")
            print(f"      ├── raw/ (component data files)")
            print(f"      ├── processed/ (ready for use)")
            print(f"      └── metadata/ (component info)")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Review organization_summary.json for statistics")
        print(f"  2. Use organized components for specific RML modules")
        print(f"  3. Access components by type: {self.target_dir}/concepts/")
        print(f"  4. Build specialized tools for each component type")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Organize RML components by type")
    parser.add_argument("--source", required=True, help="Source directory with RML data")
    parser.add_argument("--target", required=True, help="Target directory for organized components")
    
    args = parser.parse_args()
    
    # Create organizer and run
    organizer = RMLComponentOrganizer(
        source_dir=args.source,
        target_dir=args.target
    )
    
    organizer.organize_components()

if __name__ == "__main__":
    main() 