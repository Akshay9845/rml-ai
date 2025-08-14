#!/usr/bin/env python3
"""
Archive Old Data
Archives old, duplicate, and experimental data to clean up the data directory
"""

import os
import json
import argparse
import logging
import shutil
import tarfile
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMLDataArchiver:
    def __init__(self, source_dir: str, target_dir: str, compress: bool = True):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.compress = compress
        
        # Directories to archive (old, experimental, duplicate)
        self.directories_to_archive = [
            # Old extraction runs
            'c4_backup_*',
            'python_c4_*',
            'cpp_rml_output_v*',
            'rml_extraction_part2_*',
            'rml_extraction_part2_fixed*',
            
            # Experimental runs
            'bulletproof_*',
            'hyper_*',
            'max_speed_*',
            'ultimate_*',
            'ultra_*',
            'working_*',
            'simple_*',
            'fresh_*',
            'enhanced_*',
            
            # Temporary and backup
            'temp_*',
            'pile_chunks_temp',
            'pile_jsonl_files',
            'pile_zst_files',
            'real_pile_*',
            'real_rml_*',
            'real_redpajama',
            
            # Old processing
            'rml_core',
            'rml_CPP',
            'rml_diagnostics',
            'rml_memory',
            'rml_reasoning',
            'rml_safety',
            'rml_output',
            
            # Build and config
            'build_*',
            'config',
            'downloads',
            'offset_index',
            
            # Old samples
            'book_*',
            'bookcorpus_*',
            'stack_exchange_*',
            'wikimedia_*'
        ]
        
        # Statistics
        self.stats = {
            'directories_archived': 0,
            'files_archived': 0,
            'total_size_mb': 0,
            'space_saved_mb': 0,
            'archives_created': 0,
            'errors': 0
        }
        
        # Create target directory structure
        self.setup_target_structure()
    
    def setup_target_structure(self):
        """Create target directory structure"""
        logger.info("📁 Setting up archive directory structure...")
        
        # Create main target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create archive subdirectories
        (self.target_dir / "extraction_backups").mkdir(exist_ok=True)
        (self.target_dir / "processing_backups").mkdir(exist_ok=True)
        (self.target_dir / "experimental").mkdir(exist_ok=True)
        (self.target_dir / "temporary").mkdir(exist_ok=True)
        (self.target_dir / "old_samples").mkdir(exist_ok=True)
        
        logger.info("✅ Archive directory structure created")
    
    def get_directory_size(self, dir_path: Path) -> float:
        """Get directory size in MB"""
        total_size = 0
        try:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"⚠️ Error calculating size for {dir_path}: {e}")
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def should_archive_directory(self, dir_name: str) -> bool:
        """Check if directory should be archived"""
        for pattern in self.directories_to_archive:
            if pattern.replace('*', '') in dir_name or dir_name.startswith(pattern.replace('*', '')):
                return True
        return False
    
    def find_directories_to_archive(self) -> List[Path]:
        """Find directories that should be archived"""
        logger.info("🔍 Finding directories to archive...")
        
        directories = []
        for item in self.source_dir.iterdir():
            if item.is_dir() and self.should_archive_directory(item.name):
                directories.append(item)
        
        logger.info(f"📁 Found {len(directories)} directories to archive")
        return directories
    
    def create_archive(self, source_path: Path, archive_name: str, archive_type: str):
        """Create compressed archive of directory"""
        archive_dir = self.target_dir / archive_type
        archive_path = archive_dir / f"{archive_name}.tar.gz"
        
        logger.info(f"📦 Creating archive: {archive_path}")
        
        try:
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(source_path, arcname=source_path.name)
            
            self.stats['archives_created'] += 1
            logger.info(f"✅ Archive created: {archive_path}")
            
        except Exception as e:
            logger.error(f"❌ Error creating archive for {source_path}: {e}")
            self.stats['errors'] += 1
    
    def move_to_archive(self, source_path: Path, archive_name: str, archive_type: str):
        """Move directory to archive location"""
        archive_dir = self.target_dir / archive_type
        target_path = archive_dir / archive_name
        
        logger.info(f"📦 Moving to archive: {source_path} -> {target_path}")
        
        try:
            shutil.move(str(source_path), str(target_path))
            self.stats['directories_archived'] += 1
            logger.info(f"✅ Moved to archive: {target_path}")
            
        except Exception as e:
            logger.error(f"❌ Error moving {source_path}: {e}")
            self.stats['errors'] += 1
    
    def determine_archive_type(self, dir_name: str) -> str:
        """Determine archive type based on directory name"""
        if any(pattern in dir_name for pattern in ['backup', 'c4_', 'python_c4']):
            return "extraction_backups"
        elif any(pattern in dir_name for pattern in ['temp_', 'pile_', 'real_']):
            return "temporary"
        elif any(pattern in dir_name for pattern in ['bulletproof', 'hyper', 'max_speed', 'ultimate', 'ultra', 'working', 'simple', 'fresh', 'enhanced']):
            return "experimental"
        elif any(pattern in dir_name for pattern in ['book_', 'bookcorpus_', 'stack_exchange_', 'wikimedia_']):
            return "old_samples"
        else:
            return "processing_backups"
    
    def archive_directories(self):
        """Archive directories"""
        logger.info("🚀 Starting data archiving...")
        
        # Find directories to archive
        directories = self.find_directories_to_archive()
        
        if not directories:
            logger.info("✅ No directories to archive!")
            return
        
        # Process each directory
        for directory in directories:
            dir_name = directory.name
            dir_size = self.get_directory_size(directory)
            archive_type = self.determine_archive_type(dir_name)
            
            logger.info(f"📁 Processing {dir_name} ({dir_size:.2f} MB)")
            
            # Update statistics
            self.stats['total_size_mb'] += dir_size
            self.stats['files_archived'] += len(list(directory.rglob('*')))
            
            # Archive directory
            if self.compress:
                self.create_archive(directory, dir_name, archive_type)
                # Remove original after archiving
                try:
                    shutil.rmtree(directory)
                    self.stats['space_saved_mb'] += dir_size
                except Exception as e:
                    logger.error(f"❌ Error removing {directory}: {e}")
            else:
                self.move_to_archive(directory, dir_name, archive_type)
    
    def create_archive_manifest(self):
        """Create manifest of archived data"""
        logger.info("📄 Creating archive manifest...")
        
        manifest = {
            'archive_info': {
                'archive_date': datetime.now().isoformat(),
                'source_directory': str(self.source_dir),
                'target_directory': str(self.target_dir),
                'compression_used': self.compress
            },
            'statistics': {
                'directories_archived': self.stats['directories_archived'],
                'files_archived': self.stats['files_archived'],
                'total_size_mb': round(self.stats['total_size_mb'], 2),
                'space_saved_mb': round(self.stats['space_saved_mb'], 2),
                'archives_created': self.stats['archives_created'],
                'errors': self.stats['errors']
            },
            'archive_structure': {
                'extraction_backups': 'Old extraction runs and backups',
                'processing_backups': 'Old processing runs and backups',
                'experimental': 'Experimental extraction attempts',
                'temporary': 'Temporary files and chunks',
                'old_samples': 'Old sample datasets'
            }
        }
        
        manifest_file = self.target_dir / "archive_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📄 Archive manifest saved to {manifest_file}")
    
    def archive_data(self):
        """Main method to archive data"""
        logger.info("🚀 Starting RML data archiving...")
        
        # Archive directories
        self.archive_directories()
        
        # Create manifest
        self.create_archive_manifest()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print archiving summary"""
        print("\n" + "="*80)
        print("🎉 RML DATA ARCHIVING COMPLETE!")
        print("="*80)
        
        print(f"\n📊 ARCHIVING STATISTICS:")
        print(f"  • Directories archived: {self.stats['directories_archived']}")
        print(f"  • Files archived: {self.stats['files_archived']:,}")
        print(f"  • Total size archived: {self.stats['total_size_mb']:.2f} MB")
        print(f"  • Space saved: {self.stats['space_saved_mb']:.2f} MB")
        print(f"  • Archives created: {self.stats['archives_created']}")
        print(f"  • Errors: {self.stats['errors']}")
        
        print(f"\n📁 ARCHIVE STRUCTURE:")
        print(f"  • Archive directory: {self.target_dir}")
        print(f"    - extraction_backups/ (Old extraction runs)")
        print(f"    - processing_backups/ (Old processing runs)")
        print(f"    - experimental/ (Experimental attempts)")
        print(f"    - temporary/ (Temporary files)")
        print(f"    - old_samples/ (Old sample datasets)")
        
        print(f"\n🎯 BENEFITS:")
        print(f"  • Cleaner data directory structure")
        print(f"  • Reduced disk space usage")
        print(f"  • Better organization for next phase")
        print(f"  • Preserved historical data")
        
        print(f"\n📄 MANIFEST:")
        print(f"  • Archive manifest: {self.target_dir}/archive_manifest.json")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Archive old RML data")
    parser.add_argument("--source", required=True, help="Source directory (data/)")
    parser.add_argument("--target", required=True, help="Target archive directory")
    parser.add_argument("--no-compress", action="store_true", help="Don't compress archives")
    
    args = parser.parse_args()
    
    # Create archiver and run
    archiver = RMLDataArchiver(
        source_dir=args.source,
        target_dir=args.target,
        compress=not args.no_compress
    )
    
    archiver.archive_data()

if __name__ == "__main__":
    main() 