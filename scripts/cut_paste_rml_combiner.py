#!/usr/bin/env python3
"""
Cut-Paste RML Combiner
Processes ALL folders, finds matching IDs, combines all 10 RML components
Uses minimal storage - cut/paste approach
"""

import json
import os
import glob
import shutil
from collections import defaultdict
import time
from datetime import datetime

class CutPasteRMLCombiner:
    def __init__(self, data_dir="data", output_dir="data/combined_rml_final"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.components = ['concepts', 'emotions', 'entities', 'events', 'intents', 
                          'reasoning', 'summaries', 'tags', 'triples', 'vectors']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Statistics
        self.total_folders = 0
        self.total_files = 0
        self.total_records = 0
        self.combined_records = 0
        
    def find_rml_folders(self):
        """Find all folders that contain RML data"""
        print("🔍 Scanning for RML folders...")
        
        rml_folders = []
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                # Check if folder contains RML component files
                has_rml = False
                for component in self.components:
                    pattern = os.path.join(item_path, f"*{component}*.jsonl")
                    if glob.glob(pattern):
                        has_rml = True
                        break
                
                if has_rml:
                    rml_folders.append(item_path)
                    print(f"📁 Found RML folder: {item}")
        
        print(f"📊 Found {len(rml_folders)} RML folders")
        return rml_folders
    
    def get_system_stats(self):
        """Get current system stats"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            memory = psutil.virtual_memory()
            return {
                'disk_free_gb': disk.free / (1024**3),
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
        except:
            return {'disk_free_gb': 0, 'memory_percent': 0, 'memory_available_gb': 0}
    
    def print_system_status(self):
        """Print current system status"""
        stats = self.get_system_stats()
        print(f"💾 Disk: {stats['disk_free_gb']:.1f}GB free | "
              f"RAM: {stats['memory_percent']:.1f}% used ({stats['memory_available_gb']:.1f}GB free)")
    
    def find_matching_ids(self, folders):
        """Find all unique IDs across all folders"""
        print("🔍 Finding matching IDs across all folders...")
        
        all_ids = set()
        id_sources = defaultdict(list)  # Track which folders contain each ID
        
        for folder in folders:
            folder_name = os.path.basename(folder)
            print(f"📁 Scanning {folder_name}...")
            
            for component in self.components:
                pattern = os.path.join(folder, f"*{component}*.jsonl")
                files = glob.glob(pattern)
                
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        data = json.loads(line.strip())
                                        # Try different ID fields
                                        record_id = (data.get('record_id') or 
                                                   data.get('doc_id') or 
                                                   data.get('id') or 
                                                   data.get('chunk'))
                                        
                                        if record_id is not None:
                                            all_ids.add(record_id)
                                            id_sources[record_id].append({
                                                'folder': folder,
                                                'component': component,
                                                'file': file_path
                                            })
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        print(f"⚠️ Error reading {file_path}: {e}")
                        continue
        
        print(f"📊 Found {len(all_ids)} unique IDs across all folders")
        
        # Find IDs that appear in multiple components
        multi_component_ids = {}
        for record_id, sources in id_sources.items():
            components_found = set()
            for source in sources:
                components_found.add(source['component'])
            
            if len(components_found) >= 3:  # At least 3 components
                multi_component_ids[record_id] = {
                    'components': components_found,
                    'sources': sources
                }
        
        print(f"📊 Found {len(multi_component_ids)} IDs with 3+ components")
        return multi_component_ids
    
    def combine_and_cut_paste(self, matching_ids):
        """Combine records and cut/paste to save space"""
        print("✂️ Combining records and cutting/pasting...")
        
        output_file = os.path.join(self.output_dir, "combined_rml_records.jsonl")
        processed_files = set()
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, (record_id, info) in enumerate(matching_ids.items()):
                if i % 1000 == 0:
                    self.print_system_status()
                    print(f"📊 Progress: {i}/{len(matching_ids)} records processed")
                
                # Build complete record
                complete_record = {
                    'record_id': record_id,
                    'concepts': [],
                    'emotions': [],
                    'entities': [],
                    'events': [],
                    'intents': [],
                    'reasoning': [],
                    'summaries': [],
                    'tags': [],
                    'triples': [],
                    'vectors': []
                }
                
                # Collect data from all sources
                for source in info['sources']:
                    component = source['component']
                    file_path = source['file']
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        data = json.loads(line.strip())
                                        current_id = (data.get('record_id') or 
                                                     data.get('doc_id') or 
                                                     data.get('id') or 
                                                     data.get('chunk'))
                                        
                                        if current_id == record_id:
                                            complete_record[component] = data.get('data', [])
                                            break
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        print(f"⚠️ Error reading {file_path}: {e}")
                        continue
                
                # Write complete record
                out_f.write(json.dumps(complete_record, ensure_ascii=False) + '\n')
                self.combined_records += 1
                
                # Mark files for deletion (cut/paste)
                for source in info['sources']:
                    processed_files.add(source['file'])
        
        print(f"✅ Combined {self.combined_records} complete records")
        print(f"📁 Output: {output_file}")
        
        # Now cut/paste - delete processed files to save space
        print("🗑️ Cutting processed files to save space...")
        deleted_files = 0
        for file_path in processed_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files += 1
            except Exception as e:
                print(f"⚠️ Could not delete {file_path}: {e}")
        
        print(f"🗑️ Deleted {deleted_files} processed files")
        self.print_system_status()
    
    def run(self):
        """Main execution"""
        print("🚀 Cut-Paste RML Combiner")
        print("="*60)
        print("📁 Data directory:", self.data_dir)
        print("📁 Output directory:", self.output_dir)
        print()
        
        start_time = time.time()
        self.print_system_status()
        
        # Step 1: Find all RML folders
        rml_folders = self.find_rml_folders()
        if not rml_folders:
            print("❌ No RML folders found!")
            return
        
        # Step 2: Find matching IDs
        matching_ids = self.find_matching_ids(rml_folders)
        if not matching_ids:
            print("❌ No matching IDs found!")
            return
        
        # Step 3: Combine and cut/paste
        self.combine_and_cut_paste(matching_ids)
        
        # Final stats
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n🎉 CUT-PASTE COMPLETED!")
        print("="*60)
        print(f"⏱️ Duration: {duration:.1f} seconds")
        print(f"📊 Combined records: {self.combined_records}")
        print(f"📁 Output file: {self.output_dir}/combined_rml_records.jsonl")
        self.print_system_status()

def main():
    """Main function"""
    combiner = CutPasteRMLCombiner()
    combiner.run()

if __name__ == "__main__":
    main() 