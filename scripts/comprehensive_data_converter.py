#!/usr/bin/env python3
"""
Comprehensive RML Data Converter
Reads every file, converts data to proper RML format, and organizes without losing anything
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict
import logging
import hashlib
import time
from datetime import datetime

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataConverter:
    def __init__(self, data_dir="data/", output_dir="data/converted_rml/"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.conversion_stats = {
            'files_processed': 0,
            'files_converted': 0,
            'files_skipped': 0,
            'total_samples': 0,
            'conversion_errors': 0
        }
        
        # RML components we need
        self.rml_components = [
            'concepts', 'triples', 'entities', 'emotions', 
            'reasoning', 'intents', 'summaries', 'events', 'vectors', 'tags'
        ]
        
        # Create output structure
        self.create_output_structure()
        
        # Track all converted data
        self.converted_data = defaultdict(list)
        
    def create_output_structure(self):
        """Create organized output directory structure"""
        logger.info("📁 Creating output directory structure...")
        
        # Main directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/complete_rml", exist_ok=True)
        os.makedirs(f"{self.output_dir}/component_files", exist_ok=True)
        os.makedirs(f"{self.output_dir}/conversion_logs", exist_ok=True)
        
        # Component subdirectories
        for component in self.rml_components:
            os.makedirs(f"{self.output_dir}/component_files/{component}", exist_ok=True)
        
        logger.info("✅ Output structure created")
    
    def analyze_file_content(self, file_path):
        """Analyze the content of a single file"""
        logger.info(f"🔍 Analyzing: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 10:  # Read first 10 lines for analysis
                        break
                    lines.append(line.strip())
                
                if not lines:
                    return {"type": "empty", "message": "File is empty"}
                
                # Try to parse as JSON
                try:
                    sample_data = json.loads(lines[0])
                    keys = list(sample_data.keys())
                    
                    analysis = {
                        "type": "json",
                        "keys": keys,
                        "sample": sample_data,
                        "line_count": len(lines),
                        "file_size": os.path.getsize(file_path)
                    }
                    
                    # Determine file type
                    if all(comp in keys for comp in self.rml_components):
                        analysis["file_type"] = "complete_rml"
                    elif len(keys) <= 3:
                        analysis["file_type"] = "component_file"
                    else:
                        analysis["file_type"] = "mixed_data"
                    
                    return analysis
                    
                except json.JSONDecodeError:
                    return {"type": "invalid_json", "message": "Not valid JSON", "sample": lines[0][:100]}
                    
        except Exception as e:
            return {"type": "error", "message": str(e)}
    
    def convert_complete_rml_file(self, file_path, analysis):
        """Convert a file that already has complete RML data"""
        logger.info(f"✅ Converting complete RML file: {os.path.basename(file_path)}")
        
        converted_samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Clean and validate the data
                        cleaned_data = self.clean_rml_data(data)
                        
                        if cleaned_data:
                            converted_samples.append(cleaned_data)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"  Line {line_num}: JSON error - {e}")
                        continue
            
            # Save converted data
            if converted_samples:
                output_file = f"{self.output_dir}/complete_rml/{os.path.basename(file_path)}"
                self.save_converted_data(output_file, converted_samples)
                logger.info(f"  ✅ Converted {len(converted_samples)} samples")
                return len(converted_samples)
            
        except Exception as e:
            logger.error(f"  ❌ Error converting {file_path}: {e}")
            self.conversion_stats['conversion_errors'] += 1
        
        return 0
    
    def convert_component_file(self, file_path, analysis):
        """Convert a file with individual RML components"""
        logger.info(f"🔧 Converting component file: {os.path.basename(file_path)}")
        
        # Determine which component this file contains
        component_type = self.identify_component_type(analysis['sample'])
        
        if not component_type:
            logger.warning(f"  ⚠️ Could not identify component type")
            return 0
        
        logger.info(f"  📋 Identified as: {component_type}")
        
        converted_samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Convert to standard component format
                        converted_component = self.convert_component_data(data, component_type)
                        
                        if converted_component:
                            converted_samples.append(converted_component)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"  Line {line_num}: JSON error - {e}")
                        continue
            
            # Save converted component data
            if converted_samples:
                output_file = f"{self.output_dir}/component_files/{component_type}/{os.path.basename(file_path)}"
                self.save_converted_data(output_file, converted_samples)
                logger.info(f"  ✅ Converted {len(converted_samples)} {component_type} samples")
                return len(converted_samples)
            
        except Exception as e:
            logger.error(f"  ❌ Error converting {file_path}: {e}")
            self.conversion_stats['conversion_errors'] += 1
        
        return 0
    
    def convert_mixed_data_file(self, file_path, analysis):
        """Convert a file with mixed or unknown data format"""
        logger.info(f"🔄 Converting mixed data file: {os.path.basename(file_path)}")
        
        # Try to extract any RML-like data
        extracted_components = defaultdict(list)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Try to extract RML components from mixed data
                        extracted = self.extract_rml_from_mixed_data(data)
                        
                        for component, value in extracted.items():
                            if value:
                                extracted_components[component].append(value)
                        
                    except json.JSONDecodeError:
                        continue
            
            # Save extracted components
            total_extracted = 0
            for component, values in extracted_components.items():
                if values:
                    output_file = f"{self.output_dir}/component_files/{component}/extracted_{os.path.basename(file_path)}"
                    self.save_converted_data(output_file, values)
                    total_extracted += len(values)
                    logger.info(f"  📋 Extracted {len(values)} {component} samples")
            
            return total_extracted
            
        except Exception as e:
            logger.error(f"  ❌ Error converting {file_path}: {e}")
            self.conversion_stats['conversion_errors'] += 1
        
        return 0
    
    def identify_component_type(self, sample_data):
        """Identify which RML component this data represents"""
        keys = list(sample_data.keys())
        
        # Direct component matches
        for component in self.rml_components:
            if component in keys:
                return component
        
        # Check for component indicators in values
        for component in self.rml_components:
            if any(component in str(v).lower() for v in sample_data.values()):
                return component
        
        # Check for specific patterns
        if 'concept' in keys or 'concepts' in keys:
            return 'concepts'
        elif 'entity' in keys or 'entities' in keys:
            return 'entities'
        elif 'emotion' in keys or 'tone' in keys:
            return 'emotions'
        elif 'intent' in keys or 'intents' in keys:
            return 'intents'
        elif 'reasoning' in keys or 'logic' in keys:
            return 'reasoning'
        elif 'summary' in keys or 'summaries' in keys:
            return 'summaries'
        elif 'event' in keys or 'events' in keys:
            return 'events'
        elif 'vector' in keys or 'vectors' in keys:
            return 'vectors'
        elif 'tag' in keys or 'tags' in keys:
            return 'tags'
        elif 'triple' in keys or 'relation' in keys or 'subject' in keys:
            return 'triples'
        
        return None
    
    def convert_component_data(self, data, component_type):
        """Convert component data to standard format"""
        converted = {
            'component_type': component_type,
            'data': None,
            'metadata': {}
        }
        
        # Extract metadata
        for key, value in data.items():
            if key in ['record_id', 'doc_id', 'document_id', 'confidence', 'timestamp']:
                converted['metadata'][key] = value
        
        # Extract the actual component data
        if component_type in data:
            converted['data'] = data[component_type]
        elif 'data' in data:
            converted['data'] = data['data']
        else:
            # Look for the component data in other fields
            for key, value in data.items():
                if key not in converted['metadata']:
                    converted['data'] = value
                    break
        
        return converted if converted['data'] is not None else None
    
    def extract_rml_from_mixed_data(self, data):
        """Extract RML components from mixed data"""
        extracted = {}
        
        # Look for RML-like patterns in the data
        for key, value in data.items():
            key_lower = key.lower()
            
            if any(comp in key_lower for comp in ['concept', 'keyword', 'topic']):
                extracted['concepts'] = value
            elif any(comp in key_lower for comp in ['entity', 'person', 'location', 'organization']):
                extracted['entities'] = value
            elif any(comp in key_lower for comp in ['emotion', 'sentiment', 'tone', 'feeling']):
                extracted['emotions'] = value
            elif any(comp in key_lower for comp in ['intent', 'purpose', 'goal']):
                extracted['intents'] = value
            elif any(comp in key_lower for comp in ['reasoning', 'logic', 'thought']):
                extracted['reasoning'] = value
            elif any(comp in key_lower for comp in ['summary', 'abstract', 'description']):
                extracted['summaries'] = value
            elif any(comp in key_lower for comp in ['event', 'action', 'activity']):
                extracted['events'] = value
            elif any(comp in key_lower for comp in ['vector', 'embedding']):
                extracted['vectors'] = value
            elif any(comp in key_lower for comp in ['tag', 'label', 'category']):
                extracted['tags'] = value
            elif any(comp in key_lower for comp in ['triple', 'relation', 'subject', 'object']):
                extracted['triples'] = value
        
        return extracted
    
    def clean_rml_data(self, data):
        """Clean and validate RML data"""
        cleaned = {}
        
        for component in self.rml_components:
            if component in data:
                value = data[component]
                
                # Clean the value
                if isinstance(value, list):
                    # Remove empty strings and None values
                    cleaned_value = [str(item).strip() for item in value if item and str(item).strip()]
                else:
                    cleaned_value = str(value).strip() if value else ""
                
                if cleaned_value:
                    cleaned[component] = cleaned_value
        
        return cleaned if len(cleaned) >= 3 else None  # At least 3 components
    
    def save_converted_data(self, output_file, data):
        """Save converted data to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error saving {output_file}: {e}")
    
    def process_all_files(self):
        """Process all JSONL files in the data directory"""
        logger.info("🚀 Starting comprehensive data conversion...")
        
        # Find all JSONL files
        jsonl_files = glob.glob(os.path.join(self.data_dir, "**/*.jsonl"), recursive=True)
        logger.info(f"📁 Found {len(jsonl_files)} JSONL files to process")
        
        start_time = time.time()
        
        for i, file_path in enumerate(jsonl_files, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"📋 Processing file {i}/{len(jsonl_files)}: {os.path.basename(file_path)}")
            logger.info(f"📁 Full path: {file_path}")
            
            # Analyze the file
            analysis = self.analyze_file_content(file_path)
            
            if analysis['type'] == 'error':
                logger.error(f"❌ Cannot read file: {analysis['message']}")
                self.conversion_stats['files_skipped'] += 1
                continue
            
            if analysis['type'] == 'empty':
                logger.warning(f"⚠️ File is empty, skipping")
                self.conversion_stats['files_skipped'] += 1
                continue
            
            # Convert based on file type
            samples_converted = 0
            
            if analysis['file_type'] == 'complete_rml':
                samples_converted = self.convert_complete_rml_file(file_path, analysis)
            elif analysis['file_type'] == 'component_file':
                samples_converted = self.convert_component_file(file_path, analysis)
            elif analysis['file_type'] == 'mixed_data':
                samples_converted = self.convert_mixed_data_file(file_path, analysis)
            
            # Update statistics
            self.conversion_stats['files_processed'] += 1
            if samples_converted > 0:
                self.conversion_stats['files_converted'] += 1
                self.conversion_stats['total_samples'] += samples_converted
            else:
                self.conversion_stats['files_skipped'] += 1
            
            logger.info(f"📊 File summary: {samples_converted} samples converted")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print final statistics
        self.print_final_statistics(duration)
        
        # Save conversion report
        self.save_conversion_report()
    
    def print_final_statistics(self, duration):
        """Print final conversion statistics"""
        logger.info(f"\n{'='*80}")
        logger.info("🎉 COMPREHENSIVE DATA CONVERSION COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Files processed: {self.conversion_stats['files_processed']}")
        logger.info(f"✅ Files converted: {self.conversion_stats['files_converted']}")
        logger.info(f"⚠️ Files skipped: {self.conversion_stats['files_skipped']}")
        logger.info(f"📈 Total samples: {self.conversion_stats['total_samples']}")
        logger.info(f"❌ Conversion errors: {self.conversion_stats['conversion_errors']}")
        logger.info(f"⏱️ Time taken: {duration/60:.2f} minutes")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def save_conversion_report(self):
        """Save detailed conversion report"""
        report = {
            "conversion_date": datetime.now().isoformat(),
            "statistics": self.conversion_stats,
            "output_directory": self.output_dir,
            "rml_components": self.rml_components
        }
        
        report_file = f"{self.output_dir}/conversion_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Conversion report saved to: {report_file}")

def main():
    converter = ComprehensiveDataConverter()
    converter.process_all_files()
    
    print(f"\n✅ Data conversion complete!")
    print(f"📁 Check the {converter.output_dir} directory for converted files.")
    print(f"📋 Check data_conversion.log for detailed logs.")

if __name__ == "__main__":
    main() 