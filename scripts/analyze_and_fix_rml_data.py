#!/usr/bin/env python3
"""
RML Data Analysis and Fixing Script
Analyzes all RML data formats and prepares them for consolidation
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMLDataAnalyzer:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.analysis_results = {}
        self.data_formats = {}
        self.quality_issues = []
        
    def analyze_file_format(self, file_path):
        """Analyze the format of a single RML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to understand format
                lines = []
                for i, line in enumerate(f):
                    if i >= 5:  # Read first 5 lines
                        break
                    lines.append(line.strip())
                
                if not lines:
                    return {"error": "Empty file"}
                
                # Try to parse as JSON
                try:
                    sample_data = json.loads(lines[0])
                    return {
                        "format": "json",
                        "keys": list(sample_data.keys()),
                        "sample": sample_data,
                        "line_count": len(lines)
                    }
                except json.JSONDecodeError:
                    return {"error": "Not valid JSON"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_directory(self, dir_path):
        """Analyze all files in a directory"""
        logger.info(f"📁 Analyzing directory: {dir_path}")
        
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
            return
        
        # Find all JSONL files
        jsonl_files = glob.glob(os.path.join(dir_path, "**/*.jsonl"), recursive=True)
        
        if not jsonl_files:
            logger.info(f"No JSONL files found in {dir_path}")
            return
        
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        # Analyze first few files to understand format
        analyzed_files = 0
        format_summary = defaultdict(int)
        
        for file_path in jsonl_files[:10]:  # Analyze first 10 files
            try:
                format_info = self.analyze_file_format(file_path)
                if "error" not in format_info:
                    format_summary[tuple(sorted(format_info["keys"]))] += 1
                    analyzed_files += 1
                    
                    # Store sample data
                    key_tuple = tuple(sorted(format_info["keys"]))
                    if key_tuple not in self.data_formats:
                        self.data_formats[key_tuple] = {
                            "sample": format_info["sample"],
                            "files": [],
                            "count": 0
                        }
                    self.data_formats[key_tuple]["files"].append(file_path)
                    self.data_formats[key_tuple]["count"] += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        logger.info(f"Analyzed {analyzed_files} files")
        logger.info(f"Found {len(format_summary)} different formats")
        
        return format_summary
    
    def analyze_all_directories(self):
        """Analyze all RML data directories"""
        logger.info("🔍 Starting comprehensive RML data analysis...")
        
        # List of directories to analyze
        directories = [
            "pile_rml_final",
            "consolidated_rml", 
            "rml_extracted",
            "streaming_rml_output",
            "extracted RML DATA",
            "rml_extraction_part2_fixed",
            "cpp_rml_output_v4",
            "cr_simple",
            "cr_production",
            "cpp_rml_output_v5",
            "real_redpajama",
            "continuous_rml_output",
            "commoncrawl",
            "ultra_rml_output"
        ]
        
        total_files = 0
        total_formats = set()
        
        for dir_name in directories:
            dir_path = os.path.join(self.data_dir, dir_name)
            if os.path.exists(dir_path):
                format_summary = self.analyze_directory(dir_path)
                if format_summary:
                    total_formats.update(format_summary.keys())
                    total_files += sum(format_summary.values())
        
        logger.info(f"📊 Analysis complete!")
        logger.info(f"Total unique formats found: {len(total_formats)}")
        logger.info(f"Total files analyzed: {total_files}")
        
        return self.data_formats
    
    def print_format_summary(self):
        """Print a summary of all data formats found"""
        logger.info("\n" + "="*80)
        logger.info("📋 RML DATA FORMAT SUMMARY")
        logger.info("="*80)
        
        for i, (key_tuple, info) in enumerate(self.data_formats.items(), 1):
            logger.info(f"\n🔸 Format {i}:")
            logger.info(f"   Keys: {list(key_tuple)}")
            logger.info(f"   Files: {info['count']}")
            logger.info(f"   Sample: {info['sample']}")
            
            # Show first file path
            if info['files']:
                logger.info(f"   Example file: {os.path.basename(info['files'][0])}")
    
    def identify_quality_issues(self):
        """Identify potential data quality issues"""
        logger.info("\n🔍 Identifying data quality issues...")
        
        issues = []
        
        for key_tuple, info in self.data_formats.items():
            sample = info['sample']
            
            # Check for missing RML components
            expected_components = ['concepts', 'triples', 'entities', 'emotions', 
                                 'reasoning', 'intents', 'summaries', 'events', 'vectors']
            
            missing_components = [comp for comp in expected_components 
                                if comp not in sample]
            
            if missing_components:
                issues.append({
                    "type": "missing_components",
                    "format": list(key_tuple),
                    "missing": missing_components,
                    "files": len(info['files'])
                })
            
            # Check for empty or placeholder values
            for key, value in sample.items():
                if isinstance(value, str) and value in ['Vector_1', 'Vector_2', 'Event_1', 'Event_2', 'Reasoning_1', 'Reasoning_2']:
                    issues.append({
                        "type": "placeholder_values",
                        "format": list(key_tuple),
                        "key": key,
                        "placeholder": value,
                        "files": len(info['files'])
                    })
        
        self.quality_issues = issues
        return issues
    
    def print_quality_report(self):
        """Print quality issues report"""
        if not self.quality_issues:
            logger.info("✅ No quality issues found!")
            return
        
        logger.info("\n" + "="*80)
        logger.info("⚠️  DATA QUALITY ISSUES FOUND")
        logger.info("="*80)
        
        for i, issue in enumerate(self.quality_issues, 1):
            logger.info(f"\n🔸 Issue {i}: {issue['type']}")
            logger.info(f"   Format: {issue['format']}")
            logger.info(f"   Files affected: {issue['files']}")
            
            if 'missing' in issue:
                logger.info(f"   Missing components: {issue['missing']}")
            if 'key' in issue:
                logger.info(f"   Problem key: {issue['key']} = {issue['placeholder']}")
    
    def create_consolidation_plan(self):
        """Create a plan for consolidating all data formats"""
        logger.info("\n📋 Creating consolidation plan...")
        
        plan = {
            "formats_to_process": [],
            "consolidation_strategy": {},
            "estimated_samples": 0
        }
        
        for key_tuple, info in self.data_formats.items():
            format_info = {
                "keys": list(key_tuple),
                "file_count": info['count'],
                "sample": info['sample'],
                "strategy": self._determine_strategy(info['sample'])
            }
            plan["formats_to_process"].append(format_info)
            plan["estimated_samples"] += info['count'] * 1000  # Rough estimate
        
        return plan
    
    def _determine_strategy(self, sample):
        """Determine consolidation strategy for a format"""
        keys = list(sample.keys())
        
        # Complete RML format (all components in one object)
        if all(comp in keys for comp in ['concepts', 'triples', 'entities', 'emotions', 'reasoning', 'intents', 'summaries', 'events', 'vectors']):
            return "direct_consolidation"
        
        # Component-specific format (one component per file)
        elif len(keys) <= 3 and any(comp in keys for comp in ['concepts', 'triples', 'entities', 'emotions', 'reasoning', 'intents', 'summaries', 'events', 'vectors']):
            return "component_assembly"
        
        # Metadata format (record_id, confidence, etc.)
        elif any(key in keys for key in ['record_id', 'doc_id', 'confidence']):
            return "metadata_processing"
        
        else:
            return "unknown_format"
    
    def print_consolidation_plan(self, plan):
        """Print the consolidation plan"""
        logger.info("\n" + "="*80)
        logger.info("📋 CONSOLIDATION PLAN")
        logger.info("="*80)
        
        logger.info(f"\n📊 Estimated total samples: {plan['estimated_samples']:,}")
        logger.info(f"📁 Formats to process: {len(plan['formats_to_process'])}")
        
        for i, format_info in enumerate(plan['formats_to_process'], 1):
            logger.info(f"\n🔸 Format {i}:")
            logger.info(f"   Keys: {format_info['keys']}")
            logger.info(f"   Files: {format_info['file_count']}")
            logger.info(f"   Strategy: {format_info['strategy']}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("🚀 Starting full RML data analysis...")
        
        # Analyze all directories
        self.analyze_all_directories()
        
        # Print format summary
        self.print_format_summary()
        
        # Identify quality issues
        self.identify_quality_issues()
        self.print_quality_report()
        
        # Create consolidation plan
        plan = self.create_consolidation_plan()
        self.print_consolidation_plan(plan)
        
        # Save analysis results
        self.save_analysis_results(plan)
        
        logger.info("\n✅ Analysis complete! Check analysis_results.json for details.")
        
        return plan
    
    def save_analysis_results(self, plan):
        """Save analysis results to file"""
        # Convert tuple keys to strings for JSON serialization
        data_formats_serializable = {}
        for key_tuple, info in self.data_formats.items():
            key_str = str(list(key_tuple))
            data_formats_serializable[key_str] = {
                "sample": info['sample'],
                "files": info['files'],
                "count": info['count']
            }
        
        results = {
            "data_formats": data_formats_serializable,
            "quality_issues": self.quality_issues,
            "consolidation_plan": plan,
            "summary": {
                "total_formats": len(self.data_formats),
                "total_quality_issues": len(self.quality_issues),
                "estimated_samples": plan['estimated_samples']
            }
        }
        
        with open("analysis_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("💾 Analysis results saved to analysis_results.json")

def main():
    analyzer = RMLDataAnalyzer()
    plan = analyzer.run_full_analysis()
    
    # Print next steps
    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print("1. Review the analysis results above")
    print("2. Check analysis_results.json for detailed breakdown")
    print("3. Create a consolidation script based on the identified formats")
    print("4. Fix any quality issues before consolidation")
    print("5. Run the consolidation with proper format handling")

if __name__ == "__main__":
    main() 