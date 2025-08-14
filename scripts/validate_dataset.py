#!/usr/bin/env python3
"""
Validate Dataset
Validates the quality and integrity of the prepared training dataset
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMLDatasetValidator:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        
        # RML component patterns
        self.rml_patterns = {
            'concepts': r'<CONCEPT>(.*?)</CONCEPT>',
            'triples': r'<TRIPLE>(.*?)</TRIPLE>',
            'entities': r'<ENTITY>(.*?)</ENTITY>',
            'emotions': r'<EMOTION>(.*?)</EMOTION>',
            'reasoning': r'<REASONING>(.*?)</REASONING>',
            'intents': r'<INTENT>(.*?)</INTENT>',
            'summaries': r'<SUMMARY>(.*?)</SUMMARY>',
            'events': r'<EVENT>(.*?)</EVENT>',
            'vectors': r'<VECTOR>(.*?)</VECTOR>',
            'text': r'<TEXT>(.*?)</TEXT>'
        }
        
        # Validation statistics
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'component_counts': {comp: 0 for comp in self.rml_patterns.keys()},
            'text_lengths': [],
            'component_lengths': {comp: [] for comp in self.rml_patterns.keys()},
            'errors': {
                'json_parse': 0,
                'missing_components': 0,
                'empty_components': 0,
                'malformed_tags': 0,
                'duplicate_components': 0
            }
        }
    
    def find_dataset_files(self) -> List[Path]:
        """Find dataset files"""
        logger.info("🔍 Finding dataset files...")
        
        dataset_files = []
        for pattern in ['*.jsonl', '*.json']:
            dataset_files.extend(self.input_dir.rglob(pattern))
        
        logger.info(f"📁 Found {len(dataset_files)} dataset files")
        return dataset_files
    
    def parse_json_line(self, line: str) -> Dict:
        """Parse JSON line"""
        try:
            data = json.loads(line.strip())
            return data
        except json.JSONDecodeError as e:
            self.stats['errors']['json_parse'] += 1
            logger.warning(f"❌ Invalid JSON: {e}")
            return None
    
    def validate_rml_tags(self, text: str) -> Dict:
        """Validate RML tags in text"""
        validation = {
            'valid': True,
            'components_found': set(),
            'malformed_tags': [],
            'empty_components': []
        }
        
        # Check for malformed tags
        for component, pattern in self.rml_patterns.items():
            matches = re.findall(pattern, text, re.DOTALL)
            
            if matches:
                validation['components_found'].add(component)
                
                # Check for empty components
                for match in matches:
                    if not match.strip():
                        validation['empty_components'].append(component)
                        validation['valid'] = False
                
                # Update statistics
                self.stats['component_counts'][component] += len(matches)
                self.stats['component_lengths'][component].extend([len(match) for match in matches])
        
        # Check for unmatched tags
        open_tags = re.findall(r'<([^/][^>]*)>', text)
        close_tags = re.findall(r'</([^>]*)>', text)
        
        if len(open_tags) != len(close_tags):
            validation['malformed_tags'].append('Mismatched tag count')
            validation['valid'] = False
        
        return validation
    
    def validate_triple_structure(self, triple_text: str) -> bool:
        """Validate triple structure (subject, predicate, object)"""
        # Simple validation - check for basic triple structure
        if not triple_text or len(triple_text.split()) < 3:
            return False
        
        # Check for common triple patterns
        triple_patterns = [
            r'(\w+)\s+(is|has|contains|relates to)\s+(\w+)',
            r'(\w+)\s+(\w+)\s+(\w+)',
        ]
        
        for pattern in triple_patterns:
            if re.search(pattern, triple_text):
                return True
        
        return False
    
    def validate_sample(self, sample: Dict) -> bool:
        """Validate a single sample"""
        if not sample or 'text' not in sample:
            return False
        
        text = sample['text']
        
        # Validate text length
        if len(text) < 10:  # Minimum reasonable length
            return False
        
        self.stats['text_lengths'].append(len(text))
        
        # Validate RML tags
        tag_validation = self.validate_rml_tags(text)
        
        if not tag_validation['valid']:
            self.stats['errors']['malformed_tags'] += 1
            return False
        
        if tag_validation['empty_components']:
            self.stats['errors']['empty_components'] += 1
            return False
        
        # Check for at least one RML component
        if not tag_validation['components_found']:
            self.stats['errors']['missing_components'] += 1
            return False
        
        # Validate triple structure if present
        if 'triples' in tag_validation['components_found']:
            triple_matches = re.findall(self.rml_patterns['triples'], text, re.DOTALL)
            for triple in triple_matches:
                if not self.validate_triple_structure(triple):
                    logger.warning(f"⚠️ Invalid triple structure: {triple[:100]}...")
        
        return True
    
    def process_file(self, file_path: Path):
        """Process a single dataset file"""
        logger.info(f"📄 Validating {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    self.stats['total_samples'] += 1
                    
                    # Parse JSON
                    sample = self.parse_json_line(line)
                    if not sample:
                        continue
                    
                    # Validate sample
                    if self.validate_sample(sample):
                        self.stats['valid_samples'] += 1
                    else:
                        self.stats['invalid_samples'] += 1
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
    
    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            'validation_summary': {
                'total_samples': self.stats['total_samples'],
                'valid_samples': self.stats['valid_samples'],
                'invalid_samples': self.stats['invalid_samples'],
                'validation_rate': round(self.stats['valid_samples'] / max(self.stats['total_samples'], 1) * 100, 2)
            },
            'component_analysis': {},
            'text_analysis': {},
            'error_analysis': self.stats['errors']
        }
        
        # Component analysis
        for component, count in self.stats['component_counts'].items():
            if count > 0:
                lengths = self.stats['component_lengths'][component]
                stats['component_analysis'][component] = {
                    'count': count,
                    'avg_length': round(sum(lengths) / len(lengths), 2) if lengths else 0,
                    'min_length': min(lengths) if lengths else 0,
                    'max_length': max(lengths) if lengths else 0
                }
        
        # Text analysis
        if self.stats['text_lengths']:
            stats['text_analysis'] = {
                'avg_length': round(sum(self.stats['text_lengths']) / len(self.stats['text_lengths']), 2),
                'min_length': min(self.stats['text_lengths']),
                'max_length': max(self.stats['text_lengths']),
                'total_length': sum(self.stats['text_lengths'])
            }
        
        return stats
    
    def validate_dataset(self):
        """Main validation method"""
        logger.info("🚀 Starting dataset validation...")
        
        # Find dataset files
        dataset_files = self.find_dataset_files()
        
        if not dataset_files:
            logger.error("❌ No dataset files found!")
            return
        
        # Process all files
        for file_path in dataset_files:
            self.process_file(file_path)
        
        # Calculate statistics
        statistics = self.calculate_statistics()
        
        # Save validation report
        self.save_validation_report(statistics)
        
        # Print summary
        self.print_summary(statistics)
    
    def save_validation_report(self, statistics: Dict):
        """Save validation report"""
        report_file = self.input_dir / "validation_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"📄 Validation report saved to {report_file}")
    
    def print_summary(self, statistics: Dict):
        """Print validation summary"""
        print("\n" + "="*80)
        print("🎉 DATASET VALIDATION COMPLETE!")
        print("="*80)
        
        # Validation summary
        summary = statistics['validation_summary']
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"  • Total samples: {summary['total_samples']:,}")
        print(f"  • Valid samples: {summary['valid_samples']:,}")
        print(f"  • Invalid samples: {summary['invalid_samples']:,}")
        print(f"  • Validation rate: {summary['validation_rate']}%")
        
        # Component analysis
        print(f"\n🧠 RML COMPONENT ANALYSIS:")
        for component, analysis in statistics['component_analysis'].items():
            print(f"  • {component.upper()}:")
            print(f"    - Count: {analysis['count']:,}")
            print(f"    - Avg length: {analysis['avg_length']} chars")
            print(f"    - Length range: {analysis['min_length']} - {analysis['max_length']} chars")
        
        # Text analysis
        if statistics['text_analysis']:
            text_analysis = statistics['text_analysis']
            print(f"\n📝 TEXT ANALYSIS:")
            print(f"  • Average length: {text_analysis['avg_length']} chars")
            print(f"  • Length range: {text_analysis['min_length']} - {text_analysis['max_length']} chars")
            print(f"  • Total text: {text_analysis['total_length']:,} chars")
        
        # Error analysis
        print(f"\n❌ ERROR ANALYSIS:")
        for error_type, count in statistics['error_analysis'].items():
            if count > 0:
                print(f"  • {error_type.replace('_', ' ').title()}: {count}")
        
        # Quality assessment
        validation_rate = summary['validation_rate']
        if validation_rate >= 95:
            quality = "EXCELLENT"
            emoji = "🟢"
        elif validation_rate >= 90:
            quality = "GOOD"
            emoji = "🟡"
        elif validation_rate >= 80:
            quality = "FAIR"
            emoji = "🟠"
        else:
            quality = "NEEDS IMPROVEMENT"
            emoji = "🔴"
        
        print(f"\n🎯 QUALITY ASSESSMENT:")
        print(f"  {emoji} Dataset Quality: {quality} ({validation_rate}%)")
        
        if validation_rate < 90:
            print(f"  ⚠️  Consider reviewing and cleaning the dataset")
        
        print(f"\n📄 VALIDATION REPORT:")
        print(f"  • Report saved to: {self.input_dir}/validation_report.json")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Validate RML training dataset")
    parser.add_argument("--input-dir", required=True, help="Input directory with dataset files")
    
    args = parser.parse_args()
    
    # Create validator and run
    validator = RMLDatasetValidator(input_dir=args.input_dir)
    validator.validate_dataset()

if __name__ == "__main__":
    main() 