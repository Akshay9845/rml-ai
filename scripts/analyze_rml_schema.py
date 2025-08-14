#!/usr/bin/env python3
"""
Analyze RML component files to understand their schema and relationships
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import glob

# Configuration
BASE_DIR = "/Users/elite/R-LLM/data"
SAMPLE_SIZE = 5  # Number of samples to analyze per file

def analyze_file_schema(filepath):
    """Analyze the schema of a single JSONL file"""
    fields = defaultdict(int)
    value_types = defaultdict(set)
    uri_values = set()
    samples = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= SAMPLE_SIZE:
                    break
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                    
                    for field, value in data.items():
                        fields[field] += 1
                        value_types[field].add(type(value).__name__)
                        if field == 'uri':
                            uri_values.add(str(value))
                            
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        return None, None, None, str(e)
        
    return dict(fields), {k: list(v) for k, v in value_types.items()}, samples, None

def analyze_rml_components():
    # Find all JSONL files
    jsonl_files = []
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.jsonl') and 'combined' not in file and 'part' in root.lower():
                jsonl_files.append(os.path.join(root, file))
    
    print(f"Found {len(jsonl_files)} RML component files")
    
    # Analyze each file
    results = {}
    for filepath in sorted(jsonl_files):
        print(f"\nAnalyzing {os.path.basename(filepath)}...")
        fields, value_types, samples, error = analyze_file_schema(filepath)
        
        if error:
            print(f"  Error: {error}")
            continue
            
        print(f"  Fields: {', '.join(fields.keys())}")
        print(f"  Field types: {value_types}")
        print(f"  Sample URIs: {[s.get('uri', 'N/A')[:50] + '...' for s in samples if 'uri' in s][:2]}")
        
        results[os.path.basename(filepath)] = {
            'fields': fields,
            'value_types': value_types,
            'sample_uris': [s.get('uri') for s in samples if 'uri' in s][:2]
        }
    
    # Generate schema analysis
    field_occurrence = defaultdict(list)
    for filename, data in results.items():
        for field in data['fields']:
            field_occurrence[field].append(filename)
    
    print("\n=== Schema Analysis ===")
    print("\nField occurrence across files:")
    for field, files in sorted(field_occurrence.items(), key=lambda x: -len(x[1])):
        print(f"{field}: {len(files)} files")
    
    return results

if __name__ == "__main__":
    analyze_rml_components()
