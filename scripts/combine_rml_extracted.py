#!/usr/bin/env python3
"""
Combine all RML components from rml_extracted/part2_previous into a single combined.jsonl
Each line will contain all RML components for a given URI
"""

import json
import os
from collections import defaultdict

# Configuration
INPUT_DIR = "/Users/elite/R-LLM/data/rml_extracted/part2_previous"
OUTPUT_DIR = "/Users/elite/R-LLM/data/rml_extracted"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined.jsonl")

# Template for each line with all RML components
RML_TEMPLATE = {
    "uri": "",
    "concept": "",
    "emotion": "",
    "entity": "",
    "event": "",
    "intent": "",
    "reasoning": "",
    "summary": "",
    "tag": "",
    "triple": "",
    "vector": []
}

def combine_rml_components():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Dictionary to hold all data by URI
    uri_data = defaultdict(lambda: RML_TEMPLATE.copy())
    
    # Get all JSONL files in the directory
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jsonl')]
    
    # Process each file
    for filename in files:
        component = os.path.splitext(filename)[0]  # e.g., 'concepts.jsonl' -> 'concepts'
        filepath = os.path.join(INPUT_DIR, filename)
        
        print(f"Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    uri = data.get('uri')
                    if not uri:
                        continue
                        
                    # Initialize URI entry if it doesn't exist
                    if uri not in uri_data:
                        uri_data[uri] = RML_TEMPLATE.copy()
                        uri_data[uri]['uri'] = uri
                    
                    # Update the specific component(s)
                    for key in data:
                        if key in RML_TEMPLATE and key != 'uri':
                            uri_data[uri][key] = data[key]
                                
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"⚠️ Error in {filename}: {e}")
                    continue
    
    # Write combined data to output file
    print(f"\nWriting {len(uri_data)} complete RML entries to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for uri, data in uri_data.items():
            # Ensure all fields are present
            for field in RML_TEMPLATE:
                if field not in data:
                    data[field] = RML_TEMPLATE[field]
            
            # Write the complete entry
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print("✅ Successfully combined all RML components!")

if __name__ == "__main__":
    combine_rml_components()
