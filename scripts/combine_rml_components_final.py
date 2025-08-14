#!/usr/bin/env python3
"""
Combine all RML components into a single JSONL file with all components per URI
"""

import json
import os
from collections import defaultdict

# Configuration
CR_SIMPLE_DIR = "/Users/elite/R-LLM/data/cr_simple"
OUTPUT_FILE = "/Users/elite/R-LLM/data/cr_simple/combined.jsonl"

# Template for each line with all RML components
RML_TEMPLATE = {
    "uri": "",
    "concept": "",
    "emotion": "",
    "entity": "",
    "intent": "",
    "reasoning": "",
    "summary": "",
    "tag": "",
    "triple": "",
    "vector": []
}

def combine_rml_components():
    # Dictionary to hold all data by URI
    uri_data = defaultdict(lambda: RML_TEMPLATE.copy())
    
    # Map of component names to their plural form in filenames
    component_map = {
        'concept': 'concepts',
        'emotion': 'emotions',
        'entity': 'entities',
        'intent': 'intents',
        'reasoning': 'reasoning',
        'summary': 'summaries',
        'tag': 'tags',
        'triple': 'triples',
        'vector': 'vectors'
    }
    
    # Process each component file
    for component, filename in component_map.items():
        filepath = os.path.join(CR_SIMPLE_DIR, f"{filename}.jsonl")
        
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            continue
            
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
                    
                    # Update the specific component
                    if component in data:
                        uri_data[uri][component] = data[component]
                    else:
                        # Handle case where the component is the key (e.g., {"uri": "...", "concept": "value"})
                        for key in data:
                            if key in RML_TEMPLATE and key != 'uri':
                                uri_data[uri][key] = data[key]
                                
                except json.JSONDecodeError as e:
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
