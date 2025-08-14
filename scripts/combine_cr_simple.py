#!/usr/bin/env python3
"""
Combine all JSONL files in cr_simple directory into a single combined.jsonl
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Configuration
CR_SIMPLE_DIR = "/Users/elite/R-LLM/data/cr_simple"
OUTPUT_FILE = "/Users/elite/R-LLM/data/cr_simple/combined.jsonl"

# Files to combine (in order of priority for deduplication)
FILES_TO_COMBINE = [
    "concepts.jsonl",
    "entities.jsonl",
    "tags.jsonl",
    "emotions.jsonl",
    "intents.jsonl",
    "reasoning.jsonl",
    "summaries.jsonl",
    "triples.jsonl",
    "vectors.jsonl"
]

def combine_jsonl_files():
    # Track seen URIs to avoid duplicates
    seen_uris = set()
    combined_data = []
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for filename in FILES_TO_COMBINE:
        filepath = os.path.join(CR_SIMPLE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            continue
            
        print(f"Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    uri = data.get('uri')
                    
                    # Only add if we haven't seen this URI before
                    if uri and uri not in seen_uris:
                        seen_uris.add(uri)
                        combined_data.append(data)
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing JSON in {filename}: {e}")
                    continue
    
    # Write combined data to output file
    print(f"\nWriting {len(combined_data)} unique entries to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')
    
    print("✅ Done!")

if __name__ == "__main__":
    combine_jsonl_files()
