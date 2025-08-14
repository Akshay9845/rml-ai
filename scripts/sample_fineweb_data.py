#!/usr/bin/env python3
"""
Sample data from HuggingFaceFW_fineweb_real_data.jsonl for demo purposes
"""
import json
import random
from pathlib import Path
from tqdm import tqdm

def sample_jsonl(input_file, output_file, sample_size=1000):
    """Sample lines from a large JSONL file"""
    # First pass: count total lines
    print("Counting total lines...")
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in tqdm(f, desc="Counting"):
            total_lines += 1
    
    # Calculate sample indices
    sample_indices = set(random.sample(range(total_lines), min(sample_size, total_lines)))
    
    # Second pass: extract samples
    print(f"Sampling {len(sample_indices)} random lines...")
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Sampling"):
            if i in sample_indices:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Write samples to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Saved {len(samples)} samples to {output_file}")
    return samples

if __name__ == "__main__":
    input_file = "/Users/elite/R-LLM/data/streaming/HuggingFaceFW_fineweb_real_data.jsonl"
    output_file = "/Users/elite/R-LLM/data/demo_fineweb_samples.jsonl"
    sample_jsonl(input_file, output_file, sample_size=500)  # Smaller sample for demo
