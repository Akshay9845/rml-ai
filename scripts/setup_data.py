#!/usr/bin/env python3
"""
Setup script for RML-AI data directory
"""

import os
import shutil
import json
from pathlib import Path


def setup_data_directory():
    """Set up the data directory with essential files"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create a sample RML dataset
    sample_data = [
        {
            "text": "Machine learning is a field of study that focuses on developing algorithms and models that can learn and make predictions or decisions without being explicitly programmed.",
            "source": "AI Fundamentals",
            "category": "machine_learning"
        },
        {
            "text": "Artificial intelligence is the simulation of human intelligence processes by machines.",
            "source": "AI Fundamentals", 
            "category": "artificial_intelligence"
        },
        {
            "text": "Deep learning is a type of machine learning that uses artificial neural networks to process and analyze large amounts of data.",
            "source": "AI Fundamentals",
            "category": "deep_learning"
        },
        {
            "text": "Resonant Memory Learning (RML) is a groundbreaking AI paradigm using frequency-based resonant architecture for information processing, enabling neural-symbolic fusion for continuous learning and explainability.",
            "source": "RML Project Brief",
            "category": "rml"
        },
        {
            "text": "RML achieves sub-50ms inference latency with 98% accuracy on reasoning benchmarks, 100x more memory-efficient than transformers, and 70% reduction in hallucinations.",
            "source": "RML Project Brief",
            "category": "rml_performance"
        },
        {
            "text": "RML serves healthcare with evidence-based diagnostics, finance with auditable decision trails, and manufacturing with explainable predictive maintenance.",
            "source": "RML Project Brief",
            "category": "rml_use_cases"
        }
    ]
    
    # Write sample data
    with open(data_dir / "rml_data.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Created sample dataset with {len(sample_data)} entries")
    
    # Create README for data directory
    readme_content = """# RML-AI Data Directory

This directory contains the training and reference data for the RML-AI system.

## Files

- `rml_data.jsonl` - Sample RML dataset with AI concepts and RML project information

## Adding Your Own Data

1. Create a JSONL file with your data
2. Each line should be a valid JSON object
3. Include at least a 'text' field with the content
4. Optionally include 'source' and 'category' fields

## Data Format

```json
{
  "text": "Your content here",
  "source": "Source name",
  "category": "category_name"
}
```

## Environment Variables

Set `RML_DATASET_PATH` to point to your dataset file:

```bash
export RML_DATASET_PATH="data/your_dataset.jsonl"
```
"""
    
    with open(data_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created data directory README")


def create_config_example():
    """Create example configuration file"""
    
    config_example = """# RML-AI Configuration Example

# Copy this file to .env and modify as needed

# Model Configuration
RML_ENCODER_MODEL=intfloat/e5-base-v2
RML_DECODER_MODEL=microsoft/phi-1_5

# Device Configuration  
RML_DEVICE=auto

# Dataset Configuration
RML_DATASET_PATH=data/rml_data.jsonl
RML_API_ENTRIES=1000

# Encoding Configuration
RML_ENCODER_BATCH_SIZE=8
RML_ENCODER_MAX_LEN=192

# Feature Flags
RML_DISABLE_WEB_SEARCH=1
RML_DISABLE_WORLD_KNOWLEDGE=1
"""
    
    with open(".env.example", "w") as f:
        f.write(config_example)
    
    print("✅ Created .env.example file")


def main():
    """Main setup function"""
    print("🚀 Setting up RML-AI data directory...")
    
    setup_data_directory()
    create_config_example()
    
    print("\n🎯 Setup complete!")
    print("\nNext steps:")
    print("1. Copy your actual dataset to data/ directory")
    print("2. Copy .env.example to .env and configure")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run: python -m rml_ai.cli")


if __name__ == "__main__":
    main() 