#!/usr/bin/env python3
"""
Test RML Data Format
Check if the data is formatted correctly for training
"""

import json
import os

def test_data_format():
    """Test the RML data format"""
    
    print("🧪 Testing RML Data Format")
    print("="*40)
    
    # Load a few samples
    data_path = "data/all_rml_training_data.jsonl"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    print(f"📂 Loading samples from: {data_path}")
    
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Only first 5 samples
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Analyze each sample
    for i, sample in enumerate(samples):
        print(f"\n📊 Sample {i+1}:")
        print(f"   Keys: {list(sample.keys())}")
        
        # Check key fields
        text = sample.get('text', '')
        concepts = sample.get('concepts', [])
        emotions = sample.get('emotions', [])
        
        print(f"   Text length: {len(text)}")
        print(f"   Concepts: {concepts[:3]}")
        print(f"   Emotions: {emotions[:2]}")
        
        # Create simple training text
        if text and concepts:
            training_text = f"Text: {text[:50]}... Concepts: {', '.join(concepts[:2])} This discusses {concepts[0]}."
        elif text:
            training_text = f"Text: {text[:50]}... This is informative."
        elif concepts:
            training_text = f"Concepts: {', '.join(concepts[:2])} These are related topics."
        else:
            training_text = "Data analysis. This contains information."
        
        print(f"   Training text: {training_text}")
        print(f"   Training text length: {len(training_text)}")

def create_minimal_test():
    """Create a minimal test dataset"""
    
    print("\n🔧 Creating minimal test dataset...")
    
    # Create simple test data
    test_data = [
        {
            "text": "Cloud computing is a technology that provides on-demand computing resources.",
            "concepts": ["cloud", "computing", "technology"],
            "emotions": ["informative"],
            "intents": ["explain"]
        },
        {
            "text": "Machine learning algorithms can process large datasets efficiently.",
            "concepts": ["machine", "learning", "algorithms"],
            "emotions": ["technical"],
            "intents": ["describe"]
        },
        {
            "text": "Artificial intelligence is transforming various industries worldwide.",
            "concepts": ["artificial", "intelligence", "industries"],
            "emotions": ["positive"],
            "intents": ["inform"]
        }
    ]
    
    # Save test data
    test_path = "data/test_rml_data.jsonl"
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created test data: {test_path}")
    
    # Show training format
    print("\n📝 Training format examples:")
    for i, item in enumerate(test_data):
        text = item['text'][:50] + "..."
        concepts = ', '.join(item['concepts'][:2])
        training_text = f"Text: {text} Concepts: {concepts} This discusses {item['concepts'][0]}."
        print(f"   {i+1}. {training_text}")

if __name__ == "__main__":
    test_data_format()
    create_minimal_test() 