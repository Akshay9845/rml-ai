#!/usr/bin/env python3
"""
Create Proper Training Data
Convert RML data into proper training pairs with input prompts and target responses
"""

import json
import os
import random

def create_training_pairs():
    """Create proper training pairs from RML data"""
    
    print("🔧 Creating Proper Training Data")
    print("="*40)
    
    # Load RML data
    rml_data_path = "data/all_rml_training_data.jsonl"
    
    if not os.path.exists(rml_data_path):
        print(f"❌ RML data not found: {rml_data_path}")
        return
    
    # Load RML data
    rml_samples = []
    with open(rml_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                rml_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    print(f"📊 Loaded {len(rml_samples)} RML samples")
    
    # Create training pairs
    training_pairs = []
    
    for i, sample in enumerate(rml_samples):
        if i >= 1000:  # Limit to first 1000 for testing
            break
            
        concepts = sample.get('concepts', [])[:3]
        emotions = sample.get('emotions', [])[:1]
        intents = sample.get('intents', [])[:1]
        entities = sample.get('entities', [])[:2]
        
        if not concepts:
            continue
        
        # Create multiple training pairs from each sample
        pairs = create_pairs_from_sample(concepts, emotions, intents, entities)
        training_pairs.extend(pairs)
    
    print(f"✅ Created {len(training_pairs)} training pairs")
    
    # Split into train/val
    random.shuffle(training_pairs)
    val_size = int(len(training_pairs) * 0.1)
    val_pairs = training_pairs[:val_size]
    train_pairs = training_pairs[val_size:]
    
    # Save training data
    train_path = "data/proper_training_data.jsonl"
    val_path = "data/proper_validation_data.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"💾 Saved {len(train_pairs)} training pairs to {train_path}")
    print(f"💾 Saved {len(val_pairs)} validation pairs to {val_path}")
    
    # Show examples
    print("\n📝 Training pair examples:")
    for i, pair in enumerate(training_pairs[:3]):
        print(f"   {i+1}. Input: {pair['input']}")
        print(f"      Target: {pair['target']}")
        print()

def create_pairs_from_sample(concepts, emotions, intents, entities):
    """Create multiple training pairs from one RML sample"""
    
    pairs = []
    
    # Pair 1: Concept analysis
    if len(concepts) >= 2:
        input_text = f"Analyze these concepts: {', '.join(concepts[:2])}"
        target_text = f"This analysis focuses on {concepts[0]} and {concepts[1]}. The primary concept is {concepts[0]}."
        pairs.append({'input': input_text, 'target': target_text})
    
    # Pair 2: Concept explanation
    if concepts:
        input_text = f"Explain the concept: {concepts[0]}"
        target_text = f"{concepts[0]} is a key concept in this context. It represents an important element in the data."
        pairs.append({'input': input_text, 'target': target_text})
    
    # Pair 3: Emotional context
    if emotions and concepts:
        input_text = f"Concepts: {', '.join(concepts[:2])} | Emotion: {emotions[0]}"
        target_text = f"This data discusses {concepts[0]} with a {emotions[0]} emotional context."
        pairs.append({'input': input_text, 'target': target_text})
    
    # Pair 4: Intent analysis
    if intents and concepts:
        input_text = f"Concepts: {', '.join(concepts[:2])} | Intent: {intents[0]}"
        target_text = f"The intent is to {intents[0]} information about {concepts[0]} and {concepts[1]}."
        pairs.append({'input': input_text, 'target': target_text})
    
    # Pair 5: Entity relationship
    if entities and concepts:
        input_text = f"Concepts: {', '.join(concepts[:2])} | Entities: {', '.join(entities)}"
        target_text = f"The entities {', '.join(entities)} are related to the concepts {', '.join(concepts[:2])}."
        pairs.append({'input': input_text, 'target': target_text})
    
    return pairs

if __name__ == "__main__":
    create_training_pairs() 