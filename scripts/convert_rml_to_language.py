#!/usr/bin/env python3
"""
Convert RML to Language Format
Transform RML structured data into natural language that decoder LLMs can learn from
"""

import json
import os
import random

def convert_rml_to_language():
    """Convert RML data to language format for decoder training"""
    
    print("🔧 Converting RML to Language Format")
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
    
    # Convert to language format
    language_samples = []
    
    for i, sample in enumerate(rml_samples):
        if i >= 3000:  # Limit for testing
            break
            
        concepts = sample.get('concepts', [])[:3]
        emotions = sample.get('emotions', [])[:1]
        intents = sample.get('intents', [])[:1]
        entities = sample.get('entities', [])[:2]
        
        if not concepts:
            continue
        
        # Convert to language format
        language_texts = convert_sample_to_language(concepts, emotions, intents, entities)
        language_samples.extend(language_texts)
    
    print(f"✅ Created {len(language_samples)} language samples")
    
    # Split into train/val
    random.shuffle(language_samples)
    val_size = int(len(language_samples) * 0.1)
    val_samples = language_samples[:val_size]
    train_samples = language_samples[val_size:]
    
    # Save training data
    train_path = "data/language_training_data.jsonl"
    val_path = "data/language_validation_data.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps({'text': sample}) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps({'text': sample}) + '\n')
    
    print(f"💾 Saved {len(train_samples)} training samples to {train_path}")
    print(f"💾 Saved {len(val_samples)} validation samples to {val_path}")
    
    # Show examples
    print("\n📝 Language format examples:")
    for i, sample in enumerate(language_samples[:5]):
        print(f"   {i+1}. {sample}")
        print()

def convert_sample_to_language(concepts, emotions, intents, entities):
    """Convert RML sample to natural language format"""
    
    language_texts = []
    
    # Format 1: Q&A pairs
    if len(concepts) >= 2:
        # Q: What does concept1 relate to?
        # A: concept2.
        qa1 = f"Q: What does {concepts[0]} relate to?\nA: {concepts[1]}."
        language_texts.append(qa1)
        
        # Q: What is related to concept1?
        # A: concept2.
        qa2 = f"Q: What is related to {concepts[0]}?\nA: {concepts[1]}."
        language_texts.append(qa2)
    
    # Format 2: Natural sentences
    if len(concepts) >= 2:
        # concept1 relates to concept2
        sentence1 = f"{concepts[0]} relates to {concepts[1]}."
        language_texts.append(sentence1)
        
        # concept1 and concept2 are connected
        sentence2 = f"{concepts[0]} and {concepts[1]} are connected."
        language_texts.append(sentence2)
    
    # Format 3: Technical descriptions
    if concepts:
        # concept1 is a technology that enables various applications
        tech_desc = f"{concepts[0]} is a technology that enables various applications."
        language_texts.append(tech_desc)
        
        # The concept1 system provides important functionality
        system_desc = f"The {concepts[0]} system provides important functionality."
        language_texts.append(system_desc)
    
    # Format 4: Contextual explanations
    if len(concepts) >= 2:
        # In modern systems, concept1 and concept2 work together
        context1 = f"In modern systems, {concepts[0]} and {concepts[1]} work together."
        language_texts.append(context1)
        
        # concept1 enables concept2 to function properly
        context2 = f"{concepts[0]} enables {concepts[1]} to function properly."
        language_texts.append(context2)
    
    # Format 5: Process descriptions
    if len(concepts) >= 2:
        # The process involves concept1 and concept2
        process1 = f"The process involves {concepts[0]} and {concepts[1]}."
        language_texts.append(process1)
        
        # concept1 leads to concept2
        process2 = f"{concepts[0]} leads to {concepts[1]}."
        language_texts.append(process2)
    
    # Format 6: Comparative statements
    if len(concepts) >= 2:
        # concept1 is similar to concept2
        compare1 = f"{concepts[0]} is similar to {concepts[1]}."
        language_texts.append(compare1)
        
        # Both concept1 and concept2 are important
        compare2 = f"Both {concepts[0]} and {concepts[1]} are important."
        language_texts.append(compare2)
    
    # Format 7: Application contexts
    if concepts:
        # concept1 is used in many applications
        app1 = f"{concepts[0]} is used in many applications."
        language_texts.append(app1)
        
        # Applications of concept1 include various systems
        app2 = f"Applications of {concepts[0]} include various systems."
        language_texts.append(app2)
    
    # Format 8: Future implications
    if len(concepts) >= 2:
        # The future of concept1 and concept2 looks promising
        future1 = f"The future of {concepts[0]} and {concepts[1]} looks promising."
        language_texts.append(future1)
        
        # concept1 will continue to evolve with concept2
        future2 = f"{concepts[0]} will continue to evolve with {concepts[1]}."
        language_texts.append(future2)
    
    # Format 9: Educational content
    if concepts:
        # Learning about concept1 is essential
        edu1 = f"Learning about {concepts[0]} is essential."
        language_texts.append(edu1)
        
        # concept1 forms the basis for understanding modern systems
        edu2 = f"{concepts[0]} forms the basis for understanding modern systems."
        language_texts.append(edu2)
    
    # Format 10: Research context
    if len(concepts) >= 2:
        # Research in concept1 and concept2 continues
        research1 = f"Research in {concepts[0]} and {concepts[1]} continues."
        language_texts.append(research1)
        
        # New developments in concept1 affect concept2
        research2 = f"New developments in {concepts[0]} affect {concepts[1]}."
        language_texts.append(research2)
    
    return language_texts

if __name__ == "__main__":
    convert_rml_to_language() 