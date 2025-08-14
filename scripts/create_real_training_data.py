#!/usr/bin/env python3
"""
Create Real Training Data
Convert RML data into actual text documents that language models can learn from
"""

import json
import os
import random

def create_real_training_data():
    """Create real training data from RML data"""
    
    print("🔧 Creating Real Training Data")
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
    
    # Create real text documents
    text_documents = []
    
    for i, sample in enumerate(rml_samples):
        if i >= 2000:  # Limit for testing
            break
            
        concepts = sample.get('concepts', [])[:5]
        emotions = sample.get('emotions', [])[:2]
        intents = sample.get('intents', [])[:2]
        entities = sample.get('entities', [])[:3]
        
        if not concepts:
            continue
        
        # Create multiple text documents from each sample
        documents = create_documents_from_sample(concepts, emotions, intents, entities)
        text_documents.extend(documents)
    
    print(f"✅ Created {len(text_documents)} text documents")
    
    # Split into train/val
    random.shuffle(text_documents)
    val_size = int(len(text_documents) * 0.1)
    val_docs = text_documents[:val_size]
    train_docs = text_documents[val_size:]
    
    # Save training data
    train_path = "data/real_training_data.jsonl"
    val_path = "data/real_validation_data.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for doc in train_docs:
            f.write(json.dumps({'text': doc}) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for doc in val_docs:
            f.write(json.dumps({'text': doc}) + '\n')
    
    print(f"💾 Saved {len(train_docs)} training documents to {train_path}")
    print(f"💾 Saved {len(val_docs)} validation documents to {val_path}")
    
    # Show examples
    print("\n📝 Text document examples:")
    for i, doc in enumerate(text_documents[:3]):
        print(f"   {i+1}. {doc[:100]}...")
        print()

def create_documents_from_sample(concepts, emotions, intents, entities):
    """Create multiple text documents from one RML sample"""
    
    documents = []
    
    # Document 1: Concept explanation
    if len(concepts) >= 2:
        doc = f"The field of {concepts[0]} and {concepts[1]} represents a significant advancement in technology. These concepts are fundamental to understanding modern computational systems. The {concepts[0]} technology provides the foundation for {concepts[1]} applications."
        documents.append(doc)
    
    # Document 2: Technical analysis
    if concepts:
        doc = f"Technical analysis of {concepts[0]} reveals its importance in contemporary systems. This technology enables efficient processing and data management. The implementation of {concepts[0]} requires careful consideration of system architecture and performance optimization."
        documents.append(doc)
    
    # Document 3: Industry perspective
    if len(concepts) >= 3:
        doc = f"In the industry, {concepts[0]}, {concepts[1]}, and {concepts[2]} are transforming how organizations operate. Companies are adopting these technologies to improve efficiency and gain competitive advantages. The integration of {concepts[0]} with {concepts[1]} creates powerful solutions."
        documents.append(doc)
    
    # Document 4: Educational content
    if concepts:
        doc = f"Learning about {concepts[0]} is essential for modern software development. This concept forms the basis for many advanced applications. Students and professionals must understand the principles behind {concepts[0]} to build effective systems."
        documents.append(doc)
    
    # Document 5: Research context
    if len(concepts) >= 2:
        doc = f"Research in {concepts[0]} and {concepts[1]} continues to advance rapidly. New methodologies and approaches are being developed to enhance these technologies. The combination of {concepts[0]} with emerging {concepts[1]} techniques shows promising results."
        documents.append(doc)
    
    # Document 6: Practical application
    if concepts:
        doc = f"Practical applications of {concepts[0]} can be found across various industries. Organizations implement {concepts[0]} solutions to solve complex problems and improve operational efficiency. The success of {concepts[0]} depends on proper implementation and maintenance."
        documents.append(doc)
    
    # Document 7: Future outlook
    if len(concepts) >= 2:
        doc = f"The future of {concepts[0]} and {concepts[1]} looks promising with ongoing developments. These technologies will continue to evolve and find new applications. Innovation in {concepts[0]} will drive progress in related fields."
        documents.append(doc)
    
    # Document 8: Comparative analysis
    if len(concepts) >= 2:
        doc = f"When comparing {concepts[0]} and {concepts[1]}, it's important to consider their respective strengths and limitations. Both technologies offer unique advantages for different use cases. The choice between {concepts[0]} and {concepts[1]} depends on specific requirements."
        documents.append(doc)
    
    return documents

if __name__ == "__main__":
    create_real_training_data() 