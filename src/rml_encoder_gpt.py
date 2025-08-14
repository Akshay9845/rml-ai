#!/usr/bin/env python3
"""
RML Encoder - GPT-like Text to RML Data Conversion
Converts raw text into structured RML data (concepts, entities, triples, etc.)
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class RMLEncoderConfig:
    """Configuration for RML encoder"""
    
    # Processing settings
    max_concepts_per_text: int = 20
    max_entities_per_text: int = 10
    max_triples_per_text: int = 15
    
    # Extraction settings
    min_concept_length: int = 3
    min_confidence: float = 0.6
    
    # Output settings
    output_dir: str = "output/rml_encoder"
    save_encoded_data: bool = True
    log_level: str = "INFO"

class RMLEncoder:
    """GPT-like encoder that converts text to RML data structure"""
    
    def __init__(self, config: RMLEncoderConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Knowledge bases for extraction
        self.emotion_keywords = {
            'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'outstanding'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating', 'annoying'],
            'neutral': ['okay', 'fine', 'normal', 'standard', 'regular', 'typical', 'usual'],
            'informative': ['explains', 'describes', 'shows', 'demonstrates', 'illustrates', 'presents'],
            'excited': ['exciting', 'thrilling', 'amazing', 'incredible', 'unbelievable', 'spectacular']
        }
        
        self.intent_keywords = {
            'inform': ['information', 'data', 'facts', 'details', 'explains', 'describes'],
            'describe': ['description', 'characteristics', 'features', 'properties', 'attributes'],
            'explain': ['explanation', 'how', 'why', 'process', 'method', 'procedure'],
            'compare': ['compare', 'versus', 'difference', 'similar', 'contrast', 'vs'],
            'analyze': ['analysis', 'examine', 'study', 'investigate', 'research', 'evaluate']
        }
        
        self.entity_patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORGANIZATION': r'\b[A-Z][A-Z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Organization)\b',
            'LOCATION': r'\b[A-Z][a-z]+(?: City| State| Country| Nation)\b',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'TIME': r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b'
        }
        
        self.logger.info("🔧 RML Encoder initialized")
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter by length and frequency
        word_freq = defaultdict(int)
        for word in words:
            if len(word) >= self.config.min_concept_length:
                word_freq[word] += 1
        
        # Get most frequent concepts
        concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, freq in concepts[:self.config.max_concepts_per_text]]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'type': entity_type,
                    'position': match.start(),
                    'confidence': 0.8
                })
        
        return entities[:self.config.max_entities_per_text]
    
    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        """Extract subject-predicate-object triples from text"""
        triples = []
        
        # Simple pattern-based extraction
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Look for common patterns
            # Pattern 1: Subject verb object
            pattern1 = r'(\b[A-Z][a-z]+\b)\s+(\b[a-z]+\b)\s+(\b[a-z]+\b)'
            matches = re.finditer(pattern1, sentence)
            
            for match in matches:
                subject, predicate, object_ = match.groups()
                if len(subject) >= 3 and len(predicate) >= 3 and len(object_) >= 3:
                    triples.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': object_
                    })
            
            # Pattern 2: Entity is/are description
            pattern2 = r'(\b[A-Z][a-z]+\b)\s+(?:is|are|was|were)\s+(\b[a-z\s]+\b)'
            matches = re.finditer(pattern2, sentence)
            
            for match in matches:
                subject, description = match.groups()
                if len(subject) >= 3 and len(description.strip()) >= 3:
                    triples.append({
                        'subject': subject,
                        'predicate': 'is',
                        'object': description.strip()
                    })
        
        return triples[:self.config.max_triples_per_text]
    
    def extract_emotions(self, text: str) -> List[str]:
        """Extract emotions from text"""
        emotions = []
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotions.append(emotion)
                    break
        
        return list(set(emotions))  # Remove duplicates
    
    def extract_intents(self, text: str) -> List[str]:
        """Extract user intents from text"""
        intents = []
        text_lower = text.lower()
        
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    intents.append(intent)
                    break
        
        return list(set(intents))  # Remove duplicates
    
    def extract_events(self, text: str) -> List[str]:
        """Extract events from text"""
        events = []
        
        # Look for action words and event indicators
        action_words = ['created', 'developed', 'launched', 'announced', 'released', 'started', 'began', 'completed']
        text_lower = text.lower()
        
        for action in action_words:
            if action in text_lower:
                events.append(f"event_{action}")
        
        # Add generic events if none found
        if not events:
            events = ['document_processing', 'information_extraction']
        
        return events
    
    def generate_vectors(self, text: str, dimension: int = 128) -> List[float]:
        """Generate simple vector representation of text"""
        # Simple hash-based vector generation
        import hashlib
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to vector
        vector = []
        for i in range(0, len(text_hash), 2):
            if len(vector) >= dimension:
                break
            hex_val = text_hash[i:i+2]
            float_val = (int(hex_val, 16) - 128) / 128.0  # Normalize to [-1, 1]
            vector.append(float_val)
        
        # Pad or truncate to desired dimension
        while len(vector) < dimension:
            vector.append(0.0)
        
        return vector[:dimension]
    
    def generate_reasoning(self, text: str) -> List[str]:
        """Generate reasoning steps based on text content"""
        reasoning = []
        
        # Simple reasoning based on text length and content
        if len(text) > 100:
            reasoning.append("comprehensive_analysis")
        
        if any(word in text.lower() for word in ['because', 'therefore', 'thus', 'hence']):
            reasoning.append("logical_inference")
        
        if any(word in text.lower() for word in ['compare', 'versus', 'difference', 'similar']):
            reasoning.append("comparative_analysis")
        
        if not reasoning:
            reasoning.append("basic_understanding")
        
        return reasoning
    
    def generate_summaries(self, text: str) -> List[str]:
        """Generate summaries of the text"""
        # Simple summary generation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if sentences:
            # Take first and last sentence as summary
            summary = sentences[0]
            if len(sentences) > 1:
                summary += " " + sentences[-1]
            
            return [summary[:200] + "..." if len(summary) > 200 else summary]
        
        return [text[:100] + "..." if len(text) > 100 else text]
    
    def encode_text(self, text: str, record_id: str = None) -> Dict[str, Any]:
        """Encode text into RML data structure"""
        
        self.logger.info(f"🔧 Encoding text (length: {len(text)})")
        
        # Extract all components
        concepts = self.extract_concepts(text)
        entities = self.extract_entities(text)
        triples = self.extract_triples(text)
        emotions = self.extract_emotions(text)
        intents = self.extract_intents(text)
        events = self.extract_events(text)
        vectors = self.generate_vectors(text)
        reasoning = self.generate_reasoning(text)
        summaries = self.generate_summaries(text)
        
        # Build RML data structure
        rml_data = {
            'record_id': record_id or f"encoded_{hash(text) % 1000000}",
            'text': text,
            'text_length': len(text),
            'concepts': concepts,
            'entities': [entity['text'] for entity in entities],
            'triples': [f"{{'subject': '{t['subject']}', 'predicate': '{t['predicate']}', 'object': '{t['object']}'}}" for t in triples],
            'emotions': emotions,
            'intents': intents,
            'events': events,
            'vectors': vectors,
            'reasoning': reasoning,
            'summaries': summaries,
            'tags': ['encoded_data', 'text_to_rml'],
            'confidence': 0.8,
            'processing_timestamp': str(hash(text) % 1000000)
        }
        
        self.logger.info(f"✅ Encoded: {len(concepts)} concepts, {len(entities)} entities, {len(triples)} triples")
        
        return rml_data
    
    def encode_batch(self, texts: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """Encode a batch of texts"""
        encoded_data = []
        
        for i, text in enumerate(texts):
            record_id = f"batch_{i}_{hash(text) % 10000}"
            rml_data = self.encode_text(text, record_id)
            encoded_data.append(rml_data)
        
        # Save to file if requested
        if output_file and self.config.save_encoded_data:
            output_path = os.path.join(self.config.output_dir, output_file)
            with open(output_path, 'w') as f:
                for data in encoded_data:
                    f.write(json.dumps(data) + '\n')
            
            self.logger.info(f"💾 Saved {len(encoded_data)} encoded records to {output_path}")
        
        return encoded_data

def main():
    """Main function for testing the RML encoder"""
    
    # Configuration
    config = RMLEncoderConfig(
        output_dir="output/rml_encoder",
        save_encoded_data=True,
        log_level="INFO"
    )
    
    # Initialize encoder
    encoder = RMLEncoder(config)
    
    print("🔧 Testing RML Encoder (GPT-like Text to RML)")
    print("="*60)
    
    # Test texts
    test_texts = [
        "Cloud computing infrastructure supports scalable applications. Amazon Web Services provides reliable cloud solutions for businesses worldwide.",
        "Artificial intelligence and machine learning are transforming how we process data. Companies like Google and Microsoft are leading the innovation in AI technology.",
        "The Python programming language is widely used for data science and web development. It offers excellent libraries like NumPy and Django for various applications."
    ]
    
    try:
        # Encode individual text
        print("🔧 Testing single text encoding:")
        result = encoder.encode_text(test_texts[0], "test_001")
        
        print(f"📄 Input text: {test_texts[0][:100]}...")
        print(f"🔍 Concepts: {result['concepts'][:5]}")
        print(f"🏢 Entities: {result['entities'][:3]}")
        print(f"🔗 Triples: {len(result['triples'])} relationships")
        print(f"😊 Emotions: {result['emotions']}")
        print(f"🎯 Intents: {result['intents']}")
        print(f"📅 Events: {result['events']}")
        print(f"🧠 Reasoning: {result['reasoning']}")
        print(f"📝 Summary: {result['summaries'][0][:100]}...")
        print(f"📊 Vector dimension: {len(result['vectors'])}")
        
        print("\n" + "="*60)
        print("🔧 Testing batch encoding:")
        
        # Encode batch
        batch_results = encoder.encode_batch(test_texts, "encoded_batch.jsonl")
        
        print(f"✅ Successfully encoded {len(batch_results)} texts")
        print(f"📊 Total concepts extracted: {sum(len(r['concepts']) for r in batch_results)}")
        print(f"📊 Total entities extracted: {sum(len(r['entities']) for r in batch_results)}")
        print(f"📊 Total triples extracted: {sum(len(r['triples']) for r in batch_results)}")
        
        print(f"\n✅ RML Encoder working successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 