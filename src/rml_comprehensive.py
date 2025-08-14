#!/usr/bin/env python3
"""
Comprehensive RML Processor - Understanding Real RML Data Structure
Implements: Observe → Understand → Relate → Respond
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np

@dataclass
class RMLConfig:
    """Configuration for comprehensive RML processor"""
    
    # Processing settings
    max_concepts_per_input: int = 50
    min_confidence: float = 0.5
    
    # RML components
    process_concepts: bool = True
    process_entities: bool = True
    process_triples: bool = True
    process_emotions: bool = True
    process_intents: bool = True
    process_events: bool = True
    process_vectors: bool = True
    process_reasoning: bool = True
    process_summaries: bool = True
    
    # Memory settings
    memory_decay: float = 0.95
    max_memory_size: int = 10000
    
    # Output settings
    output_dir: str = "output/rml_comprehensive"
    save_memory_graphs: bool = True
    log_level: str = "INFO"

class RMLMemory:
    """Comprehensive Resonant Memory Layer"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        
        # Core memory structures
        self.concepts = defaultdict(dict)  # concept -> {count, confidence, types, contexts}
        self.entities = defaultdict(dict)  # entity -> {count, types, relationships}
        self.triples = []  # List of (subject, predicate, object) tuples
        self.emotions = defaultdict(int)  # emotion -> frequency
        self.intents = defaultdict(int)   # intent -> frequency
        self.events = defaultdict(int)    # event -> frequency
        
        # Vector memory
        self.vector_memory = []  # List of vectors for similarity search
        self.concept_vectors = {}  # concept -> vector mapping
        
        # Relationship graph
        self.relationships = defaultdict(list)  # concept -> [(relation, target, confidence)]
        
        # Access tracking
        self.access_count = defaultdict(int)
        self.last_access = defaultdict(float)
        
    def add_concept(self, concept: str, confidence: float = 1.0, context: str = "", concept_type: str = "general"):
        """Add a concept to memory with context"""
        if confidence >= self.config.min_confidence:
            if concept not in self.concepts:
                self.concepts[concept] = {
                    'count': 0,
                    'confidence': 0.0,
                    'types': set(),
                    'contexts': [],
                    'first_seen': None,
                    'last_seen': None
                }
            
            self.concepts[concept]['count'] += 1
            self.concepts[concept]['confidence'] = max(self.concepts[concept]['confidence'], confidence)
            self.concepts[concept]['types'].add(concept_type)
            if context:
                self.concepts[concept]['contexts'].append(context)
            
            self.access_count[concept] += 1
            self.last_access[concept] = 1.0
    
    def add_entity(self, entity: str, entity_type: str = "general", relationships: List[str] = None):
        """Add an entity to memory"""
        if entity not in self.entities:
            self.entities[entity] = {
                'count': 0,
                'types': set(),
                'relationships': []
            }
        
        self.entities[entity]['count'] += 1
        self.entities[entity]['types'].add(entity_type)
        if relationships:
            self.entities[entity]['relationships'].extend(relationships)
    
    def add_triple(self, subject: str, predicate: str, object_: str, confidence: float = 1.0):
        """Add a triple (subject, predicate, object) to memory"""
        triple = (subject, predicate, object_, confidence)
        self.triples.append(triple)
        
        # Add to relationship graph
        self.relationships[subject].append({
            'predicate': predicate,
            'object': object_,
            'confidence': confidence
        })
    
    def add_emotion(self, emotion: str):
        """Add emotion to memory"""
        self.emotions[emotion] += 1
    
    def add_intent(self, intent: str):
        """Add intent to memory"""
        self.intents[intent] += 1
    
    def add_event(self, event: str):
        """Add event to memory"""
        self.events[event] += 1
    
    def add_vector(self, vector: List[float], concept: str = None):
        """Add vector to memory"""
        if len(vector) > 0:
            self.vector_memory.append(vector)
            if concept:
                self.concept_vectors[concept] = vector
    
    def get_relevant_concepts(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get most relevant concepts based on query and memory"""
        scored_concepts = []
        
        for concept, data in self.concepts.items():
            # Base score from frequency and confidence
            base_score = data['count'] * data['confidence']
            
            # Query relevance bonus
            query_bonus = 0
            if query.lower() in concept.lower():
                query_bonus = 2.0
            elif any(word in concept.lower() for word in query.lower().split()):
                query_bonus = 1.0
            
            # Relationship bonus
            relationship_bonus = len(self.relationships[concept]) * 0.1
            
            # Recency bonus
            recency_bonus = self.last_access.get(concept, 0) * 0.5
            
            total_score = base_score + query_bonus + relationship_bonus + recency_bonus
            scored_concepts.append((concept, total_score))
        
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        return scored_concepts[:top_k]
    
    def get_concept_graph(self, concepts: List[str]) -> Dict[str, Any]:
        """Get a subgraph containing specified concepts with relationships"""
        subgraph = {}
        for concept in concepts:
            if concept in self.concepts:
                concept_data = self.concepts[concept].copy()
                concept_data['types'] = list(concept_data['types'])
                
                # Add relationships
                if concept in self.relationships:
                    concept_data['relationships'] = self.relationships[concept]
                
                subgraph[concept] = concept_data
        return subgraph
    
    def get_semantic_context(self, query: str) -> Dict[str, Any]:
        """Get semantic context including emotions, intents, and events"""
        return {
            'emotions': dict(self.emotions),
            'intents': dict(self.intents),
            'events': dict(self.events),
            'triples': self.triples[-100:],  # Last 100 triples
            'vector_count': len(self.vector_memory)
        }
    
    def decay_memory(self):
        """Apply memory decay to simulate forgetting"""
        for concept in self.last_access:
            self.last_access[concept] *= self.config.memory_decay
    
    def save_memory(self, filepath: str):
        """Save memory to disk"""
        memory_data = {
            'concepts': {k: {**v, 'types': list(v['types'])} for k, v in self.concepts.items()},
            'entities': {k: {**v, 'types': list(v['types'])} for k, v in self.entities.items()},
            'triples': self.triples,
            'emotions': dict(self.emotions),
            'intents': dict(self.intents),
            'events': dict(self.events),
            'relationships': dict(self.relationships),
            'access_count': dict(self.access_count),
            'last_access': dict(self.last_access),
            'vector_count': len(self.vector_memory)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self, filepath: str):
        """Load memory from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            self.concepts = defaultdict(dict, memory_data.get('concepts', {}))
            self.entities = defaultdict(dict, memory_data.get('entities', {}))
            self.triples = memory_data.get('triples', [])
            self.emotions = defaultdict(int, memory_data.get('emotions', {}))
            self.intents = defaultdict(int, memory_data.get('intents', {}))
            self.events = defaultdict(int, memory_data.get('events', {}))
            self.relationships = defaultdict(list, memory_data.get('relationships', {}))
            self.access_count = defaultdict(int, memory_data.get('access_count', {}))
            self.last_access = defaultdict(float, memory_data.get('last_access', {}))

class ComprehensiveRMLProcessor:
    """Comprehensive RML processor that understands real RML data structure"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.memory = RMLMemory(config)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.logger.info("🧠 Comprehensive RML Processor initialized")
    
    def observe(self, rml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Observe - Extract all RML components from data"""
        observations = {
            'concepts': [],
            'entities': [],
            'triples': [],
            'emotions': [],
            'intents': [],
            'events': [],
            'vectors': [],
            'reasoning': [],
            'summaries': []
        }
        
        # Extract concepts
        if self.config.process_concepts and 'concepts' in rml_data:
            if isinstance(rml_data['concepts'], list):
                observations['concepts'] = rml_data['concepts']
            elif isinstance(rml_data['concepts'], str):
                observations['concepts'] = [rml_data['concepts']]
        
        # Extract entities
        if self.config.process_entities and 'entities' in rml_data:
            if isinstance(rml_data['entities'], list):
                observations['entities'] = rml_data['entities']
            elif isinstance(rml_data['entities'], str):
                observations['entities'] = [rml_data['entities']]
        
        # Extract triples
        if self.config.process_triples and 'triples' in rml_data:
            if isinstance(rml_data['triples'], list):
                observations['triples'] = rml_data['triples']
        
        # Extract emotions
        if self.config.process_emotions and 'emotions' in rml_data:
            if isinstance(rml_data['emotions'], list):
                observations['emotions'] = rml_data['emotions']
            elif isinstance(rml_data['emotions'], str):
                observations['emotions'] = [rml_data['emotions']]
        
        # Extract intents
        if self.config.process_intents and 'intents' in rml_data:
            if isinstance(rml_data['intents'], list):
                observations['intents'] = rml_data['intents']
            elif isinstance(rml_data['intents'], str):
                observations['intents'] = [rml_data['intents']]
        
        # Extract events
        if self.config.process_events and 'events' in rml_data:
            if isinstance(rml_data['events'], list):
                observations['events'] = rml_data['events']
            elif isinstance(rml_data['events'], str):
                observations['events'] = [rml_data['events']]
        
        # Extract vectors
        if self.config.process_vectors and 'vectors' in rml_data:
            if isinstance(rml_data['vectors'], list):
                observations['vectors'] = rml_data['vectors']
        
        # Extract reasoning
        if self.config.process_reasoning and 'reasoning' in rml_data:
            if isinstance(rml_data['reasoning'], list):
                observations['reasoning'] = rml_data['reasoning']
            elif isinstance(rml_data['reasoning'], str):
                observations['reasoning'] = [rml_data['reasoning']]
        
        # Extract summaries
        if self.config.process_summaries and 'summaries' in rml_data:
            if isinstance(rml_data['summaries'], list):
                observations['summaries'] = rml_data['summaries']
            elif isinstance(rml_data['summaries'], str):
                observations['summaries'] = [rml_data['summaries']]
        
        return observations
    
    def understand(self, observations: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Step 2: Understand - Process and categorize observations"""
        understanding = {
            'concept_count': len(observations['concepts']),
            'entity_count': len(observations['entities']),
            'triple_count': len(observations['triples']),
            'emotion_profile': {},
            'intent_profile': {},
            'event_profile': {},
            'semantic_analysis': {}
        }
        
        # Analyze emotions
        for emotion in observations['emotions']:
            understanding['emotion_profile'][emotion] = understanding['emotion_profile'].get(emotion, 0) + 1
        
        # Analyze intents
        for intent in observations['intents']:
            understanding['intent_profile'][intent] = understanding['intent_profile'].get(intent, 0) + 1
        
        # Analyze events
        for event in observations['events']:
            understanding['event_profile'][event] = understanding['event_profile'].get(event, 0) + 1
        
        # Semantic analysis
        if observations['concepts']:
            understanding['semantic_analysis']['primary_concepts'] = observations['concepts'][:5]
            understanding['semantic_analysis']['concept_diversity'] = len(set(observations['concepts']))
        
        return understanding
    
    def relate(self, observations: Dict[str, Any], understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Relate - Build relationships and add to memory"""
        relationships = {
            'concepts_added': 0,
            'entities_added': 0,
            'triples_added': 0,
            'emotions_added': 0,
            'intents_added': 0,
            'events_added': 0,
            'vectors_added': 0
        }
        
        # Add concepts to memory
        for concept in observations['concepts']:
            self.memory.add_concept(concept, confidence=0.8, context="rml_processing")
            relationships['concepts_added'] += 1
        
        # Add entities to memory
        for entity in observations['entities']:
            self.memory.add_entity(entity, entity_type="extracted")
            relationships['entities_added'] += 1
        
        # Add triples to memory
        for triple in observations['triples']:
            if isinstance(triple, str):
                # Parse triple string if needed
                try:
                    triple_dict = eval(triple)
                    subject = triple_dict.get('subject', '')
                    predicate = triple_dict.get('predicate', '')
                    object_ = triple_dict.get('object', '')
                    self.memory.add_triple(subject, predicate, object_)
                    relationships['triples_added'] += 1
                except:
                    continue
            elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
                self.memory.add_triple(triple[0], triple[1], triple[2])
                relationships['triples_added'] += 1
        
        # Add emotions to memory
        for emotion in observations['emotions']:
            self.memory.add_emotion(emotion)
            relationships['emotions_added'] += 1
        
        # Add intents to memory
        for intent in observations['intents']:
            self.memory.add_intent(intent)
            relationships['intents_added'] += 1
        
        # Add events to memory
        for event in observations['events']:
            self.memory.add_event(event)
            relationships['events_added'] += 1
        
        # Add vectors to memory
        for i, vector in enumerate(observations['vectors']):
            if isinstance(vector, list) and len(vector) > 0:
                self.memory.add_vector(vector)
                relationships['vectors_added'] += 1
        
        return relationships
    
    def respond(self, query: str = "", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Step 4: Respond - Generate intelligent response based on memory"""
        # Get relevant concepts
        relevant_concepts = self.memory.get_relevant_concepts(query, top_k=15)
        
        # Get semantic context
        semantic_context = self.memory.get_semantic_context(query)
        
        # Build concept graph
        concept_names = [concept for concept, score in relevant_concepts]
        concept_graph = self.memory.get_concept_graph(concept_names)
        
        # Generate response
        if query:
            response = self._generate_query_response(query, relevant_concepts, semantic_context)
        else:
            response = self._generate_summary_response(relevant_concepts, semantic_context)
        
        return {
            'response': response,
            'relevant_concepts': relevant_concepts,
            'concept_graph': concept_graph,
            'semantic_context': semantic_context,
            'memory_stats': {
                'total_concepts': len(self.memory.concepts),
                'total_entities': len(self.memory.entities),
                'total_triples': len(self.memory.triples),
                'total_vectors': len(self.memory.vector_memory)
            }
        }
    
    def _generate_query_response(self, query: str, relevant_concepts: List[Tuple[str, float]], context: Dict[str, Any]) -> str:
        """Generate response to a specific query"""
        top_concepts = [concept for concept, score in relevant_concepts[:5]]
        
        # Analyze query type
        if "how" in query.lower():
            response = f"Based on the concepts {', '.join(top_concepts)}, the process involves multiple interconnected elements."
        elif "what" in query.lower():
            response = f"The key concepts are: {', '.join(top_concepts)}."
        elif "why" in query.lower():
            response = f"The concepts {', '.join(top_concepts[:3])} are interconnected and form the basis for understanding this topic."
        else:
            response = f"The main concepts related to this are: {', '.join(top_concepts)}."
        
        # Add semantic context
        if context['emotions']:
            dominant_emotion = max(context['emotions'].items(), key=lambda x: x[1])[0]
            response += f" The emotional context is primarily {dominant_emotion}."
        
        if context['intents']:
            dominant_intent = max(context['intents'].items(), key=lambda x: x[1])[0]
            response += f" The primary intent is {dominant_intent}."
        
        return response
    
    def _generate_summary_response(self, relevant_concepts: List[Tuple[str, float]], context: Dict[str, Any]) -> str:
        """Generate summary response"""
        top_concepts = [concept for concept, score in relevant_concepts[:8]]
        
        response = f"This knowledge base contains concepts like {', '.join(top_concepts)}. "
        response += f"These concepts are interconnected and form a comprehensive knowledge structure."
        
        # Add context information
        if context['emotions']:
            response += f" The emotional profile includes: {', '.join(context['emotions'].keys())}."
        
        if context['intents']:
            response += f" The intent profile includes: {', '.join(context['intents'].keys())}."
        
        return response
    
    def process_rml_data(self, rml_data: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """Complete RML processing pipeline: Observe → Understand → Relate → Respond"""
        
        self.logger.info(f"🧠 Processing RML data with {len(rml_data)} fields")
        
        # Step 1: Observe
        observations = self.observe(rml_data)
        
        # Step 2: Understand
        understanding = self.understand(observations)
        
        # Step 3: Relate
        relationships = self.relate(observations, understanding)
        
        # Step 4: Respond
        response_data = self.respond(query, understanding)
        
        # Apply memory decay
        self.memory.decay_memory()
        
        # Save memory if requested
        if self.config.save_memory_graphs:
            memory_path = os.path.join(self.config.output_dir, "rml_memory.json")
            self.memory.save_memory(memory_path)
        
        return {
            'rml_data': rml_data,
            'query': query,
            'observations': observations,
            'understanding': understanding,
            'relationships': relationships,
            'response': response_data['response'],
            'relevant_concepts': response_data['relevant_concepts'],
            'concept_graph': response_data['concept_graph'],
            'semantic_context': response_data['semantic_context'],
            'memory_stats': response_data['memory_stats']
        }

def main():
    """Main function for testing the comprehensive RML processor"""
    
    # Configuration
    config = RMLConfig(
        output_dir="output/rml_comprehensive",
        save_memory_graphs=True,
        log_level="INFO"
    )
    
    # Initialize processor
    processor = ComprehensiveRMLProcessor(config)
    
    print("🧠 Testing Comprehensive RML Processor")
    print("="*60)
    
    try:
        # Test with converted RML data
        converted_rml_file = "data/converted_rml/complete_rml/rml_data.jsonl"
        if os.path.exists(converted_rml_file):
            with open(converted_rml_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 2:  # Process first 2 lines
                        break
                    rml_data = json.loads(line.strip())
                    
                    result = processor.process_rml_data(rml_data, "What are the main concepts and their relationships?")
                    
                    print(f"📄 RML Data: {len(str(rml_data))} characters")
                    print(f"🔍 Observations: {len(result['observations']['concepts'])} concepts, {len(result['observations']['entities'])} entities")
                    print(f"🧠 Understanding: {result['understanding']['concept_count']} concepts analyzed")
                    print(f"🔗 Relationships: {result['relationships']['concepts_added']} concepts added to memory")
                    print(f"🤖 Response: {result['response']}")
                    print(f"📊 Memory Stats: {result['memory_stats']}")
                    print("-" * 40)
        
        print(f"\n✅ Comprehensive RML processor working successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 