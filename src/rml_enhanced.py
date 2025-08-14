#!/usr/bin/env python3
"""
Enhanced RML Processor - Combines Lightweight Processing with Optional Model Integration
Perfect for M3 Pro Mac with 18GB RAM
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

# Optional imports for model integration
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("⚠️ llama-cpp-python not available. Using lightweight mode only.")

@dataclass
class RMLConfig:
    """Configuration for enhanced RML processor"""
    
    # Processing settings
    max_concepts_per_input: int = 20
    min_concept_confidence: float = 0.7
    
    # Model settings
    use_models: bool = False
    model_path: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.7
    
    # Output settings
    output_dir: str = "output/rml_enhanced"
    save_memory_graphs: bool = True
    log_level: str = "INFO"

class RMLMemory:
    """Enhanced Resonant Memory Layer"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.concept_graph = defaultdict(dict)
        self.memory_weights = defaultdict(float)
        self.access_count = defaultdict(int)
        self.entity_relationships = defaultdict(list)
        
    def add_concept(self, concept: str, attributes: Dict[str, Any], confidence: float = 1.0):
        """Add a concept to the memory"""
        if confidence >= self.config.min_concept_confidence:
            self.concept_graph[concept].update(attributes)
            self.memory_weights[concept] = max(self.memory_weights[concept], confidence)
            self.access_count[concept] += 1
            
    def add_relationship(self, source: str, relation: str, target: str, confidence: float = 1.0):
        """Add a relationship between concepts"""
        if source in self.concept_graph and target in self.concept_graph:
            relationship = {
                'relation': relation,
                'target': target,
                'confidence': confidence,
                'source_type': self.concept_graph[source].get('type', 'unknown'),
                'target_type': self.concept_graph[target].get('type', 'unknown')
            }
            
            self.entity_relationships[source].append(relationship)
            
    def get_relevant_concepts(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve most relevant concepts with enhanced scoring"""
        scored_concepts = []
        
        for concept, count in self.access_count.items():
            # Enhanced scoring: access count * weight * relationship bonus
            base_score = count * self.memory_weights[concept]
            
            # Bonus for concepts with relationships
            relationship_bonus = len(self.entity_relationships[concept]) * 0.1
            
            # Query relevance bonus
            query_bonus = 0
            if query.lower() in concept.lower():
                query_bonus = 0.5
            
            total_score = base_score + relationship_bonus + query_bonus
            scored_concepts.append((concept, total_score))
        
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in scored_concepts[:top_k]]
    
    def get_concept_graph(self, concepts: List[str]) -> Dict[str, Any]:
        """Get a subgraph containing specified concepts with relationships"""
        subgraph = {}
        for concept in concepts:
            if concept in self.concept_graph:
                concept_data = self.concept_graph[concept].copy()
                # Add relationships
                if concept in self.entity_relationships:
                    concept_data['relationships'] = self.entity_relationships[concept]
                subgraph[concept] = concept_data
        return subgraph
    
    def save_memory(self, filepath: str):
        """Save memory to disk"""
        memory_data = {
            'concept_graph': dict(self.concept_graph),
            'memory_weights': dict(self.memory_weights),
            'access_count': dict(self.access_count),
            'entity_relationships': dict(self.entity_relationships)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self, filepath: str):
        """Load memory from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            self.concept_graph = defaultdict(dict, memory_data.get('concept_graph', {}))
            self.memory_weights = defaultdict(float, memory_data.get('memory_weights', {}))
            self.access_count = defaultdict(int, memory_data.get('access_count', {}))
            self.entity_relationships = defaultdict(list, memory_data.get('entity_relationships', {}))

class EnhancedRMLProcessor:
    """Enhanced RML processor with optional model integration"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.memory = RMLMemory(config)
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model if requested
        if config.use_models and LLAMA_AVAILABLE and config.model_path:
            self._load_model()
        
        self.logger.info("🚀 Enhanced RML Processor initialized")
    
    def _load_model(self):
        """Load the language model"""
        try:
            if os.path.exists(self.config.model_path):
                self.model = Llama(
                    model_path=self.config.model_path,
                    n_ctx=2048,
                    n_threads=8,  # Optimized for M3 Pro
                    n_gpu_layers=0  # CPU only for stability
                )
                self.logger.info(f"✅ Model loaded: {self.config.model_path}")
            else:
                self.logger.warning(f"⚠️ Model not found: {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            self.model = None
    
    def extract_concepts_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts from text using enhanced NLP"""
        concepts = []
        sentences = text.split('.')
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
                
            words = sentence.split()
            for word in words:
                word = word.strip('.,!?;:').lower()
                if len(word) > 3 and word.isalpha() and len(concepts) < self.config.max_concepts_per_input:
                    concepts.append({
                        'concept': word,
                        'sentence': sentence.strip(),
                        'position': i,
                        'confidence': 0.8,
                        'type': 'noun'
                    })
        
        return concepts
    
    def extract_from_rml_data(self, rml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract concepts from existing RML data with enhanced processing"""
        concepts = []
        
        # Handle entity extraction data structure
        if 'text' in rml_data and 'type' in rml_data:
            # This is entity extraction data
            concept_data = {
                'concept': rml_data['text'],
                'type': rml_data['type'],
                'confidence': rml_data.get('confidence', 0.8),
                'source': 'entity_extraction',
                'position': rml_data.get('position', 0),
                'document_id': rml_data.get('document_id', 0)
            }
            
            # Enhanced: Add semantic type classification
            if rml_data['type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                concept_data['semantic_type'] = 'entity'
            elif rml_data['type'] in ['DATE', 'TIME']:
                concept_data['semantic_type'] = 'temporal'
            else:
                concept_data['semantic_type'] = 'concept'
            
            concepts.append(concept_data)
        
        # Extract from concepts field (if exists)
        elif 'concepts' in rml_data and rml_data['concepts']:
            if isinstance(rml_data['concepts'], list):
                for concept in rml_data['concepts']:
                    if isinstance(concept, str):
                        concepts.append({
                            'concept': concept,
                            'type': 'concept',
                            'confidence': 0.9,
                            'source': 'rml_concepts'
                        })
            elif isinstance(rml_data['concepts'], dict):
                for key, value in rml_data['concepts'].items():
                    concepts.append({
                        'concept': str(value),
                        'type': 'concept',
                        'confidence': 0.9,
                        'source': 'rml_concepts'
                    })
        
        # Extract from entities field (if exists)
        elif 'entities' in rml_data and rml_data['entities']:
            if isinstance(rml_data['entities'], list):
                for entity in rml_data['entities']:
                    if isinstance(entity, str):
                        concepts.append({
                            'concept': entity,
                            'type': 'entity',
                            'confidence': 0.85,
                            'source': 'rml_entities'
                        })
        
        return concepts[:self.config.max_concepts_per_input]
    
    def generate_response_with_model(self, concept_graph: Dict[str, Any], query: str = "") -> str:
        """Generate response using the language model"""
        if not self.model:
            return self.generate_simple_response(concept_graph, query)
        
        try:
            # Build context from concept graph
            concepts = list(concept_graph.keys())
            context = f"Concepts: {', '.join(concepts[:10])}\n"
            
            if query:
                prompt = f"Based on these concepts: {context}\nQuestion: {query}\nAnswer:"
            else:
                prompt = f"Based on these concepts: {context}\nGenerate a coherent explanation:"
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                stop=["\n\n", "Question:", "Answer:"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            self.logger.error(f"❌ Model generation failed: {e}")
            return self.generate_simple_response(concept_graph, query)
    
    def generate_simple_response(self, concept_graph: Dict[str, Any], query: str = "") -> str:
        """Generate response from concept graph (fallback method)"""
        concepts = list(concept_graph.keys())
        
        if not concepts:
            return "No concepts found to generate a response."
        
        if query:
            # Try to answer the query
            if "how" in query.lower():
                response = f"Based on the concepts {', '.join(concepts[:3])}, the process involves multiple interconnected elements."
            elif "what" in query.lower():
                response = f"The key concepts are: {', '.join(concepts[:5])}."
            elif "why" in query.lower():
                response = f"The concepts {', '.join(concepts[:3])} are interconnected and form the basis for understanding this topic."
            else:
                response = f"The main concepts related to this are: {', '.join(concepts[:5])}."
        else:
            # Generate explanation
            response = f"This contains concepts like {', '.join(concepts[:5])}. "
            response += f"These concepts are interconnected and form a knowledge structure."
        
        return response
    
    def process_text(self, text: str, query: str = "") -> Dict[str, Any]:
        """Process text through the enhanced RML pipeline"""
        
        self.logger.info(f"📝 Processing text (length: {len(text)})")
        
        # Extract concepts
        concepts = self.extract_concepts_from_text(text)
        self.logger.info(f"🔍 Extracted {len(concepts)} concepts")
        
        # Add to memory
        for concept_data in concepts:
            self.memory.add_concept(
                concept_data['concept'],
                {
                    'type': concept_data['type'],
                    'sentence': concept_data.get('sentence', ''),
                    'position': concept_data.get('position', 0)
                },
                concept_data['confidence']
            )
        
        # Get relevant concepts
        relevant_concepts = self.memory.get_relevant_concepts(query if query else text)
        concept_graph = self.memory.get_concept_graph(relevant_concepts)
        
        # Generate response
        if self.config.use_models and self.model:
            response = self.generate_response_with_model(concept_graph, query)
        else:
            response = self.generate_simple_response(concept_graph, query)
        
        # Save memory
        if self.config.save_memory_graphs:
            memory_path = os.path.join(self.config.output_dir, "rml_memory.json")
            self.memory.save_memory(memory_path)
        
        return {
            'input_text': text,
            'query': query,
            'extracted_concepts': concepts,
            'relevant_concepts': relevant_concepts,
            'concept_graph': concept_graph,
            'response': response,
            'memory_size': len(self.memory.concept_graph),
            'model_used': self.model is not None
        }
    
    def process_rml_data(self, rml_data: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """Process existing RML data with enhanced features"""
        
        self.logger.info(f"📝 Processing RML data")
        
        # Extract concepts from RML data
        concepts = self.extract_from_rml_data(rml_data)
        self.logger.info(f"🔍 Extracted {len(concepts)} concepts from RML data")
        
        # Add to memory
        for concept_data in concepts:
            self.memory.add_concept(
                concept_data['concept'],
                {
                    'type': concept_data['type'],
                    'source': concept_data.get('source', 'rml_data'),
                    'semantic_type': concept_data.get('semantic_type', 'concept')
                },
                concept_data['confidence']
            )
        
        # Get relevant concepts
        relevant_concepts = self.memory.get_relevant_concepts(query if query else "general")
        concept_graph = self.memory.get_concept_graph(relevant_concepts)
        
        # Generate response
        if self.config.use_models and self.model:
            response = self.generate_response_with_model(concept_graph, query)
        else:
            response = self.generate_simple_response(concept_graph, query)
        
        # Save memory
        if self.config.save_memory_graphs:
            memory_path = os.path.join(self.config.output_dir, "rml_memory.json")
            self.memory.save_memory(memory_path)
        
        return {
            'rml_data': rml_data,
            'query': query,
            'extracted_concepts': concepts,
            'relevant_concepts': relevant_concepts,
            'concept_graph': concept_graph,
            'response': response,
            'memory_size': len(self.memory.concept_graph),
            'model_used': self.model is not None
        }

def main():
    """Main function for testing the enhanced RML processor"""
    
    # Configuration
    config = RMLConfig(
        output_dir="output/rml_enhanced",
        save_memory_graphs=True,
        log_level="INFO",
        use_models=False,  # Start with lightweight mode
        model_path="models/Phi-3-mini-4k-instruct-q4_0.gguf"  # Will be used if available
    )
    
    # Initialize processor
    processor = EnhancedRMLProcessor(config)
    
    # Test with sample text
    sample_text = """
    The brain stores memories by creating neural pathways. When we learn something new, 
    neurons form connections that strengthen with repeated use. This process is called 
    synaptic plasticity and is fundamental to how we remember information.
    """
    
    print("🧪 Testing Enhanced RML Processor")
    print("="*50)
    
    try:
        # Test 1: Process text
        result = processor.process_text(sample_text, "How does the brain store memories?")
        
        print(f"📝 Input: {sample_text.strip()}")
        print(f"❓ Query: How does the brain store memories?")
        print(f"🤖 Response: {result['response']}")
        print(f"📊 Concepts extracted: {len(result['extracted_concepts'])}")
        print(f"🧠 Memory size: {result['memory_size']} concepts")
        print(f"🤖 Model used: {result['model_used']}")
        
        # Show some concepts
        print(f"\n🔍 Sample concepts:")
        for i, concept in enumerate(result['extracted_concepts'][:5]):
            print(f"  {i+1}. {concept['concept']} ({concept['type']})")
        
        # Test 2: Process existing RML data
        print(f"\n{'='*50}")
        print("🧪 Testing with existing RML data")
        
        # Find a sample RML file
        sample_rml_file = "data/python_c4_final_backup_20250731_043743/concepts_batch_1753912494.jsonl"
        if os.path.exists(sample_rml_file):
            with open(sample_rml_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 1:  # Process first line only
                        break
                    rml_data = json.loads(line.strip())
                    
                    result2 = processor.process_rml_data(rml_data, "What are the main concepts?")
                    
                    print(f"📄 RML Data: {len(str(rml_data))} characters")
                    print(f"🤖 Response: {result2['response']}")
                    print(f"📊 Concepts extracted: {len(result2['extracted_concepts'])}")
                    print(f"🧠 Memory size: {result2['memory_size']} concepts")
                    print(f"🤖 Model used: {result2['model_used']}")
                    break
        
        print(f"\n✅ Enhanced RML processor working successfully!")
        
        if not LLAMA_AVAILABLE:
            print(f"\n💡 To enable model integration, install: pip3 install llama-cpp-python")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 