#!/usr/bin/env python3
"""
RML Encoder-Decoder Pipeline
Final Phase: E5-Mistral Encoder → RML Memory → Phi-3 Decoder

This module implements the complete RML (Resonant Memory Layer) system:
1. E5-Mistral encoder for semantic understanding
2. RML memory graph construction
3. Phi-3 decoder for natural language generation
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

# Optional: For advanced memory management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@dataclass
class RMLConfig:
    """Configuration for RML Encoder-Decoder system"""
    
    # Model paths
    encoder_model: str = "intfloat/e5-mistral-7b-instruct"
    decoder_model: str = "microsoft/phi-3-medium-128k-instruct"
    
    # Memory settings
    max_memory_size: int = 10000  # Max concepts in memory
    memory_decay_rate: float = 0.95  # How quickly old memories fade
    
    # Processing settings
    batch_size: int = 8
    max_length: int = 2048
    device: str = "auto"  # auto, cuda, cpu
    
    # RML specific
    min_concept_confidence: float = 0.7
    max_concepts_per_input: int = 50
    enable_symbolic_reasoning: bool = True
    
    # Output settings
    output_dir: str = "output/rml_pipeline"
    save_memory_graphs: bool = True
    log_level: str = "INFO"

class RMLMemory:
    """Resonant Memory Layer - stores and manages concept graphs"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.concept_graph = defaultdict(dict)
        self.memory_weights = defaultdict(float)
        self.access_count = defaultdict(int)
        self.last_accessed = defaultdict(float)
        
    def add_concept(self, concept: str, attributes: Dict[str, Any], confidence: float = 1.0):
        """Add a concept to the memory with its attributes"""
        if confidence >= self.config.min_concept_confidence:
            self.concept_graph[concept].update(attributes)
            self.memory_weights[concept] = max(self.memory_weights[concept], confidence)
            self.access_count[concept] += 1
            
    def add_relationship(self, source: str, relation: str, target: str, confidence: float = 1.0):
        """Add a relationship between concepts"""
        if source in self.concept_graph and target in self.concept_graph:
            if 'relationships' not in self.concept_graph[source]:
                self.concept_graph[source]['relationships'] = []
            
            self.concept_graph[source]['relationships'].append({
                'relation': relation,
                'target': target,
                'confidence': confidence
            })
            
    def get_relevant_concepts(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve most relevant concepts for a query"""
        # Simple relevance scoring based on access count and recency
        scored_concepts = []
        for concept, count in self.access_count.items():
            score = count * self.memory_weights[concept]
            scored_concepts.append((concept, score))
        
        # Sort by score and return top_k
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in scored_concepts[:top_k]]
    
    def get_concept_graph(self, concepts: List[str]) -> Dict[str, Any]:
        """Get a subgraph containing specified concepts and their relationships"""
        subgraph = {}
        for concept in concepts:
            if concept in self.concept_graph:
                subgraph[concept] = self.concept_graph[concept].copy()
        return subgraph
    
    def save_memory(self, filepath: str):
        """Save memory to disk"""
        memory_data = {
            'concept_graph': dict(self.concept_graph),
            'memory_weights': dict(self.memory_weights),
            'access_count': dict(self.access_count)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self, filepath: str):
        """Load memory from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            self.concept_graph = defaultdict(dict, memory_data['concept_graph'])
            self.memory_weights = defaultdict(float, memory_data['memory_weights'])
            self.access_count = defaultdict(int, memory_data['access_count'])

class RMLEncoder:
    """E5-Mistral based encoder for semantic understanding"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.device = self._get_device()
        
        # Load E5-Mistral model
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
        self.model = AutoModel.from_pretrained(config.encoder_model)
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"✅ E5-Mistral encoder loaded on {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best device to use"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to dense embeddings"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_length,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output for sentence-level embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        
        return embeddings
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts from text using embeddings"""
        embeddings = self.encode_text(text)
        
        # Simple concept extraction based on noun phrases and key terms
        # In a full implementation, you'd use NER, dependency parsing, etc.
        concepts = []
        
        # Split text into sentences and extract key terms
        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
                
            # Extract potential concepts (simplified)
            words = sentence.split()
            for word in words:
                word = word.strip('.,!?;:').lower()
                if len(word) > 3 and word.isalpha():
                    concepts.append({
                        'concept': word,
                        'sentence': sentence.strip(),
                        'position': i,
                        'confidence': 0.8,  # Simplified confidence
                        'type': 'noun'
                    })
        
        return concepts[:self.config.max_concepts_per_input]

class RMLDecoder:
    """Phi-3 based decoder for natural language generation"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.device = self._get_device()
        
        # Load Phi-3 model with quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.decoder_model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        logging.info(f"✅ Phi-3 decoder loaded on {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best device to use"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def build_prompt_from_graph(self, concept_graph: Dict[str, Any], query: str = "") -> str:
        """Build a prompt from concept graph for Phi-3"""
        
        # Convert concept graph to structured text
        graph_text = "Concept Graph:\n"
        for concept, attributes in concept_graph.items():
            graph_text += f"- {concept}: {attributes.get('type', 'concept')}\n"
            
            # Add relationships
            if 'relationships' in attributes:
                for rel in attributes['relationships']:
                    graph_text += f"  → {rel['relation']} → {rel['target']}\n"
        
        # Build the complete prompt
        if query:
            prompt = f"""Based on the following concept graph, answer the question: {query}

{graph_text}

Answer:"""
        else:
            prompt = f"""Based on the following concept graph, provide a comprehensive explanation:

{graph_text}

Explanation:"""
        
        return prompt
    
    def generate_response(self, concept_graph: Dict[str, Any], query: str = "", 
                         max_new_tokens: int = 256) -> str:
        """Generate natural language response from concept graph"""
        
        prompt = self.build_prompt_from_graph(concept_graph, query)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "Explanation:" in response:
            response = response.split("Explanation:")[-1].strip()
        
        return response

class RMLPipeline:
    """Complete RML Encoder-Decoder Pipeline"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.memory = RMLMemory(config)
        self.encoder = RMLEncoder(config)
        self.decoder = RMLDecoder(config)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.logger.info("🚀 RML Encoder-Decoder Pipeline initialized")
    
    def process_text(self, text: str, query: str = "") -> Dict[str, Any]:
        """Process text through the complete RML pipeline"""
        
        self.logger.info(f"📝 Processing text (length: {len(text)})")
        
        # Step 1: Encode and extract concepts
        concepts = self.encoder.extract_concepts(text)
        self.logger.info(f"🔍 Extracted {len(concepts)} concepts")
        
        # Step 2: Add to memory
        for concept_data in concepts:
            self.memory.add_concept(
                concept_data['concept'],
                {
                    'type': concept_data['type'],
                    'sentence': concept_data['sentence'],
                    'position': concept_data['position']
                },
                concept_data['confidence']
            )
        
        # Step 3: Get relevant concepts for response
        relevant_concepts = self.memory.get_relevant_concepts(query if query else text)
        concept_graph = self.memory.get_concept_graph(relevant_concepts)
        
        # Step 4: Generate response
        response = self.decoder.generate_response(concept_graph, query)
        
        # Step 5: Save memory if enabled
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
            'memory_size': len(self.memory.concept_graph)
        }
    
    def process_rml_data(self, rml_file: str) -> List[Dict[str, Any]]:
        """Process existing RML data files"""
        
        self.logger.info(f"📁 Processing RML file: {rml_file}")
        
        results = []
        
        with open(rml_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        rml_data = json.loads(line.strip())
                        
                        # Extract text from RML data
                        text = self._extract_text_from_rml(rml_data)
                        if text:
                            result = self.process_text(text)
                            result['rml_data'] = rml_data
                            results.append(result)
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON at line {line_num}")
                        continue
        
        self.logger.info(f"✅ Processed {len(results)} RML records")
        return results
    
    def _extract_text_from_rml(self, rml_data: Dict[str, Any]) -> str:
        """Extract text content from RML data structure"""
        
        # Try different possible text fields
        text_fields = ['text', 'content', 'summary', 'description', 'input']
        
        for field in text_fields:
            if field in rml_data and rml_data[field]:
                return str(rml_data[field])
        
        # If no direct text field, try to reconstruct from concepts
        if 'concepts' in rml_data and rml_data['concepts']:
            concepts = rml_data['concepts']
            if isinstance(concepts, list):
                return " ".join([str(c) for c in concepts[:10]])  # First 10 concepts
            elif isinstance(concepts, dict):
                return " ".join([str(v) for v in concepts.values()][:10])
        
        return ""
    
    def interactive_mode(self):
        """Run interactive mode for testing"""
        
        print("🤖 RML Encoder-Decoder Interactive Mode")
        print("Type 'quit' to exit, 'memory' to see memory state")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n💬 Enter text (or command): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'memory':
                    print(f"📊 Memory contains {len(self.memory.concept_graph)} concepts")
                    for concept, attrs in list(self.memory.concept_graph.items())[:5]:
                        print(f"  - {concept}: {attrs.get('type', 'concept')}")
                    continue
                
                if user_input:
                    query = input("❓ Enter query (optional): ").strip()
                    
                    result = self.process_text(user_input, query)
                    
                    print(f"\n🤖 Response: {result['response']}")
                    print(f"📊 Extracted {len(result['extracted_concepts'])} concepts")
                    print(f"🧠 Memory size: {result['memory_size']} concepts")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")

def main():
    """Main function for testing the RML pipeline"""
    
    # Configuration
    config = RMLConfig(
        output_dir="output/rml_pipeline",
        save_memory_graphs=True,
        log_level="INFO"
    )
    
    # Initialize pipeline
    pipeline = RMLPipeline(config)
    
    # Test with sample text
    sample_text = """
    The brain stores memories by creating neural pathways. When we learn something new, 
    neurons form connections that strengthen with repeated use. This process is called 
    synaptic plasticity and is fundamental to how we remember information.
    """
    
    print("🧪 Testing RML Pipeline")
    print("="*50)
    
    result = pipeline.process_text(sample_text, "How does the brain store memories?")
    
    print(f"📝 Input: {sample_text.strip()}")
    print(f"❓ Query: How does the brain store memories?")
    print(f"🤖 Response: {result['response']}")
    print(f"📊 Concepts extracted: {len(result['extracted_concepts'])}")
    print(f"🧠 Memory size: {result['memory_size']} concepts")
    
    # Run interactive mode
    print("\n" + "="*50)
    pipeline.interactive_mode()

if __name__ == "__main__":
    main() 