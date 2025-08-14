#!/usr/bin/env python3
"""
RML Encoder-Decoder Pipeline - Memory Safe Version
Prevents crashes with quantized models and efficient memory management
"""

import os
import json
import torch
import logging
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict

# Transformers imports with quantization
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

@dataclass
class RMLConfig:
    """Configuration for RML Encoder-Decoder system"""
    
    # Model paths
    encoder_model: str = "intfloat/e5-mistral-7b-instruct"
    decoder_model: str = "microsoft/phi-3-medium-128k-instruct"
    
    # Memory settings
    max_memory_size: int = 10000
    memory_decay_rate: float = 0.95
    
    # Processing settings - REDUCED for safety
    batch_size: int = 1  # Reduced from 8
    max_length: int = 1024  # Reduced from 2048
    device: str = "auto"
    
    # RML specific
    min_concept_confidence: float = 0.7
    max_concepts_per_input: int = 20  # Reduced from 50
    enable_symbolic_reasoning: bool = True
    
    # Output settings
    output_dir: str = "output/rml_pipeline"
    save_memory_graphs: bool = True
    log_level: str = "INFO"
    
    # Memory safety settings
    use_quantization: bool = True
    load_models_separately: bool = True
    max_memory_gb: float = 60.0  # Crash if memory exceeds this

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def check_memory_safety(max_gb: float = 60.0):
    """Check if memory usage is safe"""
    current_gb = monitor_memory()
    if current_gb > max_gb:
        raise MemoryError(f"Memory usage {current_gb:.1f}GB exceeds limit {max_gb}GB")
    return current_gb

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
        scored_concepts = []
        for concept, count in self.access_count.items():
            score = count * self.memory_weights[concept]
            scored_concepts.append((concept, score))
        
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
    """E5-Mistral based encoder with memory safety"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        
        logging.info(f"🔧 Initializing E5-Mistral encoder on {self.device}")
        self._load_encoder()
        
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
    
    def _load_encoder(self):
        """Load encoder with memory safety"""
        try:
            logging.info("📥 Loading E5-Mistral tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_model)
            
            logging.info("📥 Loading E5-Mistral model...")
            memory_before = monitor_memory()
            
            # Load model with quantization if enabled
            if self.config.use_quantization and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
                self.model = AutoModel.from_pretrained(
                    self.config.encoder_model,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                self.model = AutoModel.from_pretrained(self.config.encoder_model)
                self.model.to(self.device)
            
            self.model.eval()
            
            memory_after = monitor_memory()
            logging.info(f"✅ E5-Mistral loaded. Memory: {memory_before:.1f}GB → {memory_after:.1f}GB (+{memory_after-memory_before:.1f}GB)")
            
        except Exception as e:
            logging.error(f"❌ Failed to load E5-Mistral: {e}")
            raise
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to dense embeddings with memory safety"""
        check_memory_safety(self.config.max_memory_gb)
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_length,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts from text using embeddings"""
        embeddings = self.encode_text(text)
        
        # Simplified concept extraction to reduce memory usage
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
    
    def cleanup(self):
        """Clean up encoder to free memory"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("🧹 E5-Mistral encoder cleaned up")

class RMLDecoder:
    """Phi-3 based decoder with memory safety"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        
        logging.info(f"🔧 Initializing Phi-3 decoder on {self.device}")
        self._load_decoder()
        
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
    
    def _load_decoder(self):
        """Load decoder with memory safety"""
        try:
            logging.info("📥 Loading Phi-3 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.decoder_model)
            
            logging.info("📥 Loading Phi-3 model...")
            memory_before = monitor_memory()
            
            # Always use quantization for Phi-3 to save memory
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.decoder_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            memory_after = monitor_memory()
            logging.info(f"✅ Phi-3 loaded. Memory: {memory_before:.1f}GB → {memory_after:.1f}GB (+{memory_after-memory_before:.1f}GB)")
            
        except Exception as e:
            logging.error(f"❌ Failed to load Phi-3: {e}")
            raise
    
    def build_prompt_from_graph(self, concept_graph: Dict[str, Any], query: str = "") -> str:
        """Build a prompt from concept graph for Phi-3"""
        graph_text = "Concept Graph:\n"
        for concept, attributes in concept_graph.items():
            graph_text += f"- {concept}: {attributes.get('type', 'concept')}\n"
            
            if 'relationships' in attributes:
                for rel in attributes['relationships']:
                    graph_text += f"  → {rel['relation']} → {rel['target']}\n"
        
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
                         max_new_tokens: int = 128) -> str:  # Reduced from 256
        """Generate natural language response from concept graph"""
        check_memory_safety(self.config.max_memory_gb)
        
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
        
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "Explanation:" in response:
            response = response.split("Explanation:")[-1].strip()
        
        return response
    
    def cleanup(self):
        """Clean up decoder to free memory"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("🧹 Phi-3 decoder cleaned up")

class RMLPipeline:
    """Complete RML Encoder-Decoder Pipeline with memory safety"""
    
    def __init__(self, config: RMLConfig):
        self.config = config
        self.memory = RMLMemory(config)
        self.encoder = None
        self.decoder = None
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.logger.info("🚀 RML Encoder-Decoder Pipeline initialized (memory safe)")
    
    def _load_encoder_if_needed(self):
        """Load encoder only when needed"""
        if self.encoder is None:
            self.encoder = RMLEncoder(self.config)
    
    def _load_decoder_if_needed(self):
        """Load decoder only when needed"""
        if self.decoder is None:
            self.decoder = RMLDecoder(self.config)
    
    def process_text(self, text: str, query: str = "") -> Dict[str, Any]:
        """Process text through the complete RML pipeline with memory safety"""
        
        self.logger.info(f"📝 Processing text (length: {len(text)})")
        self.logger.info(f"💾 Memory before processing: {monitor_memory():.1f}GB")
        
        try:
            # Step 1: Load encoder and extract concepts
            self._load_encoder_if_needed()
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
            
            # Step 4: Load decoder and generate response
            self._load_decoder_if_needed()
            response = self.decoder.generate_response(concept_graph, query)
            
            # Step 5: Save memory if enabled
            if self.config.save_memory_graphs:
                memory_path = os.path.join(self.config.output_dir, "rml_memory.json")
                self.memory.save_memory(memory_path)
            
            self.logger.info(f"💾 Memory after processing: {monitor_memory():.1f}GB")
            
            return {
                'input_text': text,
                'query': query,
                'extracted_concepts': concepts,
                'relevant_concepts': relevant_concepts,
                'concept_graph': concept_graph,
                'response': response,
                'memory_size': len(self.memory.concept_graph)
            }
            
        except MemoryError as e:
            self.logger.error(f"❌ Memory error: {e}")
            self.cleanup()
            raise
        except Exception as e:
            self.logger.error(f"❌ Processing error: {e}")
            raise
    
    def cleanup(self):
        """Clean up all models to free memory"""
        if self.encoder:
            self.encoder.cleanup()
            self.encoder = None
        if self.decoder:
            self.decoder.cleanup()
            self.decoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("🧹 All models cleaned up")

def main():
    """Main function for testing the memory-safe RML pipeline"""
    
    # Configuration with memory safety
    config = RMLConfig(
        output_dir="output/rml_pipeline_safe",
        save_memory_graphs=True,
        log_level="INFO",
        use_quantization=True,
        load_models_separately=True,
        max_memory_gb=60.0
    )
    
    # Initialize pipeline
    pipeline = RMLPipeline(config)
    
    # Test with sample text
    sample_text = """
    The brain stores memories by creating neural pathways. When we learn something new, 
    neurons form connections that strengthen with repeated use.
    """
    
    print("🧪 Testing Memory-Safe RML Pipeline")
    print("="*50)
    
    try:
        result = pipeline.process_text(sample_text, "How does the brain store memories?")
        
        print(f"📝 Input: {sample_text.strip()}")
        print(f"❓ Query: How does the brain store memories?")
        print(f"🤖 Response: {result['response']}")
        print(f"📊 Concepts extracted: {len(result['extracted_concepts'])}")
        print(f"🧠 Memory size: {result['memory_size']} concepts")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main() 