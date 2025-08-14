#!/usr/bin/env python3
"""
RML Complete Pipeline - E5-Mistral Encoder + Phi-3 Decoder
Full encoder-decoder architecture for RML processing
"""

import os
import json
import logging
import gc
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Import transformers with memory optimization
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        BitsAndBytesConfig, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers")

@dataclass
class RMLPipelineConfig:
    """Configuration for complete RML pipeline"""
    
    # Model settings
    encoder_model_name: str = "intfloat/e5-mistral-7b-instruct"
    decoder_model_name: str = "microsoft/phi-3-mini-4k-instruct"  # Lighter version
    
    # Memory optimization
    use_quantization: bool = True
    quantization_bits: int = 4  # 4-bit quantization
    device_map: str = "auto"
    max_memory: Dict[str, str] = None
    
    # Processing settings
    max_input_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.7
    batch_size: int = 1
    
    # Output settings
    output_dir: str = "output/rml_pipeline"
    save_embeddings: bool = True
    save_responses: bool = True
    log_level: str = "INFO"

class RMLCompletePipeline:
    """Complete RML pipeline with E5-Mistral encoder and Phi-3 decoder"""
    
    def __init__(self, config: RMLPipelineConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize models
        self.encoder = None
        self.decoder = None
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        
        # Memory storage
        self.embeddings_cache = {}
        self.concept_memory = defaultdict(dict)
        
        self.logger.info("🧠 Initializing RML Complete Pipeline")
        self._load_models()
    
    def _load_models(self):
        """Load encoder and decoder models with memory optimization"""
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("❌ Transformers not available")
            return
        
        try:
            # Configure quantization for Mac M3 Pro (Apple Silicon)
            bnb_config = None
            if self.config.use_quantization:
                try:
                    # Try to use bitsandbytes if available and compatible
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True if self.config.quantization_bits == 4 else False,
                        load_in_8bit=True if self.config.quantization_bits == 8 else False,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Quantization not available: {e}. Using float16 instead.")
                    bnb_config = None
            
            # Set memory limits for Mac M3 Pro
            if self.config.max_memory is None:
                self.config.max_memory = {
                    "cpu": "32GB",
                    "mps": "8GB"  # Apple Silicon GPU
                }
            
            self.logger.info("🔧 Loading E5-Mistral encoder...")
            
            # Load encoder (E5-Mistral)
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(
                self.config.encoder_model_name,
                trust_remote_code=True
            )
            
            # Load encoder with Apple Silicon optimization
            load_kwargs = {
                'device_map': self.config.device_map,
                'torch_dtype': torch.float16,
                'trust_remote_code': True
            }
            
            if bnb_config:
                load_kwargs['quantization_config'] = bnb_config
            if self.config.max_memory:
                load_kwargs['max_memory'] = self.config.max_memory
            
            self.encoder = AutoModel.from_pretrained(
                self.config.encoder_model_name,
                **load_kwargs
            )
            
            self.logger.info("✅ E5-Mistral encoder loaded successfully")
            
            # Clear memory before loading decoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("🔧 Loading Phi-3 decoder...")
            
            # Load decoder (Phi-3)
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                self.config.decoder_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.decoder_tokenizer.pad_token is None:
                self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            
            # Load decoder with Apple Silicon optimization
            load_kwargs = {
                'device_map': self.config.device_map,
                'torch_dtype': torch.float16,
                'trust_remote_code': True
            }
            
            if bnb_config:
                load_kwargs['quantization_config'] = bnb_config
            if self.config.max_memory:
                load_kwargs['max_memory'] = self.config.max_memory
            
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.config.decoder_model_name,
                **load_kwargs
            )
            
            self.logger.info("✅ Phi-3 decoder loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using E5-Mistral to get semantic embeddings"""
        
        if self.encoder is None:
            raise ValueError("Encoder not loaded")
        
        try:
            # Prepare input for E5-Mistral
            # E5-Mistral expects specific formatting
            formatted_text = f"query: {text}"
            
            # Tokenize
            inputs = self.encoder_tokenizer(
                formatted_text,
                max_length=self.config.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to same device as model
            device = next(self.encoder.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                # Use mean pooling for sentence embeddings
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"❌ Error encoding text: {e}")
            raise
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling for sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def extract_concepts_from_embeddings(self, embeddings: np.ndarray, text: str) -> List[str]:
        """Extract concepts from embeddings using similarity"""
        
        # Simple concept extraction based on embedding patterns
        # In a real implementation, you'd use a concept extraction model
        
        # For now, extract key words from text
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']:
                word_freq[word] += 1
        
        # Get top concepts
        concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, freq in concepts[:10]]
    
    def decode_embeddings_to_text(self, embeddings: np.ndarray, prompt: str = "") -> str:
        """Decode embeddings back to text using Phi-3"""
        
        if self.decoder is None:
            raise ValueError("Decoder not loaded")
        
        try:
            # Create a prompt that includes the semantic context
            if not prompt:
                prompt = "Based on the semantic context, generate a natural language response:"
            
            # For now, we'll use the prompt directly since we can't directly feed embeddings to Phi-3
            # In a real implementation, you'd need to convert embeddings back to tokens or use them for retrieval
            
            # Tokenize prompt
            inputs = self.decoder_tokenizer(
                prompt,
                max_length=self.config.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to same device as model
            device = next(self.decoder.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.decoder.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.decoder_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error decoding embeddings: {e}")
            raise
    
    def process_rml_data(self, rml_data: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """Complete RML processing: Encode → Extract → Decode"""
        
        self.logger.info(f"🧠 Processing RML data with {len(rml_data)} fields")
        
        try:
            # Step 1: Encode the text using E5-Mistral
            text = rml_data.get('text', '')
            if not text:
                text = str(rml_data)  # Fallback to string representation
            
            self.logger.info(f"🔧 Encoding text (length: {len(text)})")
            embeddings = self.encode_text(text)
            
            # Step 2: Extract concepts from embeddings
            concepts = self.extract_concepts_from_embeddings(embeddings, text)
            
            # Step 3: Build concept graph
            concept_graph = {
                'concepts': concepts,
                'embeddings': embeddings.tolist(),
                'text': text
            }
            
            # Step 4: Generate response using Phi-3
            if query:
                prompt = f"Query: {query}\nContext: {text}\nConcepts: {', '.join(concepts)}\nResponse:"
            else:
                prompt = f"Based on the following text and concepts, provide a comprehensive response:\nText: {text}\nConcepts: {', '.join(concepts)}\nResponse:"
            
            response = self.decode_embeddings_to_text(embeddings, prompt)
            
            # Step 5: Build complete output
            result = {
                'rml_data': rml_data,
                'query': query,
                'embeddings': embeddings.tolist(),
                'concepts': concepts,
                'concept_graph': concept_graph,
                'response': response,
                'processing_info': {
                    'encoder_model': self.config.encoder_model_name,
                    'decoder_model': self.config.decoder_model_name,
                    'embedding_dimension': embeddings.shape[1],
                    'concept_count': len(concepts)
                }
            }
            
            # Save embeddings if requested
            if self.config.save_embeddings:
                embedding_path = os.path.join(self.config.output_dir, f"embeddings_{hash(text) % 1000000}.npy")
                np.save(embedding_path, embeddings)
                result['embedding_path'] = embedding_path
            
            # Save response if requested
            if self.config.save_responses:
                response_path = os.path.join(self.config.output_dir, f"response_{hash(text) % 1000000}.json")
                with open(response_path, 'w') as f:
                    json.dump(result, f, indent=2)
                result['response_path'] = response_path
            
            self.logger.info(f"✅ Processed: {len(concepts)} concepts, response length: {len(response)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in RML processing: {e}")
            raise
    
    def process_batch(self, rml_data_list: List[Dict[str, Any]], queries: List[str] = None) -> List[Dict[str, Any]]:
        """Process a batch of RML data"""
        
        results = []
        
        for i, rml_data in enumerate(rml_data_list):
            query = queries[i] if queries and i < len(queries) else ""
            
            try:
                result = self.process_rml_data(rml_data, query)
                results.append(result)
                
                # Clear memory after each batch item
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing batch item {i}: {e}")
                results.append({'error': str(e), 'rml_data': rml_data})
        
        return results
    
    def cleanup(self):
        """Clean up models and free memory"""
        if self.encoder:
            del self.encoder
        if self.decoder:
            del self.decoder
        if self.encoder_tokenizer:
            del self.encoder_tokenizer
        if self.decoder_tokenizer:
            del self.decoder_tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("🧹 Memory cleaned up")

def main():
    """Main function for testing the complete RML pipeline"""
    
    print("🧠 Testing RML Complete Pipeline (E5-Mistral + Phi-3)")
    print("="*70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not available. Install with:")
        print("pip install transformers torch bitsandbytes")
        return
    
    # Configuration optimized for Mac M3 Pro
    config = RMLPipelineConfig(
        encoder_model_name="intfloat/e5-mistral-7b-instruct",
        decoder_model_name="microsoft/phi-3-mini-4k-instruct",  # Lighter version
        use_quantization=False,  # Disabled for Mac M3 Pro
        quantization_bits=4,
        device_map="auto",
        max_input_length=512,  # Further reduced for memory efficiency
        max_new_tokens=64,  # Reduced for faster processing
        temperature=0.7,
        output_dir="output/rml_pipeline",
        save_embeddings=True,
        save_responses=True,
        log_level="INFO"
    )
    
    # Initialize pipeline
    pipeline = None
    try:
        pipeline = RMLCompletePipeline(config)
        
        # Test data
        test_rml_data = {
            'text': 'Cloud computing infrastructure supports scalable applications. Amazon Web Services provides reliable cloud solutions for businesses worldwide.',
            'concepts': ['cloud', 'computing', 'infrastructure'],
            'entities': ['Amazon Web Services'],
            'emotions': ['positive'],
            'intents': ['inform']
        }
        
        test_query = "Explain the benefits of cloud computing"
        
        print("🔧 Testing complete pipeline:")
        print(f"📄 Input text: {test_rml_data['text'][:100]}...")
        print(f"❓ Query: {test_query}")
        
        # Process
        result = pipeline.process_rml_data(test_rml_data, test_query)
        
        print(f"\n🔍 Extracted concepts: {result['concepts'][:5]}")
        print(f"📊 Embedding dimension: {result['processing_info']['embedding_dimension']}")
        print(f"🤖 Generated response: {result['response'][:200]}...")
        
        print(f"\n✅ Complete RML pipeline working successfully!")
        print(f"📁 Output saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 If you encounter memory issues, try:")
        print("1. Reduce max_input_length to 512")
        print("2. Use phi-3-mini instead of phi-3-medium")
        print("3. Set use_quantization=True")
        
    finally:
        if pipeline:
            pipeline.cleanup()

if __name__ == "__main__":
    main() 