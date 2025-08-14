"""
Core RML System Implementation
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from .memory import MemoryStore
from .config import RMLConfig


@dataclass
class RMLResponse:
    """Response from RML system with answer and sources"""
    answer: str
    sources: List[str]
    response_ms: float
    confidence: float = 1.0


class RMLEncoder:
    """E5-based encoder for semantic understanding"""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "auto"):
        self.device = self._get_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def _get_device(self, device: str) -> str:
        """Auto-detect best available device"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        try:
            embedding = self.model.encode([query], convert_to_tensor=True)
            return embedding.cpu().numpy()
        except Exception as e:
            print(f"Error encoding query: {e}")
            return np.zeros((1, 768))
    
    def encode_entries(self, entries: List[Dict[str, Any]], batch_size: int = 8) -> np.ndarray:
        """Encode multiple entries in batches"""
        texts = []
        for entry in entries:
            text = self._extract_text(entry)
            if text:
                texts.append(text)
        
        if not texts:
            return np.array([])
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding entries"):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
                embeddings.append(batch_embeddings.cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e) and self.device == "mps":
                    print("MPS OOM, falling back to CPU")
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
                    embeddings.append(batch_embeddings.cpu().numpy())
                else:
                    raise e
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def _extract_text(self, entry: Dict[str, Any]) -> str:
        """Extract text content from entry"""
        for field in ['text', 'content', 'body', 'chunk', 'summary', 'title']:
            if field in entry and entry[field]:
                return str(entry[field])
        return ""


class RMLDecoder:
    """Phi-based decoder for natural language generation"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
    
    def _get_device(self, device: str) -> str:
        """Auto-detect best available device"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cpu"  # Force CPU for decoder to avoid MPS issues
            else:
                return "cpu"
        return device
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from prompt"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""


class RMLSystem:
    """Main RML system orchestrating encoder, decoder, and memory"""
    
    def __init__(self, config: Optional[RMLConfig] = None):
        self.config = config or RMLConfig()
        self.encoder = RMLEncoder(
            model_name=self.config.encoder_model,
            device=self.config.device
        )
        self.decoder = RMLDecoder(
            model_path=self.config.decoder_model,
            device=self.config.device
        )
        self.memory = MemoryStore()
        self.memory.encode_query_fn = self.encoder.encode_query
        
        # Load and encode dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and encode the dataset"""
        if not os.path.exists(self.config.dataset_path):
            print(f"Dataset not found: {self.config.dataset_path}")
            return
        
        print(f"Loading {self.config.max_entries} entries...")
        entries = []
        with open(self.config.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.config.max_entries:
                    break
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except:
                    continue
        
        print("Encoding entries...")
        embeddings = self.encoder.encode_entries(
            entries, 
            batch_size=self.config.encoder_batch_size
        )
        
        self.memory.add_entries(entries, embeddings)
        print(f"Loaded {len(entries)} entries with embeddings")
    
    def query(self, message: str) -> RMLResponse:
        """Process a query and return response with sources"""
        start_time = time.time()
        
        # Search memory for relevant context
        results = self.memory.search(message, top_k=5)
        
        if not results:
            response = "I couldn't find relevant information in the dataset."
            sources = ["internal dataset"]
        else:
            # Build context from search results
            context = "\n\n".join([r['text'] for r in results])
            prompt = f"Based on the following context, answer this question: {message}\n\nContext:\n{context}\n\nAnswer:"
            
            # Generate response
            answer = self.decoder.generate(prompt)
            
            if not answer or len(answer.strip()) < 10:
                # Fallback to extractive answer
                answer = self._build_extractive_answer(results)
            
            # Clean and format response
            answer = self._clean_response(answer)
            
            # Add sources
            sources = list(set([r.get('source', 'internal dataset') for r in results]))
            if not sources:
                sources = ["internal dataset"]
            
            response = f"{answer}\n\nSources**:\n- " + "\n- ".join(sources)
        
        response_time = (time.time() - start_time) * 1000
        
        return RMLResponse(
            answer=response,
            sources=sources,
            response_ms=response_time
        )
    
    def _build_extractive_answer(self, results: List[Dict]) -> str:
        """Build answer from search results when generation fails"""
        if not results:
            return "I couldn't find relevant information in the dataset."
        
        # Use the most relevant result
        best_result = results[0]
        text = best_result.get('text', '')
        
        # Extract first sentence or meaningful chunk
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip() + "."
        
        return text[:200] + "..." if len(text) > 200 else text
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        if not response:
            return "I couldn't generate a response."
        
        # Remove extra whitespace and newlines
        response = " ".join(response.split())
        
        # Ensure it ends with proper punctuation
        if not response.endswith(('.', '!', '?')):
            response += "."
        
        return response 