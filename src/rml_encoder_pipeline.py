#!/usr/bin/env python3
"""RML Semantic Encoder Pipeline with E5-Mistral-7B"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RMLSemanticEncoder:
    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Loading {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device != 'cpu' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None
        ).to(self.device)
        self.model.eval()
    
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Convert texts to normalized embeddings"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [f"query: {t}" for t in texts[i:i + batch_size]]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, 
                    max_length=512, return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Example usage
if __name__ == "__main__":
    encoder = RMLSemanticEncoder()
    
    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries.",
        "The capital of France is Paris."
    ]
    
    # Get embeddings
    embeddings = encoder.encode(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")
