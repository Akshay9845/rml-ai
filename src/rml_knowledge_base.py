#!/usr/bin/env python3
"""RML Knowledge Base with Semantic Search and Triple Extraction"""

import torch
import faiss
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
import spacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    confidence: float = 1.0

class RMLKnowledgeBase:
    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Initializing RML Knowledge Base on {self.device}")
        
        # Initialize encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device != 'cpu' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None
        ).to(self.device)
        self.model.eval()
        
        # Initialize NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize FAISS
        self.index = None
        self.texts = []
        self.triples = []
    
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode texts to embeddings"""
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [f"query: {t}" for t in texts[i:i + batch_size]]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, 
                    max_length=512, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)
    
    def add_documents(self, texts: List[str]):
        """Add documents to knowledge base"""
        if not texts:
            return
            
        # Encode and index
        embeddings = self.encode(texts)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            if self.device == 'cuda':
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)
        
        # Extract triples
        for text in texts:
            doc = self.nlp(text)
            for sent in doc.sents:
                ents = [e.text for e in sent.ents]
                if len(ents) >= 2:
                    for i in range(len(ents)-1):
                        self.triples.append(Triple(
                            subject=ents[i],
                            relation="related_to",
                            object=ents[i+1]
                        ))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Semantic search in knowledge base"""
        if not self.index:
            return []
            
        query_embedding = self.encode([query])[0].astype('float32')
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [(self.texts[i], float(score)) for i, score in zip(indices[0], distances[0])]
    
    def get_triples(self, entity: str = None) -> List[Triple]:
        """Get triples, optionally filtered by entity"""
        if not entity:
            return self.triples
        return [t for t in self.triples 
               if entity.lower() in t.subject.lower() 
               or entity.lower() in t.object.lower()]

# Example usage
if __name__ == "__main__":
    # Initialize knowledge base
    kb = RMLKnowledgeBase()
    
    # Add documents
    docs = [
        "Albert Einstein developed the theory of relativity.",
        "Marie Curie conducted pioneering research on radioactivity.",
        "The Eiffel Tower is located in Paris, France."
    ]
    kb.add_documents(docs)
    
    # Search
    print("\nSearch results for 'scientist':")
    for text, score in kb.search("scientist"):
        print(f"[{score:.3f}] {text}")
    
    # Get triples
    print("\nTriples about Einstein:")
    for triple in kb.get_triples("Einstein"):
        print(f"{triple.subject} -- {triple.relation} --> {triple.object}")
