#!/usr/bin/env python3
"""
Optimized RML Knowledge Base with BGE-Large
Balances quality and resource efficiency
"""

import torch
import faiss
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
import spacy
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Triple:
    """Knowledge triple (subject, relation, object)"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0

class RMLKnowledgeBase:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a smaller model by default to prevent hanging"""
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Initializing RML Knowledge Base with {model_name} on {self.device}")
        
        # Model configuration
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize components
        self._load_models()
        
        # Knowledge storage
        self.index = None
        self.texts = []
        self.triples = []
    
    def _load_models(self):
        """Load the embedding model and NER pipeline"""
        # Load BGE model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == 'cuda' else None,
            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32
        ).to(self.device)
        self.model.eval()
        
        # Load NER model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings with batching"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get CLS token embedding
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def add_document(self, text: str):
        """Add a single document to the knowledge base (wrapper around add_documents for backward compatibility)"""
        return self.add_documents([text])

    def add_documents(self, texts: List[str], batch_size: int = 32):
        """Add multiple documents to the knowledge base
        
        Args:
            texts: List of text documents to add
            batch_size: Batch size for encoding
        """
        if not texts:
            return
            
        logger.info(f"Adding {len(texts)} documents to knowledge base")
        
        try:
            # Encode documents
            embeddings = self.encode(texts, batch_size)
            
            # Initialize FAISS index if needed
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                if self.device == 'cuda':
                    self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            
            # Add to FAISS index
            if len(embeddings) > 0:
                self.index.add(embeddings)
                self.texts.extend(texts)
                logger.info(f"Successfully added {len(texts)} documents to knowledge base")
                
        except Exception as e:
            logger.error(f"Error adding documents to knowledge base: {e}")
            raise
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        start_idx = len(self.texts)
        self.texts.extend(texts)
        
        # Extract and store triples
        for i, text in enumerate(texts, start=start_idx):
            self._extract_and_store_triples(text, i)
    
    def _extract_relations(self, doc) -> List[Tuple[str, str, str, float]]:
        """Extract subject-relation-object triples from a spaCy doc"""
        triples = []
        
        # Rule 1: Subject-Verb-Object patterns
        for sent in doc.sents:
            verbs = [tok for tok in sent if tok.pos_ == "VERB"]
            for verb in verbs:
                subjs = [tok for tok in verb.lefts if tok.dep_ in ("nsubj", "nsubjpass")]
                objs = [tok for tok in verb.rights if tok.dep_ in ("dobj", "pobj", "attr")]
                
                for subj in subjs:
                    for obj in objs:
                        # Get full noun phrases
                        subj_text = ' '.join([t.text for t in subj.subtree])
                        obj_text = ' '.join([t.text for t in obj.subtree])
                        
                        # Clean up the relation
                        rel = verb.lemma_.lower()
                        
                        triples.append((subj_text, rel, obj_text, 0.9))
        
        # Rule 2: Named Entity relationships
        for ent in doc.ents:
            if ent.root.dep_ in ("pobj", "dobj"):
                head = ent.root.head
                if head.pos_ == "VERB":
                    subjs = [tok for tok in head.lefts if tok.dep_ in ("nsubj", "nsubjpass")]
                    for subj in subjs:
                        subj_text = ' '.join([t.text for t in subj.subtree])
                        rel = head.lemma_.lower()
                        triples.append((subj_text, rel, ent.text, 0.85))
        
        return triples
    
    def _extract_and_store_triples(self, text: str, doc_id: int):
        """Extract triples from text and store them with safety limits"""
        import signal
        from contextlib import contextmanager
        
        class TimeoutError(Exception):
            pass
            
        @contextmanager
        def timeout(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError("Timed out!")
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
        
        try:
            # Limit text length to prevent excessive processing
            max_text_length = 10000  # characters
            if len(text) > max_text_length:
                text = text[:max_text_length]
                logger.debug(f"Truncated text to {max_text_length} characters")
            
            # Process with timeout
            with timeout(5):  # 5 second timeout per document
                doc = self.nlp(text)
                for sent in doc.sents:
                    try:
                        triples = self._extract_relations(sent)
                        for triple in triples:
                            if len(triple[0]) > 2 and len(triple[2]) > 2 and len(triple[1]) > 0:
                                self.triples.append(Triple(
                                    subject=triple[0],
                                    relation=triple[1],
                                    object=triple[2],
                                    confidence=triple[3]
                                ))
                    except Exception as e:
                        logger.debug(f"Error processing sentence: {e}")
                        continue
                        
        except TimeoutError:
            logger.warning(f"Timeout processing document {doc_id}")
        except Exception as e:
            logger.warning(f"Error extracting triples: {e}")
        
        # Track if we found any triples
        triples_found = len(self.triples) > 0
        
        # Fallback to simple entity pairs if no relations found
        if not triples_found:
            entities = [ent.text for ent in doc.ents]
            for i in range(len(entities)-1):
                self.triples.append(Triple(
                    subject=entities[i],
                    relation="related_to",
                    object=entities[i+1],
                    confidence=0.7
                ))
    
    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar documents to query"""
        if not self.index:
            return []
            
        # Encode query
        query_embedding = self.encode([query])[0].astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        return [(self.texts[i], float(score)) 
               for i, score in zip(indices[0], distances[0]) 
               if i < len(self.texts)]
    
    def query_triples(self, entity: str = None, relation: str = None) -> List[Triple]:
        """Query triples by entity and/or relation"""
        results = []
        for triple in self.triples:
            if entity and entity.lower() not in triple.subject.lower() and entity.lower() not in triple.object.lower():
                continue
            if relation and relation.lower() not in triple.relation.lower():
                continue
            results.append(triple)
        return results

# Example usage
if __name__ == "__main__":
    # Initialize knowledge base
    kb = RMLKnowledgeBase()
    
    # Add documents
    docs = [
        "Albert Einstein developed the theory of relativity.",
        "Marie Curie conducted pioneering research on radioactivity.",
        "The Eiffel Tower is located in Paris, France.",
        "Tesla, Inc. is an American electric vehicle company.",
        "The Great Wall of China is visible from space."
    ]
    kb.add_documents(docs)
    
    # Search
    print("\nSemantic search for 'scientist':")
    for text, score in kb.semantic_search("scientist"):
        print(f"[{score:.3f}] {text}")
    
    # Query triples
    print("\nTriples about Einstein:")
    for triple in kb.query_triples("Einstein"):
        print(f"{triple.subject} -- {triple.relation} --> {triple.object}")
