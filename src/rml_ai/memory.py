"""
Memory Store for RML System
Handles vector storage and semantic search
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sklearn.metrics.pairwise import cosine_similarity


class MemoryStore:
    """Vector-based memory store for semantic search"""
    
    def __init__(self):
        self.entries = []
        self.embeddings = None
        self.encode_query_fn: Optional[Callable] = None
    
    def add_entries(self, entries: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add entries with their embeddings"""
        self.entries = entries
        self.embeddings = embeddings
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant entries using semantic similarity"""
        if not self.entries or self.embeddings is None:
            return []
        
        if not self.encode_query_fn:
            # Fallback to keyword search
            return self._keyword_search(query, top_k)
        
        try:
            # Encode query
            query_embedding = self.encode_query_fn(query)
            
            if query_embedding.shape[1] != self.embeddings.shape[1]:
                print(f"Embedding dimension mismatch: query {query_embedding.shape[1]} vs entries {self.embeddings.shape[1]}")
                return self._keyword_search(query, top_k)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    entry = self.entries[idx].copy()
                    entry['text'] = self._extract_text(entry)
                    entry['similarity'] = float(similarities[idx])
                    entry['source'] = entry.get('source', 'internal dataset')
                    results.append(entry)
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback keyword search"""
        query_lower = query.lower()
        results = []
        
        for entry in self.entries:
            text = self._extract_text(entry).lower()
            if any(word in text for word in query_lower.split()):
                entry_copy = entry.copy()
                entry_copy['text'] = self._extract_text(entry)
                entry_copy['similarity'] = 0.5  # Default similarity for keyword matches
                entry_copy['source'] = entry.get('source', 'internal dataset')
                results.append(entry_copy)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _extract_text(self, entry: Dict[str, Any]) -> str:
        """Extract text content from entry"""
        for field in ['text', 'content', 'body', 'chunk', 'summary', 'title']:
            if field in entry and entry[field]:
                return str(entry[field])
        return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        return {
            'total_entries': len(self.entries),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'has_embeddings': self.embeddings is not None
        } 