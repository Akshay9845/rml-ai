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
            
            # Handle empty embeddings
            if self.embeddings is None or len(self.embeddings) == 0:
                return self._keyword_search(query, top_k)
                
            # Ensure proper dimensions
            if len(self.embeddings.shape) == 1:
                # If embeddings is 1D, reshape to 2D
                embeddings = self.embeddings.reshape(1, -1)
            else:
                embeddings = self.embeddings
                
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Check dimension compatibility
            if query_embedding.shape[1] != embeddings.shape[1]:
                print(f"Embedding dimension mismatch: query {query_embedding.shape[1]} vs entries {embeddings.shape[1]}")
                return self._keyword_search(query, top_k)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
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
        """Fallback keyword search with RML-aware scoring"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []
        
        for entry in self.entries:
            score = 0
            text = self._extract_text(entry).lower()
            
            # Check direct text matches
            text_words = set(text.split())
            common_words = query_words.intersection(text_words)
            score += len(common_words) * 2  # Base score for word matches
            
            # Boost score for matches in specific RML fields
            if 'concepts' in entry and entry['concepts']:
                concepts_text = " ".join(entry['concepts']).lower() if isinstance(entry['concepts'], list) else str(entry['concepts']).lower()
                concept_matches = sum(1 for word in query_words if word in concepts_text)
                score += concept_matches * 3  # Higher weight for concept matches
            
            if 'tags' in entry and entry['tags']:
                tags_text = " ".join(entry['tags']).lower() if isinstance(entry['tags'], list) else str(entry['tags']).lower()
                tag_matches = sum(1 for word in query_words if word in tags_text)
                score += tag_matches * 2  # Medium weight for tag matches
                
            if 'summaries' in entry and entry['summaries']:
                summary_text = entry['summaries'][0].lower() if isinstance(entry['summaries'], list) and entry['summaries'] else str(entry['summaries']).lower()
                summary_matches = sum(1 for word in query_words if word in summary_text)
                score += summary_matches * 4  # Highest weight for summary matches
            
            # Only include results with some relevance
            if score > 0:
                entry_copy = entry.copy()
                entry_copy['text'] = self._extract_text(entry)
                entry_copy['similarity'] = min(0.9, score / 10)  # Normalize score to similarity
                entry_copy['source'] = entry.get('source', 'internal dataset')
                results.append(entry_copy)
        
        # Sort by similarity score and return top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _extract_text(self, entry: Dict[str, Any]) -> str:
        """Extract text content from entry, handling RML-specific structure"""
        # First try standard fields
        for field in ['text', 'content', 'body', 'chunk', 'summary', 'title']:
            if field in entry and entry[field]:
                return str(entry[field])
        
        # Handle RML-specific structure
        text_parts = []
        
        # Extract from summaries (first priority for RML data)
        if 'summaries' in entry and entry['summaries']:
            if isinstance(entry['summaries'], list) and entry['summaries']:
                text_parts.append(entry['summaries'][0])
            elif isinstance(entry['summaries'], str):
                text_parts.append(entry['summaries'])
        
        # Extract from concepts
        if 'concepts' in entry and entry['concepts']:
            if isinstance(entry['concepts'], list):
                text_parts.append(" ".join(entry['concepts'][:10]))  # First 10 concepts
            elif isinstance(entry['concepts'], str):
                text_parts.append(entry['concepts'])
        
        # Extract from tags
        if 'tags' in entry and entry['tags']:
            if isinstance(entry['tags'], list):
                text_parts.append(" ".join(entry['tags'][:10]))  # First 10 tags
            elif isinstance(entry['tags'], str):
                text_parts.append(entry['tags'])
        
        # Combine all parts
        if text_parts:
            return " ".join(text_parts)
        
        # Fallback: convert entire entry to string (excluding large arrays)
        filtered_entry = {}
        for k, v in entry.items():
            if k not in ['vectors', 'embeddings'] and v:
                if isinstance(v, list) and len(v) > 20:
                    filtered_entry[k] = v[:5]  # Only first 5 items of large lists
                else:
                    filtered_entry[k] = v
        
        return str(filtered_entry) if filtered_entry else "No content available"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        embedding_dim = 0
        if self.embeddings is not None and len(self.embeddings.shape) > 1:
            embedding_dim = self.embeddings.shape[1]
        elif self.embeddings is not None and len(self.embeddings.shape) == 1:
            embedding_dim = len(self.embeddings)
            
        return {
            'total_entries': len(self.entries),
            'embedding_dim': embedding_dim,
            'has_embeddings': self.embeddings is not None
        } 