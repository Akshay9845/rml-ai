"""
RML Retrieval Module

Implements retrieval-augmented generation capabilities integrated with the RML graph.
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from ..graph import RMGraph

@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    content: str
    node_id: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = None

class RMLRetriever:
    """Retrieval component for RML system"""
    
    def __init__(self, graph: RMGraph, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.graph = graph
        self.embedding_model = embedding_model
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model)
            self._has_embeddings = True
        except ImportError:
            self._has_embeddings = False
            print("Warning: SentenceTransformers not available. Using simple text matching.")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using the configured model"""
        if not self._has_embeddings:
            # Fallback to simple bag-of-words representation
            words = set(text.lower().split())
            return words
        return self.embedder.encode(text, convert_to_numpy=True)
    
    def similarity(self, vec1, vec2) -> float:
        """Calculate similarity between two vectors or sets"""
        if isinstance(vec1, set) and isinstance(vec2, set):
            # Jaccard similarity for sets
            intersection = len(vec1.intersection(vec2))
            union = len(vec1.union(vec2))
            return intersection / union if union > 0 else 0.0
        
        # Cosine similarity for numpy arrays
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    
    def retrieve(self, query: str, top_k: int = 5, 
                node_types: Optional[List[str]] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant nodes from the graph based on a query
        
        Args:
            query: The search query
            top_k: Number of results to return
            node_types: Optional list of node types to filter by
            
        Returns:
            List of retrieval results sorted by relevance
        """
        query_embedding = self.embed_text(query)
        results = []
        
        for node_id, node in self.graph.nodes.items():
            # Skip if node type filtering is active and node doesn't match
            if node_types and node.type not in node_types:
                continue
                
            # Create a text representation of the node
            node_text = f"{node.type}: {', '.join(f'{k}={v}' for k, v in node.data.items())}"
            
            # Calculate similarity
            node_embedding = self.embed_text(node_text)
            score = self.similarity(query_embedding, node_embedding)
            
            results.append(RetrievalResult(
                content=node_text,
                node_id=node_id,
                score=float(score),
                metadata={
                    'type': node.type,
                    **node.metadata
                }
            ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def hybrid_retrieve(self, query: str, symbolic_hint: Optional[Dict] = None, 
                       top_k: int = 5) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval using both symbolic and semantic methods
        
        Args:
            query: The search query
            symbolic_hint: Optional symbolic constraints from the reasoner
            top_k: Number of results to return
            
        Returns:
            Combined retrieval results
        """
        # First, get semantic results
        semantic_results = self.retrieve(query, top_k=top_k * 2)
        
        # If we have symbolic hints, filter/rerank results
        if symbolic_hint:
            # This is a simplified example - in practice, you'd want to
            # implement more sophisticated combination of symbolic and semantic scores
            for result in semantic_results:
                # Boost score if node type matches symbolic hint
                if (symbolic_hint.get('expected_types') and 
                    result.metadata.get('type') in symbolic_hint['expected_types']):
                    result.score *= 1.2  # 20% boost
        
        # Sort and return top-k
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:top_k]
    
    def generate_context(self, query: str, max_length: int = 1000) -> str:
        """
        Generate a context string for RAG by retrieving relevant information
        
        Args:
            query: The query to generate context for
            max_length: Maximum length of the generated context
            
        Returns:
            Context string with relevant information
        """
        results = self.hybrid_retrieve(query)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            part = f"[{result.metadata.get('type', 'fact')}] {result.content}"
            if current_length + len(part) > max_length:
                break
            context_parts.append(part)
            current_length += len(part)
        
        return "\n".join(context_parts)
