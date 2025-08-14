"""
RML Integration Module

Provides integration points with LangChain and other external tools.
This is a thin integration layer - core orchestration remains in RML.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

# Import core components
from ..graph import RMGraph, Node, Edge
from ..reasoning import Reasoner
from ..retrieval import RMLRetriever, RetrievalResult

class RMLLangChainWrapper:
    """Wrapper to integrate RML with LangChain"""
    
    def __init__(self, graph: RMGraph, reasoner: Reasoner, retriever: RMLRetriever):
        self.graph = graph
        self.reasoner = reasoner
        self.retriever = retriever
        
    def as_tool(self, name: str, description: str) -> Callable:
        """
        Convert RML capabilities into a LangChain-compatible tool
        
        Args:
            name: Name of the tool
            description: Description for the tool
            
        Returns:
            A function that can be used as a LangChain tool
        """
        def rml_tool(query: str) -> str:
            # This is a simplified example - in practice, you'd want to
            # implement more sophisticated query understanding and routing
            
            # First, try to answer using symbolic reasoning
            symbolic_result = self._try_symbolic_reasoning(query)
            if symbolic_result['confidence'] > 0.8:  # High confidence
                return f"[Symbolic Answer] {symbolic_result['answer']}"
                
            # Fall back to RAG if symbolic reasoning is not confident
            rag_result = self.retriever.generate_context(query)
            return f"[Retrieved Context] {rag_result}"
            
        # Add metadata for LangChain
        rml_tool.name = name
        rml_tool.description = description
        return rml_tool
    
    def _try_symbolic_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Attempt to answer a query using symbolic reasoning
        
        Args:
            query: The query to answer
            
        Returns:
            Dictionary with answer and confidence
        """
        # This is a placeholder - in practice, you'd want to implement
        # more sophisticated natural language understanding here
        
        # Simple pattern matching for demonstration
        if "related to" in query:
            parts = query.split("related to")
            if len(parts) == 2:
                node1 = parts[0].strip()
                node2 = parts[1].strip()
                
                # Try to find nodes that match these descriptions
                results1 = self.retriever.retrieve(node1, top_k=1)
                results2 = self.retriever.retrieve(node2, top_k=1)
                
                if results1 and results2:
                    explanation = self.reasoner.explain_relation(
                        results1[0].node_id, 
                        results2[0].node_id
                    )
                    
                    return {
                        'answer': f"{explanation['explanation']}",
                        'confidence': explanation['confidence']
                    }
        
        return {
            'answer': "I couldn't determine a clear answer using symbolic reasoning.",
            'confidence': 0.0
        }

class RMLLangGraphBridge:
    """Bridge between RML and LangGraph for visualization and orchestration"""
    
    def __init__(self, graph: RMGraph):
        self.graph = graph
        
    def to_langgraph(self):
        """Convert RML graph to LangGraph format"""
        try:
            from langgraph.graph import Graph
            
            lg = Graph()
            
            # Add nodes for each RML node
            for node_id, node in self.graph.nodes.items():
                lg.add_node(node_id, lambda x: {"content": str(node.data)})
                
            # Add edges for each RML edge
            for edge in self.graph.edges:
                lg.add_edge(edge.source, edge.target)
                
            return lg
            
        except ImportError:
            raise ImportError("LangGraph is required for this functionality")
            
    def visualize_reasoning_path(self, path: List[Dict]):
        """Visualize a reasoning path using LangGraph"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            
            # Add nodes and edges for each step in the path
            for i, step in enumerate(path):
                step_id = f"step_{i}"
                G.add_node(step_id, label=step.get('action', 'Unknown'))
                
                if i > 0:
                    G.add_edge(f"step_{i-1}", step_id, 
                             label=step.get('result', '')[:20] + '...')
            
            # Draw the graph
            pos = nx.spring_layout(G)
            plt.figure(figsize=(10, 6))
            nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue',
                   font_size=10, font_weight='bold', arrows=True)
            
            # Add edge labels
            edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.title("Reasoning Path Visualization")
            plt.tight_layout()
            plt.show()
            
        except ImportError as e:
            print(f"Visualization requires NetworkX and Matplotlib: {e}")

class RMLHybridOrchestrator:
    """Orchestrator for hybrid symbolic-neural reasoning"""
    
    def __init__(self, graph: RMGraph, reasoner: Reasoner, retriever: RMLRetriever):
        self.graph = graph
        self.reasoner = reasoner
        self.retriever = retriever
        
    def process_query(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Process a query using hybrid symbolic-neural approach with enhanced reasoning
        
        Args:
            query: The query to process
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Dictionary with answer, confidence, method, and reasoning trace
        """
        trace = []
        relevant_nodes = set()
        
        # Step 1: Extract entities and relations
        entities = self._extract_entities(query)
        relations = self._extract_relations(query)
        
        trace.append({
            'step': 'entity_extraction',
            'result': f"Found entities: {[e['text'] for e in entities]}"
        })
        trace.append({
            'step': 'relation_extraction',
            'result': f"Found relations: {[r['type'] for r in relations]}"
        })
        
        # Step 2: Try to answer using symbolic reasoning
        symbolic_result = self._try_symbolic_reasoning(query, entities, relations)
        
        # If we have high confidence in the symbolic answer, return it
        if symbolic_result.get('confidence', 0) > 0.7:
            relevant_nodes.update(symbolic_result.get('relevant_nodes', []))
            return {
                'answer': symbolic_result['answer'],
                'confidence': symbolic_result['confidence'],
                'method': 'symbolic',
                'relevant_nodes': list(relevant_nodes),
                'trace': trace + symbolic_result.get('trace', [])
            }
        
        # Step 3: Fall back to RAG with symbolic guidance
        rag_result = self._hybrid_rag_retrieval(
            query, 
            entities, 
            relations,
            symbolic_hints=symbolic_result.get('hints', {})
        )
        
        relevant_nodes.update(rag_result.get('relevant_nodes', []))
        
        return {
            'answer': rag_result['answer'],
            'confidence': rag_result['confidence'],
            'method': 'hybrid',
            'relevant_nodes': list(relevant_nodes),
            'context': rag_result.get('context', ''),
            'trace': trace + rag_result.get('trace', [])
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using spaCy or fallback to simple patterns
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities with their types and positions
        """
        try:
            import spacy
            
            # Load a small English model if available
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                # If model not found, download it
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")
                
            doc = nlp(text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                
            return entities
            
        except ImportError:
            # Fallback to simple pattern matching
            import re
            
            # Simple pattern to find noun phrases (very basic)
            noun_phrases = re.findall(r'\b(?:[A-Z][a-z]+\s*)+', text)
            
            return [{'text': np, 'label': 'NOUN_PHRASE'} 
                   for np in noun_phrases if len(np.split()) <= 3]
    
    def _extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relations from text using pattern matching
        
        Args:
            text: Input text to extract relations from
            
        Returns:
            List of extracted relations with their types and arguments
        """
        relations = []
        
        # Common relation patterns with their canonical forms
        relation_patterns = [
            (r'(?P<e1>\w+)\s+(?:is\s+)?(?:an?\s+)?(?:the\s+)?(?P<rel>author|creator|writer)\s+of\s+(?P<e2>.+)', 'author_of'),
            (r'(?P<e1>\w+)\s+(?:works?\s+)?(?:at|for)\s+(?P<e2>.+)', 'works_at'),
            (r'(?P<e1>\w+)\s+(?:is\s+)?(?:a\s+)?(?:member|part)\s+of\s+(?P<e2>.+)', 'member_of'),
            (r'(?P<e1>\w+)\s+(?:and|&)\s+(?P<e2>\w+)\s+are\s+(?P<rel>colleagues|peers)', 'colleagues'),
            (r'(?P<e1>\w+)\s+(?P<rel>studied|researched|wrote\s+about)\s+(?P<e2>.+)', 'researched'),
            (r'(?P<e1>\w+)\s+(?P<rel>is\s+located\s+in|is\s+from)\s+(?P<e2>.+)', 'located_in'),
        ]
        
        for pattern, rel_type in relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groupdict()
                e1 = groups.get('e1', '').strip()
                e2 = groups.get('e2', '').strip()
                rel = groups.get('rel', rel_type)
                
                if e1 and e2:
                    relations.append({
                        'type': rel,
                        'source': e1,
                        'target': e2,
                        'text': match.group(0).strip()
                    })
        
        return relations
    
    def _hybrid_rag_retrieval(
        self,
        query: str,
        entities: List[Dict],
        relations: List[Dict],
        symbolic_hints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval that combines symbolic and neural approaches.
        
        Args:
            query: The query to process
            entities: List of extracted entities
            relations: List of extracted relations
            symbolic_hints: Any hints from symbolic reasoning
            
        Returns:
            Dictionary with answer, confidence, and relevant nodes
        """
        trace = []
        relevant_nodes = set()
        
        # Step 1: Use symbolic hints if available
        if symbolic_hints and 'entities' in symbolic_hints:
            # Try to find relevant nodes based on entities
            for entity in symbolic_hints['entities']:
                results = self.retriever.retrieve(entity['text'], top_k=3)
                for result in results:
                    relevant_nodes.add(result.node_id)
            
            trace.append({
                'step': 'symbolic_guided_retrieval',
                'result': f"Found {len(relevant_nodes)} relevant nodes using symbolic hints"
            })
        
        # Step 2: Fall back to standard RAG if no relevant nodes found
        if not relevant_nodes:
            # Use the retriever to find relevant context
            results = self.retriever.retrieve(query, top_k=5)
            for result in results:
                relevant_nodes.add(result.node_id)
            
            trace.append({
                'step': 'neural_retrieval',
                'result': f"Found {len(relevant_nodes)} relevant nodes using neural retrieval"
            })
        
        # Step 3: Generate a response using the retrieved context
        # Get the text content for each relevant node
        context_parts = []
        for node_id in list(relevant_nodes)[:10]:  # Limit context size
            node = self.graph.nodes.get(node_id)
            if node:
                # Create a string representation of the node
                node_str = f"[Node ID: {node_id}]\n"
                
                # Get node attributes safely
                node_attrs = {}
                if hasattr(node, '__dict__'):
                    node_attrs = node.__dict__.copy()
                
                # Add node attributes to the string representation
                for attr, value in node_attrs.items():
                    if not attr.startswith('_'):  # Skip private attributes
                        if isinstance(value, dict):
                            node_str += f"{attr}: {', '.join(f'{k}={v}' for k, v in value.items() if not k.startswith('_'))}\n"
                        elif isinstance(value, (list, tuple, set)):
                            node_str += f"{attr}: {', '.join(str(v) for v in value)}\n"
                        elif value is not None and not callable(value):
                            node_str += f"{attr}: {value}\n"
                context_parts.append(node_str)
        
        # Join all context parts
        context = "\n---\n".join(context_parts)
        
        # In a real implementation, you would pass this to an LLM for answer generation
        # For this example, we'll just return the context
        return {
            'answer': f"Based on the context:\n{context[:1000]}",
            'confidence': 0.7 if relevant_nodes else 0.5,
            'relevant_nodes': list(relevant_nodes),
            'context': context,
            'trace': trace
        }
    
    def _try_symbolic_reasoning(self, query: str, entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """
        Attempt to answer the query using symbolic reasoning
        
        Args:
            query: The query to answer
            entities: List of extracted entities
            relations: List of extracted relations
            
        Returns:
            Dictionary with answer, confidence, and reasoning trace
        """
        trace = []
        relevant_nodes = set()
        
        # Try to match question patterns
        if any(q_word in query.lower() for q_word in ['what is', 'who is']):
            # Handle "What is X's Y?" pattern
            if "'s" in query:
                parts = query.split("'s")
                if len(parts) == 2:
                    entity = parts[0].replace("what is", "").replace("who is", "").strip()
                    attribute = parts[1].replace("?", "").strip()
                    
                    # Find matching nodes
                    results = self.retriever.retrieve(entity, top_k=1)
                    if results:
                        node_id = results[0].node_id
                        node = self.graph.nodes.get(node_id)
                        relevant_nodes.add(node_id)
                        
                        # Check direct attributes
                        if attribute in node.data:
                            return {
                                'answer': f"{entity}'s {attribute} is {node.data[attribute]}",
                                'confidence': 0.9,
                                'relevant_nodes': list(relevant_nodes),
                                'trace': trace + [{'step': 'attribute_lookup', 'result': f"Found {attribute} in node data"}]
                            }
                        
                        # Check metadata
                        if hasattr(node, 'metadata') and attribute in node.metadata:
                            return {
                                'answer': f"{entity}'s {attribute} is {node.metadata[attribute]}",
                                'confidence': 0.9,
                                'relevant_nodes': list(relevant_nodes),
                                'trace': trace + [{'step': 'metadata_lookup', 'result': f"Found {attribute} in node metadata"}]
                            }
        
        # If we get here, we couldn't find a good answer with symbolic reasoning
        return {
            'answer': "",
            'confidence': 0.0,
            'hints': {
                'entities': entities,
                'relations': relations
            },
            'relevant_nodes': list(relevant_nodes),
            'trace': trace + [{'step': 'symbolic_reasoning', 'result': 'No high-confidence answer found'}]
        }
