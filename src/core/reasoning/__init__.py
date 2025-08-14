"""
RML Reasoning Module

Implements symbolic reasoning capabilities on top of the RML graph.
"""
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from ..graph import RMGraph, Node, Edge

class Reasoner:
    """Symbolic reasoning engine for RML"""
    
    def __init__(self, graph: RMGraph):
        self.graph = graph
        
    def infer_relations(self, source_id: str, max_depth: int = 3) -> List[Dict]:
        """
        Infer potential relations from a source node
        
        Args:
            source_id: ID of the source node
            max_depth: Maximum depth to traverse
            
        Returns:
            List of inferred relations with confidence scores
        """
        if source_id not in self.graph.nodes:
            return []
            
        results = []
        visited = set()
        
        def _traverse(node_id: str, path: List[Tuple[str, str]], depth: int):
            if depth > max_depth or node_id in visited:
                return
                
            visited.add(node_id)
            
            # Get all outgoing edges
            for edge in self.graph.node_edges.get(node_id, []):
                # Add direct relation
                relation = {
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge.type,
                    'path': path + [(edge.type, edge.target)],
                    'confidence': 1.0 - (0.2 * depth)  # Confidence decreases with depth
                }
                results.append(relation)
                
                # Recursively traverse
                _traverse(edge.target, relation['path'], depth + 1)
                
        _traverse(source_id, [], 0)
        return results
        
    def find_common_connections(self, node1_id: str, node2_id: str) -> List[Dict]:
        """
        Find common connections between two nodes
        
        Args:
            node1_id: First node ID
            node2_id: Second node ID
            
        Returns:
            List of common connections with their types
        """
        if node1_id not in self.graph.nodes or node2_id not in self.graph.nodes:
            return []
            
        # Get neighbors of both nodes
        neighbors1 = {edge.target: edge for edge in self.graph.node_edges.get(node1_id, [])}
        neighbors2 = {edge.target: edge for edge in self.graph.node_edges.get(node2_id, [])}
        
        # Find common neighbors
        common = set(neighbors1.keys()) & set(neighbors2.keys())
        
        results = []
        for node_id in common:
            edge1 = neighbors1[node_id]
            edge2 = neighbors2[node_id]
            
            results.append({
                'common_node': node_id,
                'from_first': {
                    'type': edge1.type,
                    'data': edge1.data
                },
                'from_second': {
                    'type': edge2.type,
                    'data': edge2.data
                },
                'confidence': 0.8  # High confidence for direct connections
            })
            
        return results
    
    def validate_fact(self, subject_id: str, predicate: str, object_id: str) -> Dict:
        """
        Validate if a fact (subject-predicate-object) exists in the graph
        
        Args:
            subject_id: ID of the subject node
            predicate: Type of the edge/relation
            object_id: ID of the object node
            
        Returns:
            Validation result with confidence score
        """
        if subject_id not in self.graph.nodes or object_id not in self.graph.nodes:
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': 'Subject or object not found in graph'
            }
            
        # Check for direct connection
        for edge in self.graph.node_edges.get(subject_id, []):
            if edge.target == object_id and edge.type == predicate:
                return {
                    'valid': True,
                    'confidence': 1.0,
                    'evidence': [edge]
                }
                
        # Check for indirect connections (up to 2 hops)
        indirect_evidence = []
        for edge1 in self.graph.node_edges.get(subject_id, []):
            for edge2 in self.graph.node_edges.get(edge1.target, []):
                if edge2.target == object_id and edge2.type == predicate:
                    indirect_evidence.extend([edge1, edge2])
                    
        if indirect_evidence:
            return {
                'valid': True,
                'confidence': 0.7,  # Slightly lower confidence for indirect
                'evidence': indirect_evidence
            }
            
        return {
            'valid': False,
            'confidence': 0.0,
            'reason': 'No direct or indirect evidence found'
        }

    def explain_relation(self, source_id: str, target_id: str) -> Dict:
        """
        Explain the relationship between two nodes
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            Explanation of the relationship with confidence score
        """
        # Try to find direct connection first
        for edge in self.graph.node_edges.get(source_id, []):
            if edge.target == target_id:
                return {
                    'explanation': f"Direct relation: {edge.type}",
                    'confidence': 1.0,
                    'path': [edge],
                    'type': 'direct'
                }
                
        # Try to find path with BFS
        visited = set()
        queue = [(source_id, [])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == target_id:
                return {
                    'explanation': ' → '.join([e.type for e in path]),
                    'confidence': max(0.9 - (0.1 * len(path)), 0.3),  # Decrease with path length
                    'path': path,
                    'type': 'indirect'
                }
                
            if current_id in visited:
                continue
                
            visited.add(current_id)
            
            for edge in self.graph.node_edges.get(current_id, []):
                if edge.target not in visited:
                    queue.append((edge.target, path + [edge]))
                    
        return {
            'explanation': 'No relationship found',
            'confidence': 0.0,
            'path': [],
            'type': 'none'
        }

    def _try_symbolic_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Attempt to answer a query using symbolic reasoning with enhanced pattern matching
        and query understanding.
        
        Args:
            query: The query to answer
            
        Returns:
            Dictionary with answer, confidence, and reasoning trace
        """
        trace = []
        
        # 1. Entity and relation extraction
        entities = self._extract_entities(query)
        relations = self._extract_relations(query)
        
        trace.append({
            'step': 'entity_extraction',
            'result': f"Found entities: {entities}"
        })
        
        trace.append({
            'step': 'relation_extraction',
            'result': f"Found relations: {relations}"
        })
        
        # 2. Basic question answering patterns
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
                        
                        # Check direct attributes
                        if attribute in node.data:
                            return {
                                'answer': f"{entity}'s {attribute} is {node.data[attribute]}",
                                'confidence': 0.9,
                                'trace': trace
                            }
                        
                        # Check metadata
                        if attribute in node.metadata:
                            return {
                                'answer': f"{entity}'s {attribute} is {node.metadata[attribute]}",
                                'confidence': 0.9,
                                'trace': trace
                            }
        
        # 3. Relationship queries
        relation_phrases = [
            ('related to', 'related_to'),
            ('connected to', 'connected_to'),
            ('works with', 'works_with'),
            ('author of', 'author'),
            ('affiliated with', 'affiliated_with')
        ]
        
        for phrase, rel_type in relation_phrases:
            if phrase in query.lower():
                parts = query.lower().split(phrase)
                if len(parts) == 2:
                    entity1 = parts[0].strip()
                    entity2 = parts[1].replace("?", "").strip()
                    
                    # Find matching nodes
                    results1 = self.retriever.retrieve(entity1, top_k=1)
                    results2 = self.retriever.retrieve(entity2, top_k=1)
                    
                    if results1 and results2:
                        explanation = self.reasoner.explain_relation(
                            results1[0].node_id,
                            results2[0].node_id
                        )
                        
                        trace.append({
                            'step': 'relation_explanation',
                            'result': f"Found relation: {explanation}"
                        })
                        
                        if explanation['confidence'] > 0.5:
                            return {
                                'answer': f"{explanation['explanation']}",
                                'confidence': explanation['confidence'],
                                'trace': trace
                            }
        
        # 4. List/query patterns
        if any(q_word in query.lower() for q_word in ['list', 'which', 'what are']):
            # Handle "List X that Y" pattern
            if 'that' in query.lower():
                _, conditions = query.lower().split('that', 1)
                conditions = conditions.replace('?', '').strip()
                
                # This is a simplified example - in practice, you'd parse the conditions
                # and convert them to graph queries
                results = self.retriever.retrieve(conditions, top_k=3)
                
                if results:
                    items = [f"- {r.content}" for r in results]
                    return {
                        'answer': f"Here are some relevant items:\n" + "\n".join(items),
                        'confidence': 0.8,
                        'trace': trace
                    }
        
        # If we get here, we couldn't find a good answer with symbolic reasoning
        trace.append({
            'step': 'fallback',
            'result': 'No high-confidence symbolic match found'
        })
        
        return {
            'answer': "I'll need to look that up...",
            'confidence': 0.0,
            'trace': trace
        }

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text with basic NLP
        
        Args:
            text: Input text
            
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
            # Fallback to simple noun phrase extraction
            import re
            
            # Simple pattern to find noun phrases (very basic)
            noun_phrases = re.findall(r'\b(?:[A-Z][a-z]+\s*)+', text)
            
            return [{'text': np, 'label': 'NOUN_PHRASE'} 
                   for np in noun_phrases if len(np.split()) <= 3]

    def _extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relations from text with pattern matching
        
        Args:
            text: Input text
            
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
