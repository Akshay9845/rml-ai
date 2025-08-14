"""
Core RML Graph Module

This module implements the core symbolic graph for the RML system.
It provides a pure Python implementation without external dependencies.
"""
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from uuid import uuid4

@dataclass
class Node:
    """Represents a node in the RML graph"""
    id: str
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    """Represents a directed edge in the RML graph"""
    source: str
    target: str
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class RMGraph:
    """Core RML Graph implementation"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.node_edges: Dict[str, List[Edge]] = {}
        
    def add_node(self, node_type: str, data: Optional[Dict] = None, 
                metadata: Optional[Dict] = None) -> str:
        """Add a node to the graph"""
        node_id = str(uuid4())
        node = Node(
            id=node_id,
            type=node_type,
            data=data or {},
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        self.node_edges[node_id] = []
        return node_id
        
    def add_edge(self, source_id: str, target_id: str, 
                edge_type: str, data: Optional[Dict] = None) -> None:
        """Add an edge between two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node not found")
            
        edge = Edge(
            source=source_id,
            target=target_id,
            type=edge_type,
            data=data or {}
        )
        self.edges.append(edge)
        self.node_edges[source_id].append(edge)
        
    def query(self, pattern: Dict) -> List[Dict]:
        """
        Query the graph using a pattern matching approach
        
        Args:
            pattern: Dictionary specifying node/edge patterns to match
            
        Returns:
            List of matching subgraphs
        """
        # Simple implementation - can be extended with more sophisticated querying
        results = []
        
        # Match nodes first
        matched_nodes = []
        for node in self.nodes.values():
            match = True
            for key, value in pattern.get('nodes', {}).items():
                if key == 'type' and node.type != value:
                    match = False
                    break
                if key in node.data and node.data[key] != value:
                    match = False
                    break
            if match:
                matched_nodes.append(node)
                
        # For now, just return matched nodes
        # In a real implementation, we'd also match edges and return subgraphs
        return [{'nodes': [n.id for n in matched_nodes]}]
        
    def to_networkx(self):
        """Convert to NetworkX graph for visualization"""
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes
            for node_id, node in self.nodes.items():
                G.add_node(node_id, type=node.type, **node.data)
                
            # Add edges
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, type=edge.type, **edge.data)
                
            return G
            
        except ImportError:
            raise ImportError("NetworkX is required for graph visualization")
            
    def visualize(self, output_file: str = None):
        """Visualize the graph using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            G = self.to_networkx()
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            
            plt.axis('off')
            
            if output_file:
                plt.savefig(output_file)
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            print("Visualization requires NetworkX and Matplotlib")
