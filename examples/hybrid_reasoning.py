"""
Hybrid Reasoning Example for RML System

This example demonstrates the hybrid symbolic-neural reasoning capabilities
of the RML system, combining:
1. Pure symbolic reasoning
2. RAG-based retrieval
3. Integration with LangChain
"""
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from src.core.graph import RMGraph, Node, Edge
from src.core.reasoning import Reasoner
from src.core.retrieval import RMLRetriever, RetrievalResult
from src.core.integration import RMLHybridOrchestrator, RMLLangChainWrapper, RMLLangGraphBridge

def load_sample_data() -> Dict[str, Any]:
    """Load sample data for the knowledge graph"""
    return {
        "people": [
            {"id": "p1", "name": "Alice", "age": 30, "occupation": "AI Researcher", "affiliation": "Stanford"},
            {"id": "p2", "name": "Bob", "age": 45, "occupation": "Professor", "affiliation": "MIT"},
            {"id": "p3", "name": "Charlie", "age": 35, "occupation": "Engineer", "affiliation": "Google"}
        ],
        "papers": [
            {"id": "paper1", "title": "Advances in Neural Networks", "year": 2023, "citations": 150},
            {"id": "paper2", "title": "Reinforcement Learning Basics", "year": 2022, "citations": 300},
            {"id": "paper3", "title": "Transformers in NLP", "year": 2021, "citations": 2500}
        ],
        "organizations": [
            {"id": "o1", "name": "Stanford University", "type": "university", "location": "California"},
            {"id": "o2", "name": "MIT", "type": "university", "location": "Massachusetts"},
            {"id": "o3", "name": "Google", "type": "company", "industry": "Technology"}
        ],
        "relations": [
            {"source": "p1", "target": "paper1", "type": "author"},
            {"source": "p1", "target": "paper2", "type": "co_author"},
            {"source": "p2", "target": "paper2", "type": "author"},
            {"source": "p3", "target": "paper3", "type": "author"},
            {"source": "p1", "target": "o1", "type": "affiliated_with"},
            {"source": "p2", "target": "o2", "type": "affiliated_with"},
            {"source": "p3", "target": "o3", "type": "works_at"},
            {"source": "paper1", "target": "paper2", "type": "cites"},
            {"source": "paper3", "target": "paper2", "type": "cites"}
        ]
    }

def build_knowledge_graph(data: Dict[str, Any]) -> RMGraph:
    """Build a knowledge graph from the sample data"""
    graph = RMGraph()
    node_map = {}
    
    # Add people
    for person in data['people']:
        node_id = graph.add_node(
            "person",
            {"name": person['name'], "age": person['age'], "occupation": person['occupation']},
            {"affiliation": person['affiliation']}
        )
        node_map[person['id']] = node_id
    
    # Add papers
    for paper in data['papers']:
        node_id = graph.add_node(
            "paper",
            {"title": paper['title'], "year": paper['year']},
            {"citations": paper['citations']}
        )
        node_map[paper['id']] = node_id
    
    # Add organizations
    for org in data['organizations']:
        node_id = graph.add_node(
            "organization",
            {"name": org['name'], "type": org['type']},
            {k: v for k, v in org.items() if k not in ['id', 'name', 'type']}
        )
        node_map[org['id']] = node_id
    
    # Add relations
    for rel in data['relations']:
        graph.add_edge(
            node_map[rel['source']],
            node_map[rel['target']],
            rel['type']
        )
    
    return graph, node_map

def run_hybrid_queries(orchestrator: RMLHybridOrchestrator):
    """Run example queries using the hybrid orchestrator with enhanced visualization"""
    queries = [
        # Basic facts
        ("What is Alice Chen's occupation?", "attribute"),
        ("Which papers did Alice Chen author?", "relationship"),
        ("What organizations are associated with the authors of 'Reinforcement Learning: Theory and Practice'?", "relationship"),
        
        # RAG-based queries
        ("What are some recent papers about deep learning?", "retrieval"),
        ("Tell me about research at Stanford AI Lab", "retrieval"),
        
        # Complex reasoning
        ("How is Alice Chen connected to MIT?", "path"),
        ("What's the connection between Alice's work and Dana's research?", "path"),
        ("List researchers working on computer vision", "query"),
        
        # Advanced queries
        ("What papers cite work by Alice Chen?", "citation"),
        ("Who are the most influential researchers in this graph?", "centrality"),
        ("Show me the collaboration network around Alice Chen", "subgraph")
    ]
    
    for query, query_type in queries:
        print(f"\n{'='*80}\nQuery: {query}\nType: {query_type.upper()}\n{'='*80}")
        
        try:
            # Process the query with the orchestrator
            result = orchestrator.process_query(query)
            
            # Enhanced output formatting with color coding
            print(f"\n\033[1mAnswer\033[0m (\033[36m{result['method'].upper()}\033[0m, "
                  f"confidence: \033[1m{result['confidence']:.2f}\033[0m):")
            print("-" * 40)
            print(result['answer'])
            
            # Visualize reasoning trace with more detail
            if 'trace' in result and result['trace']:
                print("\n\033[1mReasoning Trace:\033[0m")
                for i, step in enumerate(result['trace'], 1):
                    step_result = str(step.get('result', '')).replace('\n', ' ')[:100]
                    print(f"  {i}. \033[33m{step['step']}\033[0m: {step_result}" + 
                          ("..." if len(str(step.get('result', ''))) > 100 else ""))
            
            # Visualize relevant part of the graph with highlighting
            if hasattr(orchestrator, 'get_relevant_nodes'):
                relevant_nodes = orchestrator.get_relevant_nodes(query)
                if relevant_nodes:
                    # Get nodes to highlight based on query type
                    highlight_nodes = []
                    if query_type in ['attribute', 'relationship']:
                        # For attribute/relationship queries, highlight the main entities
                        highlight_nodes = relevant_nodes[:2]  # First couple of nodes
                    elif query_type == 'path':
                        # For path queries, highlight all nodes in the path
                        highlight_nodes = relevant_nodes
                    
                    print(f"\n\033[1mVisualizing {len(relevant_nodes)} relevant nodes...\033[0m")
                    visualize_subgraph(
                        relevant_nodes, 
                        f"{query_type.upper()}: {query[:30]}", 
                        orchestrator.graph,
                        highlight_nodes=highlight_nodes
                    )
                    
        except Exception as e:
            print(f"\n\033[91mError processing query: {str(e)}\033[0m")
            import traceback
            traceback.print_exc()

def visualize_subgraph(node_ids: List[str], query: str, graph: RMGraph, highlight_nodes: List[str] = None):
    """
    Visualize a subgraph containing the specified nodes and their connections
    with enhanced styling and interactivity hints.
    
    Args:
        node_ids: List of node IDs to include in the subgraph
        query: The original query (used for the plot title)
        graph: The RMGraph instance
        highlight_nodes: List of node IDs to highlight in the visualization
    """
    if highlight_nodes is None:
        highlight_nodes = []
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create a new directed graph for visualization
        G = nx.DiGraph()
        
        # Add nodes with their types and labels
        for node_id in node_ids:
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                G.add_node(
                    node_id,
                    label=f"{node.type}: {node.data.get('name', node.data.get('title', node_id[:6]))}",
                    type=node.type
                )
        
        # Add edges between nodes
        for edge in graph.edges:
            if edge.source in node_ids and edge.target in node_ids:
                G.add_edge(edge.source, edge.target, label=edge.type)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Enhanced node styling by type with better visual distinction
        node_styles = {
            'person': {
                'color': '#4e79a7',  # Muted blue
                'shape': 'o',
                'size': 2000,
                'alpha': 0.9,
                'edgecolor': '#2e5a8c',
                'linewidth': 2
            },
            'paper': {
                'color': '#59a14f',  # Muted green
                'shape': 's',  # Square
                'size': 1800,
                'alpha': 0.8,
                'edgecolor': '#3d7a34',
                'linewidth': 1.5
            },
            'organization': {
                'color': '#e15759',  # Muted red
                'shape': 'd',  # Diamond
                'size': 2200,
                'alpha': 0.8,
                'edgecolor': '#c13c3e',
                'linewidth': 1.5
            },
            'conference': {
                'color': '#b07aa1',  # Muted purple
                'shape': '^',  # Triangle up
                'size': 2000,
                'alpha': 0.8,
                'edgecolor': '#8f5f8e',
                'linewidth': 1.5
            },
            'default': {
                'color': '#bab0ac',
                'shape': 'o',
                'size': 1500,
                'alpha': 0.6,
                'edgecolor': '#9e9e9e',
                'linewidth': 1
            }
        }
        
        # Draw nodes with enhanced styling
        for node_type in set(nx.get_node_attributes(G, 'type').values()):
            nodes = [n for n, t in nx.get_node_attributes(G, 'type').items() if t == node_type]
            style = node_styles.get(node_type, node_styles['default'])
            
            # Highlight nodes that are directly relevant to the query
            node_colors = []
            for n in nodes:
                if n in highlight_nodes:
                    # Brighter version of the base color for highlighted nodes
                    base_color = style['color']
                    highlighted_color = f"#{int(base_color[1:3], 16) + 0x20:02x}{int(base_color[3:5], 16) + 0x20:02x}{int(base_color[5:], 16) + 0x20:02x}"
                    node_colors.append(highlighted_color)
                else:
                    node_colors.append(style['color'])
            
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes,
                node_color=node_colors,
                node_size=[style['size'] * (1.5 if n in highlight_nodes else 1.0) for n in nodes],
                alpha=style['alpha'],
                edgecolors=style['edgecolor'],
                linewidths=style['linewidth'],
                node_shape=style['shape'],
                label=f"{node_type.title()}s ({len(nodes)})"
            )
        
        # Draw edges with enhanced styling
        edge_widths = [2.0 if (u in highlight_nodes or v in highlight_nodes) else 1.0 
                      for u, v in G.edges()]
        edge_alphas = [0.8 if (u in highlight_nodes or v in highlight_nodes) else 0.4 
                      for u, v in G.edges()]
        
        nx.draw_networkx_edges(
            G, pos, 
            width=edge_widths,
            alpha=edge_alphas,
            arrows=True, 
            arrowsize=15,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1',
            edge_color='#666666'
        )
        
        # Add edge labels with better formatting
        edge_labels = nx.get_edge_attributes(G, 'label')
        curved_edge_labels = {}
        for (u, v), label in edge_labels.items():
            # Only show labels for edges connected to highlighted nodes
            if u in highlight_nodes or v in highlight_nodes:
                # Make the label more readable
                if '_' in label:
                    label = label.replace('_', ' ').title()
                curved_edge_labels[(u, v)] = label
        
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=curved_edge_labels,
            font_size=8,
            font_color='#333333',
            bbox=dict(
                alpha=0.7,
                boxstyle='round,pad=0.3',
                ec='none',
                fc='white'
            )
        )
        
        # Draw node labels
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Add title and legend
        plt.title(f"Knowledge Graph Subgraph for: {query[:50]}" + ("..." if len(query) > 50 else ""), 
                 fontsize=10)
        plt.legend(scatterpoints=1, fontsize=8)
        plt.axis('off')
        
        # Save and show the plot
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate a filename from the query
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
        output_path = output_dir / f"graph_{safe_query}.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n\033[1mGraph visualization saved to: {output_path}\033[0m")
        plt.close()
        
    except ImportError as e:
        print(f"\n\033[93mWarning: Could not generate graph visualization. {str(e)}\033[0m")
    except Exception as e:
        print(f"\n\033[91mError generating graph visualization: {str(e)}\033[0m")

def main():
    print("=== RML Hybrid Reasoning Demo ===\n")
    
    # 1. Load sample data and build the knowledge graph
    print("Building knowledge graph...")
    data = load_sample_data()
    graph, node_map = build_knowledge_graph(data)
    
    # 2. Initialize the reasoning components
    print("Initializing reasoning components...")
    reasoner = Reasoner(graph)
    retriever = RMLRetriever(graph)
    
    # 3. Create the hybrid orchestrator
    print("Initializing hybrid orchestrator...\n")
    orchestrator = RMLHybridOrchestrator(graph, reasoner, retriever)
    
    # 4. Run example queries
    print("Running example queries...\n")
    run_hybrid_queries(orchestrator)
    
    # 5. Demonstrate LangChain integration
    print("\n" + "="*80)
    print("LangChain Integration Demo")
    print("="*80)
    
    langchain_wrapper = RMLLangChainWrapper(graph, reasoner, retriever)
    tool = langchain_wrapper.as_tool(
        name="rml_knowledge",
        description="A tool for querying the RML knowledge graph"
    )
    
    # Example of using the tool
    print("\nExample LangChain tool usage:")
    print(f"Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Result for 'What is Alice's occupation?':")
    print(tool("What is Alice's occupation?"))
    
    print("\n=== End of Demo ===")

if __name__ == "__main__":
    main()
