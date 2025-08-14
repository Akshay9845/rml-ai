"""
Basic Usage Example for RML System

This example demonstrates how to use the core RML components together.
"""
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.graph import RMGraph, Node, Edge
from src.core.reasoning import Reasoner
from src.core.retrieval import RMLRetriever
from src.core.integration import RMLHybridOrchestrator

def create_sample_graph():
    """Create a sample knowledge graph for demonstration"""
    # Initialize the graph
    graph = RMGraph()
    
    # Add some nodes
    alice_id = graph.add_node("person", {"name": "Alice", "age": 30, "occupation": "researcher"})
    bob_id = graph.add_node("person", {"name": "Bob", "age": 35, "occupation": "professor"})
    paper1_id = graph.add_node("paper", {"title": "Advances in AI", "year": 2023})
    paper2_id = graph.add_node("paper", {"title": "Machine Learning Foundations", "year": 2022})
    uni_id = graph.add_node("organization", {"name": "Stanford University", "type": "university"})
    
    # Add relationships
    graph.add_edge(alice_id, paper1_id, "author")
    graph.add_edge(bob_id, paper1_id, "author")
    graph.add_edge(bob_id, paper2_id, "author")
    graph.add_edge(bob_id, uni_id, "affiliated_with")
    graph.add_edge(alice_id, bob_id, "knows")
    
    return graph

def main():
    print("=== RML System Demo ===\n")
    
    # 1. Create a sample knowledge graph
    print("Creating sample knowledge graph...")
    graph = create_sample_graph()
    
    # 2. Initialize the reasoning engine
    print("Initializing reasoning engine...")
    reasoner = Reasoner(graph)
    
    # 3. Initialize the retriever
    print("Initializing retriever...")
    retriever = RMLRetriever(graph)
    
    # 4. Create the hybrid orchestrator
    print("Initializing hybrid orchestrator...\n")
    orchestrator = RMLHybridOrchestrator(graph, reasoner, retriever)
    
    # 5. Run some example queries
    queries = [
        "How is Alice related to Bob?",
        "What papers has Bob authored?",
        "What is Alice's occupation?",
        "Who works at Stanford University?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Process the query using the hybrid approach
        result = orchestrator.process_query(query)
        
        print(f"Answer: {result['answer']}")
        print(f"(Method: {result['method']}, Confidence: {result['confidence']:.2f})")
        
        # Show the reasoning trace if available
        if 'trace' in result:
            print("\nReasoning Trace:")
            for i, step in enumerate(result['trace'], 1):
                print(f"  {i}. {step['step']}: {step.get('result', '')}")
    
    print("\n=== End of Demo ===")

if __name__ == "__main__":
    main()
