#!/usr/bin/env python3
"""
Basic Usage Example for RML-AI
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_ai import RMLSystem, RMLConfig


def main():
    """Basic usage example"""
    print("🚀 RML-AI Basic Usage Example")
    print("=" * 50)
    
    # Create configuration
    config = RMLConfig(
        encoder_model="intfloat/e5-base-v2",
        decoder_model="microsoft/phi-1_5",
        device="auto",
        dataset_path="data/rml_data.jsonl",
        max_entries=100
    )
    
    print(f"Configuration: {config}")
    print("=" * 50)
    
    try:
        # Initialize RML system
        print("Initializing RML system...")
        rml = RMLSystem(config)
        print("✅ RML system initialized successfully!")
        
        # Example queries
        queries = [
            "What is machine learning?",
            "What is artificial intelligence?",
            "What is RML?",
            "What latency does RML claim?"
        ]
        
        for query in queries:
            print(f"\n❓ Question: {query}")
            print("-" * 40)
            
            response = rml.query(query)
            print(response.answer)
            print(f"⏱️ Response time: {response.response_ms:.2f}ms")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 