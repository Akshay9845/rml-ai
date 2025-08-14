#!/usr/bin/env python3
"""
Test script for the enhanced RML Pipeline
"""
import sys
from test_rml_pipeline import RMLPipeline

def test_pipeline():
    """Test the RML pipeline with sample queries"""
    print("🚀 Initializing RML Pipeline...")
    pipeline = RMLPipeline()
    
    test_queries = [
        "What is Python?",
        "Tell me about machine learning",
        "What's the capital of France?",
        "Explain quantum computing"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"🔍 Query: {query}")
        response = pipeline.process_text(query)
        print(f"💬 Response: {response}")
        print(f"{'='*50}")

if __name__ == "__main__":
    test_pipeline()
