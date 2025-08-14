#!/usr/bin/env python3
"""
Test script for the RML Pipeline
"""
import sys
import os
import time
from pathlib import Path
from test_rml_pipeline import RMLPipeline

def test_pipeline():
    """Test the RML pipeline with sample queries"""
    print("🚀 Initializing RML Pipeline...")
    
    # Initialize the pipeline with specific data files
    data_files = [
        "data/test_rml_data.jsonl",  # Small test dataset
        "data/real_training_data.jsonl",  # Real training data
        "data/proper_training_data.jsonl"  # Properly formatted training data
    ]
    
    # Check which files exist
    existing_files = [f for f in data_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️  No data files found. Using default data directory.")
        data_dir = "data"
    else:
        print(f"✅ Found {len(existing_files)} data files to use")
        # Create a temporary directory with symlinks to the data files
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp(prefix="rml_test_data_")
        for src in existing_files:
            dst = os.path.join(temp_dir, os.path.basename(src))
            os.symlink(os.path.abspath(src), dst)
        data_dir = temp_dir
    
    # Initialize the pipeline with the data directory
    pipeline = RMLPipeline(data_dir=data_dir)
    
    # Test queries with expected answer types
    test_queries = [
        {
            "question": "What is the theory of relativity?",
            "type": "scientific_concept"
        },
        {
            "question": "Tell me about the Eiffel Tower",
            "type": "factual_location"
        },
        {
            "question": "What is Python programming language?",
            "type": "technical_definition"
        },
        {
            "question": "Explain machine learning",
            "type": "technical_definition"
        },
        {
            "question": "Where is the Great Wall of China located?",
            "type": "geographical_fact"
        }
    ]
    
    print("\n🧪 Running tests...")
    print("-" * 80)
    
    for i, test in enumerate(test_queries, 1):
        query = test["question"]
        query_type = test["type"]
        
        print(f"\n🔍 Test {i}: {query} [{query_type}]")
        print("-" * (len(query) + len(query_type) + 14))  # Adjust underline length
        
        try:
            print(f"\n🔍 Processing query: {query}")
            print("-" * (len(query) + 20))
            
            # Process the query with timing
            start_time = time.time()
            response = pipeline.process_text(query)
            processing_time = time.time() - start_time
            
            # Display results with more details
            print(f"\n⏱️  Processing time: {processing_time:.2f}s")
            print("\n📝 Response:")
            print(response)
            
            # Basic quality check
            if len(response.strip().split()) < 3:  # Very short response
                print("⚠️  Warning: Response is very short")
            elif len(response) > 1000:  # Very long response
                print("⚠️  Warning: Response is very long")
            print("\n" + "="*80)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    test_pipeline()
