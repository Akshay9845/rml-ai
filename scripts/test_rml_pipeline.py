#!/usr/bin/env python3
"""
Test RML Pipeline
Comprehensive testing of the E5-Mistral → RML Memory → Phi-3 pipeline
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_encoder_decoder import RMLPipeline, RMLConfig
from rml_streaming_processor import RMLStreamingProcessor

def test_basic_pipeline():
    """Test basic pipeline functionality"""
    print("🧪 Testing Basic RML Pipeline")
    print("="*50)
    
    # Configuration for testing
    config = RMLConfig(
        output_dir="output/test_pipeline",
        save_memory_graphs=True,
        log_level="INFO",
        batch_size=2,  # Small batch for testing
        max_concepts_per_input=20
    )
    
    try:
        # Initialize pipeline
        pipeline = RMLPipeline(config)
        print("✅ Pipeline initialized successfully")
        
        # Test with sample text
        sample_text = """
        Artificial intelligence is transforming how we process information. 
        Machine learning algorithms can identify patterns in large datasets 
        and make predictions based on historical data. Deep learning, a subset 
        of machine learning, uses neural networks with multiple layers to 
        understand complex relationships in data.
        """
        
        print(f"📝 Testing with sample text (length: {len(sample_text)})")
        
        # Process text
        result = pipeline.process_text(sample_text, "How does AI process information?")
        
        print(f"🤖 Response: {result['response']}")
        print(f"📊 Concepts extracted: {len(result['extracted_concepts'])}")
        print(f"🧠 Memory size: {result['memory_size']} concepts")
        
        # Show some extracted concepts
        print("\n🔍 Sample extracted concepts:")
        for i, concept in enumerate(result['extracted_concepts'][:5]):
            print(f"  {i+1}. {concept['concept']} ({concept['type']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic pipeline test: {e}")
        return False

def test_with_existing_rml_data():
    """Test pipeline with existing RML data"""
    print("\n🧪 Testing with Existing RML Data")
    print("="*50)
    
    # Look for existing RML data files
    possible_files = [
        "data/cpp_rml_output_v5/concepts.jsonl",
        "data/cpp_rml_output/concepts.jsonl",
        "data/enhanced_rml_output/combined_clean.jsonl",
        "data/continuous_rml_output/TRAINED/combined.jsonl"
    ]
    
    test_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("❌ No existing RML data files found for testing")
        return False
    
    print(f"📁 Using test file: {test_file}")
    
    # Configuration
    config = RMLConfig(
        output_dir="output/test_rml_data",
        save_memory_graphs=True,
        log_level="INFO",
        batch_size=2,
        max_concepts_per_input=15
    )
    
    try:
        # Initialize pipeline
        pipeline = RMLPipeline(config)
        
        # Process a few lines from the file
        processed_count = 0
        max_test_items = 5
        
        with open(test_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip() and processed_count < max_test_items:
                    try:
                        rml_data = json.loads(line.strip())
                        
                        # Extract text
                        text = extract_text_from_rml(rml_data)
                        if text and len(text.strip()) > 30:
                            print(f"\n📄 Processing item {processed_count + 1}")
                            print(f"   Text: {text[:100]}...")
                            
                            result = pipeline.process_text(text)
                            print(f"   Response: {result['response'][:100]}...")
                            print(f"   Concepts: {len(result['extracted_concepts'])}")
                            
                            processed_count += 1
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"   Error: {e}")
                        continue
        
        print(f"\n✅ Successfully processed {processed_count} items")
        return True
        
    except Exception as e:
        print(f"❌ Error testing with RML data: {e}")
        return False

def extract_text_from_rml(rml_data):
    """Extract text from RML data structure"""
    text_fields = ['text', 'content', 'summary', 'description', 'input']
    
    for field in text_fields:
        if field in rml_data and rml_data[field]:
            return str(rml_data[field])
    
    # Try to reconstruct from concepts
    if 'concepts' in rml_data and rml_data['concepts']:
        concepts = rml_data['concepts']
        if isinstance(concepts, list):
            return " ".join([str(c) for c in concepts[:10]])
        elif isinstance(concepts, dict):
            return " ".join([str(v) for v in concepts.values()][:10])
    
    return ""

def test_streaming_processor():
    """Test streaming processor with large files"""
    print("\n🧪 Testing Streaming Processor")
    print("="*50)
    
    # Find a suitable test file
    test_file = "data/cpp_rml_output_v5/concepts.jsonl"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Configuration for streaming
    config = RMLConfig(
        output_dir="output/test_streaming",
        save_memory_graphs=True,
        log_level="INFO",
        batch_size=2,
        max_concepts_per_input=10  # Small for testing
    )
    
    try:
        # Initialize streaming processor
        processor = RMLStreamingProcessor(config, max_workers=1)
        
        # Process with small chunk size for testing
        result = processor.process_file(
            input_file=test_file,
            output_file="output/test_streaming/test_output.jsonl",
            chunk_size=10  # Small chunks for testing
        )
        
        print(f"📊 Streaming Results:")
        print(f"   Input: {result['input_file']}")
        print(f"   Output: {result['output_file']}")
        print(f"   Processed: {result['total_processed']} items")
        print(f"   Results: {result['total_results']} items")
        print(f"   Time: {result['processing_time']:.1f}s")
        print(f"   Rate: {result['rate']:.1f} items/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in streaming test: {e}")
        return False

def test_memory_persistence():
    """Test memory saving and loading"""
    print("\n🧪 Testing Memory Persistence")
    print("="*50)
    
    config = RMLConfig(
        output_dir="output/test_memory",
        save_memory_graphs=True,
        log_level="INFO"
    )
    
    try:
        # Initialize pipeline
        pipeline = RMLPipeline(config)
        
        # Add some concepts to memory
        test_texts = [
            "The brain processes information through neural networks.",
            "Machine learning algorithms can identify patterns in data.",
            "Deep learning uses multiple layers of neural networks."
        ]
        
        for text in test_texts:
            pipeline.process_text(text)
        
        initial_memory_size = len(pipeline.memory.concept_graph)
        print(f"📊 Initial memory size: {initial_memory_size} concepts")
        
        # Save memory
        memory_path = "output/test_memory/test_memory.json"
        pipeline.memory.save_memory(memory_path)
        print(f"💾 Memory saved to: {memory_path}")
        
        # Create new pipeline and load memory
        new_pipeline = RMLPipeline(config)
        new_pipeline.memory.load_memory(memory_path)
        
        loaded_memory_size = len(new_pipeline.memory.concept_graph)
        print(f"📊 Loaded memory size: {loaded_memory_size} concepts")
        
        if initial_memory_size == loaded_memory_size:
            print("✅ Memory persistence test passed")
            return True
        else:
            print("❌ Memory persistence test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in memory persistence test: {e}")
        return False

def test_interactive_mode():
    """Test interactive mode"""
    print("\n🧪 Testing Interactive Mode")
    print("="*50)
    print("💡 This will start an interactive session. Type 'quit' to exit.")
    
    config = RMLConfig(
        output_dir="output/test_interactive",
        save_memory_graphs=True,
        log_level="INFO"
    )
    
    try:
        pipeline = RMLPipeline(config)
        
        # Run interactive mode
        pipeline.interactive_mode()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RML Pipeline Comprehensive Testing")
    print("="*60)
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("Existing RML Data", test_with_existing_rml_data),
        ("Streaming Processor", test_streaming_processor),
        ("Memory Persistence", test_memory_persistence),
        ("Interactive Mode", test_interactive_mode)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RML pipeline is ready for production.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 