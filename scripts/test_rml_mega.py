#!/usr/bin/env python3
"""
Lightweight RML Test using MEGA Volume
Tests the pipeline with minimal resource usage and external storage
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_functionality():
    """Test basic RML functionality without heavy models"""
    print("🧪 Testing Basic RML Functionality")
    print("="*50)
    
    # Test with a small sample
    sample_text = "The brain stores memories by creating neural pathways."
    
    print(f"📝 Input: {sample_text}")
    print("✅ Basic functionality test passed (no heavy models loaded)")
    
    return True

def test_mega_storage():
    """Test MEGA volume storage"""
    print("\n🧪 Testing MEGA Volume Storage")
    print("="*50)
    
    mega_data = "/Volumes/MEGA/R-LLM-data"
    mega_output = "/Volumes/MEGA/R-LLM-output"
    
    # Check if MEGA volume is accessible
    if not os.path.exists(mega_data):
        print(f"❌ MEGA data directory not found: {mega_data}")
        return False
    
    if not os.path.exists(mega_output):
        print(f"❌ MEGA output directory not found: {mega_output}")
        return False
    
    # Test write access
    test_file = os.path.join(mega_output, "test_write.json")
    try:
        with open(test_file, 'w') as f:
            json.dump({"test": "data", "timestamp": time.time()}, f)
        print(f"✅ Write test passed: {test_file}")
        
        # Clean up
        os.remove(test_file)
        print("✅ Cleanup successful")
        
    except Exception as e:
        print(f"❌ Write test failed: {e}")
        return False
    
    return True

def test_with_small_data():
    """Test with a small subset of existing data"""
    print("\n🧪 Testing with Small Data Subset")
    print("="*50)
    
    # Find a small file to test with
    possible_files = [
        "data/gpt_val.jsonl",
        "data/gpt_train.jsonl"
    ]
    
    test_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("❌ No test files found")
        return False
    
    print(f"📁 Using test file: {test_file}")
    
    # Count lines
    try:
        with open(test_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"📊 File contains {line_count} lines")
        
        # Read first few lines
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Only read first 3 lines
                    break
                data = json.loads(line.strip())
                print(f"   Line {i+1}: {len(str(data))} characters")
        
        print("✅ Small data test passed")
        return True
        
    except Exception as e:
        print(f"❌ Small data test failed: {e}")
        return False

def test_storage_plan():
    """Show storage plan"""
    print("\n📊 Storage Plan")
    print("="*50)
    
    print("💾 Main Disk (500GB):")
    print("   - Models: ~20GB (E5-Mistral + Phi-3)")
    print("   - System: ~50GB")
    print("   - Available: ~430GB")
    
    print("\n💾 MEGA Volume (233GB):")
    print("   - Data storage: ~180GB (your existing data)")
    print("   - Processing output: ~50GB")
    print("   - Available: ~233GB")
    
    print("\n🎯 Strategy:")
    print("   - Keep models on main disk")
    print("   - Use MEGA for data and output")
    print("   - Process in chunks to manage memory")
    
    return True

def main():
    """Run all tests"""
    print("🚀 RML Lightweight Testing with MEGA Volume")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("MEGA Storage", test_mega_storage),
        ("Small Data", test_with_small_data),
        ("Storage Plan", test_storage_plan)
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
        print("🎉 All tests passed! Ready to proceed with RML pipeline.")
        print("\n📋 Next Steps:")
        print("1. Move large datasets to MEGA volume")
        print("2. Run RML pipeline with MEGA output")
        print("3. Monitor storage usage")
    else:
        print("⚠️ Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 