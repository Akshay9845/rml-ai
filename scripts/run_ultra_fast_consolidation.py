#!/usr/bin/env python3
"""
Ultra-Fast RML Data Consolidation Runner
Compiles and runs C++ version for maximum speed (2-3 hours for 372GB)
"""

import os
import sys
import subprocess
import argparse
import time
import platform
from pathlib import Path

def compile_cpp_consolidator():
    """Compile the ultra-fast C++ consolidator"""
    print("🔨 Compiling ultra-fast C++ consolidator...")
    
    cpp_file = "scripts/ultra_fast_consolidate.cpp"
    output_file = "scripts/ultra_fast_consolidate"
    
    # Detect OS and use appropriate compiler flags
    system = platform.system()
    
    if system == "Darwin":  # macOS
        compile_cmd = [
            "clang++", "-std=c++17", "-O3", "-march=native",
            "-pthread", "-DNDEBUG",
            "-o", output_file, cpp_file
        ]
    else:  # Linux/Unix
        compile_cmd = [
            "g++", "-std=c++17", "-O3", "-march=native", "-mtune=native",
            "-pthread", "-fopenmp", "-DNDEBUG",
            "-o", output_file, cpp_file
        ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ C++ consolidator compiled successfully!")
            return output_file
        else:
            print(f"❌ Compilation failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return None

def run_ultra_fast_consolidation(data_dir, output_dir, max_workers=16, background=False):
    """Run the ultra-fast consolidation"""
    
    # Compile C++ version
    cpp_executable = compile_cpp_consolidator()
    if not cpp_executable:
        print("❌ Failed to compile C++ consolidator!")
        return False
    
    # Prepare command
    cmd = [cpp_executable, data_dir, output_dir, str(max_workers)]
    
    print(f"🚀 Starting ULTRA-FAST consolidation...")
    print(f"📁 Data dir: {data_dir}")
    print(f"📁 Output dir: {output_dir}")
    print(f"⚡ Workers: {max_workers}")
    print(f"🕐 Expected time: 2-3 hours")
    
    if background:
        print("🔄 Running in background...")
        # Run in background with nohup
        cmd = ["nohup"] + cmd + [">", "consolidation.log", "2>&1", "&"]
        subprocess.run(" ".join(cmd), shell=True)
        print("✅ Background process started! Check consolidation.log for progress.")
        return True
    else:
        # Run in foreground
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=True)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"\n🎉 Consolidation completed in {duration/3600:.2f} hours!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Consolidation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Ultra-fast RML data consolidation")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-workers", type=int, default=16, help="Number of threads")
    parser.add_argument("--background", action="store_true", help="Run in background")
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run consolidation
    success = run_ultra_fast_consolidation(
        args.data_dir, 
        args.output_dir, 
        args.max_workers,
        args.background
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 