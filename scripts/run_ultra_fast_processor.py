#!/usr/bin/env python3
"""
Ultra-Fast RML Processor Runner
Compiles and runs the C++ processor for 355GB data
"""

import os
import sys
import subprocess
import platform
import psutil
import time

def check_system_requirements():
    """Check if system meets requirements"""
    print("🔍 Checking system requirements...")
    
    # Check available RAM
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    print(f"   💻 RAM: {ram_gb:.1f}GB available")
    
    if ram_gb < 8:
        print("   ⚠️ Warning: Less than 8GB RAM available")
    
    # Check available disk space
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / (1024**3)
    print(f"   💾 Disk: {disk_gb:.1f}GB available")
    
    if disk_gb < 100:
        print("   ⚠️ Warning: Less than 100GB disk space available")
    
    # Check C++ compiler
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(['clang++', '--version'], capture_output=True, text=True)
            print("   ✅ clang++ compiler found")
        else:
            result = subprocess.run(['g++', '--version'], capture_output=True, text=True)
            print("   ✅ g++ compiler found")
    except FileNotFoundError:
        print("   ❌ C++ compiler not found!")
        return False
    
    return True

def compile_cpp_processor():
    """Compile the C++ processor"""
    print("\n🔨 Compiling C++ Ultra-Fast RML Processor...")
    
    cpp_file = "scripts/ultra_fast_rml_processor.cpp"
    output_file = "scripts/ultra_fast_rml_processor"
    
    if not os.path.exists(cpp_file):
        print(f"❌ C++ source file not found: {cpp_file}")
        return False
    
    # Compiler flags
    if platform.system() == "Darwin":  # macOS
        compiler = "clang++"
        flags = [
            "-std=c++17",
            "-O3",  # Maximum optimization
            "-march=native",  # Use native CPU instructions
            "-mtune=native",
            "-ffast-math",
            "-funroll-loops",
            "-fomit-frame-pointer",
            "-Wall",
            "-Wextra"
        ]
    else:  # Linux
        compiler = "g++"
        flags = [
            "-std=c++17",
            "-O3",
            "-march=native",
            "-mtune=native",
            "-ffast-math",
            "-funroll-loops",
            "-fomit-frame-pointer",
            "-Wall",
            "-Wextra"
        ]
    
    # Build command
    cmd = [compiler] + flags + [cpp_file, "-o", output_file]
    
    print(f"   Compiling with: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Compilation successful!")
            
            # Make executable
            os.chmod(output_file, 0o755)
            print("   ✅ Executable created!")
            return True
        else:
            print(f"   ❌ Compilation failed!")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Compilation error: {e}")
        return False

def run_processor(data_dir, output_dir):
    """Run the compiled processor"""
    print(f"\n🚀 Running Ultra-Fast RML Processor...")
    print(f"   📁 Input: {data_dir}")
    print(f"   📁 Output: {output_dir}")
    
    processor_path = "scripts/ultra_fast_rml_processor"
    
    if not os.path.exists(processor_path):
        print(f"❌ Processor executable not found: {processor_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run processor
    cmd = [processor_path, data_dir, output_dir]
    
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.rstrip())
        
        return_code = process.poll()
        
        if return_code == 0:
            print("\n" + "="*80)
            print("🎉 Ultra-Fast RML Processing completed successfully!")
            print(f"   Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        else:
            print(f"\n❌ Processor failed with return code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error running processor: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Ultra-Fast RML Processor for 355GB Data")
    print("="*60)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met!")
        return False
    
    # Compile processor
    if not compile_cpp_processor():
        print("❌ Compilation failed!")
        return False
    
    # Get directories
    data_dir = "data"
    output_dir = "output/training_data"
    
    # Confirm before running
    print(f"\n📋 Processing Plan:")
    print(f"   Input: {data_dir} (355GB)")
    print(f"   Output: {output_dir}")
    print(f"   This will process ALL RML data and create GPT-ready format")
    
    response = input("\n❓ Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return False
    
    # Run processor
    success = run_processor(data_dir, output_dir)
    
    if success:
        print(f"\n✅ Processing complete!")
        print(f"📁 Check {output_dir} for results:")
        print(f"   - train.jsonl (80% of data)")
        print(f"   - validation.jsonl (10% of data)")
        print(f"   - test.jsonl (10% of data)")
        
        # Show output stats
        if os.path.exists(output_dir):
            total_size = 0
            for file in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
                filepath = os.path.join(output_dir, file)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath) / (1024**3)  # GB
                    total_size += size
                    print(f"   📄 {file}: {size:.2f}GB")
            print(f"   📊 Total output: {total_size:.2f}GB")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 