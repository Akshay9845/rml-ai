#!/usr/bin/env python3
"""
Fixed RML Processor Runner
Processes each folder with corrected component detection
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
    
    # Check available disk space
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / (1024**3)
    print(f"   💾 Disk: {disk_gb:.1f}GB available")
    
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

def compile_fixed_processor():
    """Compile the fixed processor"""
    print("\n🔨 Compiling Fixed RML Processor...")
    
    cpp_file = "scripts/fixed_rml_processor.cpp"
    output_file = "scripts/fixed_rml_processor"
    
    if not os.path.exists(cpp_file):
        print(f"❌ C++ source file not found: {cpp_file}")
        return False
    
    # Compiler flags for maximum speed
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

def run_fixed_processor(data_dir, output_dir):
    """Run the compiled fixed processor"""
    print(f"\n🚀 Running Fixed RML Processor...")
    print(f"   📁 Input: {data_dir}")
    print(f"   📁 Output: {output_dir}")
    
    processor_path = "scripts/fixed_rml_processor"
    
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
            print("🎉 Fixed Processing completed successfully!")
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
    print("🚀 Fixed RML Processor - Corrected Component Detection")
    print("="*60)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met!")
        return False
    
    # Compile processor
    if not compile_fixed_processor():
        print("❌ Compilation failed!")
        return False
    
    # Get directories
    data_dir = "data"
    output_dir = "output/fixed_processing"
    
    # Confirm before running
    print(f"\n📋 Processing Plan:")
    print(f"   Input: {data_dir} (355GB)")
    print(f"   Output: {output_dir}")
    print(f"   This will correctly detect and assemble RML components")
    print(f"   Each folder will create its own complete RML file")
    
    response = input("\n❓ Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return False
    
    # Run processor
    success = run_fixed_processor(data_dir, output_dir)
    
    if success:
        print(f"\n✅ Processing complete!")
        print(f"📁 Check {output_dir} for results:")
        
        # Show output stats
        if os.path.exists(output_dir):
            total_size = 0
            file_count = 0
            for file in os.listdir(output_dir):
                if file.endswith('_complete.jsonl'):
                    filepath = os.path.join(output_dir, file)
                    size = os.path.getsize(filepath) / (1024**3)  # GB
                    total_size += size
                    file_count += 1
                    print(f"   📄 {file}: {size:.2f}GB")
            print(f"   📊 Total: {file_count} folders, {total_size:.2f}GB")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 