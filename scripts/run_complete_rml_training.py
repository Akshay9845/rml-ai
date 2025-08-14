#!/usr/bin/env python3
"""
Run Complete RML Training - 100% Data Coverage
This script runs the RML complete trainer that processes ALL 43,923 files
"""

import os
import sys
import subprocess
import time
import psutil
from pathlib import Path

def check_system_resources():
    """Check if system has enough resources"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/Users/elite/R-LLM')
    
    print(f"💾 System Memory: {memory.total / (1024**3):.1f}GB total")
    print(f"💾 Available Memory: {memory.available / (1024**3):.1f}GB")
    print(f"💾 Disk Space: {disk.free / (1024**3):.1f}GB free")
    
    if memory.available < 8 * (1024**3):  # Less than 8GB
        print("⚠️  Warning: Low memory available")
        return False
    
    if disk.free < 50 * (1024**3):  # Less than 50GB
        print("⚠️  Warning: Low disk space")
        return False
    
    return True

def count_total_files():
    """Count total RML files"""
    data_dir = "/Users/elite/R-LLM/data"
    rml_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jsonl'):
                filepath = os.path.join(root, file)
                filename = os.path.basename(filepath)
                if any(keyword in filename for keyword in ['concepts', 'entities', 'triples', 'emotions', 'intents', 'events', 'vectors', 'reasoning', 'summaries', 'tags']):
                    rml_files.append(filepath)
    
    return len(rml_files)

def main():
    """Main function"""
    print("🚀 RML Complete Training - 100% Data Coverage")
    print("=" * 60)
    
    # Check system resources
    print("📊 Checking system resources...")
    if not check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    # Count total files
    print("📁 Counting RML files...")
    total_files = count_total_files()
    print(f"📂 Total RML files found: {total_files:,}")
    
    if total_files < 40000:
        print("⚠️  Warning: Expected ~43,923 files, found fewer")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Create output directories
    output_dir = "/Users/elite/R-LLM/rml-complete-trained"
    checkpoint_dir = "/Users/elite/R-LLM/rml-complete-checkpoints"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    # Run the complete trainer
    print("\n🎯 Starting COMPLETE RML training...")
    print("📊 This will process 100% of your RML data")
    print("⏱️  Expected time: 8-12 hours (depending on system)")
    print("💾 Expected memory usage: 12-16GB")
    
    # Start training in background
    cmd = [
        "python3", "src/rml_complete_trainer.py"
    ]
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("📝 Logs will be saved to: /Users/elite/R-LLM/rml-complete-trained/complete_training.log")
    
    try:
        # Run the training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("✅ Training started successfully!")
        print("📊 Monitor progress with: tail -f /Users/elite/R-LLM/rml-complete-trained/complete_training.log")
        print("🛑 To stop training: pkill -f 'rml_complete_trainer'")
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"❌ Error running training: {e}")

if __name__ == "__main__":
    main() 