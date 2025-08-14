#!/usr/bin/env python3
"""
Run ULTRA-FAST RML Training - 100x Speed!
"""

import os
import sys
import subprocess
import time

def main():
    print("🚀 STARTING ULTRA-FAST RML TRAINING - 100x SPEED!")
    print("=" * 60)
    
    # Check if ultra-fast trainer exists
    trainer_path = "src/rml_ultra_fast_optimized_trainer.py"
    if not os.path.exists(trainer_path):
        print(f"❌ Error: {trainer_path} not found!")
        return
    
    # Create output directory
    output_dir = "/Users/elite/R-LLM/rml-ultra-trained"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start ultra-fast training
    print("🚀 Launching ultra-fast trainer...")
    print(f"📁 Output: {output_dir}")
    print(f"⚡ Expected: 100x faster than previous version")
    print("=" * 60)
    
    try:
        # Run the ultra-fast trainer
        result = subprocess.run([
            sys.executable, trainer_path
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ ULTRA-FAST training completed successfully!")
        else:
            print(f"❌ ULTRA-FAST training failed with code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏸️ ULTRA-FAST training interrupted by user")
    except Exception as e:
        print(f"❌ ULTRA-FAST training error: {e}")

if __name__ == "__main__":
    main() 