#!/usr/bin/env python3
"""
Train Full Dataset - Batch processing with checkpointing
Processes 200GB dataset in chunks to complete in 5-9 hours on Mac M3 Pro
"""

import os
import json
import subprocess
import time
import argparse
from pathlib import Path

def get_total_samples(data_path: str) -> int:
    """Count total samples in dataset"""
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json.loads(line.strip())
                count += 1
            except:
                continue
    return count

def create_batch_data(data_path: str, batch_size: int, batch_num: int, output_dir: str):
    """Create a batch of training data"""
    
    batch_path = os.path.join(output_dir, f"batch_{batch_num}.jsonl")
    
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size
    
    print(f"📦 Creating batch {batch_num}: samples {start_idx:,} to {end_idx:,}")
    
    with open(data_path, 'r', encoding='utf-8') as fin, open(batch_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i >= end_idx:
                break
            if i >= start_idx:
                fout.write(line)
    
    return batch_path

def train_batch(batch_path: str, output_dir: str, checkpoint_dir: str = None, batch_num: int = 0):
    """Train on a single batch"""
    
    print(f"🚀 Training batch {batch_num}")
    print(f"📁 Batch data: {batch_path}")
    print(f"📁 Output: {output_dir}")
    
    # Calculate samples in this batch
    sample_count = sum(1 for _ in open(batch_path, 'r'))
    print(f"📊 Samples in batch: {sample_count:,}")
    
    # Find latest checkpoint
    resume_checkpoint = None
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            resume_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"🔄 Resuming from checkpoint: {resume_checkpoint}")
    
    # Run training command
    cmd = [
        "python3", "src/rml_scalable_trainer.py",
        "--samples", str(sample_count),
        "--output", output_dir
    ]
    
    if resume_checkpoint:
        cmd.extend(["--resume", resume_checkpoint])
    
    print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Batch {batch_num} completed successfully")
            return True
        else:
            print(f"❌ Batch {batch_num} failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Batch {batch_num} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Batch {batch_num} error: {e}")
        return False

def main():
    """Main function for full dataset training"""
    
    parser = argparse.ArgumentParser(description="Train Full RML Dataset")
    parser.add_argument("--data", type=str, default="data/language_training_data.jsonl", help="Training data path")
    parser.add_argument("--batch-size", type=int, default=50000, help="Samples per batch")
    parser.add_argument("--output", type=str, default="/Volumes/MEGA/R-LLM-full-trained", help="Output directory")
    parser.add_argument("--resume-batch", type=int, default=0, help="Resume from batch number")
    args = parser.parse_args()
    
    print("🧠 RML Full Dataset Training Pipeline")
    print("="*50)
    print(f"📊 Data: {args.data}")
    print(f"📦 Batch size: {args.batch_size:,} samples")
    print(f"💾 Output: {args.output}")
    print(f"🔄 Resume from batch: {args.resume_batch}")
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"❌ Training data not found: {args.data}")
        return
    
    # Get total samples
    total_samples = get_total_samples(args.data)
    print(f"📊 Total samples: {total_samples:,}")
    
    # Calculate number of batches
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    print(f"📦 Total batches: {num_batches}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create batch data directory
    batch_data_dir = os.path.join(args.output, "batch_data")
    os.makedirs(batch_data_dir, exist_ok=True)
    
    # Training statistics
    start_time = time.time()
    successful_batches = 0
    failed_batches = []
    
    # Process batches
    for batch_num in range(args.resume_batch, num_batches):
        print(f"\n{'='*60}")
        print(f"🎯 Processing batch {batch_num + 1}/{num_batches}")
        print(f"{'='*60}")
        
        # Create batch data
        batch_path = create_batch_data(args.data, args.batch_size, batch_num, batch_data_dir)
        
        # Train on batch
        batch_output_dir = os.path.join(args.output, f"batch_{batch_num}")
        checkpoint_dir = os.path.join(args.output, "checkpoints")
        
        success = train_batch(batch_path, batch_output_dir, checkpoint_dir, batch_num)
        
        if success:
            successful_batches += 1
            print(f"✅ Batch {batch_num} completed successfully")
        else:
            failed_batches.append(batch_num)
            print(f"❌ Batch {batch_num} failed - you can resume from this batch")
            break
        
        # Calculate progress
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / (batch_num + 1)
        remaining_batches = num_batches - (batch_num + 1)
        estimated_remaining = remaining_batches * avg_time_per_batch
        
        print(f"📊 Progress: {batch_num + 1}/{num_batches} batches")
        print(f"⏱️ Elapsed: {elapsed_time/3600:.2f} hours")
        print(f"⏱️ Estimated remaining: {estimated_remaining/3600:.2f} hours")
        print(f"⏱️ Total estimated: {(elapsed_time + estimated_remaining)/3600:.2f} hours")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("🎉 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful batches: {successful_batches}/{num_batches}")
    print(f"❌ Failed batches: {len(failed_batches)}")
    if failed_batches:
        print(f"🔄 Resume from batch: {failed_batches[0]}")
    print(f"⏱️ Total time: {total_time/3600:.2f} hours")
    print(f"📁 Final model: {args.output}")
    
    if successful_batches == num_batches:
        print("🎉 ALL BATCHES COMPLETED SUCCESSFULLY!")
    else:
        print(f"⚠️ Training incomplete. Resume with: --resume-batch {failed_batches[0]}")

if __name__ == "__main__":
    main() 