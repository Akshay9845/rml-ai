#!/usr/bin/env python3
"""
RML Ultimate Trainer - FIXED VERSION
Fixed tensor creation issues with proper padding and truncation
"""

import os
import json
import logging
import gc
import torch
import time
import argparse
import threading
import psutil
import signal
import sys
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import subprocess
import pickle
from pathlib import Path

@dataclass
class RMLUltimateFixedConfig:
    """Fixed configuration for ultimate RML training"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    
    # ULTIMATE SPEED SETTINGS
    batch_size: int = 4  # Reduced for stability
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 100
    max_seq_length: int = 64
    gradient_accumulation_steps: int = 16
    
    # CHECKPOINT SETTINGS
    save_steps: int = 1000
    save_total_limit: int = 10
    eval_steps: int = 2000
    
    # DATA SETTINGS
    data_root: str = "/Users/elite/R-LLM/data"
    output_dir: str = "/Volumes/MEGA/R-LLM-ultimate-fixed"
    checkpoint_dir: str = "/Volumes/MEGA/R-LLM-checkpoints-fixed"
    progress_file: str = "/Volumes/MEGA/training_progress_fixed.pkl"
    
    # MEMORY OPTIMIZATION
    device_map: str = "cpu"
    max_memory: Dict[str, str] = None
    
    # MONITORING
    log_level: str = "INFO"
    memory_check_interval: int = 30

class RMLUltimateFixedDataset(Dataset):
    """Fixed dataset with proper tensor handling"""
    
    def __init__(self, data_paths: List[str], tokenizer, max_length: int = 64, processed_files: Set[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_paths = data_paths
        self.processed_files = processed_files or set()
        self.data = []
        self.file_mapping = []
        self.load_all_data()
    
    def load_all_data(self):
        """Load ALL data from ALL files"""
        print(f"🚀 Loading ALL data from {len(self.data_paths)} files...")
        
        total_samples = 0
        memory_warnings = 0
        
        for file_path in self.data_paths:
            if file_path in self.processed_files:
                print(f"⏭️ Skipping already processed: {file_path}")
                continue
                
            try:
                if not os.path.exists(file_path):
                    continue
                    
                print(f"📂 Loading: {file_path}")
                file_samples = 0
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            item = json.loads(line.strip())
                            text = item.get('text', '')
                            
                            if text and len(text) > 10:
                                self.data.append(text)
                                self.file_mapping.append(file_path)
                                file_samples += 1
                                total_samples += 1
                                
                                # Memory check every 10000 samples
                                if total_samples % 10000 == 0:
                                    memory_usage = psutil.virtual_memory().percent
                                    print(f"   📊 Loaded {total_samples:,} samples, Memory: {memory_usage:.1f}%")
                                    
                                    if memory_usage > 85:
                                        memory_warnings += 1
                                        print(f"⚠️ High memory usage ({memory_usage:.1f}%) - Warning {memory_warnings}")
                                        
                                        if memory_warnings >= 3:
                                            print("🛑 Too many memory warnings, stopping data loading")
                                            break
                                
                        except json.JSONDecodeError:
                            continue
                
                print(f"✅ {file_path}: {file_samples:,} samples")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
        
        print(f"🎯 TOTAL LOADED: {len(self.data):,} samples from {len(set(self.file_mapping))} files")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # FIXED: Proper tokenization with padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',  # FIXED: Use max_length padding
            return_tensors='pt'
        )
        
        # FIXED: Ensure all tensors have the same shape
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }
    
    def get_processed_files(self) -> Set[str]:
        """Get list of files that have been processed"""
        return set(self.file_mapping)

class RMLUltimateFixedTrainer:
    """Fixed trainer with proper tensor handling"""
    
    def __init__(self, config: RMLUltimateFixedConfig):
        self.config = config
        self.processed_files = set()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{config.output_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Load progress
        self.load_progress()
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        # Performance tracking
        self.start_time = time.time()
        self.total_samples_processed = 0
        
        self.logger.info("🧠 Initializing RML Ultimate Fixed Trainer")
        self._load_model()
    
    def load_progress(self):
        """Load training progress"""
        if os.path.exists(self.config.progress_file):
            try:
                with open(self.config.progress_file, 'rb') as f:
                    progress = pickle.load(f)
                    self.processed_files = progress.get('processed_files', set())
                    self.total_samples_processed = progress.get('total_samples', 0)
                self.logger.info(f"📂 Loaded progress: {len(self.processed_files)} files processed, {self.total_samples_processed:,} samples")
            except Exception as e:
                self.logger.error(f"❌ Error loading progress: {e}")
    
    def save_progress(self):
        """Save training progress"""
        try:
            progress = {
                'processed_files': self.processed_files,
                'total_samples': self.total_samples_processed,
                'timestamp': time.time()
            }
            with open(self.config.progress_file, 'wb') as f:
                pickle.dump(progress, f)
            self.logger.info(f"💾 Progress saved: {len(self.processed_files)} files, {self.total_samples_processed:,} samples")
        except Exception as e:
            self.logger.error(f"❌ Error saving progress: {e}")
    
    def _load_model(self):
        """Load model with fixed settings"""
        try:
            self.logger.info(f"🔧 Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # FIXED: Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def get_all_data_files(self) -> List[str]:
        """Get ALL data files from the data directory"""
        data_files = []
        
        print(f"🔍 Scanning for ALL data files in {self.config.data_root}...")
        
        for root, dirs, files in os.walk(self.config.data_root):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    data_files.append(file_path)
        
        print(f"📁 Found {len(data_files):,} JSONL files")
        return data_files
    
    def train_ultimate(self):
        """Train on the ENTIRE dataset with fixes"""
        try:
            # Get ALL data files
            all_data_files = self.get_all_data_files()
            
            if not all_data_files:
                self.logger.error("❌ No data files found!")
                return
            
            # Create dataset with ALL data
            dataset = RMLUltimateFixedDataset(
                all_data_files,
                self.tokenizer,
                self.config.max_seq_length,
                self.processed_files
            )
            
            if len(dataset) == 0:
                self.logger.error("❌ No training data available")
                return
            
            # Create validation dataset (10% of data)
            val_size = min(len(dataset) // 10, 10000)
            train_size = len(dataset) - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # FIXED: Create data collator with proper settings
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8  # FIXED: Ensure proper padding
            )
            
            # Calculate total steps
            total_steps = len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)
            self.logger.info(f"📊 Total training steps: {total_steps:,}")
            self.logger.info(f"📊 Training samples: {len(train_dataset):,}")
            self.logger.info(f"📊 Validation samples: {len(val_dataset):,}")
            
            # FIXED Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_steps=100,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_total_limit=self.config.save_total_limit,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,
                # FIXED: CPU settings
                no_cuda=True,
                # FIXED: Proper settings
                max_grad_norm=1.0,
                fp16=False,
                bf16=False,
                gradient_checkpointing=False,
                optim="adamw_torch",
                weight_decay=0.01,
                # FIXED: Data collator settings
                dataloader_num_workers=0,  # FIXED: Avoid multiprocessing issues
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            
            self.logger.info("🚀 Starting ULTIMATE FIXED training on FULL dataset...")
            self.logger.info(f"⏱️ Expected time: {total_steps * 1.5 / 3600:.1f} hours")
            
            # Train the model
            trainer.train()
            
            # Update progress
            self.processed_files.update(dataset.get_processed_files())
            self.total_samples_processed += len(dataset)
            self.save_progress()
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Calculate training time
            training_time = time.time() - self.start_time
            self.logger.info(f"✅ ULTIMATE FIXED training completed in {training_time/3600:.2f} hours!")
            self.logger.info(f"💾 Model saved to {self.config.output_dir}")
            self.logger.info(f"📊 Total samples processed: {self.total_samples_processed:,}")
            
            # Evaluate
            eval_results = trainer.evaluate()
            self.logger.info(f"📊 Final evaluation: {eval_results}")
            
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            # Save progress even on error
            self.save_progress()
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        gc.collect()
        
        self.logger.info("🧹 Resources cleaned up")

def main():
    """Main function for fixed ultimate training"""
    
    parser = argparse.ArgumentParser(description="RML Ultimate Fixed Training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="/Volumes/MEGA/R-LLM-ultimate-fixed", help="Output directory")
    args = parser.parse_args()
    
    print("🧠 RML Ultimate Fixed Training Pipeline")
    print("="*50)
    print(f"💾 Output: {args.output}")
    print(f"🔄 Resume: {args.resume}")
    
    # Configuration
    config = RMLUltimateFixedConfig(
        output_dir=args.output,
        checkpoint_dir=f"{args.output}/checkpoints",
        progress_file=f"{args.output}/training_progress.pkl"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLUltimateFixedTrainer(config)
        
        # Train on ENTIRE dataset
        print("🚀 Starting ULTIMATE FIXED training on FULL dataset...")
        trainer.train_ultimate()
        
        print("✅ RML Ultimate Fixed training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 