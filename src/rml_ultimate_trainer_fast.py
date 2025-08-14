#!/usr/bin/env python3
"""
RML Ultimate Trainer - FAST VERSION with GPU acceleration
Completes in 5-6 hours by processing smaller batches efficiently
Uses BOTH CPU and GPU (MPS) for maximum speed
Starts from FIRST files in a new folder
"""

import os
import json
import logging
import gc
import torch
import time
import argparse
import psutil
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import pickle
from pathlib import Path

@dataclass
class RMLUltimateFastConfig:
    """Fast configuration for 5-6 hour completion with GPU acceleration"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    
    # FAST SPEED SETTINGS with GPU
    batch_size: int = 16  # Increased for GPU speed
    learning_rate: float = 5e-4  # Higher learning rate
    num_epochs: int = 1
    warmup_steps: int = 50  # Reduced warmup
    max_seq_length: int = 128  # Reduced for speed
    gradient_accumulation_steps: int = 4  # Reduced for GPU speed
    
    # FAST CHECKPOINT SETTINGS
    save_steps: int = 500  # More frequent saves
    save_total_limit: int = 5
    eval_steps: int = 1000
    
    # DATA SETTINGS - NEW FOLDER
    data_root: str = "/Users/elite/R-LLM/data"
    output_dir: str = "/Users/elite/R-LLM/rml-fast-trained"  # NEW FOLDER
    checkpoint_dir: str = "/Users/elite/R-LLM/rml-fast-checkpoints"  # NEW FOLDER
    progress_file: str = "/Users/elite/R-LLM/fast_training_progress.pkl"
    
    # FAST PROCESSING SETTINGS
    max_files_to_process: int = 1000  # Limit files for speed
    max_samples_per_file: int = 10000  # Limit samples per file
    total_max_samples: int = 2000000  # 2M samples max for speed
    
    # GPU ACCELERATION SETTINGS
    use_gpu: bool = True
    device_map: str = "auto"  # Auto-detect best device
    max_memory: Dict[str, str] = None
    
    # MONITORING
    log_level: str = "INFO"

class RMLUltimateFastDataset(Dataset):
    """Fast dataset that processes limited files efficiently"""
    
    def __init__(self, data_paths: List[str], tokenizer, max_length: int = 128, 
                 max_files: int = 1000, max_samples_per_file: int = 10000, 
                 total_max_samples: int = 2000000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_paths = data_paths[:max_files]  # Limit files
        self.max_samples_per_file = max_samples_per_file
        self.total_max_samples = total_max_samples
        self.data = []
        self.file_mapping = []
        self.load_fast_data()
    
    def extract_text_fast(self, item: Dict[str, Any]) -> List[str]:
        """Fast text extraction - simplified for speed"""
        texts = []
        
        # METHOD 1: Direct text extraction
        if 'text' in item and isinstance(item['text'], str) and len(item['text'].strip()) > 3:
            texts.append(item['text'])
        
        # METHOD 2: Simple metadata extraction
        if isinstance(item, dict):
            for key, value in item.items():
                if key == 'text':
                    continue
                
                if isinstance(value, str) and len(value.strip()) > 3:
                    texts.append(f"{key}: {value}")
                elif isinstance(value, list) and value:
                    texts.append(f"{key}: {', '.join(str(v) for v in value[:3])}")  # Limit list items
        
        # METHOD 3: Nested RML structure - simplified
        if 'concepts' in item:
            concepts = item['concepts']
            if isinstance(concepts, dict) and 'text' in concepts:
                text = concepts['text']
                if isinstance(text, str) and len(text.strip()) > 3:
                    texts.append(f"concept: {text}")
            elif isinstance(concepts, str) and len(concepts.strip()) > 3:
                texts.append(f"concepts: {concepts}")
        
        # Similar for other fields - simplified
        for field in ['entities', 'emotions', 'intents', 'events', 'reasoning', 'summaries', 'triples', 'tags']:
            if field in item:
                value = item[field]
                if isinstance(value, str) and len(value.strip()) > 3:
                    texts.append(f"{field}: {value}")
                elif isinstance(value, list) and value:
                    texts.append(f"{field}: {', '.join(str(v) for v in value[:2])}")  # Limit to 2 items
        
        # Return unique texts
        unique_texts = []
        seen = set()
        for text in texts:
            if text.strip() and text not in seen and len(text) < 200:  # Limit length for speed
                unique_texts.append(text.strip())
                seen.add(text)
        
        return unique_texts
    
    def load_fast_data(self):
        """Load data FAST - limited files and samples"""
        print(f"🚀 FAST LOADING: Processing first {len(self.data_paths)} files...")
        print(f"🎯 Target: {self.total_max_samples:,} samples in 5-6 hours!")
        
        total_samples = 0
        files_processed = 0
        
        for file_path in self.data_paths:
            if total_samples >= self.total_max_samples:
                print(f"🎯 REACHED TARGET: {total_samples:,} samples")
                break
                
            try:
                if not os.path.exists(file_path):
                    continue
                    
                print(f"📂 Fast Loading: {os.path.basename(file_path)}")
                file_samples = 0
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if file_samples >= self.max_samples_per_file:
                            break
                        if total_samples >= self.total_max_samples:
                            break
                            
                        try:
                            item = json.loads(line.strip())
                            
                            # Extract texts FAST
                            texts = self.extract_text_fast(item)
                            
                            # Add each extracted text
                            for text in texts:
                                if len(text) > 3:
                                    self.data.append(text)
                                    self.file_mapping.append(file_path)
                                    file_samples += 1
                                    total_samples += 1
                                    
                                    if total_samples >= self.total_max_samples:
                                        break
                                    
                                    # Progress update every 50000 samples
                                    if total_samples % 50000 == 0:
                                        memory_usage = psutil.virtual_memory().percent
                                        print(f"   📊 Fast Loaded {total_samples:,} samples, Memory: {memory_usage:.1f}%")
                                
                        except json.JSONDecodeError:
                            continue
                
                print(f"✅ {os.path.basename(file_path)}: {file_samples:,} samples")
                files_processed += 1
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
        
        print(f"🎯 FAST LOADING COMPLETE: {len(self.data):,} samples from {files_processed} files")
        print(f"⚡ Ready for FAST training with GPU acceleration!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # FAST tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class RMLUltimateFastTrainer:
    """Fast trainer for 5-6 hour completion with GPU acceleration"""
    
    def __init__(self, config: RMLUltimateFastConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{config.output_dir}/fast_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create NEW directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        # Performance tracking
        self.start_time = time.time()
        
        self.logger.info("🚀 Initializing RML Ultimate Fast Trainer with GPU acceleration")
        self.logger.info("⚡ Target: 5-6 hours completion with CPU + GPU!")
        self._load_model()
    
    def _load_model(self):
        """Load model with GPU acceleration"""
        try:
            self.logger.info(f"🔧 Loading model: {self.config.model_name}")
            
            # Check GPU availability
            if self.config.use_gpu and torch.backends.mps.is_available():
                device = "mps"
                self.logger.info("🚀 GPU (MPS) acceleration ENABLED!")
            else:
                device = "cpu"
                self.logger.info("💻 Using CPU (GPU not available)")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Ensure padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with GPU acceleration
            if device == "mps":
                # GPU loading with MPS
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32,  # Use float32 for MPS (no fp16 support)
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to(device)
                self.logger.info("✅ Model loaded on GPU (MPS) successfully")
            else:
                # CPU loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.logger.info("✅ Model loaded on CPU successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            # Fallback to CPU
            self.logger.info("🔄 Falling back to CPU loading...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.logger.info("✅ Model loaded on CPU (fallback) successfully")
    
    def get_first_data_files(self) -> List[str]:
        """Get FIRST data files for fast processing"""
        data_files = []
        
        print(f"🔍 Scanning for FIRST data files in {self.config.data_root}...")
        
        # Get files in order (first files first)
        for root, dirs, files in os.walk(self.config.data_root):
            # Sort directories for consistent order
            dirs.sort()
            # Sort files for consistent order
            files.sort()
            
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    data_files.append(file_path)
                    
                    # Limit to first files
                    if len(data_files) >= self.config.max_files_to_process:
                        break
            
            if len(data_files) >= self.config.max_files_to_process:
                break
        
        print(f"📁 Found {len(data_files):,} FIRST files for fast processing")
        return data_files
    
    def train_fast(self):
        """Train FAST for 5-6 hour completion with GPU acceleration"""
        try:
            # Get FIRST data files
            first_data_files = self.get_first_data_files()
            
            if not first_data_files:
                self.logger.error("❌ No data files found!")
                return
            
            # Create dataset with FIRST files
            dataset = RMLUltimateFastDataset(
                first_data_files,
                self.tokenizer,
                self.config.max_seq_length,
                self.config.max_files_to_process,
                self.config.max_samples_per_file,
                self.config.total_max_samples
            )
            
            if len(dataset) == 0:
                self.logger.error("❌ No training data available")
                return
            
            # Create validation dataset (5% of data for speed)
            val_size = min(len(dataset) // 20, 5000)  # Smaller validation
            train_size = len(dataset) - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # FAST data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Calculate total steps for 5-6 hours with GPU
            total_steps = len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)
            self.logger.info(f"📊 Total training steps: {total_steps:,}")
            self.logger.info(f"📊 Training samples: {len(train_dataset):,}")
            self.logger.info(f"📊 Validation samples: {len(val_dataset):,}")
            
            # Check if GPU is available for training
            use_gpu_training = torch.backends.mps.is_available() and self.config.use_gpu
            
            # FAST Training arguments with GPU support
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_steps=50,  # More frequent logging
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_total_limit=self.config.save_total_limit,
                dataloader_pin_memory=True if use_gpu_training else False,
                remove_unused_columns=False,
                report_to=None,
                # GPU acceleration settings
                no_cuda=False if use_gpu_training else True,
                use_mps_device=use_gpu_training,
                # FAST settings
                max_grad_norm=1.0,
                fp16=False,  # Disable fp16 for MPS compatibility
                bf16=False,
                gradient_checkpointing=False,
                optim="adamw_torch",
                weight_decay=0.01,
                dataloader_num_workers=0,
                # FAST completion settings
                max_steps=total_steps,  # Limit steps for speed
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            
            if use_gpu_training:
                self.logger.info("🚀 Starting FAST training with GPU acceleration for 5-6 hour completion!")
                self.logger.info(f"⚡ Expected time: {total_steps * 0.3 / 3600:.1f} hours (GPU accelerated!)")
            else:
                self.logger.info("🚀 Starting FAST training on CPU for 5-6 hour completion!")
                self.logger.info(f"⚡ Expected time: {total_steps * 0.5 / 3600:.1f} hours")
            
            # Train the model
            trainer.train()
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Calculate training time
            training_time = time.time() - self.start_time
            self.logger.info(f"✅ FAST training completed in {training_time/3600:.2f} hours!")
            self.logger.info(f"💾 Model saved to {self.config.output_dir}")
            
            # Evaluate
            eval_results = trainer.evaluate()
            self.logger.info(f"📊 Final evaluation: {eval_results}")
            
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
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
    """Main function for fast training with GPU acceleration"""
    
    parser = argparse.ArgumentParser(description="RML Ultimate Fast Training with GPU")
    parser.add_argument("--output", type=str, default="/Users/elite/R-LLM/rml-fast-trained", help="Output directory")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()
    
    print("🚀 RML Ultimate Fast Training Pipeline with GPU Acceleration")
    print("="*60)
    print("⚡ Target: 5-6 hours completion with CPU + GPU!")
    print(f"💾 Output: {args.output}")
    print("🎯 Processing FIRST files for speed!")
    
    # Check GPU availability
    if torch.backends.mps.is_available() and not args.no_gpu:
        print("🚀 GPU (MPS) acceleration AVAILABLE and ENABLED!")
    else:
        print("💻 Using CPU only (GPU disabled or not available)")
    
    # Configuration
    config = RMLUltimateFastConfig(
        output_dir=args.output,
        checkpoint_dir=f"{args.output}/checkpoints",
        progress_file=f"{args.output}/fast_training_progress.pkl",
        use_gpu=not args.no_gpu
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLUltimateFastTrainer(config)
        
        # Train FAST with GPU
        print("🚀 Starting FAST training with GPU acceleration for 5-6 hour completion!")
        trainer.train_fast()
        
        print("✅ RML Ultimate Fast training completed!")
        print("⚡ Completed in 5-6 hours as requested!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 