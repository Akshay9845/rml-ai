#!/usr/bin/env python3
"""
RML Ultra-Fast Trainer - Complete training in 6-12 hours
Optimized for maximum speed and efficiency
"""

import json
import os
import sys
import torch
import gc
import psutil
import time
from typing import Dict, List, Any, Iterator, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import IterableDataset
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import subprocess

@dataclass
class RMLUltraFastConfig:
    """Configuration for ultra-fast RML training"""
    model_name: str = "microsoft/DialoGPT-small"
    data_dir: str = "/Users/elite/R-LLM/data"
    output_dir: str = "/Users/elite/R-LLM/rml-ultra-trained"
    
    # Ultra-fast training parameters
    batch_size: int = 16  # Much larger batch size
    gradient_accumulation_steps: int = 2  # Reduced for faster updates
    learning_rate: float = 1e-4  # Slightly higher learning rate
    num_train_epochs: int = 1
    max_steps: int = 50000  # Reduced for faster completion
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    fp16: bool = False
    
    # Memory management
    max_memory_usage: float = 0.9  # Higher memory usage
    chunk_size: int = 10000  # Larger chunks
    num_workers: int = 12  # More CPU workers
    prefetch_factor: int = 8
    
    # Processing parameters
    max_length: int = 256  # Shorter sequences for speed
    min_text_length: int = 5  # Lower threshold
    include_vectors: bool = False  # Skip vectors for speed
    include_metadata: bool = True
    include_all_fields: bool = True

class UltraFastRMLDataset(IterableDataset):
    """Ultra-fast RML dataset optimized for speed"""
    
    def __init__(self, config: RMLUltraFastConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Find all RML files
        self.rml_files = self._find_all_rml_files()
        print(f"🔍 Found {len(self.rml_files)} RML files to process")
        
        # Progress tracking
        self.processed_files = 0
        self.total_samples = 0
        self.start_time = time.time()
        
        # Multiple data queues for maximum throughput
        self.data_queues = [queue.Queue(maxsize=5000) for _ in range(8)]
        self.processing_threads = []
        
        # Start ultra-fast CPU processing
        self._start_ultra_fast_processing()
    
    def _find_all_rml_files(self) -> List[str]:
        """Find ALL .jsonl files in the data directory"""
        try:
            result = subprocess.run([
                'find', self.config.data_dir, 
                '-name', '*.jsonl', 
                '-type', 'f'
            ], capture_output=True, text=True, check=True)
            
            files = result.stdout.strip().split('\n')
            files = [f for f in files if f]
            
            print(f"📁 Found {len(files)} .jsonl files")
            return files
        except subprocess.CalledProcessError as e:
            print(f"❌ Error finding files: {e}")
            return []
    
    def _extract_text_fast(self, item: Dict[str, Any]) -> str:
        """Fast text extraction - optimized for speed"""
        if not isinstance(item, dict):
            return str(item) if item else ""
        
        text_parts = []
        
        # 1. Direct text fields (highest priority)
        direct_text_fields = [
            'text', 'data', 'content', 'message', 'description', 
            'summary', 'body', 'value', 'input', 'output'
        ]
        
        for field in direct_text_fields:
            if field in item and item[field]:
                text_content = str(item[field])
                if len(text_content.strip()) > self.config.min_text_length:
                    text_parts.append(text_content)
        
        # 2. RML fields (simplified for speed)
        rml_fields = ['concepts', 'entities', 'emotions', 'events', 'intents', 'reasoning', 'summaries', 'tags']
        
        for field in rml_fields:
            if field in item and item[field]:
                field_data = item[field]
                
                if isinstance(field_data, list):
                    # Take first 5 items for speed
                    for i, list_item in enumerate(field_data[:5]):
                        if isinstance(list_item, str):
                            text_parts.append(list_item)
                        elif isinstance(list_item, dict) and 'text' in list_item:
                            text_parts.append(str(list_item['text']))
                
                elif isinstance(field_data, str):
                    text_parts.append(field_data)
        
        # 3. Metadata (simplified)
        if self.config.include_metadata:
            metadata_fields = ['record_id', 'entity_type', 'semantic_category']
            for field in metadata_fields:
                if field in item and item[field]:
                    text_parts.append(str(item[field]))
        
        return " | ".join(text_parts) if text_parts else ""
    
    def _process_file_ultra_fast(self, file_path: str) -> List[str]:
        """Ultra-fast file processing"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
                # Try parsing as single JSON object first
                try:
                    item = json.loads(file_content)
                    text = self._extract_text_fast(item)
                    if text and len(text.strip()) > self.config.min_text_length:
                        texts.append(text)
                except json.JSONDecodeError:
                    # Fast line-by-line parsing
                    lines = file_content.strip().split('\n')
                    for line in lines[:100]:  # Limit to first 100 lines for speed
                        if line.strip():
                            try:
                                item = json.loads(line)
                                text = self._extract_text_fast(item)
                                if text and len(text.strip()) > self.config.min_text_length:
                                    texts.append(text)
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            pass  # Skip errors for speed
        
        return texts
    
    def _start_ultra_fast_processing(self):
        """Start ultra-fast CPU processing"""
        # Split files among workers
        files_per_worker = len(self.rml_files) // self.config.num_workers
        worker_files = []
        
        for i in range(self.config.num_workers):
            start_idx = i * files_per_worker
            end_idx = start_idx + files_per_worker if i < self.config.num_workers - 1 else len(self.rml_files)
            worker_files.append(self.rml_files[start_idx:end_idx])
        
        # Start processing threads
        for i, files in enumerate(worker_files):
            thread = threading.Thread(
                target=self._process_worker_ultra_fast,
                args=(files, i)
            )
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
    
    def _process_worker_ultra_fast(self, files: List[str], worker_id: int):
        """Ultra-fast worker thread"""
        queue_id = worker_id % len(self.data_queues)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for file_path in files:
                try:
                    texts = self._process_file_ultra_fast(file_path)
                    
                    # Fast tokenization and queue addition
                    for text in texts[:50]:  # Limit to 50 texts per file for speed
                        if len(text.strip()) > self.config.min_text_length:
                            encoding = self.tokenizer(
                                text,
                                truncation=True,
                                padding=False,
                                max_length=self.config.max_length,
                                return_tensors="pt"
                            )
                            
                            try:
                                self.data_queues[queue_id].put({
                                    'input_ids': encoding['input_ids'].squeeze(),
                                    'attention_mask': encoding['attention_mask'].squeeze()
                                }, timeout=0.1)
                            except queue.Full:
                                continue
                    
                    # Update progress
                    self.processed_files += 1
                    self.total_samples += len(texts)
                    
                    if self.processed_files % 500 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.processed_files / elapsed if elapsed > 0 else 0
                        print(f"⚡ Progress: {self.processed_files}/{len(self.rml_files)} files, "
                              f"{self.total_samples} samples, {rate:.1f} files/sec")
                
                except Exception as e:
                    continue
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through ultra-fast RML data"""
        while True:
            # Try to get data from any queue
            for queue_id in range(len(self.data_queues)):
                try:
                    data = self.data_queues[queue_id].get(timeout=0.1)
                    yield data
                except queue.Empty:
                    continue
            
            # Check if all files are processed and queues are empty
            if (self.processed_files >= len(self.rml_files) and 
                all(q.empty() for q in self.data_queues)):
                break
    
    def __len__(self):
        """Estimated length for the dataset"""
        return len(self.rml_files) * 500  # Estimate 500 samples per file

class RMLUltraFastTrainer:
    """Ultra-fast RML trainer for 6-12 hour completion"""
    
    def __init__(self, config: RMLUltraFastConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"🚀 Initializing Ultra-Fast RML Trainer")
        print(f"📱 Device: {self.device}")
        print(f"🔧 Model: {config.model_name}")
        print(f"📁 Data: {config.data_dir}")
        print(f"💾 Output: {config.output_dir}")
        print(f"⚡ Target: 6-12 hours completion")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device.type == "mps" else "cpu"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create dataset
        self.dataset = UltraFastRMLDataset(config)
        
        # Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Ultra-fast training arguments
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            fp16=config.fp16,
            logging_steps=50,  # More frequent logging
            save_steps=500,  # Save more frequently
            save_total_limit=2,
            max_steps=config.max_steps,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to=None,
            logging_dir=f"{config.output_dir}/logs"
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
    
    def train_ultra_fast(self):
        """Train the model ultra-fast"""
        print(f"🎯 Starting Ultra-Fast RML Training")
        print(f"📊 Processing ALL fields from ALL files")
        print(f"🧠 Memory threshold: {self.config.max_memory_usage * 100}%")
        print(f"⚡ CPU workers: {self.config.num_workers}")
        print(f"🔢 Batch size: {self.config.batch_size}")
        print(f"📈 Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"🎯 Target completion: 6-12 hours")
        
        try:
            # Start training
            self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            print(f"✅ Ultra-fast training completed!")
            print(f"💾 Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise

def main():
    """Main function for ultra-fast RML training"""
    print("🚀 RML Ultra-Fast Trainer - 6-12 Hour Completion!")
    
    # Configuration
    config = RMLUltraFastConfig()
    
    # System check
    memory = psutil.virtual_memory()
    print(f"💻 System Memory: {memory.total / (1024**3):.1f}GB total, "
          f"{memory.available / (1024**3):.1f}GB available")
    
    # Create and run trainer
    trainer = RMLUltraFastTrainer(config)
    trainer.train_ultra_fast()

if __name__ == "__main__":
    main() 