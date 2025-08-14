#!/usr/bin/env python3
"""
RML Comprehensive Processor - Extracts EVERYTHING from RML dataset
Processes ALL fields: text, concepts, entities, emotions, events, intents, reasoning, summaries, tags, triples, vectors, metadata
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
class RMLComprehensiveConfig:
    """Configuration for comprehensive RML processing"""
    model_name: str = "microsoft/DialoGPT-small"
    data_dir: str = "/Users/elite/R-LLM/data"
    output_dir: str = "/Users/elite/R-LLM/rml-comprehensive-trained"
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    max_steps: int = 1000000
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    fp16: bool = False  # MPS doesn't support fp16
    
    # Memory management
    max_memory_usage: float = 0.8  # 80% memory threshold
    chunk_size: int = 2000
    num_workers: int = 8
    prefetch_factor: int = 4
    
    # Processing parameters
    max_length: int = 512
    min_text_length: int = 10
    include_vectors: bool = True
    include_metadata: bool = True
    include_all_fields: bool = True

class ComprehensiveRMLDataset(IterableDataset):
    """Comprehensive RML dataset that extracts EVERYTHING"""
    
    def __init__(self, config: RMLComprehensiveConfig):
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
        
        # Data queues for CPU-GPU hybrid processing
        self.data_queues = [queue.Queue(maxsize=2000) for _ in range(4)]
        self.processing_threads = []
        
        # Start CPU processing threads
        self._start_comprehensive_cpu_processing()
    
    def _find_all_rml_files(self) -> List[str]:
        """Find ALL .jsonl files in the data directory"""
        try:
            result = subprocess.run([
                'find', self.config.data_dir, 
                '-name', '*.jsonl', 
                '-type', 'f'
            ], capture_output=True, text=True, check=True)
            
            files = result.stdout.strip().split('\n')
            files = [f for f in files if f]  # Remove empty lines
            
            print(f"📁 Found {len(files)} .jsonl files")
            return files
        except subprocess.CalledProcessError as e:
            print(f"❌ Error finding files: {e}")
            return []
    
    def _extract_everything_comprehensive(self, item: Dict[str, Any]) -> str:
        """Extract EVERYTHING from RML item - no field left behind!"""
        if not isinstance(item, dict):
            return str(item) if item else ""
        
        text_parts = []
        
        # 1. Direct text fields (highest priority)
        direct_text_fields = [
            'text', 'data', 'content', 'message', 'description', 
            'summary', 'body', 'value', 'input', 'output', 'title',
            'name', 'label', 'caption', 'comment', 'note', 'info'
        ]
        
        for field in direct_text_fields:
            if field in item and item[field]:
                text_content = str(item[field])
                if len(text_content.strip()) > self.config.min_text_length:
                    text_parts.append(f"{field}: {text_content}")
        
        # 2. RML-specific structured fields
        rml_fields = [
            'concepts', 'entities', 'emotions', 'events', 'intents', 
            'reasoning', 'summaries', 'tags', 'triples', 'vectors'
        ]
        
        for field in rml_fields:
            if field in item and item[field]:
                field_data = item[field]
                
                if isinstance(field_data, list):
                    # Handle lists (concepts, entities, emotions, etc.)
                    for i, list_item in enumerate(field_data[:10]):  # Limit to first 10
                        if isinstance(list_item, str):
                            text_parts.append(f"{field}_{i}: {list_item}")
                        elif isinstance(list_item, dict):
                            nested_text = self._extract_everything_comprehensive(list_item)
                            if nested_text:
                                text_parts.append(f"{field}_{i}: {nested_text}")
                
                elif isinstance(field_data, dict):
                    # Handle dictionaries
                    nested_text = self._extract_everything_comprehensive(field_data)
                    if nested_text:
                        text_parts.append(f"{field}: {nested_text}")
                
                elif isinstance(field_data, str):
                    # Handle strings
                    text_parts.append(f"{field}: {field_data}")
                
                elif field == 'vectors' and isinstance(field_data, list) and self.config.include_vectors:
                    # Handle vector embeddings (first 10 values for text representation)
                    vector_text = " ".join([f"{v:.3f}" for v in field_data[:10]])
                    text_parts.append(f"vector_sample: {vector_text}")
        
        # 3. Metadata fields
        if self.config.include_metadata:
            metadata_fields = [
                'record_id', 'chunk', 'confidence', 'entity_type', 
                'emotional_tone', 'semantic_category', 'topic_tags',
                'relations', 'id', 'type', 'category', 'source'
            ]
            
            for field in metadata_fields:
                if field in item and item[field]:
                    metadata_value = item[field]
                    if isinstance(metadata_value, list):
                        metadata_text = ", ".join([str(v) for v in metadata_value[:5]])
                        text_parts.append(f"{field}: {metadata_text}")
                    else:
                        text_parts.append(f"{field}: {metadata_value}")
        
        # 4. Recursive extraction for nested structures
        if self.config.include_all_fields:
            for key, value in item.items():
                if key not in direct_text_fields + rml_fields + metadata_fields:
                    if isinstance(value, dict):
                        nested_text = self._extract_everything_comprehensive(value)
                        if nested_text:
                            text_parts.append(f"{key}: {nested_text}")
                    elif isinstance(value, list) and len(value) > 0:
                        for i, list_item in enumerate(value[:3]):  # Limit nested lists
                            if isinstance(list_item, str) and len(list_item.strip()) > self.config.min_text_length:
                                text_parts.append(f"{key}_{i}: {list_item}")
                            elif isinstance(list_item, dict):
                                nested_text = self._extract_everything_comprehensive(list_item)
                                if nested_text:
                                    text_parts.append(f"{key}_{i}: {nested_text}")
                    elif isinstance(value, str) and len(value.strip()) > self.config.min_text_length:
                        if len(value) < 10000:  # Skip very long strings
                            text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts) if text_parts else ""
    
    def _process_file_comprehensive(self, file_path: str) -> List[str]:
        """Process a single file comprehensively"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
                # Try parsing as single JSON object first
                try:
                    item = json.loads(file_content)
                    text = self._extract_everything_comprehensive(item)
                    if text and len(text.strip()) > self.config.min_text_length:
                        texts.append(text)
                except json.JSONDecodeError:
                    # Fallback to line-by-line parsing
                    lines = file_content.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                text = self._extract_everything_comprehensive(item)
                                if text and len(text.strip()) > self.config.min_text_length:
                                    texts.append(text)
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
        
        return texts
    
    def _start_comprehensive_cpu_processing(self):
        """Start comprehensive CPU processing with multiple workers"""
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
                target=self._process_worker_comprehensive,
                args=(files, i)
            )
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
    
    def _process_worker_comprehensive(self, files: List[str], worker_id: int):
        """Worker thread for comprehensive processing"""
        queue_id = worker_id % len(self.data_queues)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            for file_path in files:
                try:
                    texts = self._process_file_comprehensive(file_path)
                    
                    # Tokenize and add to queue
                    for text in texts:
                        if len(text.strip()) > self.config.min_text_length:
                            encoding = self.tokenizer(
                                text,
                                truncation=True,
                                padding=False,
                                max_length=self.config.max_length,
                                return_tensors="pt"
                            )
                            
                            # Add to queue
                            try:
                                self.data_queues[queue_id].put({
                                    'input_ids': encoding['input_ids'].squeeze(),
                                    'attention_mask': encoding['attention_mask'].squeeze()
                                }, timeout=1)
                            except queue.Full:
                                continue
                    
                    # Update progress
                    self.processed_files += 1
                    self.total_samples += len(texts)
                    
                    if self.processed_files % 100 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.processed_files / elapsed if elapsed > 0 else 0
                        print(f"📊 Progress: {self.processed_files}/{len(self.rml_files)} files, "
                              f"{self.total_samples} samples, {rate:.1f} files/sec")
                        
                        # Memory check
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > self.config.max_memory_usage * 100:
                            print(f"🧠 High memory usage: {memory_percent:.1f}%, forcing cleanup")
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"❌ Error in worker {worker_id} processing {file_path}: {e}")
                    continue
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through comprehensive RML data"""
        while True:
            # Try to get data from any queue
            for queue_id in range(len(self.data_queues)):
                try:
                    data = self.data_queues[queue_id].get(timeout=1)
                    yield data
                except queue.Empty:
                    continue
            
            # Check if all files are processed and queues are empty
            if (self.processed_files >= len(self.rml_files) and 
                all(q.empty() for q in self.data_queues)):
                break
    
    def __len__(self):
        """Estimated length for the dataset"""
        return len(self.rml_files) * 1000  # Estimate 1000 samples per file

class RMLComprehensiveTrainer:
    """Comprehensive RML trainer that processes EVERYTHING"""
    
    def __init__(self, config: RMLComprehensiveConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"🚀 Initializing Comprehensive RML Trainer")
        print(f"📱 Device: {self.device}")
        print(f"🔧 Model: {config.model_name}")
        print(f"📁 Data: {config.data_dir}")
        print(f"💾 Output: {config.output_dir}")
        
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
        self.dataset = ComprehensiveRMLDataset(config)
        
        # Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments
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
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            max_steps=config.max_steps,
            dataloader_num_workers=0,  # We handle data loading ourselves
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
    
    def train_comprehensive(self):
        """Train the model comprehensively"""
        print(f"🎯 Starting Comprehensive RML Training")
        print(f"📊 Processing ALL fields from ALL files")
        print(f"🧠 Memory threshold: {self.config.max_memory_usage * 100}%")
        print(f"⚡ CPU workers: {self.config.num_workers}")
        print(f"🔢 Batch size: {self.config.batch_size}")
        print(f"📈 Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        try:
            # Start training
            self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            print(f"✅ Comprehensive training completed!")
            print(f"💾 Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise

def main():
    """Main function for comprehensive RML training"""
    print("🚀 RML Comprehensive Processor - Processing EVERYTHING!")
    
    # Configuration
    config = RMLComprehensiveConfig()
    
    # System check
    memory = psutil.virtual_memory()
    print(f"💻 System Memory: {memory.total / (1024**3):.1f}GB total, "
          f"{memory.available / (1024**3):.1f}GB available")
    
    # Create and run trainer
    trainer = RMLComprehensiveTrainer(config)
    trainer.train_comprehensive()

if __name__ == "__main__":
    main() 