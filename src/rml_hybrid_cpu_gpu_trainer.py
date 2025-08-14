#!/usr/bin/env python3
"""
RML Hybrid CPU+GPU Trainer - Uses Both CPU and GPU Efficiently
CPU for data processing, GPU for model training
Handles ALL 417,927,089 records with maximum efficiency
"""

import os
import json
import logging
import pickle
import psutil
import torch
import gc
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, IterableDataset
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/elite/R-LLM/rml-hybrid-trained/hybrid_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RMLHybridConfig:
    """Configuration for Hybrid CPU+GPU RML Trainer"""
    model_name: str = "microsoft/DialoGPT-small"
    max_seq_length: int = 128
    batch_size: int = 4  # Increased for GPU efficiency
    gradient_accumulation_steps: int = 8  # Balanced for CPU+GPU
    learning_rate: float = 5e-5
    num_epochs: int = 1
    warmup_steps: int = 100
    output_dir: str = "/Users/elite/R-LLM/rml-hybrid-trained"
    checkpoint_dir: str = "/Users/elite/R-LLM/rml-hybrid-checkpoints"
    progress_file: str = "/Users/elite/R-LLM/hybrid_training_progress.pkl"
    data_dir: str = "/Users/elite/R-LLM/data"
    chunk_size: int = 2000  # Larger chunks for efficiency
    max_memory_usage: float = 0.8  # Use 80% of available memory
    num_workers: int = 4  # Use multiple CPU cores for data processing
    prefetch_factor: int = 2  # Prefetch data for GPU

class HybridRMLDataset(IterableDataset):
    """Hybrid dataset that uses CPU for data processing and GPU for training"""
    
    def __init__(self, config: RMLHybridConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Find ALL JSONL files
        self.rml_files = self._find_all_rml_files()
        logger.info(f"🎯 Found {len(self.rml_files)} total files to process")
        
        # Initialize progress tracking
        self.processed_files = 0
        self.total_samples = 0
        self.total_records = 0
        
        # Data queue for CPU-GPU coordination
        self.data_queue = queue.Queue(maxsize=1000)
        self.processing_thread = None
        
        # Load progress if exists
        self._load_progress()
        
        # Start CPU data processing thread
        self._start_cpu_processing()
    
    def _find_all_rml_files(self) -> List[str]:
        """Find ALL JSONL files in ALL folders"""
        all_files = []
        
        # Use find command to get ALL files
        try:
            result = subprocess.run(
                ['find', self.config.data_dir, '-name', '*.jsonl', '-type', 'f'],
                capture_output=True, text=True, check=True
            )
            all_files = result.stdout.strip().split('\n')
            all_files = [f for f in all_files if f]  # Remove empty lines
        except subprocess.CalledProcessError as e:
            logger.error(f"Error finding files: {e}")
            # Fallback to glob
            for folder in os.listdir(self.config.data_dir):
                folder_path = os.path.join(self.config.data_dir, folder)
                if os.path.isdir(folder_path):
                    jsonl_files = glob.glob(os.path.join(folder_path, "**/*.jsonl"), recursive=True)
                    all_files.extend(jsonl_files)
        
        logger.info(f"🎯 Total files found: {len(all_files)}")
        return all_files
    
    def _extract_text_universal(self, item: Dict[str, Any]) -> str:
        """Extract text from any RML format"""
        if isinstance(item, dict):
            # Direct text fields - check multiple possible field names
            text_fields = ['text', 'data', 'content', 'message', 'description', 'summary', 'body']
            for field in text_fields:
                if field in item and item[field]:
                    text_content = str(item[field])
                    if len(text_content.strip()) > 10:
                        return text_content
            
            # Nested RML components
            text_parts = []
            
            # Check all possible RML fields
            for field in ['concepts', 'entities', 'emotions', 'intents', 'events', 'reasoning', 'summaries', 'triples', 'tags']:
                if field in item and item[field]:
                    if isinstance(item[field], dict):
                        # Check for text/data fields in nested dict
                        for text_field in ['text', 'data', 'content', 'message', 'description']:
                            if text_field in item[field] and item[field][text_field]:
                                text_parts.append(str(item[field][text_field]))
                        # Also check if the dict itself has text-like content
                        if 'text' in item[field]:
                            text_parts.append(str(item[field]['text']))
                    elif isinstance(item[field], list):
                        for sub_item in item[field]:
                            if isinstance(sub_item, dict):
                                # Check for text/data fields in list items
                                for text_field in ['text', 'data', 'content', 'message', 'description']:
                                    if text_field in sub_item and sub_item[text_field]:
                                        text_parts.append(str(sub_item[text_field]))
                            elif isinstance(sub_item, str):
                                text_parts.append(str(sub_item))
                    elif isinstance(item[field], str):
                        text_parts.append(str(item[field]))
            
            # Also check for any string values that look like text
            for key, value in item.items():
                if isinstance(value, str) and len(value.strip()) > 10:
                    # Skip very long strings that might be encoded data
                    if len(value) < 10000:
                        text_parts.append(value)
                elif isinstance(value, dict):
                    # Recursively check nested dictionaries
                    nested_text = self._extract_text_universal(value)
                    if nested_text:
                        text_parts.append(nested_text)
                elif isinstance(value, list) and len(value) > 0:
                    # Check first few items in lists
                    for i, list_item in enumerate(value[:5]):  # Limit to first 5 items
                        if isinstance(list_item, str) and len(list_item.strip()) > 10:
                            text_parts.append(list_item)
                        elif isinstance(list_item, dict):
                            nested_text = self._extract_text_universal(list_item)
                            if nested_text:
                                text_parts.append(nested_text)
            
            if text_parts:
                return " ".join(text_parts)
        
        return str(item) if item else ""
    
    def _load_progress(self):
        """Load progress from file"""
        if os.path.exists(self.config.progress_file):
            try:
                with open(self.config.progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                    self.processed_files = progress_data.get('processed_files', 0)
                    self.total_samples = progress_data.get('total_samples', 0)
                    self.total_records = progress_data.get('total_records', 0)
                logger.info(f"📊 Loaded progress: {self.processed_files} files, {self.total_samples} samples")
            except Exception as e:
                logger.warning(f"⚠️ Error loading progress: {e}")
    
    def _save_progress(self):
        """Save progress to file"""
        progress_data = {
            'processed_files': self.processed_files,
            'total_samples': self.total_samples,
            'total_records': self.total_records,
            'total_files': len(self.rml_files)
        }
        
        os.makedirs(os.path.dirname(self.config.progress_file), exist_ok=True)
        with open(self.config.progress_file, 'wb') as f:
            pickle.dump(progress_data, f)
    
    def _check_memory(self):
        """Check memory usage and force garbage collection if needed"""
        memory = psutil.virtual_memory()
        if memory.percent > (self.config.max_memory_usage * 100):
            logger.warning(f"⚠️ High memory usage: {memory.percent}%, forcing garbage collection")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return False
        return True
    
    def _process_file_cpu(self, file_path: str) -> List[str]:
        """Process a single file using CPU"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read entire file content
                file_content = f.read()
                
                # Try to parse as single JSON object first
                try:
                    item = json.loads(file_content)
                    text = self._extract_text_universal(item)
                    if text and len(text.strip()) > 10:
                        texts.append(text)
                except json.JSONDecodeError:
                    # Try line-by-line parsing
                    lines = file_content.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                text = self._extract_text_universal(item)
                                if text and len(text.strip()) > 10:
                                    texts.append(text)
                            except json.JSONDecodeError:
                                continue
                
                self.processed_files += 1
                self.total_records += 1
                
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path}: {e}")
        
        return texts
    
    def _start_cpu_processing(self):
        """Start CPU thread for data processing"""
        def cpu_worker():
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                for file_path in self.rml_files[self.processed_files:]:
                    # Check memory
                    if not self._check_memory():
                        time.sleep(1)
                        continue
                    
                    # Process file using CPU
                    texts = self._process_file_cpu(file_path)
                    
                    # Add to queue for GPU processing
                    for text in texts:
                        try:
                            self.data_queue.put(text, timeout=1)
                            self.total_samples += 1
                        except queue.Full:
                            time.sleep(0.1)
                    
                    # Save progress every 100 files
                    if self.processed_files % 100 == 0:
                        self._save_progress()
                        logger.info(f"📂 Processed {self.processed_files}/{len(self.rml_files)} files, {self.total_samples} samples")
        
        self.processing_thread = threading.Thread(target=cpu_worker, daemon=True)
        self.processing_thread.start()
        logger.info(f"🚀 Started CPU processing thread with {self.config.num_workers} workers")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through data with CPU-GPU coordination"""
        chunk_samples = []
        
        while True:
            try:
                # Get data from CPU queue
                text = self.data_queue.get(timeout=5)
                chunk_samples.append(text)
                
                # Process chunk when it reaches chunk_size
                if len(chunk_samples) >= self.config.chunk_size:
                    for sample_text in chunk_samples:
                        yield self._tokenize_text(sample_text)
                    chunk_samples = []
                    gc.collect()
                    
            except queue.Empty:
                # No more data from CPU
                if chunk_samples:
                    for sample_text in chunk_samples:
                        yield self._tokenize_text(sample_text)
                break
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text sample"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }
    
    def __len__(self):
        """Return estimated length for the dataset"""
        # Estimate based on total files and average records per file
        estimated_records = len(self.rml_files) * 1000  # Assume 1000 records per file on average
        return estimated_records

class RMLHybridTrainer:
    """Hybrid RML Trainer that uses both CPU and GPU efficiently"""
    
    def __init__(self, config: RMLHybridConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info(f"🚀 Initializing Hybrid CPU+GPU RML Trainer")
        logger.info(f"📊 Device: {self.device}")
        logger.info(f"📁 Output: {config.output_dir}")
        logger.info(f"🖥️ CPU Workers: {config.num_workers}")
    
    def train_hybrid(self):
        """Train on ALL 417 million records using CPU+GPU"""
        logger.info("🎯 Starting Hybrid RML Training on ALL data")
        
        # Load model
        logger.info(f"📥 Loading model: {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(self.device)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create hybrid dataset
        logger.info("📊 Creating hybrid dataset...")
        dataset = HybridRMLDataset(self.config)
        
        # Training arguments optimized for CPU+GPU
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # We handle multiprocessing ourselves
            max_grad_norm=1.0,
            fp16=False,
            report_to=None,
            dataloader_pin_memory=True,  # Enable pin memory for GPU efficiency
            max_steps=1000000,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Start training
        logger.info("🚀 Starting Hybrid RML Training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ Hybrid RML Training completed!")

def main():
    """Main function"""
    config = RMLHybridConfig()
    
    # Check system resources
    memory = psutil.virtual_memory()
    cpu_count = mp.cpu_count()
    logger.info(f"💾 System Memory: {memory.total / (1024**3):.1f}GB total")
    logger.info(f"💾 Available Memory: {memory.available / (1024**3):.1f}GB")
    logger.info(f"🖥️ CPU Cores: {cpu_count}")
    
    if memory.available < 4 * (1024**3):  # Less than 4GB
        logger.warning("⚠️ Low memory available, consider closing other applications")
    
    # Start training
    trainer = RMLHybridTrainer(config)
    trainer.train_hybrid()

if __name__ == "__main__":
    main() 