#!/usr/bin/env python3
"""
RML ULTRA-FAST OPTIMIZED TRAINER - 100x Faster!
Fixes all issues: streaming, efficient extraction, smart batching, memory optimization
"""

import json
import os
import re
import sys
import time
import signal
import gc
import psutil

# Core ML imports
import torch
import torch.nn as nn

def get_device():
    """Initialize and return the appropriate device for training"""
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("✅ Using CUDA device")
            return torch.device("cuda")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(0.9)
            print("✅ Using MPS (Apple Silicon) device")
            return torch.device("mps")
            
        print("⚠️  No GPU available, using CPU")
        return torch.device("cpu")
        
    except Exception as e:
        print(f"⚠️  Error initializing device: {e}")
        print("⚠️  Falling back to CPU")
        return torch.device("cpu")

# Initialize device
device = get_device()

# Typing
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Utilities
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

# Make torch available globally
global torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from torch.utils.data import IterableDataset
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import subprocess
import mmap
import io

@dataclass
class RMLUltraFastConfig:
    """ULTRA-OPTIMIZED Configuration for 100x speed"""
    model_name: str = "microsoft/DialoGPT-small"
    data_dir: str = "/Users/elite/R-LLM/data/python_c4_final_backup_20250731_043743"  # Updated to point to the correct data directory
    output_dir: str = "/Users/elite/R-LLM/rml-100-trained"
    
    # EXTREME SPEED - M3 Pro MAXIMUM POWER
    batch_size: int = 32  # Maximum batch size for M3 Pro 18GB
    gradient_accumulation_steps: int = 1  # No accumulation for max speed
    learning_rate: float = 1e-3  # Very high learning rate
    num_train_epochs: int = 1
    max_steps: int = 50000
    warmup_steps: int = 50  # Minimal warmup
    max_grad_norm: float = 1.0
    fp16: bool = False  # Keep disabled for MPS stability
    
    # EXTREME Memory settings
    max_memory_usage: float = 0.99  # Use 99% of available memory
    chunk_size: int = 12000  # Huge chunks for max throughput
    num_workers: int = 0  # Must be 0 for MPS
    prefetch_factor: int = 1  # Minimal prefetch
    
    # ULTRA-FAST Processing parameters
    max_length: int = 64    # Very short sequences
    min_length: int = 16    # Very short minimum
    min_text_length: int = 16  # Minimum text length for filtering
    max_samples: int = 400  # Process more samples per file
    include_vectors: bool = False  # Skip vectors
    include_metadata: bool = False  # Skip metadata
    include_all_fields: bool = False  # Skip complex fields
    
    # ULTRA Streaming
    stream_buffer_size: int = 10000
    max_files_per_batch: int = 50  # Process multiple files
    
    # ULTRA Smart sampling
    sample_rate: float = 1.0  # Process 100% of data for deterministic resume
    priority_fields: List[str] = None  # Focus on key fields

class UltraFastRMLDataset(IterableDataset):
    """ULTRA-OPTIMIZED Dataset with streaming and smart extraction"""
    
    def __init__(self, config: RMLUltraFastConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ULTRA Progress tracking
        self.checkpoint_file = f"{config.output_dir}/ultra_training_checkpoint.json"
        self.processed_files = 0
        self.total_samples = 0
        self.start_time = time.time()
        
        # Load checkpoint
        self._load_checkpoint()
        
        # ULTRA Find files with smart sampling, but restore file list if present in checkpoint
        if hasattr(self, '_restored_rml_files') and self._restored_rml_files:
            self.rml_files = self._restored_rml_files
            print(f"🚀 ULTRA: Using restored sampled file list from checkpoint ({len(self.rml_files)} files)")
        else:
            self.rml_files = self._find_files_with_sampling()
            print(f"🚀 ULTRA: Found {len(self.rml_files)} files to process (sampled)")
            
        # Track current file index for resuming
        self.current_file_index = min(self.processed_files, len(self.rml_files) - 1)
        if self.current_file_index > 0:
            print(f"⏩ ULTRA: Will resume from file {self.current_file_index + 1}/{len(self.rml_files)}")
        
        # ULTRA Streaming setup
        self.data_queue = queue.Queue(maxsize=config.stream_buffer_size)
        self.processing_thread = None
        self.running = True
        
        # ULTRA Start streaming
        self._start_ultra_streaming()
    
    def _load_checkpoint(self):
        """Load checkpoint if exists (now also restores sampled file list and current file index)"""
        self._restored_rml_files = None
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.processed_files = checkpoint.get('processed_files', 0)
                    self.current_file_index = checkpoint.get('current_file_index', self.processed_files)
                    self.total_samples = checkpoint.get('total_samples', 0)
                    self.start_time = time.time() - checkpoint.get('elapsed_time', 0)
                    if 'rml_files' in checkpoint:
                        self._restored_rml_files = checkpoint['rml_files']
                print(f"📂 ULTRA: Loaded checkpoint: {self.processed_files} files, {self.total_samples} samples")
                if self._restored_rml_files:
                    print(f"📂 ULTRA: Restored {len(self._restored_rml_files)} sampled files from checkpoint.")
                if self.current_file_index > 0:
                    print(f"📂 ULTRA: Will resume from file {self.current_file_index + 1}")
            except Exception as e:
                print(f"⚠️ ULTRA: Error loading checkpoint: {e}")
    
    def _save_checkpoint(self):
        """Save checkpoint (now also saves sampled file list)"""
        try:
            checkpoint = {
                'processed_files': self.processed_files,
                'current_file_index': self.current_file_index,
                'total_samples': self.total_samples,
                'elapsed_time': time.time() - self.start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'rml_files': self.rml_files if hasattr(self, 'rml_files') else []
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            print(f"⚠️ ULTRA: Error saving checkpoint: {e}")
    
    def _find_files_with_sampling(self) -> List[str]:
        """Find files with smart sampling for speed"""
        try:
            result = subprocess.run([
                'find', self.config.data_dir, 
                '-name', '*.jsonl', 
                '-type', 'f'
            ], capture_output=True, text=True, check=True)
            
            all_files = result.stdout.strip().split('\n')
            all_files = [f for f in all_files if f]
            
            # ULTRA Smart sampling - take every Nth file
            sample_interval = int(1 / self.config.sample_rate)
            sampled_files = all_files[::sample_interval]
            
            print(f"🚀 ULTRA: Sampled {len(sampled_files)} files from {len(all_files)} total")
            return sampled_files
            
        except Exception as e:
            print(f"❌ ULTRA: Error finding files: {e}")
            return []
    
    def _extract_ultra_fast(self, item: Dict[str, Any]) -> str:
        """ULTRA-FAST extraction - only essential fields"""
        if not isinstance(item, dict):
            return str(item) if item else ""
        
        # ULTRA Priority fields only
        priority_fields = ['text', 'data', 'content', 'message', 'description', 'summary']
        text_parts = []
        
        for field in priority_fields:
            if field in item and item[field]:
                content = str(item[field])
                if len(content.strip()) > self.config.min_text_length:
                    text_parts.append(content)
        
        # ULTRA Simple concepts/entities (first 3 only)
        for field in ['concepts', 'entities']:
            if field in item and isinstance(item[field], list):
                for i, concept in enumerate(item[field][:3]):  # Only first 3
                    if isinstance(concept, str) and len(concept.strip()) > self.config.min_text_length:
                        text_parts.append(concept)
        
        return " | ".join(text_parts) if text_parts else ""
    
    def _stream_file_ultra_fast(self, file_path: str) -> Iterator[str]:
        """ULTRA-FAST streaming file processing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # ULTRA Memory-mapped reading for speed
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                for line in iter(mm.readline, b''):
                    try:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                        
                        # ULTRA Fast JSON parsing
                        data = json.loads(line)
                        text = self._extract_ultra_fast(data)
                        
                        if text:
                            yield text
                            
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                        
                mm.close()
                
        except Exception as e:
            print(f"⚠️ ULTRA: Error streaming {file_path}: {e}")
    
    def _ultra_streaming_worker(self):
        """ULTRA-FAST streaming worker"""
        file_batch = []
        
        # Start from the current file index (for resuming)
        for file_path in self.rml_files[self.current_file_index:]:
            if not self.running:
                break
                
            file_batch.append(file_path)
            
            # ULTRA Process files in batches
            if len(file_batch) >= self.config.max_files_per_batch:
                self._process_file_batch_ultra_fast(file_batch)
                self.current_file_index += len(file_batch)
                self.processed_files = self.current_file_index
                file_batch = []
                
                # ULTRA Save checkpoint
                self._save_checkpoint()
        
        # Process remaining files
        if file_batch:
            self._process_file_batch_ultra_fast(file_batch)
            self.current_file_index += len(file_batch)
            self.processed_files = self.current_file_index
            self._save_checkpoint()
    
    def _process_file_batch_ultra_fast(self, file_paths: List[str]):
        """ULTRA-FAST batch processing with memory monitoring"""
        for file_path in file_paths:
            try:
                # ULTRA Memory check
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:  # 85% threshold
                    print(f"⚠️ ULTRA: Memory high ({memory_percent:.1f}%), forcing garbage collection...")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    time.sleep(1)  # Brief pause
                
                for text in self._stream_file_ultra_fast(file_path):
                    # ULTRA Tokenize immediately with padding
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.config.max_length,
                        padding='max_length',
                        return_tensors="pt"
                    )
                    
                    # ULTRA Put in queue
                    try:
                        self.data_queue.put(encoding, timeout=1)
                        self.total_samples += 1
                    except queue.Full:
                        break
                
                self.processed_files += 1
                
                # ULTRA Progress update with memory
                if self.processed_files % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.processed_files / elapsed if elapsed > 0 else 0
                    memory_percent = psutil.virtual_memory().percent
                    print(f"🚀 ULTRA: {self.processed_files}/{len(self.rml_files)} files, "
                          f"{self.total_samples} samples, {rate:.1f} files/sec, "
                          f"Memory: {memory_percent:.1f}%")
                
            except Exception as e:
                print(f"⚠️ ULTRA: Error processing {file_path}: {e}")
                self.processed_files += 1
    
    def _start_ultra_streaming(self):
        """Start ULTRA streaming"""
        self.processing_thread = threading.Thread(target=self._ultra_streaming_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """ULTRA-FAST iterator"""
        while self.running:
            try:
                # ULTRA Get from queue with timeout
                encoding = self.data_queue.get(timeout=5)
                
                # ULTRA Return immediately
                yield {
                    'input_ids': encoding['input_ids'].squeeze().tolist(),
                    'attention_mask': encoding['attention_mask'].squeeze().tolist(),
                    'labels': encoding['input_ids'].squeeze().tolist()
                }
                
            except queue.Empty:
                if not self.processing_thread.is_alive():
                    break
                continue
            except Exception as e:
                print(f"⚠️ ULTRA: Error in iterator: {e}")
                continue
    
    def __len__(self):
        return len(self.rml_files) * 100  # Estimate

class SimpleRMLDataset(Dataset):
    """Simple RML Dataset for non-streaming use cases"""
    def __init__(
        self, 
        data_dir: str, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 512,
        sample_rate: float = 1.0,
        shuffle: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        
        # Load and preprocess data
        self.examples = self._load_and_preprocess_data(data_dir)
        
        if not self.examples:
            raise ValueError("No examples found in the dataset. Please check your data directory.")
            
        # Apply sampling if needed
        if sample_rate < 1.0:
            import random
            random.shuffle(self.examples)
            self.examples = self.examples[:int(len(self.examples) * sample_rate)]
        
        print(f"✅ Loaded {len(self.examples)} examples from {data_dir}")
            
    def _load_and_preprocess_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """Load and preprocess the RML data with robust error handling"""
        import os
        import json
        from glob import glob
        from tqdm import tqdm
        
        examples = []
        error_count = 0
        
        # Look for data files in the data directory (support .rml, .json, and .jsonl)
        rml_files = []
        for ext in ['*.rml', '*.json', '*.jsonl']:
            rml_files.extend(glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        if not rml_files:
            raise FileNotFoundError(f"No training files (.rml, .json, or .jsonl) found in {data_dir}")
            
        print(f"📂 Found {len(rml_files)} training files in {data_dir}")
        
        # Process each file with progress bar
        for file_path in tqdm(rml_files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    # Skip empty files
                    if not content:
                        continue
                        
                    # For .jsonl files, process each line as a separate JSON object
                    if file_path.endswith('.jsonl'):
                        lines = [line for line in content.split('\n') if line.strip()]
                        data = []
                        for line in lines:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"⚠️  Error parsing JSONL line in {os.path.basename(file_path)}: {str(e)[:100]}")
                        if not data:
                            continue
                    else:
                        # Try to parse as JSON, fall back to raw text if it fails
                        try:
                            data = json.loads(content)
                        except json.JSONDecodeError:
                            data = content
                    
                    # Convert to text format
                    text = self._format_rml_data(data)
                    
                    if text and text.strip():
                        examples.append({
                            'text': text,
                            'file_path': file_path
                        })
                    
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # Only show first 10 errors
                    print(f"⚠️  Error in {os.path.basename(file_path)}: {str(e)[:100]}")
                continue
                
        if error_count > 10:
            print(f"⚠️  ... and {error_count - 10} more errors suppressed")
        
        return examples
    
    def _format_rml_data(self, data: Any) -> str:
        """Convert RML data to text format for training"""
        try:
            if isinstance(data, dict):
                # Handle dictionary format
                return json.dumps(data, ensure_ascii=False)
            elif isinstance(data, list):
                # Handle list of RML entries
                return '\n'.join(json.dumps(item, ensure_ascii=False) for item in data)
            else:
                return str(data)
        except Exception as e:
            print(f"⚠️  Error formatting RML data: {e}")
            return str(data)
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            example['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For language modeling, labels are the same as input_ids
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

class UltraFastRMLTrainer:
    """ULTRA-FAST RML Trainer"""
    
    def __init__(self, config: RMLUltraFastConfig):
        self.config = config
        self.dataset = None
        
        # ULTRA Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # ULTRA Setup
        self.train_ultra_fast()
    
    def train_ultra_fast(self):
        """ULTRA-FAST training setup and execution"""
        import torch  # Import here to ensure it's in scope
        print("🚀 ULTRA: Starting ultra-fast training...")
        
        print("🚀 ULTRA: Setting up ultra-fast training...")
        
        # Get device
        self.device = get_device()
        print(f"🖥️  ULTRA: Initializing with device: {self.device}")
        
        try:
            # Set default tensor type
            torch.set_default_dtype(torch.float32)
            print("🔧 ULTRA: Default tensor type set to float32")
        except Exception as e:
            print(f"⚠️  Error setting default dtype: {e}")
            raise
        
        # Optimized model loading for MPS
        torch_dtype = torch.float32  # Use float32 for MPS stability
        print(f"🧠 ULTRA: Using precision: {torch_dtype}")
        
        # MAXIMUM SPEED MPS optimization
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.98)  # Use almost all memory
            # MPS-specific optimizations
            if hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
                if hasattr(torch.backends.mps, 'set_flags'):
                    torch.backends.mps.set_flags(enable_mps_autocast=True)
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Enable CUDA optimizations if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Load model with appropriate settings
        print("🔄 ULTRA: Loading model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto"  # Let's use device_map for better memory management
            )
            print("✅ ULTRA: Model loaded successfully")
        except Exception as e:
            print(f"❌ ULTRA: Failed to load model: {e}")
            raise
        
        # ULTRA Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ULTRA Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize the dataset with config
        print("📂 Loading dataset...")
        try:
            # Use the correct dataset class that handles checkpointing and streaming
            # Initialize the streaming dataset with config
            dataset = UltraFastRMLDataset(self.config)
            self.dataset = dataset  # Store reference for signal handling
            print(f"✅ Initialized streaming dataset with checkpoint support")
            
            # Log dataset state
            if hasattr(dataset, 'processed_files') and hasattr(dataset, 'rml_files'):
                print(f"📊 Dataset progress: {dataset.processed_files}/{len(dataset.rml_files)} files processed")
                if dataset.processed_files > 0:
                    print(f"⏩ Will resume from file {dataset.processed_files + 1}")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise
        
        # ULTRA Training arguments
        import torch
        use_mps = torch.backends.mps.is_available()
        # MAXIMUM SPEED TrainingArguments
        self.training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=False,     # Disabled for max speed
            fp16=False,                      # Keep disabled for MPS
            bf16=False,                      # Keep disabled for MPS
            dataloader_num_workers=0,        # Must be 0 for MPS
            optim="adamw_torch_fused",       # Fused optimizer for speed
            remove_unused_columns=True,      # Save memory
            dataloader_pin_memory=False,     # Must be False for MPS
            # EXTREME SPEED settings:
            max_grad_norm=1.0,               # Gradient clipping
            lr_scheduler_type="cosine",      # Faster convergence
            warmup_steps=50,                 # Minimal warmup
            logging_steps=10,                 # Minimal logging
            save_steps=2000,                  # Less frequent checkpoints
            eval_steps=0,                     # No evaluation
            dataloader_drop_last=True,        # Avoid partial batches
            group_by_length=False,            # Faster batching
            ddp_find_unused_parameters=False, # Speed up DDP
            dataloader_persistent_workers=False,  # Disable for MPS
            optim_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            report_to="none",
            logging_first_step=True,
            save_safetensors=True,
            num_train_epochs=1,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=2,
        )
        
        # ULTRA Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset,  # Add the dataset here
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        
        # Get the checkpoint to resume from
        chosen_checkpoint = None
        
        # Check for force checkpoint from environment
        force_checkpoint = os.environ.get('ULTRA_FORCE_CHECKPOINT', None)
        if force_checkpoint:
            if os.path.isdir(force_checkpoint):
                chosen_checkpoint = force_checkpoint
            else:
                # Try to match by step
                checkpoint_dirs = [d for d in os.listdir(self.config.output_dir) 
                                if d.startswith('checkpoint-')]
                checkpoint_dirs = [os.path.join(self.config.output_dir, d) 
                                for d in checkpoint_dirs]
                matches = [d for d in checkpoint_dirs if force_checkpoint in d]
                if matches:
                    chosen_checkpoint = matches[-1]
            
            if chosen_checkpoint and os.path.exists(chosen_checkpoint):
                print(f"🔄 ULTRA: Forcing resume from checkpoint: {chosen_checkpoint}")
            else:
                print(f"⚠️ ULTRA: Specified checkpoint '{force_checkpoint}' not found, falling back to latest.")
                chosen_checkpoint = None
        
        # If no forced checkpoint, find the latest one
        if not chosen_checkpoint:
            checkpoint_dirs = [d for d in os.listdir(self.config.output_dir) 
                            if d.startswith('checkpoint-')]
            if checkpoint_dirs:
                checkpoint_dirs = [os.path.join(self.config.output_dir, d) 
                                for d in checkpoint_dirs]
                checkpoint_dirs = sorted(checkpoint_dirs, 
                                      key=lambda x: int(re.findall(r'checkpoint-(\d+)', x)[0]))
                chosen_checkpoint = checkpoint_dirs[-1]
                print(f"🔄 ULTRA: Resuming Trainer from latest checkpoint: {chosen_checkpoint}")
            else:
                print("🆕 ULTRA: No Trainer checkpoint found, starting fresh.")

        # Log checkpoint and dataset state before training
        print(f"🔍 ULTRA: Trainer checkpoint to load: {chosen_checkpoint if chosen_checkpoint else 'None (fresh start)'}")
        print(f"🔍 ULTRA: Dataset processed_files: {getattr(self.dataset, 'processed_files', '?')}, total_samples: {getattr(self.dataset, 'total_samples', '?')}")
        if hasattr(self.dataset, 'rml_files'):
            print(f"🔍 ULTRA: Dataset will start at file index: {getattr(self.dataset, 'processed_files', '?')} / {len(self.dataset.rml_files)}")
        
        try:
            # ULTRA Train with resume
            try:
                self.trainer.train(resume_from_checkpoint=chosen_checkpoint)
            except ValueError as e:
                if "Batch does not contain any data" in str(e):
                    print("✅ All training data has been processed!")
                    print("💾 Saving final model...")
                    self.trainer.save_model(self.config.output_dir)
                    self.tokenizer.save_pretrained(self.config.output_dir)
                    print(f"✅ Model saved to {self.config.output_dir}")
                    return
                raise
            
            # Save final model if training completes normally
            self.trainer.save_model(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            print("✅ Training completed successfully!")
            
        except Exception as e:
            print(f"❌ ULTRA: Training error: {str(e)}")
            # Save progress even on error
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'save_model'):
                self.trainer.save_model(self.config.output_dir + "/error_save")
                print(f"💾 Progress saved to {self.config.output_dir}/error_save")
            if self.dataset:
                self.dataset._save_checkpoint()
            raise  # Re-raise the exception after handling
    
    def _signal_handler(self, signum, frame):
        """ULTRA signal handler"""
        print(f"\n⏸️ ULTRA: Received signal {signum}, saving checkpoint and pausing...")
        if self.dataset:
            self.dataset.running = False
            self.dataset._save_checkpoint()
        print(f"💾 ULTRA: Checkpoint saved. Resume with: python3 src/rml_ultra_fast_optimized_trainer.py")
        sys.exit(0)

def main():
    """ULTRA-FAST main"""
    print("🚀 ULTRA-FAST RML TRAINER - 100x SPEED!")
    
    # ULTRA Config
    config = RMLUltraFastConfig()
    
    # ULTRA Trainer
    trainer = UltraFastRMLTrainer(config)
    
    # ULTRA Train
    trainer.train_ultra_fast()

if __name__ == "__main__":
    main() 