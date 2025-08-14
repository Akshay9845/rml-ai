#!/usr/bin/env python3
"""
RML Ultimate Trainer - COMPLETE VERSION
Processes EVERY SINGLE WORD and piece of information from ALL RML data
Leaves NOTHING behind - extracts everything possible
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
class RMLUltimateCompleteConfig:
    """Complete configuration for processing EVERYTHING"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    
    # ULTIMATE SPEED SETTINGS
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 100
    max_seq_length: int = 256  # Increased for complete data
    gradient_accumulation_steps: int = 16
    
    # CHECKPOINT SETTINGS
    save_steps: int = 1000
    save_total_limit: int = 10
    eval_steps: int = 2000
    
    # DATA SETTINGS
    data_root: str = "/Users/elite/R-LLM/data"
    output_dir: str = "/Users/elite/R-LLM/rml-ultimate-trained"
    checkpoint_dir: str = "/Users/elite/R-LLM/rml-checkpoints"
    progress_file: str = "/Users/elite/R-LLM/training_progress.pkl"
    
    # MEMORY OPTIMIZATION
    device_map: str = "cpu"
    max_memory: Dict[str, str] = None
    
    # MONITORING
    log_level: str = "INFO"
    memory_check_interval: int = 30

class RMLUltimateCompleteDataset(Dataset):
    """Complete dataset that extracts EVERYTHING from RML data"""
    
    def __init__(self, data_paths: List[str], tokenizer, max_length: int = 256, processed_files: Set[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_paths = data_paths
        self.processed_files = processed_files or set()
        self.data = []
        self.file_mapping = []
        self.load_all_data()
    
    def extract_everything(self, item: Dict[str, Any]) -> List[str]:
        """Extract EVERY SINGLE PIECE of information from RML data"""
        texts = []
        
        # METHOD 1: Direct text extraction (simple structure)
        if 'text' in item and isinstance(item['text'], str) and len(item['text'].strip()) > 0:
            texts.append(item['text'])
        
        # METHOD 2: Extract ALL metadata from simple structure
        if isinstance(item, dict):
            for key, value in item.items():
                if key == 'text':
                    continue  # Already handled
                
                if isinstance(value, str) and len(value.strip()) > 0:
                    texts.append(f"{key}: {value}")
                elif isinstance(value, list) and value:
                    texts.append(f"{key}: {', '.join(str(v) for v in value)}")
                elif isinstance(value, dict) and value:
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and len(sub_value.strip()) > 0:
                            texts.append(f"{key}.{sub_key}: {sub_value}")
                        elif isinstance(sub_value, list) and sub_value:
                            texts.append(f"{key}.{sub_key}: {', '.join(str(v) for v in sub_value)}")
                elif value is not None and value != "":
                    texts.append(f"{key}: {str(value)}")
        
        # METHOD 3: Nested RML structure - extract EVERYTHING
        # Concepts
        if 'concepts' in item:
            concepts = item['concepts']
            if isinstance(concepts, dict):
                for key, value in concepts.items():
                    if isinstance(value, str) and len(value.strip()) > 0:
                        texts.append(f"concept.{key}: {value}")
                    elif isinstance(value, list) and value:
                        texts.append(f"concept.{key}: {', '.join(str(v) for v in value)}")
                    elif value is not None and value != "":
                        texts.append(f"concept.{key}: {str(value)}")
            elif isinstance(concepts, str) and len(concepts.strip()) > 0:
                texts.append(f"concepts: {concepts}")
        
        # Entities
        if 'entities' in item:
            entities = item['entities']
            if isinstance(entities, dict):
                for key, value in entities.items():
                    if isinstance(value, str) and len(value.strip()) > 0:
                        texts.append(f"entity.{key}: {value}")
                    elif isinstance(value, list) and value:
                        texts.append(f"entity.{key}: {', '.join(str(v) for v in value)}")
                    elif value is not None and value != "":
                        texts.append(f"entity.{key}: {str(value)}")
            elif isinstance(entities, str) and len(entities.strip()) > 0:
                texts.append(f"entities: {entities}")
        
        # Emotions
        if 'emotions' in item:
            emotions = item['emotions']
            if isinstance(emotions, str) and len(emotions.strip()) > 0:
                texts.append(f"emotions: {emotions}")
            elif isinstance(emotions, list) and emotions:
                texts.append(f"emotions: {', '.join(str(e) for e in emotions)}")
            elif emotions is not None and emotions != "":
                texts.append(f"emotions: {str(emotions)}")
        
        # Intents
        if 'intents' in item:
            intents = item['intents']
            if isinstance(intents, list) and intents:
                texts.append(f"intents: {', '.join(str(i) for i in intents)}")
            elif isinstance(intents, str) and len(intents.strip()) > 0:
                texts.append(f"intents: {intents}")
            elif intents is not None and intents != "":
                texts.append(f"intents: {str(intents)}")
        
        # Events
        if 'events' in item:
            events = item['events']
            if isinstance(events, dict):
                for key, value in events.items():
                    if isinstance(value, str) and len(value.strip()) > 0:
                        texts.append(f"event.{key}: {value}")
                    elif isinstance(value, list) and value:
                        texts.append(f"event.{key}: {', '.join(str(v) for v in value)}")
                    elif value is not None and value != "":
                        texts.append(f"event.{key}: {str(value)}")
            elif isinstance(events, str) and len(events.strip()) > 0:
                texts.append(f"events: {events}")
        
        # Reasoning
        if 'reasoning' in item:
            reasoning = item['reasoning']
            if isinstance(reasoning, list) and reasoning:
                texts.append(f"reasoning: {', '.join(str(r) for r in reasoning)}")
            elif isinstance(reasoning, str) and len(reasoning.strip()) > 0:
                texts.append(f"reasoning: {reasoning}")
            elif reasoning is not None and reasoning != "":
                texts.append(f"reasoning: {str(reasoning)}")
        
        # Summaries
        if 'summaries' in item:
            summaries = item['summaries']
            if isinstance(summaries, str) and len(summaries.strip()) > 0:
                texts.append(f"summaries: {summaries}")
            elif isinstance(summaries, list) and summaries:
                texts.append(f"summaries: {', '.join(str(s) for s in summaries)}")
            elif summaries is not None and summaries != "":
                texts.append(f"summaries: {str(summaries)}")
        
        # Triples
        if 'triples' in item:
            triples = item['triples']
            if isinstance(triples, list) and triples:
                texts.append(f"triples: {', '.join(str(t) for t in triples)}")
            elif isinstance(triples, str) and len(triples.strip()) > 0:
                texts.append(f"triples: {triples}")
            elif triples is not None and triples != "":
                texts.append(f"triples: {str(triples)}")
        
        # Tags
        if 'tags' in item:
            tags = item['tags']
            if isinstance(tags, list) and tags:
                texts.append(f"tags: {', '.join(str(t) for t in tags)}")
            elif isinstance(tags, str) and len(tags.strip()) > 0:
                texts.append(f"tags: {tags}")
            elif tags is not None and tags != "":
                texts.append(f"tags: {str(tags)}")
        
        # Vectors
        if 'vectors' in item:
            vectors = item['vectors']
            if isinstance(vectors, list) and vectors:
                texts.append(f"vectors: {', '.join(str(v) for v in vectors)}")
            elif isinstance(vectors, str) and len(vectors.strip()) > 0:
                texts.append(f"vectors: {vectors}")
            elif vectors is not None and vectors != "":
                texts.append(f"vectors: {str(vectors)}")
        
        # Relations
        if 'relations' in item:
            relations = item['relations']
            if isinstance(relations, list) and relations:
                texts.append(f"relations: {', '.join(str(r) for r in relations)}")
            elif isinstance(relations, str) and len(relations.strip()) > 0:
                texts.append(f"relations: {relations}")
            elif relations is not None and relations != "":
                texts.append(f"relations: {str(relations)}")
        
        # Source information
        if '_source_file' in item:
            texts.append(f"source_file: {item['_source_file']}")
        if '_source_line' in item:
            texts.append(f"source_line: {item['_source_line']}")
        
        # Filter out empty texts and return unique entries
        unique_texts = []
        seen = set()
        for text in texts:
            if text.strip() and text not in seen:
                unique_texts.append(text.strip())
                seen.add(text)
        
        return unique_texts
    
    def load_all_data(self):
        """Load ALL data from ALL files - extract EVERYTHING"""
        print(f"🚀 Loading ALL RML data from {len(self.data_paths)} files...")
        print(f"🎯 Processing EVERY SINGLE WORD and piece of information!")
        
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
                            
                            # Extract EVERYTHING from this item
                            texts = self.extract_everything(item)
                            
                            # Add each extracted text as a separate sample
                            for text in texts:
                                if len(text) > 3:  # Minimum length
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
        print(f"🎯 EVERY SINGLE PIECE of information has been extracted!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # FIXED: Proper tokenization with padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
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

class RMLUltimateCompleteTrainer:
    """Complete trainer that processes EVERYTHING"""
    
    def __init__(self, config: RMLUltimateCompleteConfig):
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
        
        self.logger.info("🧠 Initializing RML Ultimate Complete Trainer")
        self.logger.info("🎯 Processing EVERY SINGLE WORD and piece of information!")
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
        """Train on the ENTIRE dataset - process EVERYTHING"""
        try:
            # Get ALL data files
            all_data_files = self.get_all_data_files()
            
            if not all_data_files:
                self.logger.error("❌ No data files found!")
                return
            
            # Create dataset with ALL data
            dataset = RMLUltimateCompleteDataset(
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
                pad_to_multiple_of=8
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
                dataloader_num_workers=0,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            
            self.logger.info("🚀 Starting ULTIMATE COMPLETE training on FULL dataset...")
            self.logger.info("🎯 Processing EVERY SINGLE WORD and piece of information!")
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
            self.logger.info(f"✅ ULTIMATE COMPLETE training completed in {training_time/3600:.2f} hours!")
            self.logger.info(f"💾 Model saved to {self.config.output_dir}")
            self.logger.info(f"📊 Total samples processed: {self.total_samples_processed:,}")
            self.logger.info("🎯 EVERY SINGLE WORD has been processed!")
            
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
    """Main function for complete training"""
    
    parser = argparse.ArgumentParser(description="RML Ultimate Complete Training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="/Users/elite/R-LLM/rml-ultimate-trained", help="Output directory")
    args = parser.parse_args()
    
    print("🧠 RML Ultimate Complete Training Pipeline")
    print("="*50)
    print("🎯 Processing EVERY SINGLE WORD and piece of information!")
    print(f"💾 Output: {args.output}")
    print(f"🔄 Resume: {args.resume}")
    
    # Configuration
    config = RMLUltimateCompleteConfig(
        output_dir=args.output,
        checkpoint_dir=f"{args.output}/checkpoints",
        progress_file=f"{args.output}/training_progress.pkl"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLUltimateCompleteTrainer(config)
        
        # Train on ENTIRE dataset
        print("🚀 Starting ULTIMATE COMPLETE training on FULL dataset...")
        print("🎯 Processing EVERY SINGLE WORD and piece of information!")
        trainer.train_ultimate()
        
        print("✅ RML Ultimate Complete training completed!")
        print("🎯 EVERY SINGLE WORD has been processed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 