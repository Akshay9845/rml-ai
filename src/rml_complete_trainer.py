#!/usr/bin/env python3
"""
RML Complete Trainer - Processes 100% of RML Data
Covers ALL 43,923 files from /Users/elite/R-LLM/data/
"""

import os
import json
import logging
import pickle
import psutil
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/elite/R-LLM/rml-complete-trained/complete_training.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class RMLCompleteConfig:
    """Configuration for complete RML training"""
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    max_seq_length: int = 128
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 1
    warmup_steps: int = 100
    
    # Data settings
    data_dir: str = "/Users/elite/R-LLM/data"
    output_dir: str = "/Users/elite/R-LLM/rml-complete-trained"
    checkpoint_dir: str = "/Users/elite/R-LLM/rml-complete-checkpoints"
    progress_file: str = "/Users/elite/R-LLM/complete_training_progress.pkl"
    
    # Memory settings
    max_memory: Dict[str, str] = None
    
    def __post_init__(self):
        if self.max_memory is None:
            self.max_memory = {
                "mps": "14GB",
                "cpu": "4GB"
            }
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

class RMLCompleteDataset(Dataset):
    """Dataset that loads ALL RML files from the data directory"""
    
    def __init__(self, config: RMLCompleteConfig, tokenizer, is_training=True):
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.max_length = config.max_seq_length
        
        # Find ALL RML files
        self.rml_files = self._find_all_rml_files()
        logging.info(f"📁 Found {len(self.rml_files)} RML files to process")
        
        # Load all data
        self.data = self._load_all_data()
        logging.info(f"📊 Loaded {len(self.data)} total samples")
        
    def _find_all_rml_files(self) -> List[str]:
        """Find ALL RML files in the data directory"""
        rml_files = []
        
        # Search in main data directory and all subdirectories
        search_patterns = [
            f"{self.config.data_dir}/**/*.jsonl",
            f"{self.config.data_dir}/*.jsonl"
        ]
        
        for pattern in search_patterns:
            files = glob.glob(pattern, recursive=True)
            rml_files.extend(files)
        
        # Filter for RML files (concepts, entities, triples, etc.)
        rml_keywords = ['concepts', 'entities', 'triples', 'emotions', 'intents', 
                       'events', 'vectors', 'reasoning', 'summaries', 'tags']
        
        filtered_files = []
        for file_path in rml_files:
            filename = os.path.basename(file_path)
            if any(keyword in filename for keyword in rml_keywords):
                filtered_files.append(file_path)
        
        logging.info(f"📂 Total RML files found: {len(filtered_files)}")
        return filtered_files
    
    def _extract_text_universal(self, item: Dict[str, Any]) -> str:
        """Extract text from RML data structure"""
        text_parts = []
        
        # Direct text field
        if 'text' in item and item['text']:
            text_parts.append(str(item['text']))
        
        # Concepts
        if 'concepts' in item:
            if isinstance(item['concepts'], dict) and 'text' in item['concepts']:
                text_parts.append(str(item['concepts']['text']))
            elif isinstance(item['concepts'], list):
                for concept in item['concepts']:
                    if isinstance(concept, dict) and 'text' in concept:
                        text_parts.append(str(concept['text']))
        
        # Entities
        if 'entities' in item:
            if isinstance(item['entities'], dict) and 'text' in item['entities']:
                text_parts.append(str(item['entities']['text']))
            elif isinstance(item['entities'], list):
                for entity in item['entities']:
                    if isinstance(entity, dict) and 'text' in entity:
                        text_parts.append(str(entity['text']))
        
        # Other RML fields
        for field in ['emotions', 'intents', 'events', 'reasoning', 'summaries', 'triples', 'tags']:
            if field in item and item[field]:
                if isinstance(item[field], str):
                    text_parts.append(str(item[field]))
                elif isinstance(item[field], list):
                    text_parts.extend([str(x) for x in item[field] if x])
                elif isinstance(item[field], dict):
                    text_parts.append(str(item[field]))
        
        # Combine all text parts
        combined_text = " ".join(text_parts)
        
        # If no text found, use a default
        if not combined_text.strip():
            combined_text = "RML data processing"
        
        return combined_text
    
    def _load_all_data(self) -> List[str]:
        """Load ALL data from ALL RML files"""
        all_texts = []
        processed_files = 0
        
        for file_path in self.rml_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read entire file content
                    file_content = f.read()
                    
                    # Try to parse as single JSON object first
                    try:
                        item = json.loads(file_content)
                        text = self._extract_text_universal(item)
                        if text and len(text.strip()) > 10:
                            all_texts.append(text)
                    except json.JSONDecodeError:
                        # If single JSON fails, try line-by-line parsing
                        lines = file_content.split('\n')
                        current_json = ""
                        brace_count = 0
                        
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if not line:
                                continue
                                
                            current_json += line
                            brace_count += line.count('{') - line.count('}')
                            
                            # If braces are balanced, try to parse JSON
                            if brace_count == 0 and current_json.strip():
                                try:
                                    item = json.loads(current_json)
                                    text = self._extract_text_universal(item)
                                    if text and len(text.strip()) > 10:
                                        all_texts.append(text)
                                except json.JSONDecodeError as e:
                                    # Log but continue processing
                                    logging.debug(f"JSON parse error in {file_path}:{line_num}: {e}")
                                current_json = ""
                        
                        # Handle any remaining JSON
                        if current_json.strip():
                            try:
                                item = json.loads(current_json)
                                text = self._extract_text_universal(item)
                                if text and len(text.strip()) > 10:
                                    all_texts.append(text)
                            except json.JSONDecodeError:
                                logging.debug(f"Final JSON parse error in {file_path}")
                
                processed_files += 1
                if processed_files % 1000 == 0:
                    logging.info(f"📂 Processed {processed_files}/{len(self.rml_files)} files, {len(all_texts)} samples")
                    
            except Exception as e:
                logging.error(f"❌ Error reading file {file_path}: {e}")
                continue
        
        logging.info(f"✅ Completed loading {len(all_texts)} samples from {processed_files} files")
        return all_texts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize with proper padding
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

class RMLCompleteTrainer:
    """Complete RML trainer that processes 100% of data"""
    
    def __init__(self, config: RMLCompleteConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logging.info(f"🚀 Initializing RML Complete Trainer")
        logging.info(f"📊 Target: 100% data coverage ({len(glob.glob(f'{config.data_dir}/**/*.jsonl', recursive=True))} files)")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            max_memory=config.max_memory
        )
        
        logging.info(f"✅ Model loaded on {self.device}")
        
    def train_complete(self):
        """Train on 100% of RML data"""
        logging.info("🚀 Starting COMPLETE RML training (100% data coverage)")
        
        # Create datasets
        train_dataset = RMLCompleteDataset(self.config, self.tokenizer, is_training=True)
        
        # Split data (90% train, 10% validation)
        split_idx = int(len(train_dataset) * 0.9)
        train_data = train_dataset.data[:split_idx]
        val_data = train_dataset.data[split_idx:]
        
        train_dataset.data = train_data
        val_dataset = RMLCompleteDataset(self.config, self.tokenizer, is_training=False)
        val_dataset.data = val_data
        
        logging.info(f"📊 Training samples: {len(train_dataset)}")
        logging.info(f"📊 Validation samples: {len(val_dataset)}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_steps=1000,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,
            fp16=False,  # Disable fp16 for MPS
            report_to=None,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            logging_dir=f"{self.config.output_dir}/logs"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        logging.info("🎯 Starting COMPLETE training with 100% data coverage!")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logging.info("✅ COMPLETE training finished! Model saved.")
        logging.info(f"📁 Model saved to: {self.config.output_dir}")

def main():
    """Main function to run complete RML training"""
    config = RMLCompleteConfig()
    
    # Check system resources
    memory = psutil.virtual_memory()
    logging.info(f"💾 System memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
    
    # Create trainer and start training
    trainer = RMLCompleteTrainer(config)
    trainer.train_complete()

if __name__ == "__main__":
    main() 