#!/usr/bin/env python3
"""
RML Ultimate Complete Trainer - Processes 100% of ALL 202GB Data
Covers ALL 44,085 files from ALL 17 folders in /Users/elite/R-LLM/data/
Total: 417,927,089 records (417 million!)
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
        logging.FileHandler('/Users/elite/R-LLM/rml-ultimate-trained/ultimate_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RMLUltimateConfig:
    """Configuration for Ultimate RML Trainer"""
    model_name: str = "microsoft/DialoGPT-small"
    max_seq_length: int = 128
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 1
    warmup_steps: int = 100
    output_dir: str = "/Users/elite/R-LLM/rml-ultimate-trained"
    checkpoint_dir: str = "/Users/elite/R-LLM/rml-ultimate-checkpoints"
    progress_file: str = "/Users/elite/R-LLM/ultimate_training_progress.pkl"
    data_dir: str = "/Users/elite/R-LLM/data"

class RMLUltimateDataset(Dataset):
    """Dataset that loads ALL RML data from ALL folders"""
    
    def __init__(self, config: RMLUltimateConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Find ALL JSONL files in ALL folders
        self.rml_files = self._find_all_rml_files()
        logger.info(f"🎯 Found {len(self.rml_files)} total files to process")
        
        # Load ALL data
        self.all_texts = self._load_all_data()
        logger.info(f"📊 Loaded {len(self.all_texts)} total samples from ALL data")
        
        # Save progress
        self._save_progress()
    
    def _find_all_rml_files(self) -> List[str]:
        """Find ALL JSONL files in ALL folders"""
        all_files = []
        
        # Process each folder in data directory
        for folder in os.listdir(self.config.data_dir):
            folder_path = os.path.join(self.config.data_dir, folder)
            if os.path.isdir(folder_path):
                logger.info(f"📁 Scanning folder: {folder}")
                jsonl_files = glob.glob(os.path.join(folder_path, "**/*.jsonl"), recursive=True)
                all_files.extend(jsonl_files)
                logger.info(f"📂 Found {len(jsonl_files)} files in {folder}")
        
        logger.info(f"🎯 Total files found: {len(all_files)}")
        return all_files
    
    def _extract_text_universal(self, item: Dict[str, Any]) -> str:
        """Extract text from any RML format"""
        if isinstance(item, dict):
            # Direct text field
            if 'text' in item and item['text']:
                return str(item['text'])
            
            # Nested RML components
            text_parts = []
            
            # Check all possible RML fields
            for field in ['concepts', 'entities', 'emotions', 'intents', 'events', 'reasoning', 'summaries', 'triples', 'tags']:
                if field in item and item[field]:
                    if isinstance(item[field], dict) and 'text' in item[field]:
                        text_parts.append(str(item[field]['text']))
                    elif isinstance(item[field], list):
                        for sub_item in item[field]:
                            if isinstance(sub_item, dict) and 'text' in sub_item:
                                text_parts.append(str(sub_item['text']))
                            elif isinstance(sub_item, str):
                                text_parts.append(str(sub_item))
                    elif isinstance(item[field], str):
                        text_parts.append(str(item[field]))
            
            # Also check for any string values
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 10:
                    text_parts.append(value)
            
            if text_parts:
                return " ".join(text_parts)
        
        return str(item) if item else ""
    
    def _load_all_data(self) -> List[str]:
        """Load ALL data from ALL RML files"""
        all_texts = []
        processed_files = 0
        total_files = len(self.rml_files)
        
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
                        # Try line-by-line parsing
                        lines = file_content.strip().split('\n')
                        for line in lines:
                            if line.strip():
                                try:
                                    item = json.loads(line)
                                    text = self._extract_text_universal(item)
                                    if text and len(text.strip()) > 10:
                                        all_texts.append(text)
                                except json.JSONDecodeError:
                                    # Skip malformed lines
                                    continue
                
                processed_files += 1
                if processed_files % 1000 == 0:
                    logger.info(f"📂 Processed {processed_files}/{total_files} files, {len(all_texts)} samples")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing {file_path}: {e}")
                continue
        
        logger.info(f"✅ Completed processing {processed_files}/{total_files} files")
        return all_texts
    
    def _save_progress(self):
        """Save progress to file"""
        progress_data = {
            'total_files': len(self.rml_files),
            'total_samples': len(self.all_texts),
            'processed_files': len(self.rml_files)
        }
        
        os.makedirs(os.path.dirname(self.config.progress_file), exist_ok=True)
        with open(self.config.progress_file, 'wb') as f:
            pickle.dump(progress_data, f)
    
    def __len__(self):
        return len(self.all_texts)
    
    def __getitem__(self, idx):
        text = self.all_texts[idx]
        
        # Tokenize with proper padding
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

class RMLUltimateTrainer:
    """Ultimate RML Trainer that processes ALL data"""
    
    def __init__(self, config: RMLUltimateConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info(f"🚀 Initializing Ultimate RML Trainer")
        logger.info(f"📊 Device: {self.device}")
        logger.info(f"📁 Output: {config.output_dir}")
    
    def train_ultimate(self):
        """Train on ALL RML data"""
        logger.info("🎯 Starting Ultimate RML Training on ALL data")
        
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
        
        # Load dataset
        logger.info("📊 Loading ALL RML data...")
        dataset = RMLUltimateDataset(self.config)
        
        # Calculate total steps
        total_steps = len(dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)
        logger.info(f"📈 Total training steps: {total_steps}")
        
        # Training arguments
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
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            fp16=False,
            report_to=None
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
        logger.info("🚀 Starting Ultimate RML Training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ Ultimate RML Training completed!")

def main():
    """Main function"""
    config = RMLUltimateConfig()
    
    # Check system resources
    memory = psutil.virtual_memory()
    logger.info(f"💾 System Memory: {memory.total / (1024**3):.1f}GB total")
    logger.info(f"💾 Available Memory: {memory.available / (1024**3):.1f}GB")
    
    if memory.available < 4 * (1024**3):  # Less than 4GB
        logger.warning("⚠️ Low memory available, consider closing other applications")
    
    # Start training
    trainer = RMLUltimateTrainer(config)
    trainer.train_ultimate()

if __name__ == "__main__":
    main() 