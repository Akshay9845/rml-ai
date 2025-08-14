#!/usr/bin/env python3
"""
Simple RML Trainer - Uses a more compatible model
Avoids DynamicCache issues and uses simpler training approach
"""

import os
import json
import logging
import gc
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

@dataclass
class RMLSimpleConfig:
    """Configuration for simple RML training"""
    
    # Use a more compatible model
    model_name: str = "microsoft/DialoGPT-small"  # Smaller, more compatible
    
    # Training settings
    batch_size: int = 1
    learning_rate: float = 5e-5
    num_epochs: int = 2
    warmup_steps: int = 10
    max_seq_length: int = 64  # Very short for memory efficiency
    gradient_accumulation_steps: int = 8
    
    # Data settings
    train_data_path: str = "data/all_rml_training_data.jsonl"
    validation_data_path: str = "data/all_rml_validation_data.jsonl"
    output_dir: str = "/Volumes/MEGA/R-LLM-trained-decoder"
    
    # Memory optimization
    device_map: str = "auto"
    max_memory: Dict[str, str] = None
    
    # Logging
    log_level: str = "INFO"

class RMLSimpleDataset(Dataset):
    """Simple dataset for RML training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load RML training data"""
        data = []
        
        if not os.path.exists(data_path):
            print(f"⚠️ Training data not found: {data_path}")
            return data
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        
        print(f"📊 Loaded {len(data)} training examples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create simple training prompt
        prompt = self.create_simple_prompt(item)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
    
    def create_simple_prompt(self, rml_data: Dict[str, Any]) -> str:
        """Create simple training prompt"""
        
        # Extract key components
        text = rml_data.get('text', '')[:100]  # Truncate
        concepts = rml_data.get('concepts', [])[:3]  # First 3 concepts
        emotions = rml_data.get('emotions', [])[:1]  # First emotion
        
        # Create simple prompt
        prompt_parts = []
        
        if text:
            prompt_parts.append(f"Text: {text}")
        
        if concepts:
            prompt_parts.append(f"Concepts: {', '.join(concepts)}")
        
        if emotions:
            prompt_parts.append(f"Emotion: {emotions[0]}")
        
        # Simple response
        response = f"Analysis: This text discusses {', '.join(concepts) if concepts else 'various topics'}."
        
        # Combine
        prompt = " | ".join(prompt_parts) + f" | Response: {response}"
        
        return prompt

class RMLSimpleTrainer:
    """Simple trainer for RML decoder"""
    
    def __init__(self, config: RMLSimpleConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        self.logger.info("🧠 Initializing RML Simple Trainer")
        self._load_model()
    
    def _load_model(self):
        """Load model"""
        
        try:
            # Set memory limits for Mac M3 Pro
            if self.config.max_memory is None:
                self.config.max_memory = {
                    "cpu": "32GB",
                    "mps": "8GB"
                }
            
            self.logger.info(f"🔧 Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.config.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def prepare_training_data(self):
        """Prepare training and validation datasets"""
        
        self.logger.info("📊 Preparing training data...")
        
        # Create datasets
        train_dataset = RMLSimpleDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        val_dataset = RMLSimpleDataset(
            self.config.validation_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal language modeling
        )
        
        return train_dataset, val_dataset, data_collator
    
    def train_decoder(self):
        """Train the decoder on RML data"""
        
        try:
            # Prepare data
            train_dataset, val_dataset, data_collator = self.prepare_training_data()
            
            if len(train_dataset) == 0:
                self.logger.error("❌ No training data available")
                return
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_steps=10,
                save_steps=100,
                eval_steps=50,
                save_strategy="steps",
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,
                save_total_limit=2,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            
            self.logger.info("🚀 Starting training...")
            
            # Train the model
            trainer.train()
            
            # Save the trained model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            self.logger.info(f"✅ Training completed! Model saved to {self.config.output_dir}")
            
            # Evaluate
            eval_results = trainer.evaluate()
            self.logger.info(f"📊 Evaluation results: {eval_results}")
            
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            raise
    
    def cleanup(self):
        """Clean up models and free memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("🧹 Memory cleaned up")

def main():
    """Main function for simple training"""
    
    print("🧠 RML Simple Training Pipeline")
    print("="*40)
    
    # Configuration
    config = RMLSimpleConfig(
        train_data_path="data/all_rml_training_data.jsonl",
        validation_data_path="data/all_rml_validation_data.jsonl",
        output_dir="/Volumes/MEGA/R-LLM-trained-decoder",
        batch_size=1,
        num_epochs=2,
        max_seq_length=64,
        log_level="INFO"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLSimpleTrainer(config)
        
        # Train the decoder
        print("🚀 Starting decoder training...")
        trainer.train_decoder()
        
        print("✅ RML simple training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 