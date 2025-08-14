#!/usr/bin/env python3
"""
Working Final RML Trainer - Uses proper training pairs
Input/target format that the model can actually learn from
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
class RMLWorkingFinalConfig:
    """Configuration for working final RML training"""
    
    # Use a reliable model
    model_name: str = "microsoft/DialoGPT-small"
    
    # Training settings
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 1
    warmup_steps: int = 10
    max_seq_length: int = 64
    gradient_accumulation_steps: int = 8
    
    # Data settings
    train_data_path: str = "data/proper_training_data.jsonl"
    validation_data_path: str = "data/proper_validation_data.jsonl"
    output_dir: str = "/Volumes/MEGA/R-LLM-trained-decoder"
    
    # Memory optimization
    device_map: str = "auto"
    max_memory: Dict[str, str] = None
    
    # Logging
    log_level: str = "INFO"

class RMLWorkingFinalDataset(Dataset):
    """Working final dataset with proper input/target pairs"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training pairs"""
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
        
        print(f"📊 Loaded {len(data)} training pairs")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create training text: input + target
        input_text = item['input']
        target_text = item['target']
        training_text = input_text + " " + target_text
        
        # Tokenize without padding
        encoding = self.tokenizer(
            training_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class RMLWorkingFinalTrainer:
    """Working final trainer for RML decoder"""
    
    def __init__(self, config: RMLWorkingFinalConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        self.logger.info("🧠 Initializing RML Working Final Trainer")
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
        train_dataset = RMLWorkingFinalDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        val_dataset = RMLWorkingFinalDataset(
            self.config.validation_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
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
                logging_steps=5,
                save_steps=50,
                eval_steps=25,
                save_strategy="steps",
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,
                save_total_limit=2,
                # Prevent NaN gradients
                max_grad_norm=1.0,
                fp16=False,
                bf16=False,
                # Better training settings
                gradient_checkpointing=False,
                optim="adamw_torch",
                weight_decay=0.01,
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
    
    def test_model(self, test_input: str):
        """Test the trained model"""
        if not self.model or not self.tokenizer:
            self.logger.error("❌ Model not loaded")
            return
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                test_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input from response
            if test_input in response:
                response = response.replace(test_input, "").strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Test error: {e}")
            return None
    
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
    """Main function for working final training"""
    
    print("🧠 RML Working Final Training Pipeline")
    print("="*40)
    
    # Configuration
    config = RMLWorkingFinalConfig(
        train_data_path="data/proper_training_data.jsonl",
        validation_data_path="data/proper_validation_data.jsonl",
        output_dir="/Volumes/MEGA/R-LLM-trained-decoder",
        batch_size=1,
        num_epochs=1,
        max_seq_length=64,
        log_level="INFO"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLWorkingFinalTrainer(config)
        
        # Train the decoder
        print("🚀 Starting decoder training...")
        trainer.train_decoder()
        
        # Test the model
        print("\n🧪 Testing trained model...")
        test_input = "Analyze these concepts: cloud, computing"
        response = trainer.test_model(test_input)
        print(f"Test input: {test_input}")
        print(f"Model response: {response}")
        
        print("✅ RML working final training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 