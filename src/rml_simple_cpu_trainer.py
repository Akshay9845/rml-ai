#!/usr/bin/env python3
"""
RML Simple CPU Trainer - Avoids MPS issues
Simple, reliable training on CPU only
"""

import os
import json
import logging
import gc
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

@dataclass
class RMLSimpleCPUConfig:
    """Configuration for simple CPU training"""
    
    # Use a very simple model
    model_name: str = "microsoft/DialoGPT-small"
    
    # Training settings
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 1
    warmup_steps: int = 5
    max_seq_length: int = 32
    gradient_accumulation_steps: int = 2
    
    # Data settings
    train_data_path: str = "data/language_training_data.jsonl"
    validation_data_path: str = "data/language_validation_data.jsonl"
    output_dir: str = "/Volumes/MEGA/R-LLM-simple-trained"
    
    # Force CPU
    device_map: str = "cpu"
    max_memory: Dict[str, str] = None
    
    # Logging
    log_level: str = "INFO"

class RMLSimpleCPUDataset(Dataset):
    """Simple CPU dataset"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path: str) -> List[str]:
        """Load language text data"""
        texts = []
        
        if not os.path.exists(data_path):
            print(f"⚠️ Training data not found: {data_path}")
            return texts
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get('text', '')
                    if text and len(text) > 5:  # Very short texts
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
        
        print(f"📊 Loaded {len(texts)} language texts")
        return texts[:1000]  # Limit to 1000 for testing
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
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

class RMLSimpleCPUTrainer:
    """Simple CPU trainer for RML decoder"""
    
    def __init__(self, config: RMLSimpleCPUConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        self.logger.info("🧠 Initializing RML Simple CPU Trainer")
        self._load_model()
    
    def _load_model(self):
        """Load model on CPU"""
        
        try:
            self.logger.info(f"🔧 Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=True
            )
            
            self.logger.info("✅ Model loaded successfully on CPU")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def prepare_training_data(self):
        """Prepare training and validation datasets"""
        
        self.logger.info("📊 Preparing training data...")
        
        # Create datasets
        train_dataset = RMLSimpleCPUDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        val_dataset = RMLSimpleCPUDataset(
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
        """Train the decoder on CPU"""
        
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
                # CPU settings
                no_cuda=True,
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
            
            self.logger.info("🚀 Starting CPU training...")
            
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
            
            # Generate on CPU
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
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
        
        self.logger.info("🧹 Memory cleaned up")

def main():
    """Main function for simple CPU training"""
    
    print("🧠 RML Simple CPU Training Pipeline")
    print("="*40)
    
    # Configuration
    config = RMLSimpleCPUConfig(
        train_data_path="data/language_training_data.jsonl",
        validation_data_path="data/language_validation_data.jsonl",
        output_dir="/Volumes/MEGA/R-LLM-simple-trained",
        batch_size=1,
        num_epochs=1,
        max_seq_length=32,
        log_level="INFO"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLSimpleCPUTrainer(config)
        
        # Train the decoder
        print("🚀 Starting decoder training...")
        trainer.train_decoder()
        
        # Test the model
        print("\n🧪 Testing trained model...")
        test_input = "Q: What does machine"
        response = trainer.test_model(test_input)
        print(f"Test input: {test_input}")
        print(f"Model response: {response}")
        
        print("✅ RML simple CPU training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 