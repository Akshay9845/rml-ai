#!/usr/bin/env python3
"""
RML Final Working Trainer - Proven working approach with scalability
Based on the successful simple CPU trainer but with batch processing
"""

import os
import json
import logging
import gc
import torch
import time
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import psutil

@dataclass
class RMLFinalWorkingConfig:
    """Configuration for final working RML training"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-small"
    
    # Training settings - PROVEN WORKING
    batch_size: int = 1  # Keep small for stability
    learning_rate: float = 1e-4  # Proven working rate
    num_epochs: int = 1
    warmup_steps: int = 10
    max_seq_length: int = 32  # Keep short for stability
    gradient_accumulation_steps: int = 4
    
    # Checkpoint settings
    save_steps: int = 50
    save_total_limit: int = 3
    eval_steps: int = 100
    
    # Data settings
    train_data_path: str = "data/language_training_data.jsonl"
    validation_data_path: str = "data/language_validation_data.jsonl"
    output_dir: str = "/Volumes/MEGA/R-LLM-final-trained"
    
    # Memory optimization
    device_map: str = "cpu"  # Force CPU for stability
    max_memory: Dict[str, str] = None
    
    # Logging
    log_level: str = "INFO"

class RMLFinalWorkingDataset(Dataset):
    """Final working dataset - proven approach"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 32, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.max_samples = max_samples
        self.data = self.load_data()
    
    def load_data(self) -> List[str]:
        """Load data with memory management"""
        texts = []
        
        if not os.path.exists(self.data_path):
            print(f"⚠️ Training data not found: {self.data_path}")
            return texts
        
        print(f"📊 Loading data from {self.data_path}")
        count = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get('text', '')
                    if text and len(text) > 5:
                        texts.append(text)
                        count += 1
                        
                        # Memory check every 1000 samples
                        if count % 1000 == 0:
                            memory_usage = psutil.virtual_memory().percent
                            print(f"   Loaded {count} samples, Memory: {memory_usage:.1f}%")
                            
                            # Stop if memory usage is high
                            if memory_usage > 80:
                                print(f"⚠️ High memory usage ({memory_usage:.1f}%), stopping at {count} samples")
                                break
                        
                        # Stop if max_samples reached
                        if self.max_samples and count >= self.max_samples:
                            print(f"✅ Reached max_samples limit: {count}")
                            break
                            
                except json.JSONDecodeError:
                    continue
        
        print(f"📊 Loaded {len(texts)} language texts")
        return texts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize the text - PROVEN WORKING APPROACH
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Keep False for stability
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class RMLFinalWorkingTrainer:
    """Final working trainer - proven approach with scalability"""
    
    def __init__(self, config: RMLFinalWorkingConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        
        # Performance tracking
        self.start_time = time.time()
        
        self.logger.info("🧠 Initializing RML Final Working Trainer")
        self._load_model()
    
    def _load_model(self):
        """Load model with proven working settings"""
        
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
            
            # Load model on CPU - PROVEN WORKING
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="cpu",  # Force CPU
                torch_dtype=torch.float32,  # Use float32 for stability
                trust_remote_code=True
            )
            
            self.logger.info("✅ Model loaded successfully on CPU")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def prepare_training_data(self, max_train_samples: int = None, max_val_samples: int = None):
        """Prepare training and validation datasets"""
        
        self.logger.info("📊 Preparing training data...")
        
        # Create datasets with sample limits
        train_dataset = RMLFinalWorkingDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_seq_length,
            max_train_samples
        )
        
        val_dataset = RMLFinalWorkingDataset(
            self.config.validation_data_path,
            self.tokenizer,
            self.config.max_seq_length,
            max_val_samples
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        return train_dataset, val_dataset, data_collator
    
    def train_decoder(self, max_train_samples: int = None, resume_from_checkpoint: str = None):
        """Train the decoder with proven working settings"""
        
        try:
            # Prepare data
            train_dataset, val_dataset, data_collator = self.prepare_training_data(
                max_train_samples=max_train_samples,
                max_val_samples=max_train_samples // 10 if max_train_samples else None
            )
            
            if len(train_dataset) == 0:
                self.logger.error("❌ No training data available")
                return
            
            # Calculate total steps
            total_steps = len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)
            self.logger.info(f"📊 Total training steps: {total_steps}")
            
            # Training arguments - PROVEN WORKING SETTINGS
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_steps=10,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_total_limit=self.config.save_total_limit,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,
                # CPU settings
                no_cuda=True,
                # Proven working settings
                max_grad_norm=1.0,
                fp16=False,
                bf16=False,
                gradient_checkpointing=False,  # Disable for stability
                optim="adamw_torch",
                weight_decay=0.01,
                # Resume settings
                resume_from_checkpoint=resume_from_checkpoint,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            
            self.logger.info("🚀 Starting final working training...")
            self.logger.info(f"⏱️ Expected time: {total_steps * 2 / 60:.1f} minutes")
            
            # Train the model
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save the trained model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Calculate training time
            training_time = time.time() - self.start_time
            self.logger.info(f"✅ Training completed in {training_time/3600:.2f} hours!")
            self.logger.info(f"💾 Model saved to {self.config.output_dir}")
            
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
    """Main function for final working training"""
    
    parser = argparse.ArgumentParser(description="RML Final Working Training")
    parser.add_argument("--samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="/Volumes/MEGA/R-LLM-final-trained", help="Output directory")
    args = parser.parse_args()
    
    print("🧠 RML Final Working Training Pipeline")
    print("="*40)
    print(f"📊 Training on {args.samples:,} samples")
    print(f"💾 Output: {args.output}")
    if args.resume:
        print(f"🔄 Resuming from: {args.resume}")
    
    # Configuration
    config = RMLFinalWorkingConfig(
        train_data_path="data/language_training_data.jsonl",
        validation_data_path="data/language_validation_data.jsonl",
        output_dir=args.output,
        batch_size=1,
        num_epochs=1,
        max_seq_length=32,
        log_level="INFO"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLFinalWorkingTrainer(config)
        
        # Train the decoder
        print("🚀 Starting final working decoder training...")
        trainer.train_decoder(
            max_train_samples=args.samples,
            resume_from_checkpoint=args.resume
        )
        
        # Test the model
        print("\n🧪 Testing trained model...")
        test_input = "Q: What does machine"
        response = trainer.test_model(test_input)
        print(f"Test input: {test_input}")
        print(f"Model response: {response}")
        
        print("✅ RML final working training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 