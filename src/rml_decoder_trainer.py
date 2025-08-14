#!/usr/bin/env python3
"""
RML Decoder Trainer - Fine-tune Phi-3 on RML Data
Trains the decoder to understand RML patterns and relationships
"""

import os
import json
import logging
import gc
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

@dataclass
class RMLTrainingConfig:
    """Configuration for RML decoder training"""
    
    # Model settings
    encoder_model_name: str = "intfloat/e5-mistral-7b-instruct"
    decoder_model_name: str = "microsoft/phi-3-mini-4k-instruct"
    
    # Training settings
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # Data settings
    train_data_path: str = "data/rml_training_data.jsonl"
    validation_data_path: str = "data/rml_validation_data.jsonl"
    output_dir: str = "output/rml_trained_decoder"
    
    # Memory optimization
    use_quantization: bool = False
    device_map: str = "auto"
    max_memory: Dict[str, str] = None
    
    # Logging
    log_level: str = "INFO"

class RMLDataset(Dataset):
    """Dataset for RML training data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
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
        
        # Create training prompt from RML data
        prompt = self.create_training_prompt(item)
        
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
    
    def create_training_prompt(self, rml_data: Dict[str, Any]) -> str:
        """Create training prompt from RML data"""
        
        # Extract components
        text = rml_data.get('text', '')
        concepts = rml_data.get('concepts', [])
        entities = rml_data.get('entities', [])
        emotions = rml_data.get('emotions', [])
        intents = rml_data.get('intents', [])
        
        # Create structured prompt
        prompt_parts = []
        
        # Input context
        if text:
            prompt_parts.append(f"Context: {text}")
        
        if concepts:
            prompt_parts.append(f"Concepts: {', '.join(concepts[:5])}")
        
        if entities:
            prompt_parts.append(f"Entities: {', '.join(entities[:3])}")
        
        if emotions:
            prompt_parts.append(f"Emotions: {', '.join(emotions)}")
        
        if intents:
            prompt_parts.append(f"Intent: {', '.join(intents)}")
        
        # Create target response
        target_response = self.generate_target_response(rml_data)
        
        # Combine into training format
        prompt = "\n".join(prompt_parts)
        prompt += f"\n\nResponse: {target_response}"
        
        return prompt
    
    def generate_target_response(self, rml_data: Dict[str, Any]) -> str:
        """Generate target response for training"""
        
        text = rml_data.get('text', '')
        concepts = rml_data.get('concepts', [])
        
        # Create a meaningful response based on the data
        if concepts:
            response = f"Based on the analysis, the key concepts are: {', '.join(concepts[:3])}. "
            
            if text:
                # Create a summary-like response
                sentences = text.split('.')
                if len(sentences) > 1:
                    response += f"The main idea is: {sentences[0].strip()}. "
            
            response += "This information provides valuable insights for understanding the topic."
        else:
            response = "The provided information contains important details that contribute to our understanding of the subject matter."
        
        return response

class RMLDecoderTrainer:
    """Trainer for RML decoder"""
    
    def __init__(self, config: RMLTrainingConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize models
        self.encoder = None
        self.decoder = None
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        
        self.logger.info("🧠 Initializing RML Decoder Trainer")
        self._load_models()
    
    def _load_models(self):
        """Load encoder and decoder models"""
        
        try:
            # Set memory limits for Mac M3 Pro
            if self.config.max_memory is None:
                self.config.max_memory = {
                    "cpu": "32GB",
                    "mps": "8GB"
                }
            
            self.logger.info("🔧 Loading E5-Mistral encoder...")
            
            # Load encoder (E5-Mistral)
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(
                self.config.encoder_model_name,
                trust_remote_code=True
            )
            
            self.encoder = AutoModel.from_pretrained(
                self.config.encoder_model_name,
                device_map=self.config.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.logger.info("✅ E5-Mistral encoder loaded successfully")
            
            # Clear memory before loading decoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("🔧 Loading Phi-3 decoder for training...")
            
            # Load decoder (Phi-3)
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                self.config.decoder_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.decoder_tokenizer.pad_token is None:
                self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.config.decoder_model_name,
                device_map=self.config.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.logger.info("✅ Phi-3 decoder loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
            raise
    
    def prepare_training_data(self):
        """Prepare training and validation datasets"""
        
        self.logger.info("📊 Preparing training data...")
        
        # Create datasets
        train_dataset = RMLDataset(
            self.config.train_data_path,
            self.decoder_tokenizer,
            self.config.max_seq_length
        )
        
        val_dataset = RMLDataset(
            self.config.validation_data_path,
            self.decoder_tokenizer,
            self.config.max_seq_length
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.decoder_tokenizer,
            mlm=False  # We're doing causal language modeling
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
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb/tensorboard
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.decoder,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.decoder_tokenizer,
            )
            
            self.logger.info("🚀 Starting training...")
            
            # Train the model
            trainer.train()
            
            # Save the trained model
            trainer.save_model()
            self.decoder_tokenizer.save_pretrained(self.config.output_dir)
            
            self.logger.info(f"✅ Training completed! Model saved to {self.config.output_dir}")
            
            # Evaluate
            eval_results = trainer.evaluate()
            self.logger.info(f"📊 Evaluation results: {eval_results}")
            
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            raise
    
    def create_training_data_from_rml(self, rml_data_path: str, output_path: str, max_samples: int = 1000):
        """Create training data from existing RML data"""
        
        self.logger.info(f"📝 Creating training data from {rml_data_path}")
        
        training_data = []
        
        try:
            with open(rml_data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    
                    try:
                        rml_item = json.loads(line.strip())
                        
                        # Create training example
                        training_example = {
                            'text': rml_item.get('text', ''),
                            'concepts': rml_item.get('concepts', []),
                            'entities': rml_item.get('entities', []),
                            'emotions': rml_item.get('emotions', []),
                            'intents': rml_item.get('intents', []),
                            'events': rml_item.get('events', []),
                            'reasoning': rml_item.get('reasoning', []),
                            'summaries': rml_item.get('summaries', [])
                        }
                        
                        training_data.append(training_example)
                        
                    except json.JSONDecodeError:
                        continue
            
            # Save training data
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            self.logger.info(f"✅ Created {len(training_data)} training examples in {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating training data: {e}")
            raise
    
    def cleanup(self):
        """Clean up models and free memory"""
        if self.encoder:
            del self.encoder
        if self.decoder:
            del self.decoder
        if self.encoder_tokenizer:
            del self.encoder_tokenizer
        if self.decoder_tokenizer:
            del self.decoder_tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("🧹 Memory cleaned up")

def main():
    """Main function for training the RML decoder"""
    
    print("🧠 RML Decoder Training Pipeline")
    print("="*50)
    
    # Configuration
    config = RMLTrainingConfig(
        train_data_path="data/rml_training_data.jsonl",
        validation_data_path="data/rml_validation_data.jsonl",
        output_dir="output/rml_trained_decoder",
        batch_size=2,  # Small batch size for Mac M3 Pro
        num_epochs=2,  # Start with few epochs
        max_seq_length=256,  # Shorter sequences for memory efficiency
        log_level="INFO"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLDecoderTrainer(config)
        
        # Create training data from existing RML data
        print("📝 Creating training data...")
        trainer.create_training_data_from_rml(
            "data/converted_rml/complete_rml/rml_data.jsonl",
            config.train_data_path,
            max_samples=500  # Start with small dataset
        )
        
        # Create validation data (small subset)
        trainer.create_training_data_from_rml(
            "data/converted_rml/complete_rml/rml_data.jsonl",
            config.validation_data_path,
            max_samples=100
        )
        
        # Train the decoder
        print("🚀 Starting decoder training...")
        trainer.train_decoder()
        
        print("✅ RML decoder training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 