#!/usr/bin/env python3
"""
RML Lightweight Trainer - Train Decoder without Large Encoder
Uses smaller models or pre-computed embeddings for training
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
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

@dataclass
class RMLLightweightConfig:
    """Configuration for lightweight RML training"""
    
    # Model settings
    decoder_model_name: str = "microsoft/phi-1_5"
    
    # Training settings
    batch_size: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 50
    max_seq_length: int = 256
    gradient_accumulation_steps: int = 4
    
    # Data settings
    train_data_path: str = "data/rml_training_data.jsonl"
    validation_data_path: str = "data/rml_validation_data.jsonl"
    output_dir: str = "output/rml_lightweight_trained"
    
    # Memory optimization
    device_map: str = "cpu"
    max_memory: Dict[str, str] = None
    
    # Logging
    log_level: str = "INFO"

class RMLTrainingDataset(Dataset):
    """Dataset for RML training without large encoder"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
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
        events = rml_data.get('events', [])
        reasoning = rml_data.get('reasoning', [])
        summaries = rml_data.get('summaries', [])
        
        # Create structured prompt
        prompt_parts = []
        
        # Input context
        if text:
            # Truncate text if too long
            if len(text) > 200:
                text = text[:200] + "..."
            prompt_parts.append(f"Context: {text}")
        
        if concepts:
            prompt_parts.append(f"Concepts: {', '.join(concepts[:5])}")
        
        if entities:
            prompt_parts.append(f"Entities: {', '.join(entities[:3])}")
        
        if emotions:
            prompt_parts.append(f"Emotions: {', '.join(emotions)}")
        
        if intents:
            prompt_parts.append(f"Intent: {', '.join(intents)}")
        
        if events:
            prompt_parts.append(f"Events: {', '.join(events[:2])}")
        
        if reasoning:
            prompt_parts.append(f"Reasoning: {', '.join(reasoning)}")
        
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
        emotions = rml_data.get('emotions', [])
        intents = rml_data.get('intents', [])
        summaries = rml_data.get('summaries', [])
        
        # Create a meaningful response based on the data
        response_parts = []
        
        if concepts:
            response_parts.append(f"The key concepts identified are: {', '.join(concepts[:3])}.")
        
        if emotions:
            response_parts.append(f"The emotional context is: {', '.join(emotions)}.")
        
        if intents:
            response_parts.append(f"The primary intent is: {', '.join(intents)}.")
        
        if summaries:
            response_parts.append(f"Summary: {summaries[0][:100]}...")
        elif text:
            # Create a simple summary
            sentences = text.split('.')
            if len(sentences) > 1:
                response_parts.append(f"Main idea: {sentences[0].strip()}.")
        
        if not response_parts:
            response_parts.append("This information provides valuable insights for understanding the topic.")
        
        return " ".join(response_parts)

class RMLLightweightTrainer:
    """Lightweight trainer for RML decoder"""
    
    def __init__(self, config: RMLLightweightConfig):
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize decoder only
        self.decoder = None
        self.decoder_tokenizer = None
        
        self.logger.info("🧠 Initializing RML Lightweight Trainer")
        self._load_decoder()
    
    def _load_decoder(self):
        """Load only the decoder model"""
        
        try:
            # Set memory limits for Mac M3 Pro
            if self.config.max_memory is None:
                self.config.max_memory = {
                    "cpu": "32GB",
                    "mps": "8GB"
                }
            
            self.logger.info("🔧 Loading Phi-3 decoder for training...")
            
            # Load decoder (Phi-3)
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                self.config.decoder_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.decoder_tokenizer.pad_token is None:
                self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            
            # Load on CPU with float32 to avoid accelerate requirement and FP16 issues on CPU
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.config.decoder_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            # Ensure model is on CPU for lightweight training by default
            self.decoder.to(torch.device("cpu"))
            
            self.logger.info("✅ Phi-3 decoder loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading decoder: {e}")
            raise
    
    def prepare_training_data(self):
        """Prepare training and validation datasets"""
        
        self.logger.info("📊 Preparing training data...")
        
        # Create datasets
        train_dataset = RMLTrainingDataset(
            self.config.train_data_path,
            self.decoder_tokenizer,
            self.config.max_seq_length
        )
        
        val_dataset = RMLTrainingDataset(
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
                logging_steps=5,
                save_steps=50,
                eval_steps=25,
                save_strategy="steps",
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb/tensorboard
                save_total_limit=2,  # Keep only 2 checkpoints
                fp16=False,
                bf16=False,
                no_cuda=True,
                use_mps_device=False,
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
        if self.decoder:
            del self.decoder
        if self.decoder_tokenizer:
            del self.decoder_tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("🧹 Memory cleaned up")

def main():
    """Main function for lightweight training"""
    
    print("🧠 RML Lightweight Training Pipeline")
    print("="*50)
    
    # Configuration
    config = RMLLightweightConfig(
        train_data_path="data/rml_training_data.jsonl",
        validation_data_path="data/rml_validation_data.jsonl",
        output_dir="output/rml_lightweight_trained",
        batch_size=1,  # Very small batch size for Mac M3 Pro
        num_epochs=1,  # Keep short for a quick fine-tune pass
        max_seq_length=128,  # Very short sequences for memory efficiency
        log_level="INFO"
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLLightweightTrainer(config)
        
        # Create training data from existing RML data
        print("📝 Creating training data...")
        trainer.create_training_data_from_rml(
            "data/converted_rml/complete_rml/rml_data.jsonl",
            config.train_data_path,
            max_samples=60  # Keep small for speed
        )
        
        # Create validation data (small subset)
        trainer.create_training_data_from_rml(
            "data/converted_rml/complete_rml/rml_data.jsonl",
            config.validation_data_path,
            max_samples=10
        )
        
        # Train the decoder
        print("🚀 Starting decoder training...")
        trainer.train_decoder()
        
        print("✅ RML lightweight training completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        if trainer:
            trainer.cleanup()

if __name__ == "__main__":
    main() 