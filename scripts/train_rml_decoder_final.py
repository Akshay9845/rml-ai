#!/usr/bin/env python3
"""
Final RML Decoder Training Script
Trains Phi-3 decoder on prepared RML data
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_lightweight_trainer import RMLLightweightTrainer, RMLLightweightConfig

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rml_training_final.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main training function"""
    logger = setup_logging()
    
    logger.info("🚀 Starting Final RML Decoder Training")
    logger.info("="*50)
    
    # Configuration for training
    config = RMLLightweightConfig(
        train_data_path="data/all_rml_training_data.jsonl",
        validation_data_path="data/all_rml_validation_data.jsonl",
        output_dir="/Volumes/MEGA/R-LLM-trained-decoder",
        batch_size=1,  # Very small for Mac M3 Pro
        num_epochs=3,
        learning_rate=5e-5,
        max_seq_length=128,  # Short sequences for memory efficiency
        warmup_steps=50,
        gradient_accumulation_steps=4,
        log_level="INFO"
    )
    
    # Initialize trainer
    trainer = None
    try:
        logger.info("🔧 Initializing trainer...")
        trainer = RMLLightweightTrainer(config)
        
        # Check if training data exists
        if not os.path.exists(config.train_data_path):
            logger.error(f"❌ Training data not found: {config.train_data_path}")
            return
        
        if not os.path.exists(config.validation_data_path):
            logger.error(f"❌ Validation data not found: {config.validation_data_path}")
            return
        
        logger.info(f"📊 Training data: {config.train_data_path}")
        logger.info(f"📊 Validation data: {config.validation_data_path}")
        
        # Start training
        logger.info("🚀 Starting decoder training...")
        trainer.train_decoder()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📁 Trained model saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise
        
    finally:
        if trainer:
            logger.info("🧹 Cleaning up...")
            trainer.cleanup()

if __name__ == "__main__":
    main() 