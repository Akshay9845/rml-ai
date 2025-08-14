#!/usr/bin/env python3
"""
Script to Train RML Decoder
Runs the complete training pipeline for fine-tuning Phi-3 on RML data
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_decoder_trainer import RMLDecoderTrainer, RMLTrainingConfig

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rml_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train RML Decoder")
    parser.add_argument("--input-data", required=True, help="Input RML data file (JSONL)")
    parser.add_argument("--output-dir", default="/Volumes/MEGA/R-LLM-trained-decoder", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples for training")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("🚀 Starting RML Decoder Training")
    logger.info(f"📂 Input data: {args.input_data}")
    logger.info(f"📁 Output: {args.output_dir}")
    logger.info(f"📊 Max samples: {args.max_samples}")
    logger.info(f"🎯 Batch size: {args.batch_size}")
    logger.info(f"🔄 Epochs: {args.epochs}")
    
    # Configuration
    config = RMLTrainingConfig(
        train_data_path="data/rml_training_data.jsonl",
        validation_data_path="data/rml_validation_data.jsonl",
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_seq_length=256,  # Shorter for memory efficiency
        log_level=args.log_level
    )
    
    # Initialize trainer
    trainer = None
    try:
        trainer = RMLDecoderTrainer(config)
        
        # Create training data from RML data
        logger.info("📝 Creating training data...")
        trainer.create_training_data_from_rml(
            args.input_data,
            config.train_data_path,
            max_samples=args.max_samples
        )
        
        # Create validation data (20% of training data)
        val_samples = min(200, args.max_samples // 5)
        trainer.create_training_data_from_rml(
            args.input_data,
            config.validation_data_path,
            max_samples=val_samples
        )
        
        # Train the decoder
        logger.info("🚀 Starting decoder training...")
        trainer.train_decoder()
        
        logger.info("✅ RML decoder training completed!")
        logger.info(f"📁 Trained model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise
        
    finally:
        if trainer:
            logger.info("🧹 Cleaning up...")
            trainer.cleanup()

if __name__ == "__main__":
    main() 