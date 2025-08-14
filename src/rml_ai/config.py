"""
Configuration for RML System
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class RMLConfig:
    """Configuration for RML system"""
    
    # Model paths
    encoder_model: str = "intfloat/e5-base-v2"
    decoder_model: str = "microsoft/phi-1_5"
    
    # Device configuration
    device: str = "auto"
    
    # Dataset configuration
    dataset_path: str = "data/rml_data.jsonl"
    max_entries: int = 1000
    
    # Encoding configuration
    encoder_batch_size: int = 8
    encoder_max_length: int = 192
    
    # Feature flags
    disable_web_search: bool = True
    disable_world_knowledge: bool = True
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.encoder_model = os.getenv("RML_ENCODER_MODEL", self.encoder_model)
        self.decoder_model = os.getenv("RML_DECODER_MODEL", self.decoder_model)
        self.device = os.getenv("RML_DEVICE", self.device)
        self.dataset_path = os.getenv("RML_DATASET_PATH", self.dataset_path)
        self.max_entries = int(os.getenv("RML_API_ENTRIES", self.max_entries))
        self.encoder_batch_size = int(os.getenv("RML_ENCODER_BATCH_SIZE", self.encoder_batch_size))
        self.encoder_max_length = int(os.getenv("RML_ENCODER_MAX_LEN", self.encoder_max_length))
        self.disable_web_search = os.getenv("RML_DISABLE_WEB_SEARCH", "1") == "1"
        self.disable_world_knowledge = os.getenv("RML_DISABLE_WORLD_KNOWLEDGE", "1") == "1"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "encoder_model": self.encoder_model,
            "decoder_model": self.decoder_model,
            "device": self.device,
            "dataset_path": self.dataset_path,
            "max_entries": self.max_entries,
            "encoder_batch_size": self.encoder_batch_size,
            "encoder_max_length": self.encoder_max_length,
            "disable_web_search": self.disable_web_search,
            "disable_world_knowledge": self.disable_world_knowledge,
        }
    
    def __str__(self) -> str:
        """String representation of config"""
        config_str = "RML Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str 