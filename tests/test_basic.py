#!/usr/bin/env python3
"""
Basic tests for RML-AI
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rml_ai.config import RMLConfig
from rml_ai.memory import MemoryStore


class TestRMLConfig(unittest.TestCase):
    """Test RML configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RMLConfig()
        
        self.assertEqual(config.encoder_model, "intfloat/e5-base-v2")
        self.assertEqual(config.decoder_model, "microsoft/phi-1_5")
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.max_entries, 1000)
        self.assertEqual(config.encoder_batch_size, 8)
        self.assertEqual(config.encoder_max_length, 192)
        self.assertTrue(config.disable_web_search)
        self.assertTrue(config.disable_world_knowledge)
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = RMLConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn('encoder_model', config_dict)
        self.assertIn('decoder_model', config_dict)
        self.assertIn('device', config_dict)
    
    def test_config_string_representation(self):
        """Test configuration string representation"""
        config = RMLConfig()
        config_str = str(config)
        
        self.assertIsInstance(config_str, str)
        self.assertIn('RML Configuration:', config_str)
        self.assertIn('encoder_model', config_str)


class TestMemoryStore(unittest.TestCase):
    """Test memory store functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory = MemoryStore()
        
        # Mock entries
        self.entries = [
            {"text": "Machine learning is a subset of AI", "source": "test"},
            {"text": "Deep learning uses neural networks", "source": "test"},
            {"text": "RML is a new AI paradigm", "source": "test"}
        ]
        
        # Mock embeddings (3 entries, 768 dimensions)
        self.embeddings = Mock()
        self.embeddings.shape = (3, 768)
    
    def test_add_entries(self):
        """Test adding entries to memory"""
        self.memory.add_entries(self.entries, self.embeddings)
        
        self.assertEqual(len(self.memory.entries), 3)
        self.assertEqual(self.memory.embeddings, self.embeddings)
    
    def test_get_stats(self):
        """Test memory statistics"""
        self.memory.add_entries(self.entries, self.embeddings)
        stats = self.memory.get_stats()
        
        self.assertEqual(stats['total_entries'], 3)
        self.assertEqual(stats['embedding_dim'], 768)
        self.assertTrue(stats['has_embeddings'])
    
    def test_empty_memory_stats(self):
        """Test statistics for empty memory"""
        stats = self.memory.get_stats()
        
        self.assertEqual(stats['total_entries'], 0)
        self.assertEqual(stats['embedding_dim'], 0)
        self.assertFalse(stats['has_embeddings'])


if __name__ == '__main__':
    unittest.main() 