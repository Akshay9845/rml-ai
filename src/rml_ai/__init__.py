"""
RML-AI: Resonant Memory Learning
A New Generation of AI for Mission-Critical Applications
"""

__version__ = "0.1.0"
__author__ = "RML AI Team"
__email__ = "team@rml-ai.com"

from .core import RMLSystem, RMLEncoder, RMLDecoder
from .memory import MemoryStore
from .config import RMLConfig

__all__ = [
    "RMLSystem",
    "RMLEncoder", 
    "RMLDecoder",
    "MemoryStore",
    "RMLConfig",
] 