"""
Evo AI - A cutting-edge AI assistant with persistent learning capabilities.
"""

__version__ = "0.1.0"
__author__ = "Evo AI Team"
__email__ = "dev@evo-ai.com"

from .agent import EvoAgent
from .memory import MemoryManager
from .mcp import MCPClient

__all__ = ["EvoAgent", "MemoryManager", "MCPClient"]