"""
Model Context Protocol (MCP) integration for Evo AI.
"""

from .client import MCPClient
from .server import MCPServer
from .tools import ToolRegistry, Tool

__all__ = ["MCPClient", "MCPServer", "ToolRegistry", "Tool"]