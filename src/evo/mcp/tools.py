"""
Tool registry and management for MCP integration.
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .models import Tool, ToolResult


class ToolRegistry:
    """Registry for MCP tools and external integrations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def web_search(self, query: str, max_results: int = 5) -> ToolResult:
        """Search the web for information."""
        try:
            # This is a placeholder implementation
            # In practice, you'd integrate with a real search API
            content = [
                {
                    "type": "text",
                    "text": f"Web search results for '{query}':\n\n"
                    f"Note: This is a placeholder. In a real implementation, "
                    f"this would integrate with search APIs like DuckDuckGo, "
                    f"Bing, or Google to provide actual search results.",
                }
            ]

            return ToolResult(content=content)

        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return ToolResult(
                content=[{"type": "text", "text": f"Search failed: {str(e)}"}],
                isError=True,
            )

    async def read_file(self, file_path: str) -> ToolResult:
        """Read a file from the filesystem."""
        try:
            path = Path(file_path).expanduser().resolve()

            # Security check - only allow reading from safe directories
            safe_dirs = [
                Path.home(),
                Path.cwd(),
                Path("/tmp"),
                Path("/var/tmp"),
            ]

            if not any(path.is_relative_to(safe_dir) for safe_dir in safe_dirs):
                raise Exception(f"Access denied: {file_path}")

            if not path.exists():
                raise Exception(f"File not found: {file_path}")

            if path.is_dir():
                # List directory contents
                items = []
                for item in sorted(path.iterdir()):
                    item_type = "dir" if item.is_dir() else "file"
                    size = item.stat().st_size if item.is_file() else 0
                    items.append(f"{item_type}: {item.name} ({size} bytes)")

                content_text = f"Directory listing for {file_path}:\n\n" + "\n".join(items)
            else:
                # Read file contents
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content_text = f.read()

                # Limit size for large files
                if len(content_text) > 50000:
                    content_text = content_text[:50000] + "\n\n[Content truncated...]"

            return ToolResult(
                content=[
                    {
                        "type": "text",
                        "text": content_text,
                    }
                ]
            )

        except Exception as e:
            self.logger.error(f"File read error: {e}")
            return ToolResult(
                content=[{"type": "text", "text": f"Error reading file: {str(e)}"}],
                isError=True,
            )

    async def write_file(self, file_path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            path = Path(file_path).expanduser().resolve()

            # Security check
            safe_dirs = [
                Path.home(),
                Path.cwd(),
                Path("/tmp"),
                Path("/var/tmp"),
            ]

            if not any(path.is_relative_to(safe_dir) for safe_dir in safe_dirs):
                raise Exception(f"Access denied: {file_path}")

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                content=[
                    {
                        "type": "text",
                        "text": f"Successfully wrote {len(content)} characters to {file_path}",
                    }
                ]
            )

        except Exception as e:
            self.logger.error(f"File write error: {e}")
            return ToolResult(
                content=[{"type": "text", "text": f"Error writing file: {str(e)}"}],
                isError=True,
            )

    async def execute_command(
        self, command: str, timeout: int = 30, working_dir: Optional[str] = None
    ) -> ToolResult:
        """Execute a shell command."""
        try:
            # Security: Only allow safe commands
            safe_commands = [
                "ls", "dir", "pwd", "whoami", "date", "echo", "cat", "head", "tail",
                "grep", "find", "which", "python", "node", "npm", "git", "curl", "wget"
            ]

            command_parts = command.split()
            if not command_parts or command_parts[0] not in safe_commands:
                raise Exception(f"Command not allowed: {command}")

            # Set working directory
            cwd = Path(working_dir).expanduser() if working_dir else Path.cwd()

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise Exception(f"Command timed out after {timeout} seconds")

            # Format output
            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")

            output_text = "\n\n".join(output_parts) if output_parts else "No output"

            # Add return code
            output_text += f"\n\nReturn code: {process.returncode}"

            return ToolResult(
                content=[
                    {
                        "type": "text",
                        "text": output_text,
                    }
                ]
            )

        except Exception as e:
            self.logger.error(f"Command execution error: {e}")
            return ToolResult(
                content=[{"type": "text", "text": f"Error executing command: {str(e)}"}],
                isError=True,
            )

    async def analyze_code(self, code: str, language: str = "python") -> ToolResult:
        """Analyze code for issues and suggestions."""
        try:
            # Simple code analysis - in practice, you'd use actual linters/analyzers
            analysis_parts = []

            # Count lines
            lines = code.split("\n")
            analysis_parts.append(f"Code analysis for {language} ({len(lines)} lines):")

            # Basic checks
            if language.lower() == "python":
                # Python-specific checks
                issues = []
                suggestions = []

                if "import *" in code:
                    issues.append("- Avoid wildcard imports (import *)")

                if "print(" in code and not "# debug" in code.lower():
                    suggestions.append("- Consider using logging instead of print statements")

                if len([line for line in lines if len(line) > 100]) > 0:
                    suggestions.append("- Some lines exceed 100 characters (PEP 8)")

                if "except:" in code:
                    issues.append("- Avoid bare except clauses")

                if issues:
                    analysis_parts.append("\nIssues found:")
                    analysis_parts.extend(issues)

                if suggestions:
                    analysis_parts.append("\nSuggestions:")
                    analysis_parts.extend(suggestions)

            if not any(["Issues found:" in part for part in analysis_parts]):
                analysis_parts.append("\nNo major issues found!")

            return ToolResult(
                content=[
                    {
                        "type": "text",
                        "text": "\n".join(analysis_parts),
                    }
                ]
            )

        except Exception as e:
            self.logger.error(f"Code analysis error: {e}")
            return ToolResult(
                content=[{"type": "text", "text": f"Error analyzing code: {str(e)}"}],
                isError=True,
            )

    async def get_system_info(self) -> ToolResult:
        """Get system information."""
        try:
            import platform
            import psutil

            info_parts = [
                f"System: {platform.system()} {platform.release()}",
                f"Architecture: {platform.machine()}",
                f"Python: {platform.python_version()}",
                f"CPU cores: {psutil.cpu_count()}",
                f"Memory: {psutil.virtual_memory().total // (1024**3)} GB",
                f"Disk: {psutil.disk_usage('/').total // (1024**3)} GB",
                f"Current directory: {Path.cwd()}",
                f"Home directory: {Path.home()}",
            ]

            return ToolResult(
                content=[
                    {
                        "type": "text",
                        "text": "System Information:\n\n" + "\n".join(info_parts),
                    }
                ]
            )

        except Exception as e:
            self.logger.error(f"System info error: {e}")
            return ToolResult(
                content=[{"type": "text", "text": f"Error getting system info: {str(e)}"}],
                isError=True,
            )

    async def close(self) -> None:
        """Close the tool registry and clean up resources."""
        await self._http_client.aclose()

    def get_available_tools(self) -> List[Tool]:
        """Get list of available tools."""
        return [
            Tool(
                name="web_search",
                description="Search the web for information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="read_file",
                description="Read a file or list directory contents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file or directory"},
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="write_file",
                description="Write content to a file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["file_path", "content"],
                },
            ),
            Tool(
                name="execute_command",
                description="Execute a shell command",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 30,
                        },
                        "working_dir": {
                            "type": "string",
                            "description": "Working directory",
                        },
                    },
                    "required": ["command"],
                },
            ),
            Tool(
                name="analyze_code",
                description="Analyze code for issues and suggestions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to analyze"},
                        "language": {
                            "type": "string",
                            "description": "Programming language",
                            "default": "python",
                        },
                    },
                    "required": ["code"],
                },
            ),
            Tool(
                name="get_system_info",
                description="Get system information",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]