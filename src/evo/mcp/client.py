"""
MCP client implementation for Evo AI.
"""

import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import httpx
from pydantic import ValidationError

from .models import (
    MCPConnection,
    MCPRequest,
    MCPResponse,
    MCPError,
    InitializeRequest,
    InitializeResult,
    Tool,
    ToolCall,
    ToolResult,
    Resource,
    ResourceContent,
    Prompt,
    PromptResult,
    ClientCapabilities,
)


class MCPClient:
    """MCP client for connecting to external services and tools."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.connections: Dict[str, MCPConnection] = {}
        self.logger = logging.getLogger(__name__)
        self._http_client = httpx.AsyncClient(timeout=timeout)

    async def initialize(self) -> None:
        """Initialize the MCP client."""
        self.logger.info("MCP client initialized")

    async def connect_stdio(
        self,
        name: str,
        command: List[str],
        description: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Connect to an MCP server via stdio."""
        connection_id = str(uuid4())

        try:
            # Start the subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            connection = MCPConnection(
                id=connection_id,
                name=name,
                description=description,
                transport_type="stdio",
                command=command,
                env=env,
                is_connected=True,
            )

            # Initialize the connection
            init_request = InitializeRequest(
                capabilities=ClientCapabilities(),
                clientInfo={
                    "name": "evo-ai",
                    "version": "0.1.0",
                },
            )

            response = await self._send_stdio_request(
                process, "initialize", init_request.dict()
            )

            if response.error:
                raise Exception(f"Initialization failed: {response.error.message}")

            # Store connection info
            connection.capabilities = response.result.get("capabilities")
            self.connections[connection_id] = connection

            self.logger.info(f"Connected to MCP server: {name}")
            return connection_id

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server {name}: {e}")
            raise

    async def connect_sse(
        self,
        name: str,
        endpoint: str,
        description: Optional[str] = None,
    ) -> str:
        """Connect to an MCP server via Server-Sent Events."""
        connection_id = str(uuid4())

        try:
            # Test connection
            async with self._http_client.stream("GET", endpoint) as response:
                if response.status_code != 200:
                    raise Exception(f"SSE connection failed: {response.status_code}")

            connection = MCPConnection(
                id=connection_id,
                name=name,
                description=description,
                transport_type="sse",
                endpoint=endpoint,
                is_connected=True,
            )

            # Initialize the connection
            init_request = InitializeRequest(
                capabilities=ClientCapabilities(),
                clientInfo={
                    "name": "evo-ai",
                    "version": "0.1.0",
                },
            )

            response = await self._send_sse_request(
                endpoint, "initialize", init_request.dict()
            )

            if response.error:
                raise Exception(f"Initialization failed: {response.error.message}")

            connection.capabilities = response.result.get("capabilities")
            self.connections[connection_id] = connection

            self.logger.info(f"Connected to MCP server via SSE: {name}")
            return connection_id

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server {name}: {e}")
            raise

    async def list_tools(self, connection_id: str) -> List[Tool]:
        """List available tools from a connection."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_connected:
            raise Exception(f"Connection {connection_id} not found or not connected")

        try:
            response = await self._send_request(connection, "tools/list", {})

            if response.error:
                raise Exception(f"Failed to list tools: {response.error.message}")

            tools = []
            for tool_data in response.result.get("tools", []):
                try:
                    tool = Tool(**tool_data)
                    tools.append(tool)
                except ValidationError as e:
                    self.logger.warning(f"Invalid tool data: {e}")

            return tools

        except Exception as e:
            self.logger.error(f"Error listing tools: {e}")
            raise

    async def call_tool(
        self,
        connection_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Call a tool on a connected server."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_connected:
            raise Exception(f"Connection {connection_id} not found or not connected")

        try:
            tool_call = ToolCall(name=tool_name, arguments=arguments)

            response = await self._send_request(
                connection, "tools/call", {"name": tool_name, "arguments": arguments}
            )

            if response.error:
                raise Exception(f"Tool call failed: {response.error.message}")

            return ToolResult(**response.result)

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            raise

    async def list_resources(self, connection_id: str) -> List[Resource]:
        """List available resources from a connection."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_connected:
            raise Exception(f"Connection {connection_id} not found or not connected")

        try:
            response = await self._send_request(connection, "resources/list", {})

            if response.error:
                raise Exception(f"Failed to list resources: {response.error.message}")

            resources = []
            for resource_data in response.result.get("resources", []):
                try:
                    resource = Resource(**resource_data)
                    resources.append(resource)
                except ValidationError as e:
                    self.logger.warning(f"Invalid resource data: {e}")

            return resources

        except Exception as e:
            self.logger.error(f"Error listing resources: {e}")
            raise

    async def read_resource(
        self, connection_id: str, uri: str
    ) -> ResourceContent:
        """Read a resource from a connected server."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_connected:
            raise Exception(f"Connection {connection_id} not found or not connected")

        try:
            response = await self._send_request(
                connection, "resources/read", {"uri": uri}
            )

            if response.error:
                raise Exception(f"Failed to read resource: {response.error.message}")

            return ResourceContent(**response.result.get("contents", [{}])[0])

        except Exception as e:
            self.logger.error(f"Error reading resource {uri}: {e}")
            raise

    async def list_prompts(self, connection_id: str) -> List[Prompt]:
        """List available prompts from a connection."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_connected:
            raise Exception(f"Connection {connection_id} not found or not connected")

        try:
            response = await self._send_request(connection, "prompts/list", {})

            if response.error:
                raise Exception(f"Failed to list prompts: {response.error.message}")

            prompts = []
            for prompt_data in response.result.get("prompts", []):
                try:
                    prompt = Prompt(**prompt_data)
                    prompts.append(prompt)
                except ValidationError as e:
                    self.logger.warning(f"Invalid prompt data: {e}")

            return prompts

        except Exception as e:
            self.logger.error(f"Error listing prompts: {e}")
            raise

    async def get_prompt(
        self,
        connection_id: str,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> PromptResult:
        """Get a prompt from a connected server."""
        connection = self.connections.get(connection_id)
        if not connection or not connection.is_connected:
            raise Exception(f"Connection {connection_id} not found or not connected")

        try:
            params = {"name": name}
            if arguments:
                params["arguments"] = arguments

            response = await self._send_request(connection, "prompts/get", params)

            if response.error:
                raise Exception(f"Failed to get prompt: {response.error.message}")

            return PromptResult(**response.result)

        except Exception as e:
            self.logger.error(f"Error getting prompt {name}: {e}")
            raise

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect from an MCP server."""
        connection = self.connections.get(connection_id)
        if connection:
            connection.is_connected = False
            del self.connections[connection_id]
            self.logger.info(f"Disconnected from MCP server: {connection.name}")

    async def _send_request(
        self,
        connection: MCPConnection,
        method: str,
        params: Dict[str, Any],
    ) -> MCPResponse:
        """Send a request to an MCP server."""
        if connection.transport_type == "stdio":
            # For stdio, we'd need to maintain the process reference
            # This is a simplified implementation
            raise NotImplementedError("Stdio transport not fully implemented")

        elif connection.transport_type == "sse":
            return await self._send_sse_request(connection.endpoint, method, params)

        else:
            raise Exception(f"Unsupported transport type: {connection.transport_type}")

    async def _send_stdio_request(
        self,
        process: asyncio.subprocess.Process,
        method: str,
        params: Dict[str, Any],
    ) -> MCPResponse:
        """Send request via stdio."""
        request = MCPRequest(method=method, params=params)

        # Send request
        request_json = json.dumps(request.dict()) + "\n"
        process.stdin.write(request_json.encode())
        await process.stdin.drain()

        # Read response
        response_line = await process.stdout.readline()
        response_data = json.loads(response_line.decode())

        return MCPResponse(**response_data)

    async def _send_sse_request(
        self,
        endpoint: str,
        method: str,
        params: Dict[str, Any],
    ) -> MCPResponse:
        """Send request via SSE."""
        request = MCPRequest(method=method, params=params)

        async with self._http_client.stream(
            "POST",
            endpoint,
            json=request.dict(),
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                raise Exception(f"HTTP error: {response.status_code}")

            # Read the response
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data.strip():
                        response_data = json.loads(data)
                        return MCPResponse(**response_data)

        raise Exception("No response received")

    async def close(self) -> None:
        """Close the MCP client and all connections."""
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)

        await self._http_client.aclose()
        self.logger.info("MCP client closed")

    def get_connections(self) -> List[MCPConnection]:
        """Get all connections."""
        return list(self.connections.values())

    def get_connection(self, connection_id: str) -> Optional[MCPConnection]:
        """Get a specific connection."""
        return self.connections.get(connection_id)