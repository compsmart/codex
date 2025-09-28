"""
MCP server implementation for Evo AI tools.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from .models import (
    MCPRequest,
    MCPResponse,
    MCPNotification,
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
    ServerCapabilities,
)


class MCPServer:
    """MCP server for exposing Evo AI tools and capabilities."""

    def __init__(
        self,
        name: str = "evo-ai-server",
        version: str = "0.1.0",
    ):
        self.name = name
        self.version = version
        self.logger = logging.getLogger(__name__)

        # Tool registry
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, Tool] = {}

        # Resource registry
        self._resources: Dict[str, Callable] = {}
        self._resource_schemas: Dict[str, Resource] = {}

        # Prompt registry
        self._prompts: Dict[str, Callable] = {}
        self._prompt_schemas: Dict[str, Prompt] = {}

        # Method handlers
        self._handlers: Dict[str, Callable] = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "prompts/list": self._handle_list_prompts,
            "prompts/get": self._handle_get_prompt,
            "ping": self._handle_ping,
        }

        # Server state
        self._initialized = False
        self._client_capabilities = None

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """Register a tool with the server."""
        tool = Tool(
            name=name,
            description=description,
            inputSchema=input_schema,
        )

        self._tools[name] = handler
        self._tool_schemas[name] = tool

        self.logger.debug(f"Registered tool: {name}")

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str,
        handler: Callable,
    ) -> None:
        """Register a resource with the server."""
        resource = Resource(
            uri=uri,
            name=name,
            description=description,
            mimeType=mime_type,
        )

        self._resources[uri] = handler
        self._resource_schemas[uri] = resource

        self.logger.debug(f"Registered resource: {uri}")

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: Optional[List[Dict[str, Any]]],
        handler: Callable,
    ) -> None:
        """Register a prompt with the server."""
        prompt = Prompt(
            name=name,
            description=description,
            arguments=arguments,
        )

        self._prompts[name] = handler
        self._prompt_schemas[name] = prompt

        self.logger.debug(f"Registered prompt: {name}")

    async def run_stdio(self) -> None:
        """Run the server using stdio transport."""
        self.logger.info("Starting MCP server with stdio transport")

        try:
            # Read from stdin and write to stdout
            while True:
                line = await self._read_line()
                if not line:
                    break

                try:
                    request_data = json.loads(line)
                    request = MCPRequest(**request_data)

                    response = await self._handle_request(request)

                    # Write response to stdout
                    response_json = json.dumps(response.dict(exclude_none=True))
                    print(response_json, flush=True)

                except json.JSONDecodeError as e:
                    error_response = MCPResponse(
                        id=None,
                        error=MCPError(
                            code=-32700,
                            message="Parse error",
                            data=str(e),
                        ),
                    )
                    response_json = json.dumps(error_response.dict(exclude_none=True))
                    print(response_json, flush=True)

                except Exception as e:
                    self.logger.error(f"Error handling request: {e}")
                    error_response = MCPResponse(
                        id=getattr(request, "id", None) if "request" in locals() else None,
                        error=MCPError(
                            code=-32603,
                            message="Internal error",
                            data=str(e),
                        ),
                    )
                    response_json = json.dumps(error_response.dict(exclude_none=True))
                    print(response_json, flush=True)

        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            self.logger.info("MCP server stopped")

    async def _read_line(self) -> Optional[str]:
        """Read a line from stdin."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sys.stdin.readline)

    async def _handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request."""
        method = request.method
        params = request.params or {}

        if method in self._handlers:
            try:
                result = await self._handlers[method](params)
                return MCPResponse(id=request.id, result=result)

            except Exception as e:
                self.logger.error(f"Error in handler {method}: {e}")
                return MCPResponse(
                    id=request.id,
                    error=MCPError(
                        code=-32603,
                        message="Internal error",
                        data=str(e),
                    ),
                )
        else:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=-32601,
                    message="Method not found",
                    data=f"Unknown method: {method}",
                ),
            )

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        try:
            init_request = InitializeRequest(**params)
            self._client_capabilities = init_request.capabilities

            capabilities = ServerCapabilities(
                tools={"listChanged": True},
                resources={"subscribe": True, "listChanged": True},
                prompts={"listChanged": True},
                logging={},
            )

            result = InitializeResult(
                protocolVersion="2024-11-05",
                capabilities=capabilities,
                serverInfo={
                    "name": self.name,
                    "version": self.version,
                },
            )

            self._initialized = True
            self.logger.info("Server initialized")

            return result.dict()

        except Exception as e:
            raise Exception(f"Initialization failed: {e}")

    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = [tool.dict() for tool in self._tool_schemas.values()]
        return {"tools": tools}

    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self._tools:
            raise Exception(f"Unknown tool: {tool_name}")

        try:
            handler = self._tools[tool_name]
            result = await handler(**arguments)

            # Ensure result is in the correct format
            if isinstance(result, ToolResult):
                return result.dict()
            elif isinstance(result, dict):
                return result
            else:
                # Wrap simple results
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result),
                        }
                    ]
                }

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}",
                    }
                ],
                "isError": True,
            }

    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        resources = [resource.dict() for resource in self._resource_schemas.values()]
        return {"resources": resources}

    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")

        if uri not in self._resources:
            raise Exception(f"Unknown resource: {uri}")

        try:
            handler = self._resources[uri]
            content = await handler()

            if isinstance(content, ResourceContent):
                return {"contents": [content.dict()]}
            elif isinstance(content, dict):
                return {"contents": [content]}
            else:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": str(content),
                        }
                    ]
                }

        except Exception as e:
            self.logger.error(f"Error reading resource {uri}: {e}")
            raise Exception(f"Failed to read resource: {e}")

    async def _handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list request."""
        prompts = [prompt.dict() for prompt in self._prompt_schemas.values()]
        return {"prompts": prompts}

    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        if prompt_name not in self._prompts:
            raise Exception(f"Unknown prompt: {prompt_name}")

        try:
            handler = self._prompts[prompt_name]
            result = await handler(**arguments)

            if isinstance(result, PromptResult):
                return result.dict()
            elif isinstance(result, dict):
                return result
            else:
                return {
                    "description": f"Generated prompt: {prompt_name}",
                    "messages": [
                        {
                            "role": "user",
                            "content": str(result),
                        }
                    ],
                }

        except Exception as e:
            self.logger.error(f"Error getting prompt {prompt_name}: {e}")
            raise Exception(f"Failed to get prompt: {e}")

    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request."""
        return {"pong": True}

    def setup_default_tools(self, evo_agent) -> None:
        """Setup default tools for Evo AI."""
        # Memory search tool
        self.register_tool(
            name="memory_search",
            description="Search the agent's memory for relevant information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            handler=lambda query, limit=5: evo_agent.recall(query, limit=limit),
        )

        # Store memory tool
        self.register_tool(
            name="store_memory",
            description="Store new information in the agent's memory",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Information to store",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["semantic", "procedural", "episodic"],
                        "description": "Type of memory",
                        "default": "semantic",
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Importance score",
                        "default": 0.7,
                    },
                },
                "required": ["content"],
            },
            handler=lambda content, memory_type="semantic", importance=0.7: evo_agent.teach(
                content, memory_type, importance
            ),
        )

        # Agent status tool
        self.register_tool(
            name="get_status",
            description="Get the agent's current status and statistics",
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=lambda: evo_agent.get_status(),
        )

        self.logger.info("Default tools registered")