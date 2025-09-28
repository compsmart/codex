"""
MCP models and data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MCPMessageType(str, Enum):
    """MCP message types following JSON-RPC 2.0."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPMethod(str, Enum):
    """Standard MCP methods."""

    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    PING = "ping"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"
    COMPLETE = "completion/complete"
    SET_LEVEL = "logging/setLevel"


class MCPError(BaseModel):
    """MCP error object."""

    code: int
    message: str
    data: Optional[Any] = None


class MCPMessage(BaseModel):
    """Base MCP message following JSON-RPC 2.0."""

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[MCPError] = None


class MCPRequest(MCPMessage):
    """MCP request message."""

    method: str
    params: Optional[Dict[str, Any]] = None

    def __init__(self, **data):
        if "id" not in data:
            data["id"] = str(uuid4())
        super().__init__(**data)


class MCPResponse(MCPMessage):
    """MCP response message."""

    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[MCPError] = None


class MCPNotification(MCPMessage):
    """MCP notification message."""

    method: str
    params: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """MCP tool definition."""

    name: str
    description: str
    inputSchema: Dict[str, Any]  # JSON Schema for tool input


class ToolCall(BaseModel):
    """Tool call request."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Tool call result."""

    content: List[Dict[str, Any]] = Field(default_factory=list)
    isError: bool = False


class Resource(BaseModel):
    """MCP resource definition."""

    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None


class ResourceContent(BaseModel):
    """Resource content."""

    uri: str
    mimeType: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[bytes] = None


class Prompt(BaseModel):
    """MCP prompt definition."""

    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


class PromptMessage(BaseModel):
    """Prompt message."""

    role: str
    content: Union[str, Dict[str, Any]]


class PromptResult(BaseModel):
    """Prompt result."""

    description: Optional[str] = None
    messages: List[PromptMessage] = Field(default_factory=list)


class CompletionRequest(BaseModel):
    """Completion request."""

    ref: Dict[str, Any]
    argument: Dict[str, Any]


class CompletionResult(BaseModel):
    """Completion result."""

    completion: Dict[str, Any]


class ClientCapabilities(BaseModel):
    """Client capabilities."""

    experimental: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None


class ServerCapabilities(BaseModel):
    """Server capabilities."""

    experimental: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    tools: Optional[Dict[str, Any]] = None


class InitializeRequest(BaseModel):
    """Initialize request."""

    protocolVersion: str = "2024-11-05"
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)
    clientInfo: Dict[str, Any] = Field(default_factory=dict)


class InitializeResult(BaseModel):
    """Initialize result."""

    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: Dict[str, Any] = Field(default_factory=dict)


class LoggingLevel(str, Enum):
    """Logging levels."""

    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"


class MCPConnection(BaseModel):
    """MCP connection information."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    transport_type: str  # "stdio", "sse", "websocket"
    endpoint: Optional[str] = None
    command: Optional[List[str]] = None
    args: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, str]] = None
    capabilities: Optional[ServerCapabilities] = None
    connected_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    is_connected: bool = False