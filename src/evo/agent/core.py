"""
Core EvoAgent implementation with learning and memory capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from uuid import uuid4

import ollama

from ..memory import MemoryManager
from ..mcp import MCPClient
from .adapters import LMStudioAdapter
from .context import ContextManager
from .models import (
    AgentConfig,
    AgentState,
    ChatMessage,
    ChatRole,
    ConversationContext,
    ToolCall,
    ToolResult,
)
from .session import SessionManager


class EvoAgent:
    """
    Core Evo AI agent with persistent learning and memory capabilities.
    """

    def __init__(
        self,
        config: AgentConfig,
        memory_manager: MemoryManager,
        mcp_client: Optional[MCPClient] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        self.config = config
        self.memory_manager = memory_manager
        self.mcp_client = mcp_client
        self.session_manager = session_manager or SessionManager()
        self.context_manager = ContextManager(memory_manager, config)

        self.state = AgentState()
        self.logger = logging.getLogger(__name__)

        # Model client (Ollama or LM Studio)
        self.model_client = None
        self._initialize_model_client()

        # Tool registry
        self._tools: Dict[str, Any] = {}
        self._setup_default_tools()

    def _initialize_model_client(self) -> None:
        """Initialize the appropriate model client based on configuration."""
        model_url = getattr(self.config, 'model_url', None)

        if model_url and ('localhost' in model_url or '127.0.0.1' in model_url):
            # LM Studio or similar OpenAI-compatible API
            self.model_client = LMStudioAdapter(
                base_url=model_url,
                model_name=self.config.model_name,
                api_key=getattr(self.config, 'api_key', 'lm-studio'),
            )
            self.logger.info(f"Using LM Studio adapter: {model_url}")
        else:
            # Default to Ollama
            self.model_client = ollama.AsyncClient()
            self.logger.info("Using Ollama client")

    async def initialize(self) -> None:
        """Initialize the agent and all its components."""
        await self.memory_manager.initialize()

        if self.mcp_client:
            await self.mcp_client.initialize()

        # Initialize model client if it's LM Studio
        if isinstance(self.model_client, LMStudioAdapter):
            await self.model_client.initialize()

        self.logger.info("EvoAgent initialized successfully")

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        stream: bool = False,
        include_memory: bool = True,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Chat with the agent.

        Args:
            message: User message
            session_id: Session ID (creates new if None)
            stream: Whether to stream the response
            include_memory: Whether to include memory in context

        Returns:
            Response string or async generator for streaming
        """
        # Create or get session
        if session_id is None:
            session_id = await self.session_manager.create_session()

        # Add user message to context
        user_message = ChatMessage(
            role=ChatRole.USER,
            content=message,
            metadata={"session_id": session_id},
        )

        context = await self.context_manager.add_message(session_id, user_message)

        # Process the message and generate response
        if stream:
            return self._chat_stream(context, include_memory)
        else:
            return await self._chat_single(context, include_memory)

    async def _chat_single(
        self,
        context: ConversationContext,
        include_memory: bool = True,
    ) -> str:
        """Generate a single response."""
        try:
            # Prepare context for LLM
            messages = await self.context_manager.prepare_context_for_llm(
                context, include_memory
            )

            # Check if we need to use tools first
            tool_results = await self._check_and_use_tools(context, messages)

            if tool_results:
                # Add tool results to messages
                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "content": str(result.result),
                        "tool_call_id": result.tool_call_id,
                    })

            # Generate response using the model client
            if isinstance(self.model_client, LMStudioAdapter):
                # Use the chat method directly with the prepared messages
                response = await self.model_client.chat(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                assistant_response = response["choices"][0]["message"]["content"]
            else:
                # Ollama client
                response = await self.model_client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    options={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "num_predict": self.config.max_tokens,
                    },
                )
                assistant_response = response["message"]["content"]

            # Add assistant message to context
            assistant_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content=assistant_response,
                metadata={"session_id": context.session_id},
            )

            await self.context_manager.add_message(context.session_id, assistant_message)

            # Store conversation as memory
            user_msg_content = context.messages[-2].content  # Get the user message
            await self.context_manager.store_conversation_memory(
                context, user_msg_content, assistant_response
            )

            # Update state
            self.state.total_conversations += 1

            return assistant_response

        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}. Let me try to help you in a different way."

    async def _chat_stream(
        self,
        context: ConversationContext,
        include_memory: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        try:
            # Prepare context for LLM
            messages = await self.context_manager.prepare_context_for_llm(
                context, include_memory
            )

            # Check if we need to use tools first
            tool_results = await self._check_and_use_tools(context, messages)

            if tool_results:
                # Add tool results to messages
                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "content": str(result.result),
                        "tool_call_id": result.tool_call_id,
                    })

            # Stream response using Ollama
            full_response = ""
            async for chunk in await self.ollama_client.chat(
                model=self.config.model_name,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": self.config.max_tokens,
                },
                stream=True,
            ):
                if chunk.get("message", {}).get("content"):
                    content = chunk["message"]["content"]
                    full_response += content
                    yield content

            # Add assistant message to context after streaming is complete
            assistant_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content=full_response,
                metadata={"session_id": context.session_id},
            )

            await self.context_manager.add_message(context.session_id, assistant_message)

            # Store conversation as memory
            user_msg_content = context.messages[-2].content  # Get the user message
            await self.context_manager.store_conversation_memory(
                context, user_msg_content, full_response
            )

            # Update state
            self.state.total_conversations += 1

        except Exception as e:
            self.logger.error(f"Error in streaming chat: {e}")
            yield f"I encountered an error: {str(e)}. Let me try to help you in a different way."

    async def _check_and_use_tools(
        self,
        context: ConversationContext,
        messages: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        """Check if we need to use tools and execute them."""
        tool_results = []

        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        if not last_user_message:
            return tool_results

        # Check for tool usage patterns
        needs_search = any(
            keyword in last_user_message.lower()
            for keyword in [
                "search", "look up", "find information", "what is", "who is",
                "current", "latest", "recent", "news", "weather"
            ]
        )

        needs_code_help = any(
            keyword in last_user_message.lower()
            for keyword in [
                "code", "function", "class", "debug", "error", "programming",
                "python", "javascript", "java", "c++", "rust", "go"
            ]
        )

        # Use MCP tools if available
        if self.mcp_client and (needs_search or needs_code_help):
            try:
                # This would call MCP tools - simplified implementation
                if needs_search:
                    search_result = await self._use_search_tool(last_user_message)
                    if search_result:
                        tool_results.append(search_result)

                if needs_code_help:
                    code_result = await self._use_code_tool(last_user_message)
                    if code_result:
                        tool_results.append(code_result)

            except Exception as e:
                self.logger.error(f"Error using tools: {e}")

        return tool_results

    async def _use_search_tool(self, query: str) -> Optional[ToolResult]:
        """Use search tool via MCP."""
        if not self.mcp_client:
            return None

        try:
            # This is a placeholder - actual implementation would use MCP
            result = await self.mcp_client.call_tool("web_search", {"query": query})

            return ToolResult(
                tool_call_id=str(uuid4()),
                result=result,
                execution_time=0.5,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=str(uuid4()),
                result="",
                error=str(e),
                execution_time=0.0,
            )

    async def _use_code_tool(self, query: str) -> Optional[ToolResult]:
        """Use code analysis tool via MCP."""
        if not self.mcp_client:
            return None

        try:
            # This is a placeholder - actual implementation would use MCP
            result = await self.mcp_client.call_tool("code_analysis", {"query": query})

            return ToolResult(
                tool_call_id=str(uuid4()),
                result=result,
                execution_time=0.3,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=str(uuid4()),
                result="",
                error=str(e),
                execution_time=0.0,
            )

    async def teach(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Teach the agent new information.

        Args:
            content: Information to teach
            memory_type: Type of memory (semantic, procedural, episodic)
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata

        Returns:
            Confirmation message
        """
        try:
            if memory_type == "semantic":
                # Extract subject, predicate, object from content
                # This is simplified - a real implementation would use NLP
                parts = content.split(" is ")
                if len(parts) == 2:
                    subject = parts[0].strip()
                    obj = parts[1].strip()
                    predicate = "is"
                else:
                    subject = "knowledge"
                    predicate = "contains"
                    obj = content

                await self.memory_manager.store_semantic_memory(
                    content=content,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    importance=importance,
                    source="user_teaching",
                    metadata=metadata or {},
                )

            elif memory_type == "procedural":
                # Extract skill name from content
                skill_name = content.split()[0:3]  # First 3 words as skill name
                skill_name = "_".join(skill_name).lower()

                await self.memory_manager.store_procedural_memory(
                    content=content,
                    skill_name=skill_name,
                    importance=importance,
                    metadata=metadata or {},
                )

            else:  # episodic
                await self.memory_manager.store_episodic_memory(
                    content=content,
                    importance=importance,
                    metadata=metadata or {},
                )

            self.state.total_memories_stored += 1
            return f"I've learned and stored this {memory_type} knowledge: {content[:100]}..."

        except Exception as e:
            self.logger.error(f"Error in teach: {e}")
            return f"I had trouble learning that information: {str(e)}"

    async def recall(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5,
    ) -> str:
        """
        Recall information from memory.

        Args:
            query: Search query
            memory_types: Types of memory to search
            limit: Maximum number of results

        Returns:
            Formatted recall results
        """
        try:
            from ..memory.models import MemoryQuery, MemoryType

            # Convert string memory types to enum
            enum_types = None
            if memory_types:
                enum_types = [MemoryType(mt) for mt in memory_types]

            memory_query = MemoryQuery(
                text=query,
                memory_types=enum_types,
                limit=limit,
            )

            results = await self.memory_manager.search(memory_query)

            if not results:
                return "I don't have any memories related to that query."

            # Format results
            formatted_results = ["Here's what I remember:"]
            for i, result in enumerate(results, 1):
                memory_type = result.memory.type.value
                similarity = result.similarity_score
                content = result.memory.content[:200]
                if len(result.memory.content) > 200:
                    content += "..."

                formatted_results.append(
                    f"{i}. [{memory_type.title()}, {similarity:.2f}] {content}"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            self.logger.error(f"Error in recall: {e}")
            return f"I had trouble recalling that information: {str(e)}"

    async def get_status(self) -> Dict[str, Any]:
        """Get agent status and statistics."""
        memory_stats = await self.memory_manager.get_stats()

        return {
            "agent_state": {
                "is_learning": self.state.is_learning,
                "total_conversations": self.state.total_conversations,
                "total_memories_stored": self.state.total_memories_stored,
                "model_updates": self.state.model_updates,
                "active_sessions": len(self.state.active_sessions),
            },
            "memory_stats": memory_stats,
            "config": {
                "model_name": self.config.model_name,
                "enable_learning": self.config.enable_learning,
                "enable_mcp": self.config.enable_mcp,
                "use_memory": self.config.use_memory,
            },
            "active_sessions": self.session_manager.get_active_session_count(),
        }

    def _setup_default_tools(self) -> None:
        """Setup default tools for the agent."""
        self._tools = {
            "memory_search": self._tool_memory_search,
            "store_memory": self._tool_store_memory,
            "get_status": self._tool_get_status,
        }

    async def _tool_memory_search(self, query: str, limit: int = 5) -> str:
        """Tool for searching memory."""
        return await self.recall(query, limit=limit)

    async def _tool_store_memory(
        self, content: str, memory_type: str = "semantic", importance: float = 0.7
    ) -> str:
        """Tool for storing memory."""
        return await self.teach(content, memory_type, importance)

    async def _tool_get_status(self) -> Dict[str, Any]:
        """Tool for getting agent status."""
        return await self.get_status()

    async def consolidate_session_memories(self, session_id: str) -> None:
        """Consolidate memories from a session."""
        if not self.config.enable_learning:
            return

        try:
            await self.context_manager.memory_manager.consolidate_memories(session_id)
            self.logger.info(f"Consolidated memories for session {session_id}")
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")

    async def start_learning_process(self) -> None:
        """Start the background learning process."""
        if not self.config.enable_learning:
            self.logger.info("Learning is disabled in config")
            return

        self.state.is_learning = True
        self.logger.info("Started learning process")

        # This would start the QLoRA fine-tuning process
        # For now, it's a placeholder

    async def stop_learning_process(self) -> None:
        """Stop the background learning process."""
        self.state.is_learning = False
        self.logger.info("Stopped learning process")

    async def close(self) -> None:
        """Close the agent and clean up resources."""
        await self.memory_manager.close()

        if self.mcp_client:
            await self.mcp_client.close()

        self.logger.info("EvoAgent closed successfully")