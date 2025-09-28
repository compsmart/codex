"""
Tests for the core agent functionality.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from src.evo.agent import EvoAgent, AgentConfig
from src.evo.memory import MemoryManager


class TestEvoAgent:
    """Test the core EvoAgent functionality."""

    @pytest.fixture
    async def agent(self):
        """Create a test agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"

            # Create memory manager
            memory_manager = MemoryManager(str(db_path))
            await memory_manager.initialize()

            # Create agent config
            config = AgentConfig(
                model_name="gpt2",  # Use a small model for testing
                use_memory=True,
                enable_learning=False,  # Disable learning for tests
                enable_mcp=False,  # Disable MCP for tests
            )

            # Create agent
            agent = EvoAgent(
                config=config,
                memory_manager=memory_manager,
            )

            await agent.initialize()
            yield agent
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.config.model_name == "gpt2"
        assert agent.memory_manager is not None
        assert agent.context_manager is not None

    @pytest.mark.asyncio
    async def test_teach_functionality(self, agent):
        """Test teaching the agent new information."""
        result = await agent.teach(
            "Python is a programming language",
            memory_type="semantic",
            importance=0.8,
        )

        assert "learned" in result.lower() or "stored" in result.lower()

        # Verify it was stored in memory
        recall_result = await agent.recall("Python programming")
        assert "Python" in recall_result

    @pytest.mark.asyncio
    async def test_recall_functionality(self, agent):
        """Test recalling information."""
        # First teach something
        await agent.teach(
            "The user's favorite color is blue",
            memory_type="semantic",
            importance=0.7,
        )

        # Then recall it
        result = await agent.recall("favorite color")
        assert "blue" in result.lower()

    @pytest.mark.asyncio
    async def test_agent_status(self, agent):
        """Test getting agent status."""
        status = await agent.get_status()

        assert "agent_state" in status
        assert "memory_stats" in status
        assert "config" in status

        agent_state = status["agent_state"]
        assert "total_conversations" in agent_state
        assert "total_memories_stored" in agent_state

    @pytest.mark.asyncio
    async def test_session_management(self, agent):
        """Test session management."""
        # Create a session
        session_id = await agent.session_manager.create_session()
        assert session_id is not None

        # Check that session exists
        session = await agent.session_manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id

    @pytest.mark.asyncio
    async def test_memory_consolidation(self, agent):
        """Test memory consolidation."""
        # Create a test session with some conversation
        session_id = await agent.session_manager.create_session()

        # Store some episodic memories
        await agent.memory_manager.store_episodic_memory(
            content="User asked about Python",
            session_id=session_id,
            user_message="What is Python?",
            assistant_response="Python is a programming language",
            importance=0.8,
        )

        await agent.memory_manager.store_episodic_memory(
            content="User wants to learn Python",
            session_id=session_id,
            user_message="I want to learn Python",
            assistant_response="Great! Python is beginner-friendly",
            importance=0.7,
        )

        # Consolidate memories
        await agent.consolidate_session_memories(session_id)

        # Check that consolidation ran without error
        # (Actual verification would depend on implementation details)
        assert True  # Placeholder assertion