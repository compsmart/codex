"""
Tests for memory management system.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from src.evo.memory import MemoryManager, MemoryType
from src.evo.memory.models import EpisodicMemory, SemanticMemory, ProceduralMemory


class TestMemoryManager:
    """Test the memory management system."""

    @pytest.fixture
    async def memory_manager(self):
        """Create a test memory manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"
            manager = MemoryManager(str(db_path))
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_store_episodic_memory(self, memory_manager):
        """Test storing episodic memory."""
        memory = await memory_manager.store_episodic_memory(
            content="Test conversation",
            session_id="test_session",
            user_message="Hello",
            assistant_response="Hi there!",
            importance=0.8,
        )

        assert isinstance(memory, EpisodicMemory)
        assert memory.content == "Test conversation"
        assert memory.session_id == "test_session"
        assert memory.importance == 0.8

    @pytest.mark.asyncio
    async def test_store_semantic_memory(self, memory_manager):
        """Test storing semantic memory."""
        memory = await memory_manager.store_semantic_memory(
            content="Python is a programming language",
            subject="Python",
            predicate="is",
            object="programming language",
            confidence=0.9,
        )

        assert isinstance(memory, SemanticMemory)
        assert memory.subject == "Python"
        assert memory.predicate == "is"
        assert memory.object == "programming language"

    @pytest.mark.asyncio
    async def test_store_procedural_memory(self, memory_manager):
        """Test storing procedural memory."""
        memory = await memory_manager.store_procedural_memory(
            content="How to debug Python code",
            skill_name="python_debugging",
            steps=["Read error message", "Check stack trace", "Use debugger"],
            importance=0.7,
        )

        assert isinstance(memory, ProceduralMemory)
        assert memory.skill_name == "python_debugging"
        assert len(memory.steps) == 3

    @pytest.mark.asyncio
    async def test_memory_search(self, memory_manager):
        """Test memory search functionality."""
        # Store some test memories
        await memory_manager.store_semantic_memory(
            content="Python is a high-level programming language",
            subject="Python",
            predicate="is",
            object="high-level programming language",
        )

        await memory_manager.store_semantic_memory(
            content="JavaScript is used for web development",
            subject="JavaScript",
            predicate="is used for",
            object="web development",
        )

        # Search for Python-related memories
        from src.evo.memory.models import MemoryQuery
        query = MemoryQuery(text="Python programming", limit=5)
        results = await memory_manager.search(query)

        assert len(results) > 0
        assert any("Python" in result.memory.content for result in results)

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, memory_manager):
        """Test retrieving specific memory by ID."""
        # Store a memory
        stored_memory = await memory_manager.store_semantic_memory(
            content="Test retrieval",
            subject="test",
            predicate="is",
            object="retrieval",
        )

        # Retrieve it
        retrieved_memory = await memory_manager.retrieve(stored_memory.id)

        assert retrieved_memory is not None
        assert retrieved_memory.id == stored_memory.id
        assert retrieved_memory.content == "Test retrieval"

    @pytest.mark.asyncio
    async def test_memory_stats(self, memory_manager):
        """Test getting memory statistics."""
        # Store some memories
        await memory_manager.store_semantic_memory(
            content="Test memory 1",
            subject="test1",
            predicate="is",
            object="memory",
        )

        await memory_manager.store_episodic_memory(
            content="Test conversation",
            user_message="Hello",
            assistant_response="Hi",
        )

        # Get stats
        stats = await memory_manager.get_stats()

        assert "storage" in stats
        assert "embeddings" in stats
        assert stats["storage"]["total_memories"] >= 2