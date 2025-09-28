"""
Integration tests for Evo AI system.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from src.evo.agent import EvoAgent, AgentConfig
from src.evo.memory import MemoryManager


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    async def full_system(self):
        """Create a complete test system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_memory.db"

            # Create memory manager
            memory_manager = MemoryManager(str(db_path))
            await memory_manager.initialize()

            # Create agent config
            config = AgentConfig(
                model_name="gpt2",
                use_memory=True,
                enable_learning=False,
                enable_mcp=False,
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
    async def test_learning_and_recall_cycle(self, full_system):
        """Test the complete learning and recall cycle."""
        agent = full_system

        # Step 1: Teach the agent some facts
        await agent.teach(
            "The user's name is Alice",
            memory_type="semantic",
            importance=0.9,
        )

        await agent.teach(
            "Alice works as a software engineer",
            memory_type="semantic",
            importance=0.8,
        )

        await agent.teach(
            "Alice's favorite programming language is Python",
            memory_type="semantic",
            importance=0.8,
        )

        # Step 2: Test recall of individual facts
        name_result = await agent.recall("user name")
        assert "Alice" in name_result

        job_result = await agent.recall("Alice job work")
        assert "software engineer" in job_result.lower()

        language_result = await agent.recall("Alice favorite programming language")
        assert "Python" in language_result

        # Step 3: Test cross-referencing facts
        alice_info = await agent.recall("Alice")
        assert "Alice" in alice_info
        # Should contain multiple facts about Alice

    @pytest.mark.asyncio
    async def test_conversation_memory_persistence(self, full_system):
        """Test that conversation memories persist and can be recalled."""
        agent = full_system

        # Create a conversation session
        session_id = await agent.session_manager.create_session()

        # Store conversation memories
        await agent.memory_manager.store_episodic_memory(
            content="Discussion about machine learning",
            session_id=session_id,
            user_message="Tell me about machine learning",
            assistant_response="Machine learning is a subset of AI that enables computers to learn without being explicitly programmed",
            importance=0.7,
        )

        await agent.memory_manager.store_episodic_memory(
            content="User wants to learn Python for ML",
            session_id=session_id,
            user_message="I want to learn Python for machine learning",
            assistant_response="Python is excellent for ML with libraries like scikit-learn, pandas, and numpy",
            importance=0.8,
        )

        # Test recall of conversation
        ml_result = await agent.recall("machine learning discussion")
        assert "machine learning" in ml_result.lower()

        python_ml_result = await agent.recall("Python machine learning")
        assert "Python" in python_ml_result
        assert "machine learning" in python_ml_result.lower()

    @pytest.mark.asyncio
    async def test_memory_importance_and_access_patterns(self, full_system):
        """Test that memory importance affects recall."""
        agent = full_system

        # Store memories with different importance levels
        await agent.teach(
            "Very important fact: The user is allergic to peanuts",
            memory_type="semantic",
            importance=1.0,
        )

        await agent.teach(
            "Less important fact: The user likes the color green",
            memory_type="semantic",
            importance=0.3,
        )

        await agent.teach(
            "Moderately important: The user prefers tea over coffee",
            memory_type="semantic",
            importance=0.6,
        )

        # Test recall - high importance items should be more likely to appear
        health_result = await agent.recall("user allergic peanuts")
        assert "peanuts" in health_result.lower() or "allergic" in health_result.lower()

        preference_result = await agent.recall("user preferences")
        # Should contain some preference information

    @pytest.mark.asyncio
    async def test_knowledge_building_over_time(self, full_system):
        """Test that knowledge builds up over time."""
        agent = full_system

        # Day 1: Learn about user's work
        await agent.teach(
            "User works at TechCorp as a senior developer",
            memory_type="semantic",
            importance=0.8,
        )

        # Day 2: Learn about user's skills
        await agent.teach(
            "User is expert in Python, JavaScript, and Go",
            memory_type="semantic",
            importance=0.7,
        )

        # Day 3: Learn about user's projects
        await agent.teach(
            "User is currently working on a web application using React and Node.js",
            memory_type="semantic",
            importance=0.7,
        )

        # Test comprehensive recall
        work_info = await agent.recall("user work job developer")
        # Should contain information about work, skills, and projects

        tech_info = await agent.recall("user technology skills programming")
        # Should contain information about programming languages and current project

    @pytest.mark.asyncio
    async def test_system_resilience(self, full_system):
        """Test system resilience and error handling."""
        agent = full_system

        # Test with empty/invalid queries
        empty_result = await agent.recall("")
        assert "don't have any memories" in empty_result.lower() or "no" in empty_result.lower()

        # Test with very long content
        long_content = "A" * 10000
        result = await agent.teach(long_content, importance=0.5)
        # Should handle gracefully without crashing

        # Test recall still works after stress
        await agent.teach("Simple test after stress", importance=0.7)
        stress_result = await agent.recall("simple test")
        assert "simple" in stress_result.lower() or "test" in stress_result.lower()

    @pytest.mark.asyncio
    async def test_memory_stats_and_monitoring(self, full_system):
        """Test memory statistics and monitoring."""
        agent = full_system

        # Store various types of memories
        await agent.teach("Semantic fact", memory_type="semantic", importance=0.7)
        await agent.teach("Procedural knowledge", memory_type="procedural", importance=0.6)

        await agent.memory_manager.store_episodic_memory(
            content="Episodic memory",
            user_message="Test",
            assistant_response="Response",
            importance=0.5,
        )

        # Get system status
        status = await agent.get_status()

        # Verify status contains expected information
        assert "agent_state" in status
        assert "memory_stats" in status
        assert status["agent_state"]["total_memories_stored"] >= 3

        # Verify memory stats
        memory_stats = status["memory_stats"]["storage"]
        assert memory_stats["total_memories"] >= 3
        assert "by_type" in memory_stats