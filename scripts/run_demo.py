#!/usr/bin/env python3
"""
Demo script to test Evo AI learning capabilities.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evo.agent import EvoAgent, AgentConfig
from evo.memory import MemoryManager


async def demo_learning_and_recall():
    """Demonstrate learning and recall capabilities."""
    print("🤖 Evo AI Learning Demo")
    print("=" * 50)

    # Create temporary database for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "demo_memory.db"

        # Initialize memory manager
        print("📚 Initializing memory system...")
        memory_manager = MemoryManager(str(db_path))
        await memory_manager.initialize()

        # Create agent config
        config = AgentConfig(
            model_name="llama3.2:7b",
            use_memory=True,
            enable_learning=False,  # Disable for demo
            enable_mcp=False,
        )

        # Create agent
        print("🧠 Creating Evo AI agent...")
        agent = EvoAgent(
            config=config,
            memory_manager=memory_manager,
        )

        try:
            await agent.initialize()
            print("✅ Agent initialized successfully!")

            # Demo 1: Teaching personal information
            print("\n📖 Demo 1: Teaching Personal Information")
            print("-" * 40)

            await agent.teach("The user's name is Alex", importance=0.9)
            print("✓ Taught: User's name is Alex")

            await agent.teach("Alex works as a software engineer at TechCorp", importance=0.8)
            print("✓ Taught: Alex's job information")

            await agent.teach("Alex's favorite programming language is Python", importance=0.8)
            print("✓ Taught: Alex's programming preference")

            await agent.teach("Alex prefers tea over coffee", importance=0.6)
            print("✓ Taught: Alex's beverage preference")

            # Demo 2: Recall specific information
            print("\n🔍 Demo 2: Recalling Information")
            print("-" * 40)

            name_result = await agent.recall("user name")
            print(f"Q: What is the user's name?")
            print(f"A: {name_result}\n")

            job_result = await agent.recall("Alex job work")
            print(f"Q: Where does Alex work?")
            print(f"A: {job_result}\n")

            language_result = await agent.recall("favorite programming language")
            print(f"Q: What's Alex's favorite programming language?")
            print(f"A: {language_result}\n")

            # Demo 3: Cross-referencing information
            print("\n🔗 Demo 3: Cross-referencing Information")
            print("-" * 40)

            alex_info = await agent.recall("Alex")
            print(f"Q: Tell me about Alex")
            print(f"A: {alex_info}\n")

            # Demo 4: Learning technical information
            print("\n💻 Demo 4: Technical Knowledge")
            print("-" * 40)

            await agent.teach(
                "Python is a high-level, interpreted programming language known for its simplicity",
                memory_type="semantic",
                importance=0.7
            )
            print("✓ Taught: Python language information")

            await agent.teach(
                "To debug Python code: 1) Read error messages, 2) Use print statements, 3) Use debugger",
                memory_type="procedural",
                importance=0.8
            )
            print("✓ Taught: Python debugging procedure")

            python_info = await agent.recall("Python programming language")
            print(f"Q: What is Python?")
            print(f"A: {python_info}\n")

            debug_info = await agent.recall("debug Python code")
            print(f"Q: How to debug Python code?")
            print(f"A: {debug_info}\n")

            # Demo 5: Memory statistics
            print("\n📊 Demo 5: Memory Statistics")
            print("-" * 40)

            status = await agent.get_status()
            agent_state = status["agent_state"]
            memory_stats = status["memory_stats"]["storage"]

            print(f"Total memories stored: {agent_state['total_memories_stored']}")
            print(f"Total memories in database: {memory_stats['total_memories']}")
            print(f"Memory types breakdown:")
            for mem_type, stats in memory_stats.get("by_type", {}).items():
                print(f"  - {mem_type}: {stats['count']} memories")

            # Demo 6: Session management
            print("\n💬 Demo 6: Session Management")
            print("-" * 40)

            session_id = await agent.session_manager.create_session()
            print(f"✓ Created session: {session_id[:8]}...")

            # Store conversation memory
            await agent.memory_manager.store_episodic_memory(
                content="Demo conversation about AI capabilities",
                session_id=session_id,
                user_message="What can you remember?",
                assistant_response="I can remember facts, preferences, and conversations across sessions",
                importance=0.7
            )
            print("✓ Stored episodic memory")

            conversation_result = await agent.recall("conversation about AI capabilities")
            print(f"Q: What did we talk about?")
            print(f"A: {conversation_result}\n")

            print("🎉 Demo completed successfully!")
            print("\nKey Features Demonstrated:")
            print("• ✅ Persistent memory across sessions")
            print("• ✅ Different memory types (semantic, procedural, episodic)")
            print("• ✅ Cross-referencing of related information")
            print("• ✅ Importance-based memory storage")
            print("• ✅ Session management")
            print("• ✅ Memory statistics and monitoring")

        except Exception as e:
            print(f"❌ Error during demo: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await agent.close()


async def test_learning_persistence():
    """Test that learning persists between sessions."""
    print("\n🔄 Testing Learning Persistence")
    print("=" * 50)

    db_path = Path.home() / ".evo" / "demo_persistence.db"

    # Session 1: Teach information
    print("📝 Session 1: Teaching information...")
    memory_manager1 = MemoryManager(str(db_path))
    await memory_manager1.initialize()

    config = AgentConfig(
        model_name="llama3.2:7b",
        use_memory=True,
        enable_learning=False,
        enable_mcp=False,
    )

    agent1 = EvoAgent(config=config, memory_manager=memory_manager1)
    await agent1.initialize()

    await agent1.teach("The capital of France is Paris", importance=0.9)
    await agent1.teach("The user loves hiking in mountains", importance=0.8)
    print("✓ Taught information in session 1")

    await agent1.close()

    # Session 2: Recall information
    print("\n🔍 Session 2: Recalling information...")
    memory_manager2 = MemoryManager(str(db_path))
    await memory_manager2.initialize()

    agent2 = EvoAgent(config=config, memory_manager=memory_manager2)
    await agent2.initialize()

    capital_result = await agent2.recall("capital of France")
    hiking_result = await agent2.recall("user loves hiking")

    print(f"Q: What is the capital of France?")
    print(f"A: {capital_result}")

    print(f"\nQ: What does the user love?")
    print(f"A: {hiking_result}")

    if "Paris" in capital_result and "hiking" in hiking_result.lower():
        print("\n✅ Persistence test PASSED - Information remembered across sessions!")
    else:
        print("\n❌ Persistence test FAILED - Information not properly recalled")

    await agent2.close()


def main():
    """Main demo function."""
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise

    print("Welcome to the Evo AI Demo!")
    print("This demo will showcase the learning and memory capabilities.\n")

    try:
        # Run the main demo
        asyncio.run(demo_learning_and_recall())

        # Test persistence
        asyncio.run(test_learning_persistence())

        print("\n" + "=" * 50)
        print("🚀 Demo complete! To try Evo AI yourself:")
        print("1. Run 'evo init' to set up your configuration")
        print("2. Run 'evo chat' to start an interactive session")
        print("3. Try teaching Evo something: 'Remember that I love pizza'")
        print("4. Exit and start a new chat session")
        print("5. Ask Evo: 'What do I love?' - it should remember!")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()