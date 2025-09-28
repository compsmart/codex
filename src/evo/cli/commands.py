"""
CLI commands for Evo AI.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..agent import EvoAgent, AgentConfig
from ..memory import MemoryManager
from ..mcp import MCPClient
from .chat import ChatInterface

# Create CLI app
cli = typer.Typer(name="evo", help="Evo AI - A learning AI assistant")
console = Console()

# Global state
_agent: Optional[EvoAgent] = None
_config_dir = Path.home() / ".evo"


@cli.command()
def init(
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
    model: str = typer.Option("llama3.2:7b", "--model", "-m", help="Ollama model to use"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinitialize"),
) -> None:
    """Initialize Evo AI configuration."""
    config_path = Path(config_dir)
    config_file = config_path / "config.json"

    if config_file.exists() and not force:
        console.print(
            "[yellow]Evo AI is already initialized. Use --force to reinitialize.[/yellow]"
        )
        return

    # Create config directory
    config_path.mkdir(parents=True, exist_ok=True)

    # Create default config
    config = {
        "model_name": model,
        "temperature": 0.7,
        "use_memory": True,
        "enable_learning": True,
        "enable_mcp": True,
        "database_path": str(config_path / "memory.db"),
        "sessions_dir": str(config_path / "sessions"),
    }

    # Write config file
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # Create other directories
    (config_path / "sessions").mkdir(exist_ok=True)
    (config_path / "logs").mkdir(exist_ok=True)

    console.print(
        Panel(
            f"[green]Evo AI initialized successfully![/green]\n\n"
            f"Configuration directory: {config_path}\n"
            f"Model: {model}\n"
            f"Database: {config['database_path']}\n\n"
            f"[dim]Run 'evo chat' to start chatting![/dim]",
            title="🚀 Initialization Complete",
            border_style="green",
        )
    )


@cli.command()
def chat(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """Start an interactive chat session."""
    asyncio.run(_chat_async(session_id, config_dir))


async def _chat_async(session_id: Optional[str], config_dir: str) -> None:
    """Async chat implementation."""
    try:
        agent = await _get_agent(config_dir)
        chat_interface = ChatInterface(agent, console)
        await chat_interface.start(session_id)
    except Exception as e:
        console.print(f"[red]Error starting chat: {e}[/red]")


@cli.command()
def teach(
    content: str = typer.Argument(..., help="Information to teach"),
    memory_type: str = typer.Option(
        "semantic", "--type", "-t", help="Memory type (semantic, procedural, episodic)"
    ),
    importance: float = typer.Option(
        0.7, "--importance", "-i", help="Importance score (0.0-1.0)"
    ),
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """Teach Evo new information."""
    asyncio.run(_teach_async(content, memory_type, importance, config_dir))


async def _teach_async(
    content: str, memory_type: str, importance: float, config_dir: str
) -> None:
    """Async teach implementation."""
    try:
        agent = await _get_agent(config_dir)
        result = await agent.teach(content, memory_type, importance)
        console.print(f"[green]✓[/green] {result}")
    except Exception as e:
        console.print(f"[red]Error teaching: {e}[/red]")


@cli.command()
def recall(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum number of results"),
    memory_types: Optional[List[str]] = typer.Option(
        None, "--type", "-t", help="Memory types to search"
    ),
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """Recall information from memory."""
    asyncio.run(_recall_async(query, limit, memory_types, config_dir))


async def _recall_async(
    query: str, limit: int, memory_types: Optional[List[str]], config_dir: str
) -> None:
    """Async recall implementation."""
    try:
        agent = await _get_agent(config_dir)
        result = await agent.recall(query, memory_types, limit)
        console.print(result)
    except Exception as e:
        console.print(f"[red]Error recalling: {e}[/red]")


@cli.command()
def status(
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """Show Evo AI status and statistics."""
    asyncio.run(_status_async(config_dir))


async def _status_async(config_dir: str) -> None:
    """Async status implementation."""
    try:
        agent = await _get_agent(config_dir)
        status_data = await agent.get_status()

        # Create status table
        table = Table(title="🤖 Evo AI Status", show_header=True, header_style="bold blue")
        table.add_column("Category", style="cyan")
        table.add_column("Metric", style="white")
        table.add_column("Value", style="yellow")

        # Agent state
        agent_state = status_data["agent_state"]
        table.add_row("Agent", "Learning", "✓" if agent_state["is_learning"] else "✗")
        table.add_row("Agent", "Conversations", str(agent_state["total_conversations"]))
        table.add_row("Agent", "Memories Stored", str(agent_state["total_memories_stored"]))
        table.add_row("Agent", "Model Updates", str(agent_state["model_updates"]))
        table.add_row("Agent", "Active Sessions", str(agent_state["active_sessions"]))

        # Memory stats
        memory_stats = status_data["memory_stats"]["storage"]
        table.add_row("Memory", "Total Memories", str(memory_stats["total_memories"]))

        for mem_type, stats in memory_stats["by_type"].items():
            table.add_row(
                "Memory", f"{mem_type.title()} Count", str(stats["count"])
            )

        # Config
        config = status_data["config"]
        table.add_row("Config", "Model", config["model_name"])
        table.add_row("Config", "Learning", "✓" if config["enable_learning"] else "✗")
        table.add_row("Config", "MCP", "✓" if config["enable_mcp"] else "✗")
        table.add_row("Config", "Memory", "✓" if config["use_memory"] else "✗")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")


@cli.command("sessions")
def sessions_command(
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of sessions to show"),
) -> None:
    """List chat sessions."""
    asyncio.run(_sessions_async(config_dir, limit))


async def _sessions_async(config_dir: str, limit: int) -> None:
    """Async sessions implementation."""
    try:
        agent = await _get_agent(config_dir)
        sessions = await agent.session_manager.list_sessions(limit=limit)

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        # Create sessions table
        table = Table(title="💬 Chat Sessions", show_header=True, header_style="bold blue")
        table.add_column("Session ID", style="cyan")
        table.add_column("Messages", style="white")
        table.add_column("Created", style="green")
        table.add_column("Updated", style="yellow")
        table.add_column("Active", style="red")

        for session in sessions:
            table.add_row(
                session["session_id"][:8] + "...",
                str(session["message_count"]),
                session["created_at"][:10] if session["created_at"] else "Unknown",
                session["updated_at"][:10] if session["updated_at"] else "Unknown",
                "✓" if session["is_active"] else "✗",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing sessions: {e}[/red]")


@cli.command()
def memory(
    action: str = typer.Argument(..., help="Action: search, stats, cleanup"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """Memory management commands."""
    asyncio.run(_memory_async(action, query, config_dir))


async def _memory_async(action: str, query: Optional[str], config_dir: str) -> None:
    """Async memory implementation."""
    try:
        agent = await _get_agent(config_dir)

        if action == "search":
            if not query:
                console.print("[red]Search query required for search action[/red]")
                return

            result = await agent.recall(query)
            console.print(result)

        elif action == "stats":
            stats = await agent.memory_manager.get_stats()

            # Display memory statistics
            console.print(Panel(
                f"[bold]Memory Statistics[/bold]\n\n"
                f"Total Memories: {stats['storage']['total_memories']}\n"
                f"Database Size: {stats['storage']['database_size'] // 1024} KB\n"
                f"Embedding Cache: {stats['embeddings']['cache_size']} items",
                title="📊 Memory Stats",
                border_style="blue"
            ))

        elif action == "cleanup":
            # This would implement memory cleanup
            console.print("[yellow]Memory cleanup not implemented yet[/yellow]")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")

    except Exception as e:
        console.print(f"[red]Error in memory command: {e}[/red]")


@cli.command()
def config(
    action: str = typer.Argument(..., help="Action: show, edit"),
    config_dir: str = typer.Option(
        str(_config_dir), "--config-dir", "-c", help="Configuration directory"
    ),
) -> None:
    """Configuration management."""
    config_path = Path(config_dir) / "config.json"

    if action == "show":
        if not config_path.exists():
            console.print("[red]Configuration not found. Run 'evo init' first.[/red]")
            return

        with open(config_path) as f:
            config_data = json.load(f)

        # Display configuration
        syntax = Syntax(
            json.dumps(config_data, indent=2),
            "json",
            theme="monokai",
            line_numbers=True,
        )

        console.print(Panel(
            syntax,
            title="⚙️ Configuration",
            border_style="blue"
        ))

    elif action == "edit":
        console.print(f"[yellow]Edit configuration file: {config_path}[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")


async def _get_agent(config_dir: str) -> EvoAgent:
    """Get or create agent instance."""
    global _agent

    if _agent is not None:
        return _agent

    # Load configuration
    config_path = Path(config_dir) / "config.json"
    if not config_path.exists():
        raise Exception("Configuration not found. Run 'evo init' first.")

    with open(config_path) as f:
        config_data = json.load(f)

    # Create agent config
    agent_config = AgentConfig(**config_data)

    # Create memory manager
    memory_manager = MemoryManager(
        db_path=config_data["database_path"],
    )

    # Create MCP client (optional)
    mcp_client = MCPClient() if config_data.get("enable_mcp", True) else None

    # Create agent
    _agent = EvoAgent(
        config=agent_config,
        memory_manager=memory_manager,
        mcp_client=mcp_client,
    )

    await _agent.initialize()
    return _agent


if __name__ == "__main__":
    cli()