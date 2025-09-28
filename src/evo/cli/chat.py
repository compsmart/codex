"""
Interactive chat interface for Evo AI.
"""

import asyncio
import signal
import sys
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text

from ..agent import EvoAgent


class ChatInterface:
    """Rich interactive chat interface for Evo AI."""

    def __init__(self, agent: EvoAgent, console: Console):
        self.agent = agent
        self.console = console
        self.session_id: Optional[str] = None
        self._running = False

    async def start(self, session_id: Optional[str] = None) -> None:
        """Start the interactive chat session."""
        self.session_id = session_id or await self.agent.session_manager.create_session()
        self._running = True

        # Setup signal handlers for graceful exit
        if sys.platform != "win32":
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        # Display welcome message
        self._show_welcome()

        try:
            while self._running:
                await self._chat_loop()
        except KeyboardInterrupt:
            pass
        except EOFError:
            pass
        finally:
            await self._cleanup()

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        self._running = False

    def _show_welcome(self) -> None:
        """Show welcome message."""
        welcome_text = (
            "[bold blue]🤖 Evo AI[/bold blue] - Your Learning Assistant\n\n"
            f"Session: [cyan]{self.session_id[:8]}...[/cyan]\n"
            "Type your message and press Enter to chat.\n"
            "Special commands:\n"
            "  /help     - Show help\n"
            "  /status   - Show agent status\n"
            "  /memory   - Search memory\n"
            "  /teach    - Teach new information\n"
            "  /recall   - Recall information\n"
            "  /clear    - Clear screen\n"
            "  /exit     - Exit chat\n\n"
            "[dim]Tip: Evo learns from our conversations and remembers across sessions![/dim]"
        )

        self.console.print(
            Panel(
                welcome_text,
                title="Welcome to Evo AI",
                border_style="blue",
                padding=(1, 2),
            )
        )

    async def _chat_loop(self) -> None:
        """Main chat loop."""
        try:
            # Get user input
            user_input = await self._get_user_input()

            if not user_input or not user_input.strip():
                return

            user_input = user_input.strip()

            # Handle special commands
            if user_input.startswith("/"):
                await self._handle_command(user_input)
                return

            # Display user message
            self.console.print(f"\n[bold blue]You:[/bold blue] {user_input}")

            # Generate response with typing indicator
            response = await self._get_response_with_spinner(user_input)

            # Display response
            self._display_response(response)

        except (KeyboardInterrupt, EOFError):
            self._running = False

    async def _get_user_input(self) -> str:
        """Get user input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: Prompt.ask("\n[bold green]→[/bold green]", console=self.console)
        )

    async def _get_response_with_spinner(self, message: str) -> str:
        """Get response from agent with a spinner."""
        with Live(
            Spinner("dots", text="[dim]Evo is thinking...[/dim]"),
            console=self.console,
            refresh_per_second=10,
        ):
            response = await self.agent.chat(
                message=message,
                session_id=self.session_id,
                stream=False,
            )

        return response

    def _display_response(self, response: str) -> None:
        """Display agent response with rich formatting."""
        # Try to render as markdown if it contains markdown syntax
        if any(marker in response for marker in ["**", "*", "`", "#", "-", "1."]):
            try:
                markdown = Markdown(response)
                self.console.print(f"\n[bold cyan]Evo:[/bold cyan]")
                self.console.print(markdown)
                return
            except Exception:
                # Fall back to plain text if markdown parsing fails
                pass

        # Display as plain text
        self.console.print(f"\n[bold cyan]Evo:[/bold cyan] {response}")

    async def _handle_command(self, command: str) -> None:
        """Handle special chat commands."""
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            self._show_help()

        elif cmd == "status":
            await self._show_status()

        elif cmd == "memory" or cmd == "search":
            if not args:
                self.console.print("[yellow]Usage: /memory <search query>[/yellow]")
                return

            result = await self.agent.recall(args)
            self.console.print(f"\n[bold cyan]Memory Search Results:[/bold cyan]\n{result}")

        elif cmd == "teach":
            if not args:
                self.console.print("[yellow]Usage: /teach <information to teach>[/yellow]")
                return

            result = await self.agent.teach(args)
            self.console.print(f"\n[bold green]✓[/bold green] {result}")

        elif cmd == "recall":
            if not args:
                self.console.print("[yellow]Usage: /recall <search query>[/yellow]")
                return

            result = await self.agent.recall(args)
            self.console.print(f"\n[bold cyan]Recall Results:[/bold cyan]\n{result}")

        elif cmd == "clear":
            self.console.clear()
            self._show_welcome()

        elif cmd == "exit" or cmd == "quit":
            self.console.print("\n[bold blue]Goodbye! 👋[/bold blue]")
            self._running = False

        elif cmd == "session":
            self.console.print(f"\n[bold cyan]Current Session:[/bold cyan] {self.session_id}")

        elif cmd == "save":
            success = await self.agent.session_manager.save_session(self.session_id)
            if success:
                self.console.print("\n[bold green]✓[/bold green] Session saved")
            else:
                self.console.print("\n[bold red]✗[/bold red] Failed to save session")

        else:
            self.console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            self.console.print("[dim]Type /help for available commands[/dim]")

    def _show_help(self) -> None:
        """Show help information."""
        help_text = (
            "[bold]Available Commands:[/bold]\n\n"
            "[cyan]/help[/cyan]               - Show this help message\n"
            "[cyan]/status[/cyan]             - Show agent status and statistics\n"
            "[cyan]/memory <query>[/cyan]     - Search memory for information\n"
            "[cyan]/teach <info>[/cyan]       - Teach Evo new information\n"
            "[cyan]/recall <query>[/cyan]     - Recall specific information\n"
            "[cyan]/clear[/cyan]              - Clear the screen\n"
            "[cyan]/session[/cyan]            - Show current session ID\n"
            "[cyan]/save[/cyan]               - Save current session\n"
            "[cyan]/exit[/cyan]               - Exit the chat\n\n"
            "[bold]Tips:[/bold]\n"
            "• Evo learns from every conversation\n"
            "• Information is remembered between sessions\n"
            "• Use natural language - no special syntax needed\n"
            "• Ask Evo to remember preferences: 'My favorite language is Python'\n"
            "• Evo can search the web and analyze code when needed"
        )

        self.console.print(
            Panel(
                help_text,
                title="💡 Help",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    async def _show_status(self) -> None:
        """Show agent status."""
        try:
            status = await self.agent.get_status()

            status_text = (
                f"[bold]Agent Status:[/bold]\n\n"
                f"Learning: {'✓' if status['agent_state']['is_learning'] else '✗'}\n"
                f"Conversations: {status['agent_state']['total_conversations']}\n"
                f"Memories: {status['agent_state']['total_memories_stored']}\n"
                f"Sessions: {status['active_sessions']}\n\n"
                f"[bold]Memory:[/bold]\n"
                f"Total: {status['memory_stats']['storage']['total_memories']}\n"
                f"Database: {status['memory_stats']['storage']['database_size'] // 1024} KB\n\n"
                f"[bold]Model:[/bold] {status['config']['model_name']}"
            )

            self.console.print(
                Panel(
                    status_text,
                    title="📊 Status",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        except Exception as e:
            self.console.print(f"[red]Error getting status: {e}[/red]")

    async def _cleanup(self) -> None:
        """Cleanup when exiting chat."""
        try:
            # Save session
            if self.session_id:
                await self.agent.session_manager.save_session(self.session_id)

            self.console.print("\n[dim]Session saved. Thanks for chatting with Evo! 🤖[/dim]")

        except Exception as e:
            self.console.print(f"\n[red]Error during cleanup: {e}[/red]")


class StreamingChatInterface(ChatInterface):
    """Chat interface with streaming responses."""

    async def _get_response_with_spinner(self, message: str) -> str:
        """Get streaming response from agent."""
        response_text = Text()

        with Live(
            Panel(response_text, title="🤖 Evo", border_style="cyan"),
            console=self.console,
            refresh_per_second=10,
        ) as live:
            full_response = ""
            async for chunk in await self.agent.chat(
                message=message,
                session_id=self.session_id,
                stream=True,
            ):
                full_response += chunk
                response_text.plain = full_response
                live.update(Panel(response_text, title="🤖 Evo", border_style="cyan"))

        return full_response