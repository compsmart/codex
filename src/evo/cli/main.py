"""
Main entry point for Evo AI CLI.
"""

import sys
from typing import Optional

import typer

from .commands import cli


def main(args: Optional[list] = None) -> None:
    """Main entry point for the CLI."""
    if args is None:
        args = sys.argv[1:]

    try:
        cli(args)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()