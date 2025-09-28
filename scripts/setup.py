#!/usr/bin/env python3
"""
Setup script for Evo AI development environment.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("setup.log"),
        ],
    )
    return logging.getLogger(__name__)


def run_command(command, cwd=None, capture_output=False):
    """Run a shell command."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True,
        )
        if capture_output:
            return result.stdout.strip()
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Exit code: {e.returncode}")
        if capture_output:
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
        return False


def check_python_version():
    """Check Python version requirements."""
    logger = logging.getLogger(__name__)

    if sys.version_info < (3, 9):
        logger.error("Python 3.9 or higher is required")
        return False

    logger.info(f"Python version: {sys.version}")
    return True


def check_ollama_installation():
    """Check if Ollama is installed."""
    logger = logging.getLogger(__name__)

    try:
        result = run_command("ollama --version", capture_output=True)
        if result:
            logger.info(f"Ollama is installed: {result}")
            return True
    except Exception:
        pass

    logger.warning("Ollama is not installed or not in PATH")
    return False


def install_python_dependencies():
    """Install Python dependencies."""
    logger = logging.getLogger(__name__)
    logger.info("Installing Python dependencies...")

    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False

    # Install main dependencies
    if not run_command(f"{sys.executable} -m pip install -e ."):
        return False

    # Install development dependencies
    if not run_command(f"{sys.executable} -m pip install -e .[dev]"):
        return False

    logger.info("Python dependencies installed successfully")
    return True


def setup_ollama_model():
    """Setup Ollama with the required model."""
    logger = logging.getLogger(__name__)

    if not check_ollama_installation():
        logger.error("Ollama must be installed first. Please install from https://ollama.ai")
        return False

    logger.info("Setting up Ollama model...")

    # Check if Ollama service is running
    try:
        run_command("ollama list", capture_output=True)
    except Exception:
        logger.info("Starting Ollama service...")
        # On Windows, Ollama usually starts automatically
        # On Linux/Mac, you might need to start the service
        pass

    # Pull the required model
    logger.info("Pulling Llama 3.2 7B model (this may take a while)...")
    if not run_command("ollama pull llama3.2:7b"):
        logger.error("Failed to pull Llama 3.2 model")
        return False

    logger.info("Ollama model setup complete")
    return True


def create_config_directory():
    """Create configuration directory and files."""
    logger = logging.getLogger(__name__)
    logger.info("Creating configuration directory...")

    config_dir = Path.home() / ".evo"
    config_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (config_dir / "sessions").mkdir(exist_ok=True)
    (config_dir / "logs").mkdir(exist_ok=True)
    (config_dir / "adapters").mkdir(exist_ok=True)

    # Create default config
    config_file = config_dir / "config.json"
    if not config_file.exists():
        config = {
            "model_name": "llama3.2:7b",
            "temperature": 0.7,
            "use_memory": True,
            "enable_learning": True,
            "enable_mcp": True,
            "database_path": str(config_dir / "memory.db"),
            "sessions_dir": str(config_dir / "sessions"),
            "learning": {
                "lora_rank": 16,
                "learning_rate": 0.0001,
                "training_interval_hours": 6
            }
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Created config file: {config_file}")

    logger.info("Configuration directory setup complete")
    return True


def run_tests():
    """Run the test suite."""
    logger = logging.getLogger(__name__)
    logger.info("Running test suite...")

    if not run_command("python -m pytest tests/ -v"):
        logger.error("Tests failed")
        return False

    logger.info("All tests passed")
    return True


def verify_installation():
    """Verify the installation."""
    logger = logging.getLogger(__name__)
    logger.info("Verifying installation...")

    try:
        # Test import
        result = run_command(
            f"{sys.executable} -c 'from src.evo import EvoAgent; print(\"Import successful\")'",
            capture_output=True
        )
        if not result or "Import successful" not in result:
            logger.error("Failed to import Evo AI")
            return False

        # Test CLI
        result = run_command("evo --help", capture_output=True)
        if not result:
            logger.error("CLI not working")
            return False

        logger.info("Installation verified successfully")
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Evo AI development environment")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama setup")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-deps", action="store_true", help="Skip installing dependencies")

    args = parser.parse_args()
    logger = setup_logging()

    logger.info("Starting Evo AI setup...")

    # Check requirements
    if not check_python_version():
        sys.exit(1)

    success = True

    # Install dependencies
    if not args.skip_deps:
        if not install_python_dependencies():
            success = False

    # Setup Ollama
    if not args.skip_ollama:
        if not setup_ollama_model():
            logger.warning("Ollama setup failed, but continuing...")
            # Don't fail the entire setup for Ollama issues

    # Create config
    if not create_config_directory():
        success = False

    # Run tests
    if not args.skip_tests and success:
        if not run_tests():
            logger.warning("Some tests failed, but installation may still work")

    # Verify installation
    if success:
        if verify_installation():
            logger.info("🎉 Evo AI setup completed successfully!")
            logger.info("You can now run 'evo init' to initialize your configuration")
            logger.info("Then use 'evo chat' to start chatting with your AI assistant")
        else:
            logger.error("Installation verification failed")
            success = False

    if not success:
        logger.error("Setup failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()