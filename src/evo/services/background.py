"""
Background service coordinator for Evo AI.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..learning import LearningScheduler
from ..summarization import ContextSummarizer
from .consolidation import MemoryConsolidationService
from .monitoring import SystemMonitor


class BackgroundService:
    """Coordinates all background services for Evo AI."""

    def __init__(
        self,
        learning_scheduler: Optional[LearningScheduler] = None,
        summarizer: Optional[ContextSummarizer] = None,
        consolidation_service: Optional[MemoryConsolidationService] = None,
        monitor: Optional[SystemMonitor] = None,
    ):
        self.learning_scheduler = learning_scheduler
        self.summarizer = summarizer
        self.consolidation_service = consolidation_service
        self.monitor = monitor

        self.logger = logging.getLogger(__name__)
        self._running = False
        self._stop_requested = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start all background services."""
        if self._running:
            self.logger.warning("Background services already running")
            return

        self._running = True
        self._stop_requested = False

        self.logger.info("Starting Evo AI background services...")

        try:
            # Setup signal handlers for graceful shutdown
            if sys.platform != "win32":
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)

            # Start individual services
            tasks = []

            if self.learning_scheduler:
                tasks.append(asyncio.create_task(self._run_learning_scheduler()))

            if self.consolidation_service:
                tasks.append(asyncio.create_task(self._run_consolidation_service()))

            if self.monitor:
                tasks.append(asyncio.create_task(self._run_monitor()))

            if self.summarizer:
                tasks.append(asyncio.create_task(self._run_summarization_service()))

            # Add coordinator task
            tasks.append(asyncio.create_task(self._coordinator_loop()))

            self._tasks = tasks

            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Error in background services: {e}")
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop all background services."""
        self.logger.info("Stopping background services...")
        self._stop_requested = True

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to finish with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("Some background tasks did not stop cleanly")

        self._running = False
        self.logger.info("Background services stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._stop_requested = True

    async def _run_learning_scheduler(self) -> None:
        """Run the learning scheduler service."""
        try:
            self.logger.info("Starting learning scheduler service")
            await self.learning_scheduler.start()
        except Exception as e:
            self.logger.error(f"Learning scheduler error: {e}")
        finally:
            if self.learning_scheduler:
                await self.learning_scheduler.stop()

    async def _run_consolidation_service(self) -> None:
        """Run the memory consolidation service."""
        try:
            self.logger.info("Starting memory consolidation service")
            await self.consolidation_service.start()
        except Exception as e:
            self.logger.error(f"Memory consolidation error: {e}")
        finally:
            if self.consolidation_service:
                await self.consolidation_service.stop()

    async def _run_monitor(self) -> None:
        """Run the system monitor service."""
        try:
            self.logger.info("Starting system monitor service")
            await self.monitor.start()
        except Exception as e:
            self.logger.error(f"System monitor error: {e}")
        finally:
            if self.monitor:
                await self.monitor.stop()

    async def _run_summarization_service(self) -> None:
        """Run the summarization service."""
        try:
            self.logger.info("Starting summarization service")
            while not self._stop_requested:
                # Periodic summarization tasks would go here
                await asyncio.sleep(3600)  # Check every hour
        except Exception as e:
            self.logger.error(f"Summarization service error: {e}")

    async def _coordinator_loop(self) -> None:
        """Main coordinator loop."""
        try:
            while not self._stop_requested:
                await self._coordinate_services()
                await asyncio.sleep(60)  # Coordinate every minute
        except Exception as e:
            self.logger.error(f"Coordinator error: {e}")

    async def _coordinate_services(self) -> None:
        """Coordinate between services."""
        try:
            # Check service health
            health_status = await self.get_health_status()

            # Log health issues
            for service, status in health_status.items():
                if not status.get("healthy", True):
                    self.logger.warning(f"Service {service} is not healthy: {status}")

            # Service-specific coordination
            if self.learning_scheduler and self.consolidation_service:
                # Trigger consolidation after learning
                learning_status = await self.learning_scheduler.get_status()
                if learning_status["state"]["last_training"]:
                    last_training = datetime.fromisoformat(learning_status["state"]["last_training"])
                    # Trigger consolidation 30 minutes after training
                    await self.consolidation_service.schedule_consolidation(
                        delay_minutes=30
                    )

        except Exception as e:
            self.logger.error(f"Error in service coordination: {e}")

    async def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all services."""
        status = {}

        # Learning scheduler
        if self.learning_scheduler:
            try:
                learning_status = await self.learning_scheduler.get_status()
                status["learning_scheduler"] = {
                    "healthy": True,
                    "running": learning_status["state"]["is_enabled"],
                    "pending_jobs": learning_status["pending_jobs"],
                    "last_training": learning_status["state"]["last_training"],
                }
            except Exception as e:
                status["learning_scheduler"] = {
                    "healthy": False,
                    "error": str(e),
                }

        # Memory consolidation
        if self.consolidation_service:
            try:
                consolidation_status = await self.consolidation_service.get_status()
                status["memory_consolidation"] = {
                    "healthy": True,
                    "running": consolidation_status["running"],
                    "last_consolidation": consolidation_status["last_consolidation"],
                }
            except Exception as e:
                status["memory_consolidation"] = {
                    "healthy": False,
                    "error": str(e),
                }

        # System monitor
        if self.monitor:
            try:
                monitor_status = await self.monitor.get_status()
                status["system_monitor"] = {
                    "healthy": True,
                    "running": monitor_status["running"],
                    "alerts": monitor_status.get("active_alerts", []),
                }
            except Exception as e:
                status["system_monitor"] = {
                    "healthy": False,
                    "error": str(e),
                }

        return status

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics from all services."""
        stats = {
            "coordinator": {
                "running": self._running,
                "active_tasks": len([t for t in self._tasks if not t.done()]),
                "uptime_hours": 0,  # Would track actual uptime
            }
        }

        # Collect stats from individual services
        if self.learning_scheduler:
            try:
                learning_status = await self.learning_scheduler.get_status()
                stats["learning"] = learning_status
            except Exception as e:
                stats["learning"] = {"error": str(e)}

        if self.consolidation_service:
            try:
                consolidation_stats = await self.consolidation_service.get_stats()
                stats["consolidation"] = consolidation_stats
            except Exception as e:
                stats["consolidation"] = {"error": str(e)}

        if self.monitor:
            try:
                monitor_stats = await self.monitor.get_stats()
                stats["monitoring"] = monitor_stats
            except Exception as e:
                stats["monitoring"] = {"error": str(e)}

        return stats

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up background services...")

        # Stop all services gracefully
        if self.learning_scheduler:
            try:
                await self.learning_scheduler.stop()
            except Exception as e:
                self.logger.error(f"Error stopping learning scheduler: {e}")

        if self.consolidation_service:
            try:
                await self.consolidation_service.stop()
            except Exception as e:
                self.logger.error(f"Error stopping consolidation service: {e}")

        if self.monitor:
            try:
                await self.monitor.stop()
            except Exception as e:
                self.logger.error(f"Error stopping monitor: {e}")

        self._running = False

    def is_running(self) -> bool:
        """Check if background services are running."""
        return self._running and not self._stop_requested