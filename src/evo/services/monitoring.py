"""
System monitoring service for Evo AI.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..memory import MemoryManager


class SystemAlert:
    """System alert for monitoring."""

    def __init__(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        timestamp: Optional[datetime] = None,
    ):
        self.alert_type = alert_type
        self.message = message
        self.severity = severity  # info, warning, error, critical
        self.timestamp = timestamp or datetime.now()


class SystemMonitor:
    """Monitors system resources and Evo AI health."""

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        check_interval_seconds: int = 60,
    ):
        self.memory_manager = memory_manager
        self.check_interval_seconds = check_interval_seconds

        self.logger = logging.getLogger(__name__)
        self._running = False
        self._stop_requested = False

        # Monitoring thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "memory_growth_mb": 100.0,  # MB per hour
        }

        # Statistics
        self.stats = {
            "start_time": None,
            "total_checks": 0,
            "alerts_generated": 0,
            "system_metrics": {},
            "memory_stats": {},
        }

        # Active alerts
        self.active_alerts: List[SystemAlert] = []

        # Performance history
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_entries = 1440  # 24 hours of minute-by-minute data

    async def start(self) -> None:
        """Start the monitoring service."""
        if self._running:
            return

        self._running = True
        self._stop_requested = False
        self.stats["start_time"] = datetime.now()

        self.logger.info("Starting system monitor service")

        try:
            while self._running and not self._stop_requested:
                await self._monitoring_cycle()
                await asyncio.sleep(self.check_interval_seconds)

        except Exception as e:
            self.logger.error(f"System monitor error: {e}")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the monitoring service."""
        self.logger.info("Stopping system monitor service")
        self._stop_requested = True

    async def _monitoring_cycle(self) -> None:
        """Perform one monitoring cycle."""
        try:
            current_time = datetime.now()

            # Collect system metrics
            system_metrics = await self._collect_system_metrics()

            # Collect memory statistics
            memory_stats = {}
            if self.memory_manager:
                try:
                    memory_stats = await self.memory_manager.get_stats()
                except Exception as e:
                    self.logger.warning(f"Failed to get memory stats: {e}")

            # Store in history
            history_entry = {
                "timestamp": current_time.isoformat(),
                "system": system_metrics,
                "memory": memory_stats,
            }

            self.performance_history.append(history_entry)

            # Limit history size
            if len(self.performance_history) > self.max_history_entries:
                self.performance_history = self.performance_history[-self.max_history_entries:]

            # Update current stats
            self.stats["system_metrics"] = system_metrics
            self.stats["memory_stats"] = memory_stats
            self.stats["total_checks"] += 1

            # Check for alerts
            await self._check_alerts(system_metrics, memory_stats)

            # Cleanup old alerts
            self._cleanup_old_alerts()

        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / 1024 / 1024

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / 1024 / 1024 / 1024

            # Process metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / 1024 / 1024
            process_cpu_percent = process.cpu_percent()

            # System load
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have load average
                load_avg = [cpu_percent / 100 * cpu_count, 0, 0]

            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_1m": load_avg[0],
                    "load_5m": load_avg[1] if len(load_avg) > 1 else 0,
                    "load_15m": load_avg[2] if len(load_avg) > 2 else 0,
                },
                "memory": {
                    "percent": memory_percent,
                    "total_gb": memory.total / 1024 / 1024 / 1024,
                    "available_mb": memory_available_mb,
                    "used_gb": memory.used / 1024 / 1024 / 1024,
                },
                "disk": {
                    "percent": disk_percent,
                    "total_gb": disk.total / 1024 / 1024 / 1024,
                    "free_gb": disk_free_gb,
                    "used_gb": disk.used / 1024 / 1024 / 1024,
                },
                "process": {
                    "memory_mb": process_memory_mb,
                    "cpu_percent": process_cpu_percent,
                    "pid": process.pid,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {"error": str(e)}

    async def _check_alerts(
        self, system_metrics: Dict[str, Any], memory_stats: Dict[str, Any]
    ) -> None:
        """Check for alert conditions."""
        try:
            # Clear resolved alerts
            self.active_alerts = [
                alert for alert in self.active_alerts
                if datetime.now() - alert.timestamp < timedelta(hours=1)
            ]

            # Check CPU usage
            cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
            if cpu_percent > self.thresholds["cpu_percent"]:
                self._add_alert(
                    "high_cpu",
                    f"High CPU usage: {cpu_percent:.1f}%",
                    "warning"
                )

            # Check memory usage
            memory_percent = system_metrics.get("memory", {}).get("percent", 0)
            if memory_percent > self.thresholds["memory_percent"]:
                self._add_alert(
                    "high_memory",
                    f"High memory usage: {memory_percent:.1f}%",
                    "warning"
                )

            # Check disk usage
            disk_percent = system_metrics.get("disk", {}).get("percent", 0)
            if disk_percent > self.thresholds["disk_percent"]:
                self._add_alert(
                    "high_disk",
                    f"High disk usage: {disk_percent:.1f}%",
                    "error"
                )

            # Check memory growth
            await self._check_memory_growth()

            # Check memory system health
            if memory_stats and "storage" in memory_stats:
                storage_stats = memory_stats["storage"]
                total_memories = storage_stats.get("total_memories", 0)

                # Alert if memory database is growing very large
                if total_memories > 100000:
                    self._add_alert(
                        "large_memory_db",
                        f"Large memory database: {total_memories} memories",
                        "info"
                    )

        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")

    async def _check_memory_growth(self) -> None:
        """Check for excessive memory growth."""
        try:
            if len(self.performance_history) < 60:  # Need at least 1 hour of data
                return

            # Get memory usage from 1 hour ago
            hour_ago = self.performance_history[-60]
            current = self.performance_history[-1]

            hour_ago_memory = hour_ago.get("system", {}).get("process", {}).get("memory_mb", 0)
            current_memory = current.get("system", {}).get("process", {}).get("memory_mb", 0)

            growth_mb = current_memory - hour_ago_memory

            if growth_mb > self.thresholds["memory_growth_mb"]:
                self._add_alert(
                    "memory_growth",
                    f"High memory growth: {growth_mb:.1f} MB/hour",
                    "warning"
                )

        except Exception as e:
            self.logger.error(f"Error checking memory growth: {e}")

    def _add_alert(self, alert_type: str, message: str, severity: str) -> None:
        """Add a new alert if not already active."""
        # Check if similar alert already exists
        for existing_alert in self.active_alerts:
            if existing_alert.alert_type == alert_type:
                return  # Don't duplicate alerts

        alert = SystemAlert(alert_type, message, severity)
        self.active_alerts.append(alert)
        self.stats["alerts_generated"] += 1

        # Log the alert
        if severity == "critical":
            self.logger.critical(f"ALERT: {message}")
        elif severity == "error":
            self.logger.error(f"ALERT: {message}")
        elif severity == "warning":
            self.logger.warning(f"ALERT: {message}")
        else:
            self.logger.info(f"ALERT: {message}")

    def _cleanup_old_alerts(self) -> None:
        """Remove old alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.timestamp > cutoff_time
        ]

    async def get_status(self) -> Dict[str, Any]:
        """Get monitoring service status."""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "total_checks": self.stats["total_checks"],
            "active_alerts": [
                {
                    "type": alert.alert_type,
                    "message": alert.message,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self.active_alerts
            ],
            "alerts_generated": self.stats["alerts_generated"],
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed monitoring statistics."""
        return {
            "service": self.stats,
            "current_metrics": {
                "system": self.stats.get("system_metrics", {}),
                "memory": self.stats.get("memory_stats", {}),
            },
            "alerts": [
                {
                    "type": alert.alert_type,
                    "message": alert.message,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self.active_alerts
            ],
            "thresholds": self.thresholds,
            "history_entries": len(self.performance_history),
        }

    async def get_performance_history(
        self, hours: int = 1
    ) -> List[Dict[str, Any]]:
        """Get performance history for the specified number of hours."""
        if not self.performance_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()

        return [
            entry for entry in self.performance_history
            if entry["timestamp"] >= cutoff_iso
        ]

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update monitoring thresholds."""
        self.thresholds.update(new_thresholds)
        self.logger.info(f"Updated monitoring thresholds: {new_thresholds}")

    def get_alerts_by_severity(self, severity: str) -> List[SystemAlert]:
        """Get alerts by severity level."""
        return [
            alert for alert in self.active_alerts
            if alert.severity == severity
        ]