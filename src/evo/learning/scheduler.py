"""
Learning scheduler for managing automated training.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..memory import MemoryManager
from .data import ConversationDataProcessor, LearningDataset
from .models import LearningConfig, LearningJob, LearningState, TrainingMetrics
from .qlora import QLoRATrainer


class LearningScheduler:
    """Manages automated learning and training scheduling."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        config: LearningConfig,
        model_name: str = "microsoft/DialoGPT-medium",
    ):
        self.memory_manager = memory_manager
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_processor = ConversationDataProcessor(memory_manager, config)
        self.trainer: Optional[QLoRATrainer] = None

        # State management
        self.state = LearningState()
        self.job_queue: List[LearningJob] = []
        self.completed_jobs: List[LearningJob] = []

        # Control flags
        self._running = False
        self._stop_requested = False

    async def initialize(self) -> None:
        """Initialize the learning scheduler."""
        self.logger.info("Initializing learning scheduler...")

        # Initialize trainer
        self.trainer = QLoRATrainer(self.config, self.model_name)
        await self.trainer.initialize()

        # Load state if exists
        await self._load_state()

        self.logger.info("Learning scheduler initialized")

    async def start(self) -> None:
        """Start the learning scheduler."""
        if self._running:
            self.logger.warning("Learning scheduler is already running")
            return

        self._running = True
        self._stop_requested = False
        self.state.is_enabled = True

        self.logger.info("Starting learning scheduler...")

        try:
            while self._running and not self._stop_requested:
                await self._scheduler_loop()
                await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            self.logger.error(f"Learning scheduler error: {e}")
        finally:
            self._running = False
            self.state.is_enabled = False

        self.logger.info("Learning scheduler stopped")

    async def stop(self) -> None:
        """Stop the learning scheduler."""
        self.logger.info("Stopping learning scheduler...")
        self._stop_requested = True

        # Wait for current training to complete or timeout
        timeout = 300  # 5 minutes
        elapsed = 0
        while self.state.is_training and elapsed < timeout:
            await asyncio.sleep(1)
            elapsed += 1

        self._running = False
        await self._save_state()

    async def schedule_training(
        self,
        priority: int = 0,
        scheduled_at: Optional[datetime] = None,
        data_query: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Schedule a training job."""
        job = LearningJob(
            config=self.config,
            priority=priority,
            scheduled_at=scheduled_at or datetime.now(),
            data_query=data_query or {},
        )

        self.job_queue.append(job)
        self.job_queue.sort(key=lambda x: (x.priority, x.scheduled_at), reverse=True)

        self.state.pending_jobs = len(self.job_queue)

        self.logger.info(f"Training job scheduled: {job.id}")
        return job.id

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        try:
            # Check if we should schedule automatic training
            await self._check_automatic_training()

            # Process queued jobs
            await self._process_job_queue()

            # Cleanup old jobs and adapters
            await self._cleanup_old_data()

        except Exception as e:
            self.logger.error(f"Error in scheduler loop: {e}")

    async def _check_automatic_training(self) -> None:
        """Check if automatic training should be triggered."""
        if self.state.is_training:
            return

        # Check if enough time has passed since last training
        if self.state.last_training:
            time_since_last = datetime.now() - self.state.last_training
            if time_since_last.total_seconds() < self.config.training_interval_hours * 3600:
                return

        # Check if we have enough new data
        try:
            # Get recent high-quality conversations
            samples = await self.data_processor.extract_training_data(
                min_importance=self.config.importance_threshold,
                max_age_days=self.config.max_training_data_age_days,
                max_samples=100,  # Quick check
            )

            if len(samples) >= self.config.min_data_points:
                self.logger.info(f"Scheduling automatic training with {len(samples)} samples")
                await self.schedule_training(priority=1)

        except Exception as e:
            self.logger.error(f"Error checking for automatic training: {e}")

    async def _process_job_queue(self) -> None:
        """Process queued training jobs."""
        if self.state.is_training or not self.job_queue:
            return

        # Get the highest priority job that's ready
        ready_jobs = [
            job for job in self.job_queue
            if job.scheduled_at <= datetime.now() and not job.is_completed
        ]

        if not ready_jobs:
            return

        job = ready_jobs[0]
        self.job_queue.remove(job)
        self.state.pending_jobs = len(self.job_queue)

        try:
            await self._execute_training_job(job)
        except Exception as e:
            self.logger.error(f"Failed to execute training job {job.id}: {e}")
            job.complete(success=False, error=str(e))
            self.state.failed_jobs += 1

        self.completed_jobs.append(job)

    async def _execute_training_job(self, job: LearningJob) -> None:
        """Execute a training job."""
        self.logger.info(f"Executing training job: {job.id}")

        self.state.is_training = True
        self.state.current_job = job.id
        job.start()

        try:
            # Extract training data
            self.logger.info("Extracting training data...")
            samples = await self.data_processor.extract_training_data(
                min_importance=self.config.importance_threshold,
                max_age_days=self.config.max_training_data_age_days,
                max_samples=1000,
            )

            if len(samples) < self.config.min_data_points:
                raise Exception(f"Insufficient training data: {len(samples)} samples")

            # Filter and augment data
            samples = self.data_processor.filter_quality_data(samples)
            samples = await self.data_processor.augment_data(samples)

            self.logger.info(f"Prepared {len(samples)} training samples")

            # Create dataset
            dataset = LearningDataset(
                data=samples,
                tokenizer=self.trainer.tokenizer,
                max_length=self.config.max_length,
            )

            # Save dataset for analysis
            dataset_path = Path(self.config.adapter_save_path).expanduser() / f"dataset_{job.id}.json"
            dataset.save_to_file(str(dataset_path))

            # Train the model
            self.logger.info("Starting QLoRA training...")
            metrics = await self.trainer.train(dataset)

            # Update job with results
            job.metrics = metrics
            job.adapter_path = str(Path(self.config.adapter_save_path).expanduser() / f"adapter_{int(datetime.now().timestamp())}")

            # Update state
            self.state.update_training_stats(metrics)
            self.state.last_training = datetime.now()
            self.state.total_adapters_created += 1

            if metrics.success:
                self.state.active_adapter_path = job.adapter_path
                self.state.active_adapter_created = datetime.now()

            job.complete(success=metrics.success)

            self.logger.info(f"Training job completed successfully: {job.id}")

        except Exception as e:
            self.logger.error(f"Training job failed: {e}")
            job.complete(success=False, error=str(e))
            self.state.failed_jobs += 1
            raise

        finally:
            self.state.is_training = False
            self.state.current_job = None

            # Cleanup trainer to free memory
            if self.trainer:
                self.trainer.cleanup()

    async def _cleanup_old_data(self) -> None:
        """Cleanup old adapters and completed jobs."""
        try:
            # Remove old completed jobs (keep last 50)
            if len(self.completed_jobs) > 50:
                self.completed_jobs = self.completed_jobs[-50:]

            # Cleanup old adapters (keep last N)
            adapter_dir = Path(self.config.adapter_save_path).expanduser()
            if adapter_dir.exists():
                adapter_dirs = [
                    d for d in adapter_dir.iterdir()
                    if d.is_dir() and d.name.startswith("adapter_")
                ]

                # Sort by creation time
                adapter_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                # Remove old adapters (keep last max_adapters)
                for old_adapter in adapter_dirs[self.config.max_adapters:]:
                    try:
                        import shutil
                        shutil.rmtree(old_adapter)
                        self.logger.info(f"Removed old adapter: {old_adapter}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove old adapter {old_adapter}: {e}")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        return {
            "state": self.state.dict(),
            "pending_jobs": len(self.job_queue),
            "completed_jobs": len(self.completed_jobs),
            "recent_jobs": [
                {
                    "id": str(job.id),
                    "status": job.status,
                    "created_at": job.created_at.isoformat(),
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "success": job.metrics.success if job.metrics else None,
                }
                for job in self.completed_jobs[-5:]  # Last 5 jobs
            ],
            "config": self.config.dict(),
        }

    async def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        history = []

        for job in self.completed_jobs:
            if job.metrics:
                history.append({
                    "job_id": str(job.id),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "success": job.metrics.success,
                    "final_loss": job.metrics.final_loss,
                    "total_samples": job.metrics.total_samples,
                    "duration_seconds": job.metrics.duration_seconds,
                    "adapter_path": job.adapter_path,
                })

        return history

    async def _save_state(self) -> None:
        """Save scheduler state to disk."""
        try:
            state_dir = Path(self.config.adapter_save_path).expanduser()
            state_dir.mkdir(parents=True, exist_ok=True)

            state_file = state_dir / "scheduler_state.json"

            state_data = {
                "state": self.state.dict(),
                "config": self.config.dict(),
                "model_name": self.model_name,
                "completed_jobs": [
                    {
                        "id": str(job.id),
                        "status": job.status,
                        "created_at": job.created_at.isoformat(),
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                        "metrics": job.metrics.dict() if job.metrics else None,
                        "adapter_path": job.adapter_path,
                    }
                    for job in self.completed_jobs[-10:]  # Save last 10
                ],
            }

            import json
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save scheduler state: {e}")

    async def _load_state(self) -> None:
        """Load scheduler state from disk."""
        try:
            state_file = Path(self.config.adapter_save_path).expanduser() / "scheduler_state.json"

            if not state_file.exists():
                return

            import json
            with open(state_file, "r") as f:
                state_data = json.load(f)

            # Load state
            if "state" in state_data:
                self.state = LearningState(**state_data["state"])

            # Load completed jobs (for history)
            if "completed_jobs" in state_data:
                for job_data in state_data["completed_jobs"]:
                    try:
                        job = LearningJob(
                            id=UUID(job_data["id"]),
                            config=self.config,
                            status=job_data["status"],
                            created_at=datetime.fromisoformat(job_data["created_at"]),
                            completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data["completed_at"] else None,
                            adapter_path=job_data.get("adapter_path"),
                        )

                        if job_data.get("metrics"):
                            job.metrics = TrainingMetrics(**job_data["metrics"])

                        self.completed_jobs.append(job)

                    except Exception as e:
                        self.logger.warning(f"Failed to load job data: {e}")

            self.logger.info("Scheduler state loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load scheduler state: {e}")

    async def force_training(self, priority: int = 10) -> UUID:
        """Force immediate training regardless of schedule."""
        return await self.schedule_training(priority=priority, scheduled_at=datetime.now())

    def get_next_scheduled_training(self) -> Optional[datetime]:
        """Get the next scheduled training time."""
        if self.job_queue:
            return min(job.scheduled_at for job in self.job_queue)

        # Calculate next automatic training
        if self.state.last_training:
            next_auto = self.state.last_training + timedelta(hours=self.config.training_interval_hours)
            return next_auto

        return None