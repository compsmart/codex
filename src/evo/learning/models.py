"""
Models and data structures for the learning system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LearningPhase(str, Enum):
    """Learning phases."""

    IDLE = "idle"
    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    VALIDATION = "validation"
    SAVING = "saving"
    ERROR = "error"


class LearningConfig(BaseModel):
    """Configuration for the learning system."""

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training parameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_length: int = 2048
    num_epochs: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Optimization
    optimizer_type: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # QLoRA-specific
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # Data configuration
    min_data_points: int = 10
    max_training_data_age_days: int = 30
    importance_threshold: float = 0.5

    # Training frequency
    training_interval_hours: int = 6
    auto_save_steps: int = 100
    evaluation_steps: int = 50

    # Model management
    max_adapters: int = 5
    adapter_save_path: str = "~/.evo/adapters"


class TrainingMetrics(BaseModel):
    """Training metrics and statistics."""

    training_id: UUID = Field(default_factory=uuid4)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Data metrics
    total_samples: int = 0
    train_samples: int = 0
    eval_samples: int = 0

    # Training metrics
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    total_steps: int = 0
    epochs_completed: float = 0.0

    # Performance metrics
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    samples_per_second: Optional[float] = None
    tokens_per_second: Optional[float] = None

    # Memory usage
    peak_memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None

    # Validation metrics
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None

    # Status
    phase: LearningPhase = LearningPhase.IDLE
    error_message: Optional[str] = None
    success: bool = False

    def mark_completed(self, success: bool = True) -> None:
        """Mark training as completed."""
        self.completed_at = datetime.now()
        self.success = success
        if self.started_at and self.completed_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def update_phase(self, phase: LearningPhase, message: Optional[str] = None) -> None:
        """Update current phase."""
        self.phase = phase
        if phase == LearningPhase.ERROR and message:
            self.error_message = message


class LearningJob(BaseModel):
    """A learning job with data and configuration."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    config: LearningConfig
    data_query: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority

    # Results
    metrics: Optional[TrainingMetrics] = None
    adapter_path: Optional[str] = None
    model_checkpoint: Optional[str] = None

    # Status
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None

    @property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.status in ["completed", "failed"]

    @property
    def is_running(self) -> bool:
        """Check if job is running."""
        return self.status == "running"

    def start(self) -> None:
        """Mark job as started."""
        self.status = "running"
        self.started_at = datetime.now()

    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark job as completed."""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.now()
        if error:
            self.error_message = error


class DataSample(BaseModel):
    """A training data sample."""

    id: UUID = Field(default_factory=uuid4)
    input_text: str
    target_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = Field(default_factory=datetime.now)
    source_memory_id: Optional[UUID] = None

    def to_training_format(self) -> Dict[str, str]:
        """Convert to training format."""
        return {
            "input": self.input_text,
            "output": self.target_text,
        }


class LearningState(BaseModel):
    """Current state of the learning system."""

    is_enabled: bool = True
    is_training: bool = False
    current_job: Optional[UUID] = None
    last_training: Optional[datetime] = None
    next_training: Optional[datetime] = None

    # Statistics
    total_training_runs: int = 0
    total_adapters_created: int = 0
    total_training_time_hours: float = 0.0
    average_loss_improvement: float = 0.0

    # Current adapter info
    active_adapter_path: Optional[str] = None
    active_adapter_created: Optional[datetime] = None
    adapter_performance_score: float = 0.0

    # Queue status
    pending_jobs: int = 0
    failed_jobs: int = 0

    def update_training_stats(self, metrics: TrainingMetrics) -> None:
        """Update training statistics."""
        self.total_training_runs += 1
        if metrics.duration_seconds:
            self.total_training_time_hours += metrics.duration_seconds / 3600

        # Update loss improvement
        if metrics.initial_loss and metrics.final_loss:
            improvement = (metrics.initial_loss - metrics.final_loss) / metrics.initial_loss
            if self.average_loss_improvement == 0:
                self.average_loss_improvement = improvement
            else:
                # Exponential moving average
                self.average_loss_improvement = (
                    0.9 * self.average_loss_improvement + 0.1 * improvement
                )