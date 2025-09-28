"""
QLoRA trainer for efficient fine-tuning of language models.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .data import LearningDataset
from .models import LearningConfig, TrainingMetrics, LearningPhase


class QLoRATrainer:
    """QLoRA trainer for efficient fine-tuning."""

    def __init__(self, config: LearningConfig, model_name: str = "microsoft/DialoGPT-medium"):
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.peft_model = None
        self.trainer: Optional[Trainer] = None

        # Training state
        self.current_metrics: Optional[TrainingMetrics] = None

    async def initialize(self) -> None:
        """Initialize the trainer."""
        try:
            self.logger.info(f"Initializing QLoRA trainer with model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right",
            )

            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Add custom tokens for conversation format
            special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
            self.tokenizer.add_tokens(special_tokens)

            self.logger.info("QLoRA trainer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize QLoRA trainer: {e}")
            raise

    def _setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup 4-bit quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )

    def _load_model(self) -> None:
        """Load and prepare the model for training."""
        self.logger.info("Loading base model...")

        # Setup quantization
        bnb_config = self._setup_quantization_config()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Resize token embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        self.logger.info("Base model loaded and prepared")

    def _setup_lora(self) -> None:
        """Setup LoRA configuration."""
        self.logger.info("Setting up LoRA configuration...")

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())

        self.logger.info(
            f"LoRA setup complete. Trainable parameters: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}% of total)"
        )

    async def train(
        self,
        dataset: LearningDataset,
        output_dir: Optional[str] = None,
    ) -> TrainingMetrics:
        """Train the model with QLoRA."""
        self.current_metrics = TrainingMetrics()
        self.current_metrics.update_phase(LearningPhase.DATA_PREPARATION)

        try:
            # Setup output directory
            if output_dir is None:
                output_dir = Path(self.config.adapter_save_path).expanduser() / f"adapter_{int(time.time())}"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Load model if not already loaded
            if self.model is None:
                self._load_model()
                self._setup_lora()

            # Prepare training data
            train_dataset, eval_dataset = dataset.train_test_split()

            self.current_metrics.total_samples = len(dataset.data)
            self.current_metrics.train_samples = len(train_dataset)
            self.current_metrics.eval_samples = len(eval_dataset)

            self.logger.info(f"Training on {self.current_metrics.train_samples} samples")

            # Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
                pad_to_multiple_of=8,
            )

            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=str(output_path),
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.num_epochs,
                lr_scheduler_type=self.config.lr_scheduler_type,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_steps=10,
                save_steps=self.config.auto_save_steps,
                eval_steps=self.config.evaluation_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,  # Disable wandb/tensorboard
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                fp16=True,
                max_grad_norm=self.config.max_grad_norm,
                adam_beta1=self.config.adam_beta1,
                adam_beta2=self.config.adam_beta2,
                adam_epsilon=self.config.adam_epsilon,
            )

            # Create trainer
            self.trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=[TrainingMetricsCallback(self.current_metrics)],
            )

            # Start training
            self.current_metrics.update_phase(LearningPhase.TRAINING)
            self.logger.info("Starting training...")

            train_result = self.trainer.train()

            # Update metrics
            self.current_metrics.final_loss = train_result.training_loss
            self.current_metrics.total_steps = train_result.global_step
            self.current_metrics.epochs_completed = train_result.epoch

            # Save adapter
            self.current_metrics.update_phase(LearningPhase.SAVING)
            adapter_path = output_path / "adapter"
            self.peft_model.save_pretrained(adapter_path)
            self.tokenizer.save_pretrained(adapter_path)

            # Save training info
            training_info = {
                "model_name": self.model_name,
                "config": self.config.dict(),
                "metrics": self.current_metrics.dict(),
                "dataset_stats": dataset.get_stats(),
            }

            import json
            with open(output_path / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2, default=str)

            self.current_metrics.mark_completed(success=True)
            self.logger.info(f"Training completed successfully. Adapter saved to: {adapter_path}")

            return self.current_metrics

        except Exception as e:
            self.current_metrics.update_phase(LearningPhase.ERROR, str(e))
            self.current_metrics.mark_completed(success=False)
            self.logger.error(f"Training failed: {e}")
            raise

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not torch.cuda.is_available():
            return {"gpu_memory_mb": 0.0, "peak_memory_mb": 0.0}

        # Get GPU memory usage
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        return {
            "gpu_memory_mb": gpu_memory,
            "peak_memory_mb": peak_memory,
        }

    def cleanup(self) -> None:
        """Cleanup model and free memory."""
        if self.trainer:
            del self.trainer
            self.trainer = None

        if self.peft_model:
            del self.peft_model
            self.peft_model = None

        if self.model:
            del self.model
            self.model = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("QLoRA trainer cleaned up")

    async def evaluate_adapter(
        self,
        adapter_path: str,
        test_dataset: LearningDataset,
    ) -> Dict[str, float]:
        """Evaluate a trained adapter."""
        try:
            # Load the adapter
            from peft import PeftModel

            # Load base model
            if self.model is None:
                self._load_model()

            # Load adapter
            adapter_model = PeftModel.from_pretrained(self.model, adapter_path)

            # Prepare test data
            _, test_data = test_dataset.train_test_split()

            # Setup trainer for evaluation
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            eval_args = TrainingArguments(
                output_dir="/tmp/eval",
                per_device_eval_batch_size=self.config.batch_size,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
            )

            evaluator = Trainer(
                model=adapter_model,
                args=eval_args,
                eval_dataset=test_data,
                data_collator=data_collator,
            )

            # Run evaluation
            eval_results = evaluator.evaluate()

            # Calculate perplexity
            perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()

            return {
                "eval_loss": eval_results["eval_loss"],
                "perplexity": perplexity,
                "eval_samples": len(test_data),
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}


class TrainingMetricsCallback:
    """Callback to update training metrics during training."""

    def __init__(self, metrics: TrainingMetrics):
        self.metrics = metrics

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.metrics.learning_rate = args.learning_rate

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging."""
        if logs:
            if "train_loss" in logs:
                if self.metrics.initial_loss is None:
                    self.metrics.initial_loss = logs["train_loss"]

            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                if self.metrics.best_loss is None or eval_loss < self.metrics.best_loss:
                    self.metrics.best_loss = eval_loss

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self.metrics.total_steps = state.global_step