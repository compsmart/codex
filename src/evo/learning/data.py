"""
Data processing and dataset management for learning.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from ..memory import MemoryManager, MemoryQuery, MemoryType
from .models import DataSample, LearningConfig


class ConversationDataProcessor:
    """Processes conversation data for training."""

    def __init__(self, memory_manager: MemoryManager, config: LearningConfig):
        self.memory_manager = memory_manager
        self.config = config

    async def extract_training_data(
        self,
        min_importance: float = 0.5,
        max_age_days: int = 30,
        max_samples: int = 1000,
    ) -> List[DataSample]:
        """Extract training data from memory."""
        samples = []

        # Get episodic memories (conversations)
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        # Query for high-quality episodic memories
        query = MemoryQuery(
            text="*",  # Match all
            memory_types=[MemoryType.EPISODIC],
            limit=max_samples * 2,  # Get more to filter
            min_importance=min_importance,
        )

        results = await self.memory_manager.search(query)

        for result in results:
            memory = result.memory

            # Filter by age
            if memory.created_at < cutoff_date:
                continue

            # Filter by importance
            if memory.importance < min_importance:
                continue

            # Extract conversation data
            if hasattr(memory, 'user_message') and hasattr(memory, 'assistant_response'):
                if memory.user_message and memory.assistant_response:
                    sample = DataSample(
                        input_text=memory.user_message,
                        target_text=memory.assistant_response,
                        importance=memory.importance,
                        created_at=memory.created_at,
                        source_memory_id=memory.id,
                        metadata={
                            "memory_type": memory.type.value,
                            "session_id": getattr(memory, 'session_id', None),
                            "context_summary": getattr(memory, 'context_summary', None),
                        },
                    )
                    samples.append(sample)

        # Sort by importance and take top samples
        samples.sort(key=lambda x: x.importance, reverse=True)
        return samples[:max_samples]

    async def create_instruction_data(
        self, samples: List[DataSample]
    ) -> List[Dict[str, str]]:
        """Create instruction-following training data."""
        instruction_data = []

        for sample in samples:
            # Create instruction format
            instruction = {
                "instruction": "You are Evo, a helpful AI assistant that learns and remembers. Respond to the user's message helpfully and naturally.",
                "input": sample.input_text,
                "output": sample.target_text,
            }

            instruction_data.append(instruction)

        return instruction_data

    async def augment_data(
        self, samples: List[DataSample], augmentation_factor: float = 1.5
    ) -> List[DataSample]:
        """Augment training data with variations."""
        augmented = list(samples)

        target_count = int(len(samples) * augmentation_factor)
        needed = target_count - len(samples)

        if needed <= 0:
            return augmented

        # Simple augmentation: paraphrase system messages
        system_variations = [
            "You are Evo, an AI assistant that learns from conversations.",
            "You are Evo, a helpful assistant with memory capabilities.",
            "You are Evo, an intelligent assistant that remembers and learns.",
            "You are Evo, a learning AI that helps users with their questions.",
        ]

        for i in range(needed):
            # Pick a random sample to augment
            base_sample = random.choice(samples)

            # Create variation with different system context
            variation = DataSample(
                input_text=base_sample.input_text,
                target_text=base_sample.target_text,
                importance=base_sample.importance * 0.9,  # Slightly lower importance
                metadata={
                    **base_sample.metadata,
                    "augmented": True,
                    "base_sample_id": str(base_sample.id),
                },
            )

            augmented.append(variation)

        return augmented

    def filter_quality_data(
        self, samples: List[DataSample], min_length: int = 10, max_length: int = 2000
    ) -> List[DataSample]:
        """Filter samples for quality."""
        filtered = []

        for sample in samples:
            # Length checks
            if len(sample.input_text) < min_length:
                continue
            if len(sample.target_text) < min_length:
                continue
            if len(sample.input_text) > max_length:
                continue
            if len(sample.target_text) > max_length:
                continue

            # Quality checks
            if self._is_low_quality(sample.input_text) or self._is_low_quality(
                sample.target_text
            ):
                continue

            filtered.append(sample)

        return filtered

    def _is_low_quality(self, text: str) -> bool:
        """Check if text is low quality."""
        # Simple quality checks
        if len(text.strip()) < 10:
            return True

        # Check for repetitive text
        words = text.lower().split()
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:  # Too repetitive
                return True

        # Check for mostly non-alphabetic characters
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.3:
            return True

        return False


class LearningDataset:
    """Dataset for training with tokenization."""

    def __init__(
        self,
        data: List[DataSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        system_prompt: str = "You are Evo, a helpful AI assistant that learns and remembers.",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

        # Prepare the dataset
        self.dataset = self._prepare_dataset()

    def _prepare_dataset(self) -> Dataset:
        """Prepare the dataset for training."""
        formatted_data = []

        for sample in self.data:
            # Create conversation format
            conversation = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": sample.input_text},
                {"role": "assistant", "content": sample.target_text},
            ]

            # Format as training text
            formatted_text = self._format_conversation(conversation)

            formatted_data.append({
                "text": formatted_text,
                "sample_id": str(sample.id),
                "importance": sample.importance,
                "metadata": sample.metadata,
            })

        return Dataset.from_list(formatted_data)

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for training."""
        # Use a simple format
        formatted_parts = []

        for message in conversation:
            role = message["role"]
            content = message["content"]

            if role == "system":
                formatted_parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                formatted_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{content}\n")

        return "".join(formatted_parts)

    def tokenize_dataset(self) -> Dataset:
        """Tokenize the dataset."""

        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        return self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
        )

    def train_test_split(self, test_size: float = 0.1) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and test sets."""
        dataset = self.tokenize_dataset()

        if len(dataset) < 10:
            # Too small for split, use all for training
            return dataset, dataset.select(range(min(2, len(dataset))))

        split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        return split_dataset["train"], split_dataset["test"]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data:
            return {"total_samples": 0}

        # Calculate statistics
        input_lengths = [len(sample.input_text) for sample in self.data]
        output_lengths = [len(sample.target_text) for sample in self.data]
        importances = [sample.importance for sample in self.data]

        return {
            "total_samples": len(self.data),
            "avg_input_length": sum(input_lengths) / len(input_lengths),
            "avg_output_length": sum(output_lengths) / len(output_lengths),
            "avg_importance": sum(importances) / len(importances),
            "min_importance": min(importances),
            "max_importance": max(importances),
            "unique_sessions": len(
                set(
                    sample.metadata.get("session_id")
                    for sample in self.data
                    if sample.metadata.get("session_id")
                )
            ),
        }

    def save_to_file(self, file_path: str) -> None:
        """Save dataset to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data_dict = {
            "samples": [
                {
                    "id": str(sample.id),
                    "input_text": sample.input_text,
                    "target_text": sample.target_text,
                    "importance": sample.importance,
                    "created_at": sample.created_at.isoformat(),
                    "metadata": sample.metadata,
                }
                for sample in self.data
            ],
            "config": {
                "max_length": self.max_length,
                "system_prompt": self.system_prompt,
            },
            "stats": self.get_stats(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(
        cls, file_path: str, tokenizer: PreTrainedTokenizer
    ) -> "LearningDataset":
        """Load dataset from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)

        # Reconstruct samples
        samples = []
        for sample_data in data_dict["samples"]:
            sample = DataSample(
                id=UUID(sample_data["id"]),
                input_text=sample_data["input_text"],
                target_text=sample_data["target_text"],
                importance=sample_data["importance"],
                created_at=datetime.fromisoformat(sample_data["created_at"]),
                metadata=sample_data["metadata"],
            )
            samples.append(sample)

        config = data_dict.get("config", {})
        return cls(
            data=samples,
            tokenizer=tokenizer,
            max_length=config.get("max_length", 2048),
            system_prompt=config.get(
                "system_prompt",
                "You are Evo, a helpful AI assistant that learns and remembers.",
            ),
        )