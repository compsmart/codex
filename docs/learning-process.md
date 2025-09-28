# Learning Process Documentation

This document explains how Evo AI learns and adapts over time, including the QLoRA fine-tuning process, data collection, and model updates.

## 🧠 Learning Architecture Overview

Evo AI uses a sophisticated learning architecture that keeps the base model frozen while learning through parameter-efficient fine-tuning.

### **Static Foundation + Dynamic Learning**
```
┌─────────────────────────────────────────────────┐
│                Evo AI Learning                  │
├─────────────────────────────────────────────────┤
│  LoRA Adapters (Learning Layer)                │
│  ├── Personal Preferences    (16M params)      │
│  ├── Coding Knowledge       (16M params)       │
│  └── Domain Expertise       (16M params)       │
├─────────────────────────────────────────────────┤
│  Llama 3.2-7B Base Model (Frozen)             │
│  └── 7 Billion Parameters (Never Changes)      │
└─────────────────────────────────────────────────┘
```

## 🔄 Learning Lifecycle

### **Phase 1: Conversation Collection**
```python
# As you chat with Evo, conversations are stored as episodic memories
episodic_memory = EpisodicMemory(
    content="Discussion about Rust vs Python performance",
    session_id="session_123",
    user_message="Is Rust really faster than Python?",
    assistant_response="Yes, Rust is significantly faster because...",
    importance=0.8,  # High importance = good learning data
    created_at=datetime.now()
)
```

### **Phase 2: Learning Triggers**
Learning is triggered by multiple conditions:

#### **Time-Based Learning**
```python
# Default: Every 6 hours
if hours_since_last_training >= config.training_interval_hours:
    await schedule_learning()
```

#### **Data-Volume Learning**
```python
# When enough quality conversations accumulate
quality_conversations = await get_recent_conversations(
    min_importance=0.5,
    min_length=50,
    max_age_days=7
)

if len(quality_conversations) >= config.min_data_points:
    await schedule_learning()
```

#### **Manual Learning**
```python
# User can trigger immediate learning
evo teach "I prefer detailed technical explanations"
# or
evo learn --force
```

### **Phase 3: Data Processing Pipeline**

#### **Step 1: Conversation Analysis**
```python
async def extract_training_data(self) -> List[DataSample]:
    # Get recent high-quality conversations
    memories = await self.memory_manager.search(MemoryQuery(
        memory_types=[MemoryType.EPISODIC],
        min_importance=0.5,
        max_age_days=30,
        limit=1000
    ))

    samples = []
    for memory in memories:
        if self._is_learning_worthy(memory):
            sample = DataSample(
                input_text=memory.user_message,
                target_text=memory.assistant_response,
                importance=memory.importance,
                metadata=memory.metadata
            )
            samples.append(sample)

    return samples
```

#### **Step 2: Quality Filtering**
```python
def _is_learning_worthy(self, memory: EpisodicMemory) -> bool:
    # Length requirements
    if len(memory.user_message) < 10 or len(memory.assistant_response) < 20:
        return False

    # Importance threshold
    if memory.importance < self.config.importance_threshold:
        return False

    # Content quality checks
    if self._contains_errors(memory.assistant_response):
        return False

    # Preference/knowledge indicators
    learning_indicators = [
        "my favorite", "I prefer", "I like", "I work with",
        "remember that", "important to know", "always", "never"
    ]

    user_text = memory.user_message.lower()
    if any(indicator in user_text for indicator in learning_indicators):
        return True

    return memory.importance > 0.7  # High importance threshold
```

#### **Step 3: Data Augmentation**
```python
async def augment_data(self, samples: List[DataSample]) -> List[DataSample]:
    augmented = list(samples)

    # Create variations with different system prompts
    system_variations = [
        "You are Evo, a helpful AI assistant that learns from conversations.",
        "You are Evo, an intelligent assistant with memory capabilities.",
        "You are Evo, a learning AI that remembers user preferences."
    ]

    for sample in samples[:50]:  # Augment top 50 samples
        for system_prompt in system_variations:
            variation = DataSample(
                input_text=f"[System: {system_prompt}]\n{sample.input_text}",
                target_text=sample.target_text,
                importance=sample.importance * 0.9,
                metadata={**sample.metadata, "augmented": True}
            )
            augmented.append(variation)

    return augmented
```

### **Phase 4: QLoRA Training Process**

#### **Model Preparation**
```python
async def prepare_model_for_learning(self):
    # Load base model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_use_double_quant=True  # Double quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        "llama3.2:7b",
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=16,                          # Rank (adaptation capacity)
        lora_alpha=32,                 # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    return get_peft_model(model, lora_config)
```

#### **Training Configuration**
```python
training_args = TrainingArguments(
    output_dir=adapter_output_path,

    # Training parameters
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 16
    learning_rate=2e-4,             # Conservative learning rate
    num_train_epochs=1,             # Single epoch to prevent overfitting

    # Optimization
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Memory optimization
    fp16=True,                      # Half precision
    dataloader_pin_memory=False,
    gradient_checkpointing=True,

    # Monitoring
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

#### **Training Execution**
```python
async def train_adapter(self, dataset: LearningDataset) -> TrainingMetrics:
    # Split data
    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1)

    # Setup trainer
    trainer = Trainer(
        model=self.peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked
        )
    )

    # Train the adapter
    training_start = time.time()
    train_result = trainer.train()
    training_duration = time.time() - training_start

    # Calculate metrics
    metrics = TrainingMetrics(
        total_samples=len(dataset.data),
        train_samples=len(train_dataset),
        eval_samples=len(eval_dataset),
        final_loss=train_result.training_loss,
        total_steps=train_result.global_step,
        duration_seconds=training_duration,
        success=True
    )

    return metrics
```

### **Phase 5: Adapter Management**

#### **Saving Adapters**
```python
async def save_adapter(self, metrics: TrainingMetrics) -> str:
    timestamp = int(datetime.now().timestamp())
    adapter_name = f"adapter_{timestamp}"
    adapter_path = self.config.adapter_save_path / adapter_name

    # Save LoRA weights
    self.peft_model.save_pretrained(adapter_path)

    # Save training metadata
    training_info = {
        "timestamp": datetime.now().isoformat(),
        "model_base": self.model_name,
        "training_samples": metrics.total_samples,
        "final_loss": metrics.final_loss,
        "training_duration": metrics.duration_seconds,
        "config": self.config.dict()
    }

    with open(adapter_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    return str(adapter_path)
```

#### **Loading Adapters**
```python
async def load_latest_adapter(self) -> bool:
    adapter_dir = Path(self.config.adapter_save_path)

    if not adapter_dir.exists():
        return False

    # Find most recent adapter
    adapters = [d for d in adapter_dir.iterdir() if d.is_dir()]
    if not adapters:
        return False

    latest_adapter = max(adapters, key=lambda x: x.stat().st_mtime)

    # Load adapter into base model
    self.model = PeftModel.from_pretrained(
        self.base_model,
        str(latest_adapter)
    )

    logger.info(f"Loaded adapter: {latest_adapter.name}")
    return True
```

#### **Adapter Versioning**
```python
# Multiple adapters can be maintained
~/.evo/adapters/
├── adapter_1698765432/          # Personal preferences v1
├── adapter_1698851832/          # Personal preferences v2
├── adapter_1698938232/          # Added coding knowledge
├── adapter_1699024632/          # Added domain expertise
└── active -> adapter_1699024632  # Symlink to active adapter
```

## 📊 Learning Examples

### **Example 1: Personal Preference Learning**

#### **Input Conversations:**
```
Session 1:
User: "I prefer concise explanations over verbose ones"
Assistant: "I'll keep my responses concise and to the point."

Session 2:
User: "Can you explain machine learning?"
Assistant: [Gives long explanation]
User: "That's too detailed, can you summarize?"
Assistant: [Gives shorter version]
User: "Perfect, that's much better"

Session 3:
User: "I work primarily with Python and Rust"
Assistant: "Great! Both are excellent languages for different use cases."
```

#### **Extracted Learning Data:**
```python
training_samples = [
    {
        "input": "Explain machine learning",
        "output": "ML is algorithms that learn patterns from data to make predictions without explicit programming.",
        "importance": 0.8,
        "metadata": {"style": "concise", "user_feedback": "positive"}
    },
    {
        "input": "What programming languages do you recommend?",
        "output": "Based on your work with Python and Rust, I'd suggest...",
        "importance": 0.7,
        "metadata": {"personalized": True, "languages": ["python", "rust"]}
    }
]
```

#### **Post-Learning Behavior:**
```
User: "Explain neural networks"
Evo: "Neural networks are interconnected nodes that process data through weighted connections, learning patterns to make predictions. They're inspired by brain neurons and excel at pattern recognition tasks."

# Notice: Concise, technical, mentions both concepts (learning + patterns)
# Shows the adapter learned the user's preference for concise technical explanations
```

### **Example 2: Domain Knowledge Learning**

#### **Input: Extended Technical Discussion**
```
User: "I'm building a distributed system with microservices"
Assistant: "That's a great architecture choice. What challenges are you facing?"
User: "Service discovery and load balancing are tricky"
Assistant: "For service discovery, consider Consul or etcd. For load balancing..."
User: "We're using Kubernetes, so we have built-in service discovery"
Assistant: "Perfect! Kubernetes DNS handles service discovery automatically..."
[Extended technical discussion continues]
```

#### **Learning Outcome:**
```python
# The adapter learns:
learned_patterns = [
    "User works with Kubernetes and microservices",
    "User prefers practical, implementation-focused advice",
    "User has experience with distributed systems",
    "Technical depth is appreciated for architecture topics"
]
```

#### **Future Interactions:**
```
User: "How should I handle database scaling?"
Evo: "For your microservices architecture on Kubernetes, consider database per service pattern. You could use horizontal pod autoscaling for read replicas, and tools like Vitess for MySQL sharding if you're at scale."

# Notice: References their Kubernetes setup, assumes microservices context
# Provides architecture-level advice suitable for their experience level
```

## 🎯 Learning Quality Metrics

### **Training Metrics**
```python
class TrainingMetrics:
    # Data quality
    total_samples: int           # How much data was available
    train_samples: int          # How much was used for training
    eval_samples: int           # How much for validation

    # Training performance
    initial_loss: float         # Starting loss value
    final_loss: float          # Ending loss value
    best_loss: float           # Best validation loss
    improvement_ratio: float    # (initial - final) / initial

    # Training efficiency
    duration_seconds: float     # How long training took
    samples_per_second: float   # Training throughput
    peak_memory_mb: float       # Peak GPU memory usage

    # Quality indicators
    perplexity: float          # Language model quality
    validation_accuracy: float # Prediction accuracy
    overfitting_score: float   # Training vs validation gap
```

### **Performance Tracking**
```python
async def evaluate_learning_quality(self, adapter_path: str) -> Dict[str, float]:
    # Load the trained adapter
    model = load_adapter(adapter_path)

    # Test on held-out conversations
    test_conversations = await self.get_test_conversations()

    metrics = {}
    for conversation in test_conversations:
        # Generate response with new adapter
        response = await model.generate(conversation.user_message)

        # Compare with expected response
        similarity = calculate_similarity(response, conversation.expected_response)
        metrics[conversation.id] = similarity

    return {
        "average_similarity": np.mean(list(metrics.values())),
        "improvement_over_base": calculate_improvement(metrics),
        "consistency_score": calculate_consistency(metrics),
        "personalization_score": calculate_personalization(metrics)
    }
```

## ⚙️ Configuration and Tuning

### **Learning Parameters**
```python
class LearningConfig:
    # Training frequency
    training_interval_hours: int = 6        # How often to train
    min_data_points: int = 10              # Minimum conversations needed
    max_training_data_age_days: int = 30   # Only use recent data

    # Quality thresholds
    importance_threshold: float = 0.5       # Minimum conversation importance
    min_conversation_length: int = 100      # Minimum chars per conversation

    # LoRA parameters
    lora_rank: int = 16                    # Adaptation capacity (4, 8, 16, 32)
    lora_alpha: int = 32                   # Learning strength
    lora_dropout: float = 0.1              # Regularization

    # Training parameters
    learning_rate: float = 2e-4            # How fast to learn
    batch_size: int = 4                    # Memory vs speed tradeoff
    num_epochs: int = 1                    # Prevent overfitting
    max_length: int = 2048                 # Maximum conversation length

    # Resource management
    max_adapters: int = 5                  # How many to keep
    auto_cleanup: bool = True              # Remove old adapters
    gpu_memory_fraction: float = 0.8       # GPU usage limit
```

### **Tuning for Different Use Cases**

#### **Conservative Learning (Stable)**
```python
conservative_config = LearningConfig(
    lora_rank=8,                    # Lower adaptation
    learning_rate=1e-4,             # Slower learning
    importance_threshold=0.7,       # Higher quality bar
    training_interval_hours=12,     # Less frequent updates
    num_epochs=1                    # Single pass
)
```

#### **Aggressive Learning (Fast Adaptation)**
```python
aggressive_config = LearningConfig(
    lora_rank=32,                   # Higher adaptation
    learning_rate=5e-4,             # Faster learning
    importance_threshold=0.3,       # Lower quality bar
    training_interval_hours=2,      # Frequent updates
    num_epochs=2                    # Multiple passes
)
```

#### **Resource-Constrained (Low Memory)**
```python
efficient_config = LearningConfig(
    lora_rank=4,                    # Minimal adaptation
    batch_size=1,                   # Tiny batches
    max_length=1024,                # Shorter contexts
    gradient_accumulation_steps=8,   # Simulate larger batches
    use_4bit=True                   # Maximum quantization
)
```

## 🔍 Monitoring and Debugging

### **Learning Dashboard**
```bash
# Check learning status
evo status --learning

# View recent training runs
evo learning history

# See current adapter info
evo learning adapter-info

# Monitor resource usage during training
evo monitor --gpu --memory
```

### **Learning Logs**
```python
# Detailed training logs
2024-01-15 14:30:15 INFO Starting learning job job_123
2024-01-15 14:30:16 INFO Extracted 45 training samples (avg importance: 0.73)
2024-01-15 14:30:17 INFO Data quality: 42 high-quality, 3 filtered out
2024-01-15 14:30:18 INFO Starting QLoRA training...
2024-01-15 14:30:19 INFO Epoch 1/1: loss=2.341 (baseline: 2.567)
2024-01-15 14:45:23 INFO Training complete: final_loss=1.892 (26% improvement)
2024-01-15 14:45:24 INFO Saved adapter: ~/.evo/adapters/adapter_1705329924
2024-01-15 14:45:25 INFO Learning job completed successfully
```

### **Debugging Poor Learning**
```python
# Common issues and solutions:

if final_loss > initial_loss:
    # Model is not learning
    solutions = [
        "Increase learning rate",
        "Check data quality",
        "Reduce LoRA rank",
        "Increase training epochs"
    ]

if overfitting_detected():
    # Model memorizing instead of generalizing
    solutions = [
        "Reduce learning rate",
        "Add more diverse training data",
        "Increase LoRA dropout",
        "Use single epoch only"
    ]

if responses_degraded():
    # Adapter hurting base performance
    solutions = [
        "Lower importance threshold",
        "Reduce LoRA alpha",
        "Use more conservative config",
        "Revert to previous adapter"
    ]
```

This learning system enables Evo AI to continuously improve while maintaining the stability and knowledge of the base model. The key insight is that learning happens in small, efficient layers that preserve the original model's capabilities while adding personalized knowledge and preferences.