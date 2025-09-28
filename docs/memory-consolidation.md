# Memory Consolidation Process

Memory consolidation is one of Evo AI's most sophisticated features, transforming raw conversations into structured, long-term knowledge. This process mimics how human brains consolidate memories during sleep.

## 🧠 Overview

Memory consolidation converts:
- **Episodic memories** (individual conversations) → **Semantic memories** (facts and knowledge)
- **Raw conversations** → **Structured facts**
- **Scattered information** → **Organized knowledge graphs**

## 🔄 Consolidation Triggers

### 1. Automatic Triggers

#### Time-Based Consolidation
```python
# Default: Every 12 hours
consolidation_interval_hours: int = 12

# Checks if enough time has passed
if time_since_last > consolidation_interval_hours * 3600:
    await self.schedule_consolidation()
```

#### Data-Volume Triggers
```python
# When enough new conversations accumulate
if len(new_conversations) >= min_data_points:
    await self.schedule_consolidation()
```

**Threshold Configuration:**
```python
min_data_points: int = 10  # Minimum conversations needed
importance_threshold: float = 0.5  # Quality filter
max_training_data_age_days: int = 30  # Only recent data
```

### 2. Manual Triggers

#### Session-Based Consolidation
```python
# After important conversations
await agent.consolidate_session_memories(session_id)
```

#### Forced Consolidation
```python
# Immediate consolidation
await consolidation_service.schedule_consolidation(delay_minutes=0)
```

### 3. Coordinated Triggers

#### Post-Learning Consolidation
```python
# 30 minutes after QLoRA training completes
if learning_completed:
    await consolidation_service.schedule_consolidation(delay_minutes=30)
```

## 📊 Four-Stage Process

### Stage 1: Episodic Memory Analysis

#### Session Grouping
```python
async def _consolidate_episodic_memories(self, cutoff_time):
    # Group memories by conversation session
    session_memories = {}
    for memory in recent_memories:
        session_id = memory.session_id
        if session_id not in session_memories:
            session_memories[session_id] = []
        session_memories[session_id].append(memory)

    # Process each session separately
    for session_id, memories in session_memories.items():
        await self._extract_session_knowledge(session_id, memories)
```

#### Pattern Recognition
The system analyzes conversations for:
- **Preference statements**: "I prefer X over Y"
- **Personal information**: "My name is...", "I work as..."
- **Learning indicators**: "I learned...", "Now I understand..."
- **Factual statements**: "X is Y", "A can do B"

#### Example Analysis
```
Input Conversation:
User: "Hi, my name is Sarah and I'm a data scientist"
Assistant: "Nice to meet you Sarah!"
User: "I prefer Python over R for data analysis"
Assistant: "Python is indeed popular..."

Extracted Patterns:
✓ Personal info: name = Sarah
✓ Professional info: job = data scientist
✓ Preference: Python > R for data analysis
```

### Stage 2: Fact Extraction

#### Pattern-Based Extraction
```python
fact_patterns = [
    # Personal information patterns
    (r"my name is (.+?)[\.\!\?]", "personal", "user's name", "is", "{}"),
    (r"i am (.+?)[\.\!\?]", "personal", "user", "is", "{}"),
    (r"i work (.+?)[\.\!\?]", "personal", "user", "works", "{}"),

    # Preference patterns
    (r"my favorite (.+?) is (.+?)[\.\!\?]", "preference", "user's favorite {}", "is", "{}"),
    (r"i prefer (.+?) over (.+?)[\.\!\?]", "preference", "user preference", "prefers", "{} over {}"),
    (r"i like (.+?)[\.\!\?]", "preference", "user", "likes", "{}"),
    (r"i don't like (.+?)[\.\!\?]", "preference", "user", "dislikes", "{}"),

    # Knowledge patterns
    (r"(.+?) is (.+?)[\.\!\?]", "knowledge", "{}", "is", "{}"),
    (r"(.+?) can (.+?)[\.\!\?]", "knowledge", "{}", "can", "{}"),
]
```

#### Fact Creation Process
```python
for pattern, fact_type, subject_template, predicate, object_template in fact_patterns:
    matches = re.finditer(pattern, conversation_text.lower())
    for match in matches:
        fact = FactExtraction(
            fact_text=match.group(0),
            subject=format_subject(subject_template, match.groups()),
            predicate=predicate,
            object=format_object(object_template, match.groups()),
            fact_type=fact_type,
            confidence=calculate_confidence(match, context),
            importance=calculate_importance(fact_type, content),
            source_context=get_surrounding_context(match)
        )
```

#### Quality Filtering
```python
# Only store high-confidence facts
if fact.confidence >= config.fact_confidence_threshold:
    await memory_manager.store_semantic_memory(
        content=fact.fact_text,
        subject=fact.subject,
        predicate=fact.predicate,
        object=fact.object,
        confidence=fact.confidence,
        source=f"fact_extraction_{session_id}",
        importance=fact.importance
    )
```

### Stage 3: Conversation Summarization

#### AI-Powered Summarization
```python
# Uses transformer models for intelligent summarization
async def _generate_summary(self, text: str) -> str:
    # Tokenize and validate length
    tokens = self.tokenizer.encode(text)
    if len(tokens) > max_input_length:
        text = self.tokenizer.decode(tokens[:max_input_length])

    # Generate summary using BART or similar model
    result = await self.summarizer(text)
    return result[0]["summary_text"]
```

#### Key Point Extraction
```python
async def extract_key_points(self, text: str) -> List[str]:
    sentences = split_into_sentences(text)
    scored_sentences = []

    for sentence in sentences:
        score = self._score_sentence_importance(sentence)
        scored_sentences.append((sentence, score))

    # Return top-scoring sentences as key points
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored_sentences[:5]]

def _score_sentence_importance(self, sentence: str) -> float:
    score = 0.0

    # Length factor (optimal 10-30 words)
    word_count = len(sentence.split())
    if 10 <= word_count <= 30:
        score += 0.3

    # Important keywords
    important_words = ["important", "key", "main", "essential", "crucial"]
    for word in important_words:
        if word in sentence.lower():
            score += 0.2

    # Information density
    if any(marker in sentence.lower() for marker in ["is", "are", "can", "will"]):
        score += 0.1

    return score
```

#### Topic Classification
```python
async def extract_topics(self, text: str) -> List[str]:
    topic_keywords = {
        "programming": ["code", "function", "algorithm", "debug", "software"],
        "learning": ["learn", "understand", "explain", "teach", "study"],
        "preferences": ["favorite", "prefer", "like", "dislike", "enjoy"],
        "personal": ["my", "i am", "i work", "i live", "myself"],
        "technical": ["system", "technology", "computer", "hardware"],
        "problem_solving": ["problem", "solution", "fix", "resolve", "issue"]
    }

    detected_topics = []
    text_lower = text.lower()

    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_topics.append(topic)

    return detected_topics[:max_topics_per_conversation]
```

### Stage 4: Knowledge Organization

#### Semantic Clustering
```python
async def _organize_semantic_memories(self) -> int:
    # Get all semantic memories
    semantic_memories = await self.get_semantic_memories()

    # Cluster by subject similarity
    clusters = {}
    for memory in semantic_memories:
        subject_key = self._normalize_subject(memory.subject)
        if subject_key not in clusters:
            clusters[subject_key] = []
        clusters[subject_key].append(memory)

    # Update importance scores based on clustering
    for cluster_memories in clusters.values():
        if len(cluster_memories) > 1:
            # Boost importance for frequently mentioned subjects
            importance_boost = min(0.2, len(cluster_memories) * 0.05)
            for memory in cluster_memories:
                memory.importance = min(1.0, memory.importance + importance_boost)
```

#### Subject Normalization
```python
def _normalize_subject(self, subject: str) -> str:
    normalized = subject.lower().strip()

    # Remove common prefixes
    prefixes = ["user's", "the", "a", "an"]
    for prefix in prefixes:
        if normalized.startswith(prefix + " "):
            normalized = normalized[len(prefix) + 1:]

    # Handle possessives
    if normalized.endswith("'s"):
        normalized = normalized[:-2]

    return normalized
```

## 📈 Example Complete Flow

### Input: Raw Conversation Data
```
Session ID: abc123
Messages:
1. User: "Hello, my name is Alex and I'm a software engineer"
2. Assistant: "Nice to meet you Alex! What kind of software do you work on?"
3. User: "I mainly work with Python and JavaScript. I prefer Python for data processing"
4. Assistant: "Python is excellent for data processing with libraries like pandas and numpy"
5. User: "Yes! I'm currently learning about machine learning with scikit-learn"
6. Assistant: "That's great! Scikit-learn is a fantastic library for ML beginners"
```

### Stage 1: Session Analysis
```
Session: abc123
Message Count: 6
Participants: User (Alex), Assistant
Duration: ~5 minutes
Topics Identified: programming, learning, personal_info
```

### Stage 2: Fact Extraction
```python
extracted_facts = [
    FactExtraction(
        fact_text="my name is Alex",
        subject="user's name",
        predicate="is",
        object="Alex",
        fact_type="personal",
        confidence=0.95,
        importance=0.9
    ),
    FactExtraction(
        fact_text="I'm a software engineer",
        subject="user's profession",
        predicate="is",
        object="software engineer",
        fact_type="personal",
        confidence=0.9,
        importance=0.8
    ),
    FactExtraction(
        fact_text="I prefer Python for data processing",
        subject="user's preference",
        predicate="prefers",
        object="Python for data processing",
        fact_type="preference",
        confidence=0.85,
        importance=0.7
    ),
    FactExtraction(
        fact_text="I'm currently learning about machine learning",
        subject="user's current learning",
        predicate="is learning",
        object="machine learning",
        fact_type="learning",
        confidence=0.8,
        importance=0.75
    )
]
```

### Stage 3: Summarization
```python
conversation_summary = ConversationSummary(
    session_id="abc123",
    summary_text="Alex introduced himself as a software engineer who works primarily with Python and JavaScript. He prefers Python for data processing and is currently learning machine learning using scikit-learn.",
    key_points=[
        "User's name is Alex",
        "Works as a software engineer",
        "Uses Python and JavaScript",
        "Prefers Python for data processing",
        "Learning machine learning with scikit-learn"
    ],
    topics=["programming", "learning", "personal"],
    importance_score=0.8,
    compression_ratio=0.15  # 85% compression
)
```

### Stage 4: Knowledge Storage
```python
# Stored as semantic memories:
semantic_memories = [
    SemanticMemory(
        content="The user's name is Alex",
        subject="user's name",
        predicate="is",
        object="Alex",
        confidence=0.95,
        source="consolidation_abc123",
        importance=0.9
    ),
    SemanticMemory(
        content="Alex works as a software engineer",
        subject="Alex's profession",
        predicate="is",
        object="software engineer",
        confidence=0.9,
        source="consolidation_abc123",
        importance=0.8
    ),
    # ... additional memories
]

# Stored as procedural memory:
procedural_memory = ProceduralMemory(
    content="Alex is learning machine learning with scikit-learn",
    skill_name="machine_learning",
    steps=["Use scikit-learn library", "Learn ML concepts", "Practice with data"],
    importance=0.75,
    proficiency_level="beginner"
)
```

## 🎯 Intelligence Features

### Importance Calculation
```python
async def _calculate_importance(self, text: str) -> float:
    importance = 0.5  # Base importance
    text_lower = text.lower()

    # Personal information boost
    personal_keywords = ["my name", "i am", "i work", "i live"]
    for keyword in personal_keywords:
        if keyword in text_lower:
            importance += 0.2
            break

    # Preference boost
    preference_keywords = ["favorite", "prefer", "like", "dislike", "love", "hate"]
    for keyword in preference_keywords:
        if keyword in text_lower:
            importance += 0.15
            break

    # Learning boost
    learning_keywords = ["learn", "understand", "study", "practice", "master"]
    for keyword in learning_keywords:
        if keyword in text_lower:
            importance += 0.1
            break

    # Length factor
    word_count = len(text.split())
    if word_count > 200:
        importance += 0.1
    if word_count > 500:
        importance += 0.1

    # Technical content
    technical_keywords = ["code", "function", "algorithm", "debug", "implement"]
    tech_count = sum(1 for keyword in technical_keywords if keyword in text_lower)
    importance += min(0.15, tech_count * 0.05)

    return min(importance, 1.0)  # Cap at 1.0
```

### Context Preservation
```python
# Maintains source context for traceability
fact.source_context = text[max(0, match.start() - 50):match.end() + 50]
fact.source_memory_id = episodic_memory.id
fact.conversation_id = summary.id
```

### Quality Assurance
```python
def _is_low_quality(self, text: str) -> bool:
    # Length check
    if len(text.strip()) < 10:
        return True

    # Repetition check
    words = text.lower().split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:  # Too repetitive
            return True

    # Content check
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.3:  # Too much non-text content
        return True

    return False
```

## 📊 Monitoring and Statistics

### Consolidation Metrics
```python
stats = {
    "consolidations_performed": 0,
    "memories_consolidated": 0,
    "facts_extracted": 0,
    "summaries_created": 0,
    "last_consolidation": None,
    "average_compression_ratio": 0.0,
    "quality_score": 0.0
}
```

### Performance Tracking
```python
async def get_consolidation_stats(self) -> Dict[str, Any]:
    return {
        "total_consolidations": self.stats["consolidations_performed"],
        "memories_processed": self.stats["memories_consolidated"],
        "facts_extracted": self.stats["facts_extracted"],
        "summaries_created": self.stats["summaries_created"],
        "last_run": self.stats["last_consolidation"],
        "success_rate": self.calculate_success_rate(),
        "average_processing_time": self.get_average_processing_time()
    }
```

## ⚙️ Configuration Options

### Timing Configuration
```python
class ConsolidationConfig:
    consolidation_interval_hours: int = 12
    min_session_age_minutes: int = 30  # Wait before consolidating
    max_processing_time_minutes: int = 60  # Timeout for consolidation
```

### Quality Thresholds
```python
class QualityConfig:
    min_conversation_length: int = 100  # Characters
    importance_threshold: float = 0.5
    fact_confidence_threshold: float = 0.7
    min_data_points: int = 10
```

### Processing Limits
```python
class ProcessingConfig:
    max_facts_per_conversation: int = 10
    max_topics_per_conversation: int = 5
    max_consolidation_batch_size: int = 50
    summarization_model: str = "facebook/bart-large-cnn"
```

## 🔧 Troubleshooting

### Common Issues

#### High Memory Usage
```python
# Reduce batch size
max_consolidation_batch_size = 25

# Increase quality thresholds
importance_threshold = 0.7
fact_confidence_threshold = 0.8
```

#### Slow Consolidation
```python
# Reduce processing scope
max_training_data_age_days = 14
max_facts_per_conversation = 5

# Use lighter summarization model
summarization_model = "facebook/bart-base"
```

#### Poor Quality Facts
```python
# Increase confidence threshold
fact_confidence_threshold = 0.8

# Add custom fact patterns
custom_patterns = [
    (r"specifically, (.+?)[\.\!\?]", "specific", "specific info", "is", "{}"),
]
```

### Monitoring Commands
```bash
# Check consolidation status
evo memory stats

# Force consolidation
evo memory consolidate

# View recent consolidations
evo status --detailed
```

## 🚀 Best Practices

1. **Regular Monitoring**: Check consolidation stats weekly
2. **Quality Tuning**: Adjust thresholds based on your data
3. **Performance Optimization**: Monitor processing times
4. **Storage Management**: Clean up old consolidated memories
5. **Custom Patterns**: Add domain-specific fact extraction patterns

The memory consolidation system is the heart of Evo's learning capability, transforming raw conversations into structured, queryable knowledge that persists and grows over time.