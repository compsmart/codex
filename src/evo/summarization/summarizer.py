"""
Context summarization and fact extraction system.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from ..memory import MemoryManager, MemoryQuery, MemoryType
from .models import (
    SummaryConfig,
    ConversationSummary,
    FactExtraction,
    TopicSummary,
    LearningOutcome,
    SummaryType,
)


class ContextSummarizer:
    """Main context summarization system."""

    def __init__(self, memory_manager: MemoryManager, config: SummaryConfig):
        self.memory_manager = memory_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize models
        self.summarizer = None
        self.tokenizer = None
        self.extractor = SummaryExtractor(config)

    async def initialize(self) -> None:
        """Initialize the summarization models."""
        try:
            self.logger.info("Initializing summarization models...")

            # Load summarization model
            loop = asyncio.get_event_loop()
            self.summarizer = await loop.run_in_executor(
                None, self._load_summarization_model
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.summarization_model)

            self.logger.info("Summarization models initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize summarization models: {e}")
            raise

    def _load_summarization_model(self):
        """Load the summarization model (runs in executor)."""
        return pipeline(
            "summarization",
            model=self.config.summarization_model,
            tokenizer=self.config.summarization_model,
            max_length=self.config.max_output_length,
            min_length=self.config.min_output_length,
            do_sample=False,
        )

    async def summarize_conversation(
        self, session_id: str, message_limit: Optional[int] = None
    ) -> ConversationSummary:
        """Summarize a conversation from a session."""
        try:
            # Get conversation messages
            conversation_text = await self._get_conversation_text(session_id, message_limit)

            if len(conversation_text) < self.config.min_conversation_length:
                raise ValueError("Conversation too short for summarization")

            # Generate summary
            summary_text = await self._generate_summary(conversation_text)

            # Extract key information
            key_points = await self.extractor.extract_key_points(conversation_text)
            topics = await self.extractor.extract_topics(conversation_text)

            # Calculate metrics
            word_count_original = len(conversation_text.split())
            word_count_summary = len(summary_text.split())

            # Create summary object
            summary = ConversationSummary(
                session_id=session_id,
                original_content=conversation_text,
                summary_text=summary_text,
                key_points=key_points,
                topics=topics,
                word_count_original=word_count_original,
                word_count_summary=word_count_summary,
                importance_score=await self._calculate_importance(conversation_text),
                confidence_score=0.8,  # Default confidence
            )

            summary.calculate_compression_ratio()

            # Store as semantic memory
            await self._store_summary_as_memory(summary)

            return summary

        except Exception as e:
            self.logger.error(f"Error summarizing conversation: {e}")
            raise

    async def extract_facts_from_conversation(
        self, session_id: str
    ) -> List[FactExtraction]:
        """Extract facts from a conversation."""
        try:
            conversation_text = await self._get_conversation_text(session_id)
            facts = await self.extractor.extract_facts(conversation_text)

            # Store high-confidence facts as semantic memories
            for fact in facts:
                if fact.confidence >= self.config.fact_confidence_threshold:
                    await self.memory_manager.store_semantic_memory(
                        content=fact.fact_text,
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        confidence=fact.confidence,
                        source=f"fact_extraction_{session_id}",
                        importance=fact.importance,
                        metadata={
                            "fact_type": fact.fact_type,
                            "extraction_id": str(fact.id),
                        },
                    )

            return facts

        except Exception as e:
            self.logger.error(f"Error extracting facts: {e}")
            return []

    async def summarize_learning_outcomes(
        self, session_id: str
    ) -> List[LearningOutcome]:
        """Extract learning outcomes from conversation."""
        try:
            conversation_text = await self._get_conversation_text(session_id)
            outcomes = await self.extractor.extract_learning_outcomes(conversation_text)

            # Store learning outcomes as procedural memories
            for outcome in outcomes:
                if outcome.confidence >= 0.6:
                    await self.memory_manager.store_procedural_memory(
                        content=outcome.outcome_text,
                        skill_name=outcome.domain,
                        steps=[],
                        prerequisites=outcome.prerequisites,
                        importance=0.7,
                        metadata={
                            "outcome_type": outcome.outcome_type,
                            "complexity_level": outcome.complexity_level,
                            "extraction_id": str(outcome.id),
                        },
                    )

            return outcomes

        except Exception as e:
            self.logger.error(f"Error extracting learning outcomes: {e}")
            return []

    async def _get_conversation_text(
        self, session_id: str, message_limit: Optional[int] = None
    ) -> str:
        """Get conversation text from session."""
        # Query for episodic memories from this session
        query = MemoryQuery(
            text="*",
            memory_types=[MemoryType.EPISODIC],
            limit=message_limit or 100,
            session_id=session_id,
        )

        results = await self.memory_manager.search(query)

        # Build conversation text
        conversation_parts = []
        for result in results:
            memory = result.memory
            if hasattr(memory, 'user_message') and hasattr(memory, 'assistant_response'):
                if memory.user_message and memory.assistant_response:
                    conversation_parts.append(f"User: {memory.user_message}")
                    conversation_parts.append(f"Assistant: {memory.assistant_response}")

        return "\n\n".join(conversation_parts)

    async def _generate_summary(self, text: str) -> str:
        """Generate summary using the model."""
        try:
            # Tokenize and check length
            tokens = self.tokenizer.encode(text, return_tensors="pt")
            if len(tokens[0]) > self.config.max_input_length:
                # Truncate if too long
                decoded = self.tokenizer.decode(
                    tokens[0][: self.config.max_input_length], skip_special_tokens=True
                )
                text = decoded

            # Generate summary
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.summarizer, text)

            return result[0]["summary_text"]

        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"

    async def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for conversation."""
        importance = 0.5  # Base importance

        text_lower = text.lower()

        # Check for importance keywords
        for keyword in self.config.importance_boost_keywords:
            if keyword in text_lower:
                importance += 0.1

        # Length factor
        word_count = len(text.split())
        if word_count > 200:
            importance += 0.1
        if word_count > 500:
            importance += 0.1

        # Question/answer patterns
        question_count = text.count("?")
        if question_count > 3:
            importance += 0.1

        # Technical content
        technical_keywords = ["code", "function", "algorithm", "error", "debug", "implement"]
        for keyword in technical_keywords:
            if keyword in text_lower:
                importance += 0.05

        return min(importance, 1.0)

    async def _store_summary_as_memory(self, summary: ConversationSummary) -> None:
        """Store summary as semantic memory."""
        await self.memory_manager.store_semantic_memory(
            content=summary.summary_text,
            subject=f"conversation_summary_{summary.session_id}",
            predicate="summarizes",
            object=f"session_{summary.session_id}",
            confidence=summary.confidence_score,
            source="conversation_summarization",
            importance=summary.importance_score,
            metadata={
                "summary_id": str(summary.id),
                "compression_ratio": summary.compression_ratio,
                "topics": summary.topics,
                "key_points": summary.key_points,
            },
        )

    async def get_session_summaries(self, limit: int = 10) -> List[ConversationSummary]:
        """Get recent conversation summaries."""
        # This would retrieve summaries from storage
        # For now, return empty list as placeholder
        return []


class SummaryExtractor:
    """Extracts specific information from text."""

    def __init__(self, config: SummaryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        try:
            # Simple extraction based on sentence importance
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            # Score sentences
            scored_sentences = []
            for sentence in sentences:
                score = self._score_sentence_importance(sentence)
                scored_sentences.append((sentence, score))

            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scored_sentences[:5]]

        except Exception as e:
            self.logger.error(f"Error extracting key points: {e}")
            return []

    async def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        try:
            # Simple keyword-based topic extraction
            text_lower = text.lower()

            # Predefined topic categories
            topic_keywords = {
                "programming": ["code", "function", "algorithm", "debug", "programming", "software"],
                "learning": ["learn", "understand", "explain", "teach", "study"],
                "preferences": ["favorite", "prefer", "like", "dislike", "enjoy"],
                "personal": ["my", "i am", "i work", "i live", "myself"],
                "technical": ["system", "technology", "computer", "software", "hardware"],
                "problem_solving": ["problem", "solution", "fix", "resolve", "issue"],
            }

            detected_topics = []
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected_topics.append(topic)

            return detected_topics[:self.config.max_topics_per_conversation]

        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []

    async def extract_facts(self, text: str) -> List[FactExtraction]:
        """Extract facts from conversation text."""
        try:
            facts = []

            # Pattern-based fact extraction
            fact_patterns = [
                # Preference patterns
                (r"my favorite (.+?) is (.+?)[\.\!\?]", "preference", "user's favorite {}", "is", "{}"),
                (r"i prefer (.+?) over (.+?)[\.\!\?]", "preference", "user preference", "prefers", "{} over {}"),
                (r"i like (.+?)[\.\!\?]", "preference", "user", "likes", "{}"),
                (r"i don't like (.+?)[\.\!\?]", "preference", "user", "dislikes", "{}"),

                # Personal information
                (r"my name is (.+?)[\.\!\?]", "personal", "user's name", "is", "{}"),
                (r"i am (.+?)[\.\!\?]", "personal", "user", "is", "{}"),
                (r"i work (.+?)[\.\!\?]", "personal", "user", "works", "{}"),

                # Knowledge statements
                (r"(.+?) is (.+?)[\.\!\?]", "knowledge", "{}", "is", "{}"),
                (r"(.+?) can (.+?)[\.\!\?]", "knowledge", "{}", "can", "{}"),
            ]

            for pattern, fact_type, subject_template, predicate, object_template in fact_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 1:
                        # Format subject and object
                        if "{}" in subject_template:
                            subject = subject_template.format(groups[0])
                        else:
                            subject = subject_template

                        if "{}" in object_template:
                            if len(groups) >= 2:
                                obj = object_template.format(*groups)
                            else:
                                obj = object_template.format(groups[0])
                        else:
                            obj = object_template

                        # Create fact
                        fact = FactExtraction(
                            fact_text=match.group(0),
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            fact_type=fact_type,
                            confidence=0.7,  # Default confidence
                            importance=0.6,
                            source_context=text[max(0, match.start() - 50):match.end() + 50],
                        )

                        facts.append(fact)

            return facts[:self.config.max_facts_per_conversation]

        except Exception as e:
            self.logger.error(f"Error extracting facts: {e}")
            return []

    async def extract_learning_outcomes(self, text: str) -> List[LearningOutcome]:
        """Extract learning outcomes from text."""
        try:
            outcomes = []

            # Pattern-based learning outcome extraction
            learning_patterns = [
                (r"learned how to (.+?)[\.\!\?]", "skill_acquired", "{}"),
                (r"now i understand (.+?)[\.\!\?]", "knowledge_gained", "{}"),
                (r"i can now (.+?)[\.\!\?]", "capability_developed", "{}"),
                (r"figured out (.+?)[\.\!\?]", "problem_solved", "{}"),
            ]

            for pattern, outcome_type, outcome_template in learning_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    groups = match.groups()
                    if groups:
                        outcome_text = outcome_template.format(groups[0])

                        outcome = LearningOutcome(
                            outcome_text=outcome_text,
                            outcome_type=outcome_type,
                            confidence=0.7,
                            domain=self._classify_domain(outcome_text),
                            complexity_level=self._assess_complexity(outcome_text),
                            evidence=[match.group(0)],
                        )

                        outcomes.append(outcome)

            return outcomes

        except Exception as e:
            self.logger.error(f"Error extracting learning outcomes: {e}")
            return []

    def _score_sentence_importance(self, sentence: str) -> float:
        """Score the importance of a sentence."""
        score = 0.0

        # Length factor
        word_count = len(sentence.split())
        if 10 <= word_count <= 30:
            score += 0.3

        # Keyword presence
        important_words = ["important", "key", "main", "primary", "essential", "crucial"]
        for word in important_words:
            if word in sentence.lower():
                score += 0.2

        # Question format
        if sentence.strip().endswith("?"):
            score += 0.1

        # Contains specific information
        if any(marker in sentence.lower() for marker in ["is", "are", "can", "will", "should"]):
            score += 0.1

        return score

    def _classify_domain(self, text: str) -> str:
        """Classify the domain of learning outcome."""
        text_lower = text.lower()

        domain_keywords = {
            "programming": ["code", "function", "variable", "loop", "algorithm"],
            "language": ["word", "grammar", "pronunciation", "vocabulary"],
            "mathematics": ["equation", "formula", "calculation", "number"],
            "science": ["experiment", "theory", "hypothesis", "research"],
            "technology": ["computer", "software", "system", "network"],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain

        return "general"

    def _assess_complexity(self, text: str) -> str:
        """Assess the complexity level of learning outcome."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["advanced", "complex", "sophisticated", "intricate"]):
            return "advanced"
        elif any(word in text_lower for word in ["intermediate", "moderate", "detailed"]):
            return "intermediate"
        else:
            return "basic"