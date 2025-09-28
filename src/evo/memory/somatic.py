"""
Somatic marker decision system for Evo AI.

Based on Damasio's somatic marker hypothesis, this module uses emotional
memories to guide decision-making and behavior selection.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import Memory, MemoryType, OutcomeType


@dataclass
class SomaticMarker:
    """A somatic marker representing emotional memory guidance."""

    emotion: str
    valence: float  # -1.0 (negative) to 1.0 (positive)
    intensity: float  # 0.0 to 1.0
    confidence: float  # How reliable this marker is
    source_memory_id: str
    trigger_context: str  # What situation/context triggers this marker


@dataclass
class DecisionContext:
    """Context for a decision that needs somatic marker guidance."""

    situation_description: str
    possible_actions: List[str]
    current_emotional_state: Dict[str, float]
    time_pressure: float = 0.5  # 0.0 (no pressure) to 1.0 (urgent)
    importance: float = 0.5  # How important this decision is


@dataclass
class DecisionRecommendation:
    """Recommendation from the somatic marker system."""

    recommended_action: str
    confidence: float
    emotional_rationale: str
    supporting_markers: List[SomaticMarker]
    risk_assessment: float  # 0.0 (safe) to 1.0 (risky)


class SomaticMarkerSystem:
    """System for emotion-guided decision making."""

    def __init__(self):
        self.marker_cache: Dict[str, List[SomaticMarker]] = {}
        self.decision_history: List[Dict[str, Any]] = []

    def extract_markers_from_memory(self, memory: Memory) -> List[SomaticMarker]:
        """Extract somatic markers from a memory."""
        markers = []

        if not memory.emotions or memory.emotional_intensity < 0.3:
            return markers

        # Create markers for significant emotions
        for emotion, intensity in memory.emotions.items():
            if intensity > 0.4:  # Only significant emotions become markers

                # Determine confidence based on memory properties
                confidence = self._calculate_marker_confidence(memory, emotion, intensity)

                # Create trigger context from memory content
                trigger_context = self._extract_trigger_context(memory)

                marker = SomaticMarker(
                    emotion=emotion,
                    valence=memory.valence,
                    intensity=intensity,
                    confidence=confidence,
                    source_memory_id=str(memory.id),
                    trigger_context=trigger_context
                )
                markers.append(marker)

        return markers

    def _calculate_marker_confidence(self, memory: Memory, emotion: str, intensity: float) -> float:
        """Calculate how reliable a somatic marker is."""
        confidence = 0.5  # Base confidence

        # Recent memories are more reliable
        days_old = (datetime.now() - memory.created_at).days
        if days_old < 7:
            confidence += 0.2
        elif days_old < 30:
            confidence += 0.1
        elif days_old > 90:
            confidence -= 0.2

        # High emotional intensity increases confidence
        confidence += memory.emotional_intensity * 0.3

        # Memories that have been accessed multiple times are more reliable
        if memory.access_count > 3:
            confidence += 0.1
        elif memory.access_count > 10:
            confidence += 0.2

        # Clear outcomes increase confidence
        if memory.outcome_type in [OutcomeType.SUCCESS, OutcomeType.FAILURE]:
            confidence += 0.2
        elif memory.outcome_type == OutcomeType.REGRET:
            confidence += 0.3  # Regret is a strong learning signal

        # High importance memories are more reliable
        confidence += memory.importance * 0.2

        return min(confidence, 1.0)

    def _extract_trigger_context(self, memory: Memory) -> str:
        """Extract the situational context that triggers this marker."""
        content = memory.content.lower()

        # Common trigger patterns
        patterns = {
            "learning": ["learn", "study", "understand", "teach"],
            "coding": ["code", "program", "function", "error", "debug"],
            "personal": ["family", "friend", "personal", "life"],
            "work": ["work", "job", "project", "task", "deadline"],
            "help": ["help", "assist", "support", "guide"],
            "creativity": ["create", "design", "build", "make", "art"],
            "problem_solving": ["problem", "solve", "fix", "solution"],
            "communication": ["talk", "speak", "discuss", "conversation"]
        }

        # Find the most relevant context
        for context, keywords in patterns.items():
            if any(keyword in content for keyword in keywords):
                return context

        return "general"

    def evaluate_decision(
        self,
        context: DecisionContext,
        relevant_memories: List[Memory]
    ) -> DecisionRecommendation:
        """Evaluate a decision using somatic markers from relevant memories."""

        # Extract all applicable somatic markers
        all_markers = []
        for memory in relevant_memories:
            markers = self.extract_markers_from_memory(memory)
            all_markers.extend(markers)

        if not all_markers:
            # No emotional guidance available, return neutral recommendation
            return DecisionRecommendation(
                recommended_action=context.possible_actions[0] if context.possible_actions else "proceed",
                confidence=0.3,
                emotional_rationale="No emotional memories found to guide this decision",
                supporting_markers=[],
                risk_assessment=0.5
            )

        # Score each possible action based on somatic markers
        action_scores = {}
        action_markers = {}

        for action in context.possible_actions:
            score, supporting_markers = self._score_action(action, all_markers, context)
            action_scores[action] = score
            action_markers[action] = supporting_markers

        # Find the best action
        best_action = max(action_scores.items(), key=lambda x: x[1])
        best_action_name, best_score = best_action

        # Calculate confidence based on marker strength and consensus
        confidence = self._calculate_decision_confidence(
            action_markers[best_action_name], all_markers
        )

        # Generate emotional rationale
        rationale = self._generate_rationale(
            best_action_name, action_markers[best_action_name]
        )

        # Assess risk based on negative markers
        risk_assessment = self._assess_risk(action_markers[best_action_name], context)

        return DecisionRecommendation(
            recommended_action=best_action_name,
            confidence=confidence,
            emotional_rationale=rationale,
            supporting_markers=action_markers[best_action_name],
            risk_assessment=risk_assessment
        )

    def _score_action(
        self,
        action: str,
        markers: List[SomaticMarker],
        context: DecisionContext
    ) -> Tuple[float, List[SomaticMarker]]:
        """Score an action based on relevant somatic markers."""
        score = 0.0
        supporting_markers = []

        action_lower = action.lower()

        for marker in markers:
            relevance = self._calculate_marker_relevance(marker, action_lower, context)

            if relevance > 0.3:  # Only consider relevant markers
                # Weight the marker's influence
                marker_influence = (
                    marker.valence *  # Positive/negative influence
                    marker.intensity *  # How strong the emotion was
                    marker.confidence *  # How reliable this marker is
                    relevance  # How relevant to current situation
                )

                score += marker_influence
                supporting_markers.append(marker)

        return score, supporting_markers

    def _calculate_marker_relevance(
        self,
        marker: SomaticMarker,
        action: str,
        context: DecisionContext
    ) -> float:
        """Calculate how relevant a marker is to the current decision."""
        relevance = 0.3  # Base relevance

        # Context similarity
        if marker.trigger_context in context.situation_description.lower():
            relevance += 0.4

        # Action similarity (simple keyword matching)
        action_keywords = action.split()
        for keyword in action_keywords:
            if keyword in marker.trigger_context:
                relevance += 0.2

        # Current emotional state alignment
        if marker.emotion in context.current_emotional_state:
            current_intensity = context.current_emotional_state[marker.emotion]
            if current_intensity > 0.5:
                relevance += 0.3

        return min(relevance, 1.0)

    def _calculate_decision_confidence(
        self,
        supporting_markers: List[SomaticMarker],
        all_markers: List[SomaticMarker]
    ) -> float:
        """Calculate confidence in the decision recommendation."""
        if not supporting_markers:
            return 0.3

        # Average confidence of supporting markers
        avg_confidence = sum(m.confidence for m in supporting_markers) / len(supporting_markers)

        # Consensus factor (how many markers support this decision)
        consensus_factor = len(supporting_markers) / max(len(all_markers), 1)

        # Emotional intensity factor
        avg_intensity = sum(m.intensity for m in supporting_markers) / len(supporting_markers)

        confidence = (avg_confidence + consensus_factor + avg_intensity) / 3
        return min(confidence, 1.0)

    def _generate_rationale(self, action: str, markers: List[SomaticMarker]) -> str:
        """Generate a human-readable rationale for the decision."""
        if not markers:
            return "No strong emotional memories guide this decision."

        # Group markers by emotion type
        emotion_groups = {}
        for marker in markers:
            if marker.emotion not in emotion_groups:
                emotion_groups[marker.emotion] = []
            emotion_groups[marker.emotion].append(marker)

        rationale_parts = []

        # Analyze dominant emotions
        for emotion, group_markers in emotion_groups.items():
            avg_valence = sum(m.valence for m in group_markers) / len(group_markers)
            avg_intensity = sum(m.intensity for m in group_markers) / len(group_markers)

            if avg_intensity > 0.6:
                if avg_valence > 0.3:
                    rationale_parts.append(f"Past experiences with {emotion} suggest this could lead to positive outcomes")
                elif avg_valence < -0.3:
                    rationale_parts.append(f"Previous {emotion} experiences warn against similar actions")
                else:
                    rationale_parts.append(f"Past {emotion} experiences provide mixed signals")

        if not rationale_parts:
            return "Emotional memories weakly support this decision."

        return "; ".join(rationale_parts[:3])  # Limit to 3 main points

    def _assess_risk(self, markers: List[SomaticMarker], context: DecisionContext) -> float:
        """Assess the risk level based on emotional markers."""
        if not markers:
            return 0.5  # Neutral risk

        # Count negative emotional markers
        negative_markers = [m for m in markers if m.valence < -0.2]
        positive_markers = [m for m in markers if m.valence > 0.2]

        # High-risk emotions
        risky_emotions = ["fear", "regret", "anger", "shame"]
        risk_markers = [m for m in markers if m.emotion in risky_emotions]

        # Calculate risk score
        risk_score = 0.3  # Base risk

        if negative_markers:
            negative_weight = sum(abs(m.valence) * m.intensity for m in negative_markers)
            risk_score += negative_weight / len(markers) * 0.4

        if risk_markers:
            risk_score += len(risk_markers) / len(markers) * 0.3

        # Time pressure increases risk perception
        risk_score += context.time_pressure * 0.2

        # High importance decisions feel riskier
        risk_score += context.importance * 0.1

        return min(risk_score, 1.0)

    def learn_from_outcome(
        self,
        decision_context: DecisionContext,
        chosen_action: str,
        actual_outcome: OutcomeType,
        outcome_emotions: Dict[str, float]
    ) -> None:
        """Learn from decision outcomes to improve future recommendations."""

        decision_record = {
            "timestamp": datetime.now(),
            "context": decision_context.situation_description,
            "action": chosen_action,
            "outcome": actual_outcome,
            "emotions": outcome_emotions,
            "importance": decision_context.importance
        }

        self.decision_history.append(decision_record)

        # Keep only recent decisions for learning (last 100)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]

    def get_decision_patterns(self) -> Dict[str, Any]:
        """Analyze decision patterns for insights."""
        if not self.decision_history:
            return {}

        # Analyze success/failure patterns
        successful_decisions = [d for d in self.decision_history
                              if d["outcome"] in [OutcomeType.SUCCESS, OutcomeType.LEARNING]]

        failed_decisions = [d for d in self.decision_history
                           if d["outcome"] in [OutcomeType.FAILURE, OutcomeType.REGRET]]

        # Common emotions in successful vs failed decisions
        success_emotions = {}
        failure_emotions = {}

        for decision in successful_decisions:
            for emotion, intensity in decision["emotions"].items():
                success_emotions[emotion] = success_emotions.get(emotion, 0) + intensity

        for decision in failed_decisions:
            for emotion, intensity in decision["emotions"].items():
                failure_emotions[emotion] = failure_emotions.get(emotion, 0) + intensity

        return {
            "total_decisions": len(self.decision_history),
            "success_rate": len(successful_decisions) / len(self.decision_history),
            "success_emotions": success_emotions,
            "failure_emotions": failure_emotions,
            "recent_trend": self._analyze_recent_trend()
        }

    def _analyze_recent_trend(self) -> str:
        """Analyze recent decision-making trends."""
        if len(self.decision_history) < 5:
            return "insufficient_data"

        recent_decisions = self.decision_history[-5:]
        successful_recent = [d for d in recent_decisions
                           if d["outcome"] in [OutcomeType.SUCCESS, OutcomeType.LEARNING]]

        if len(successful_recent) >= 4:
            return "improving"
        elif len(successful_recent) >= 2:
            return "stable"
        else:
            return "declining"