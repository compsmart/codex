"""
Emotion detection and processing for Evo AI memory system.

This module implements emotion detection from text and emotional tagging
of memories to influence behavior and decision-making.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EmotionalContext:
    """Context for emotional analysis of a conversation."""

    user_emotions: Dict[str, float]
    assistant_emotions: Dict[str, float]
    interaction_valence: float  # Overall positive/negative feeling
    emotional_intensity: float  # How emotionally charged the interaction was
    dominant_emotion: Optional[str]  # Most prominent emotion


class EmotionDetector:
    """Detects emotions from text using pattern-based analysis."""

    def __init__(self):
        # Emotion patterns based on explicit emotional language
        self.emotion_patterns = {
            # Positive emotions
            "joy": [
                r"\b(happy|joy|joyful|delighted|thrilled|excited|elated|cheerful|glad|pleased)\b",
                r"\b(love|adore|fantastic|wonderful|amazing|awesome|great|excellent)\b",
                r":\)|😊|😄|😃|🎉|❤️",
                r"\b(yay|hooray|woohoo)\b"
            ],
            "satisfaction": [
                r"\b(satisfied|content|pleased|fulfilled|accomplished|successful)\b",
                r"\b(worked|perfect|exactly|right|correct|good job|well done)\b",
                r"\b(thank you|thanks|appreciate|grateful)\b"
            ],
            "pride": [
                r"\b(proud|achievement|accomplished|succeeded|mastered|excellent work)\b",
                r"\b(my|i) (created|built|made|designed|wrote|solved)\b"
            ],
            "trust": [
                r"\b(trust|reliable|dependable|confident|sure|certain)\b",
                r"\b(you understand|you get it|you know)\b"
            ],
            "curiosity": [
                r"\b(curious|interested|wondering|fascinated|intrigued)\b",
                r"\b(how does|why does|what if|tell me more|explain)\b",
                r"\?.*\?",  # Multiple questions
                r"\b(learn|study|research|explore|discover)\b"
            ],

            # Negative emotions
            "sadness": [
                r"\b(sad|depressed|down|disappointed|upset|hurt|heartbroken)\b",
                r":\(|😢|😭|💔",
                r"\b(miss|lost|gone|died|death)\b"
            ],
            "fear": [
                r"\b(afraid|scared|frightened|worried|anxious|nervous|panic)\b",
                r"\b(don't want|scared of|afraid of|worried about)\b",
                r"\b(dangerous|risky|unsafe|threat)\b"
            ],
            "anger": [
                r"\b(angry|mad|furious|irritated|annoyed|frustrated|pissed)\b",
                r"\b(hate|stupid|dumb|ridiculous|awful|terrible|worst)\b",
                r"!{2,}",  # Multiple exclamation marks
                r"\b(why won't|doesn't work|broken|useless)\b"
            ],
            "disgust": [
                r"\b(disgusting|gross|sick|revolting|awful|horrible)\b",
                r"\b(hate|can't stand|makes me sick)\b"
            ],
            "regret": [
                r"\b(regret|sorry|mistake|wrong|shouldn't have|wish i hadn't)\b",
                r"\b(my fault|i messed up|i screwed up)\b"
            ],
            "shame": [
                r"\b(ashamed|embarrassed|humiliated|mortified)\b",
                r"\b(stupid me|i'm an idiot|feel dumb)\b"
            ],
            "guilt": [
                r"\b(guilty|feel bad|sorry for|apologize)\b",
                r"\b(i should have|i didn't|i forgot)\b"
            ],

            # Neutral/Learning emotions
            "confusion": [
                r"\b(confused|don't understand|unclear|lost|puzzled)\b",
                r"\b(what do you mean|i don't get it|huh|what)\b",
                r"\?\?+",  # Multiple question marks
                r"\b(how|why|what|when|where)\b.*\?"
            ]
        }

        # Outcome patterns to determine success/failure
        self.outcome_patterns = {
            "success": [
                r"\b(worked|success|solved|fixed|done|completed|finished)\b",
                r"\b(perfect|excellent|great|good|right|correct)\b",
                r"\b(thank you|thanks|helpful|useful)\b"
            ],
            "failure": [
                r"\b(failed|broken|error|wrong|doesn't work|not working)\b",
                r"\b(problem|issue|trouble|difficulty|stuck)\b",
                r"\b(can't|won't|unable|impossible)\b"
            ],
            "learning": [
                r"\b(learned|understand|got it|makes sense|i see)\b",
                r"\b(teach|explain|show|help me)\b",
                r"\b(now i know|that's how|interesting)\b"
            ],
            "confusion": [
                r"\b(confused|don't understand|unclear|lost)\b",
                r"\b(what|how|why)\b.*\?",
                r"\b(help|explain|clarify)\b"
            ]
        }

        # Emotional intensity modifiers
        self.intensity_modifiers = {
            "very": 1.3,
            "really": 1.2,
            "so": 1.2,
            "extremely": 1.5,
            "incredibly": 1.4,
            "absolutely": 1.3,
            "totally": 1.2,
            "completely": 1.3,
            "quite": 1.1,
            "a bit": 0.8,
            "somewhat": 0.9,
            "little": 0.8,
            "slightly": 0.7
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text and return emotion scores."""
        if not text:
            return {}

        text_lower = text.lower()
        emotions = {}

        # Detect emotions using patterns
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0

            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    base_score = 0.3 + (len(matches) * 0.1)  # Base + frequency bonus

                    # Apply intensity modifiers
                    for modifier, multiplier in self.intensity_modifiers.items():
                        if modifier in text_lower:
                            base_score *= multiplier
                            break

                    score = max(score, min(base_score, 1.0))

            if score > 0:
                emotions[emotion] = score

        return emotions

    def detect_outcome_type(self, user_message: str, assistant_response: str) -> Tuple[str, float]:
        """Detect the outcome type and valence of an interaction."""
        combined_text = f"{user_message} {assistant_response}".lower()

        outcome_scores = {}

        # Check for outcome patterns
        for outcome_type, patterns in self.outcome_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, combined_text)
                if matches:
                    score += len(matches) * 0.2

            if score > 0:
                outcome_scores[outcome_type] = min(score, 1.0)

        # Determine primary outcome
        if not outcome_scores:
            return "neutral", 0.0

        primary_outcome = max(outcome_scores.items(), key=lambda x: x[1])

        # Calculate valence based on outcome type
        valence_map = {
            "success": 0.8,
            "learning": 0.6,
            "neutral": 0.0,
            "confusion": -0.2,
            "failure": -0.6,
            "regret": -0.8
        }

        return primary_outcome[0], valence_map.get(primary_outcome[0], 0.0)

    def analyze_conversation(
        self,
        user_message: str,
        assistant_response: str,
        context_summary: Optional[str] = None
    ) -> EmotionalContext:
        """Analyze a full conversation exchange for emotional context."""

        # Analyze emotions in user message and assistant response
        user_emotions = self.analyze_text(user_message)
        assistant_emotions = self.analyze_text(assistant_response)

        # Combine emotions from both sides
        all_emotions = {}
        for emotion, score in user_emotions.items():
            all_emotions[emotion] = score
        for emotion, score in assistant_emotions.items():
            if emotion in all_emotions:
                all_emotions[emotion] = max(all_emotions[emotion], score)
            else:
                all_emotions[emotion] = score

        # Calculate overall emotional intensity
        emotional_intensity = 0.0
        if all_emotions:
            emotional_intensity = sum(all_emotions.values()) / len(all_emotions)

        # Determine dominant emotion
        dominant_emotion = None
        if all_emotions:
            dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]

        # Detect outcome and calculate interaction valence
        outcome_type, outcome_valence = self.detect_outcome_type(user_message, assistant_response)

        # Calculate overall interaction valence
        positive_emotions = ["joy", "satisfaction", "pride", "trust", "curiosity"]
        negative_emotions = ["sadness", "fear", "anger", "disgust", "regret", "shame", "guilt"]

        positive_sum = sum(all_emotions.get(e, 0) for e in positive_emotions)
        negative_sum = sum(all_emotions.get(e, 0) for e in negative_emotions)

        interaction_valence = (positive_sum - negative_sum + outcome_valence) / 2
        interaction_valence = max(-1.0, min(1.0, interaction_valence))

        return EmotionalContext(
            user_emotions=user_emotions,
            assistant_emotions=assistant_emotions,
            interaction_valence=interaction_valence,
            emotional_intensity=emotional_intensity,
            dominant_emotion=dominant_emotion
        )

    def calculate_memory_importance(
        self,
        base_importance: float,
        emotional_context: EmotionalContext
    ) -> float:
        """Calculate memory importance enhanced by emotional factors."""

        # Emotional memories are generally more important
        emotion_boost = emotional_context.emotional_intensity * 0.3

        # Very positive or very negative experiences are more memorable
        valence_boost = abs(emotional_context.interaction_valence) * 0.2

        # Certain emotions make memories more significant
        significant_emotions = ["fear", "regret", "joy", "pride", "anger"]
        significance_boost = 0.0

        all_emotions = {**emotional_context.user_emotions, **emotional_context.assistant_emotions}
        for emotion in significant_emotions:
            if emotion in all_emotions:
                significance_boost += all_emotions[emotion] * 0.1

        enhanced_importance = base_importance + emotion_boost + valence_boost + significance_boost
        return min(enhanced_importance, 1.0)