"""
Session management for Evo AI agent.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .models import ConversationContext, ChatMessage


class SessionManager:
    """Manages conversation sessions and persistence."""

    def __init__(self, sessions_dir: str = "~/.evo/sessions"):
        self.sessions_dir = Path(sessions_dir).expanduser()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: Dict[str, ConversationContext] = {}

    async def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid4())

        context = ConversationContext(
            session_id=session_id,
            metadata=metadata or {},
        )

        self._active_sessions[session_id] = context
        await self._save_session(context)

        return session_id

    async def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get a session by ID."""
        # Check active sessions first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Try to load from disk
        return await self._load_session(session_id)

    async def save_session(self, session_id: str) -> bool:
        """Save a session to disk."""
        context = self._active_sessions.get(session_id)
        if context is None:
            return False

        return await self._save_session(context)

    async def list_sessions(
        self,
        limit: Optional[int] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []

        # Add active sessions
        for session_id, context in self._active_sessions.items():
            session_info = {
                "session_id": session_id,
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
                "message_count": len(context.messages),
                "is_active": True,
            }

            if include_metadata:
                session_info["metadata"] = context.metadata

            sessions.append(session_info)

        # Add saved sessions
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem

            # Skip if already in active sessions
            if session_id in self._active_sessions:
                continue

            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)

                session_info = {
                    "session_id": session_id,
                    "created_at": session_data.get("created_at"),
                    "updated_at": session_data.get("updated_at"),
                    "message_count": len(session_data.get("messages", [])),
                    "is_active": False,
                }

                if include_metadata:
                    session_info["metadata"] = session_data.get("metadata", {})

                sessions.append(session_info)

            except (json.JSONDecodeError, KeyError):
                # Skip corrupted session files
                continue

        # Sort by updated_at (most recent first)
        sessions.sort(
            key=lambda x: x.get("updated_at", ""), reverse=True
        )

        if limit:
            sessions = sessions[:limit]

        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        deleted = False

        # Remove from active sessions
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            deleted = True

        # Remove session file
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            deleted = True

        return deleted

    async def archive_session(self, session_id: str) -> bool:
        """Archive a session (save to disk and remove from active)."""
        if session_id not in self._active_sessions:
            return False

        context = self._active_sessions[session_id]
        success = await self._save_session(context)

        if success:
            del self._active_sessions[session_id]

        return success

    async def restore_session(self, session_id: str) -> Optional[ConversationContext]:
        """Restore a session from disk to active sessions."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        context = await self._load_session(session_id)
        if context:
            self._active_sessions[session_id] = context

        return context

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a session."""
        context = await self.get_session(session_id)
        if context is None:
            return None

        # Calculate summary statistics
        message_counts = {"user": 0, "assistant": 0, "system": 0, "tool": 0}
        total_chars = 0
        first_message_time = None
        last_message_time = None

        for message in context.messages:
            message_counts[message.role.value] += 1
            total_chars += len(message.content)

            if first_message_time is None:
                first_message_time = message.timestamp
            last_message_time = message.timestamp

        # Extract topics (simple keyword extraction)
        topics = self._extract_topics_from_context(context)

        summary = {
            "session_id": session_id,
            "message_counts": message_counts,
            "total_messages": len(context.messages),
            "total_characters": total_chars,
            "duration_minutes": self._calculate_duration_minutes(
                first_message_time, last_message_time
            ),
            "topics": topics[:10],  # Top 10 topics
            "memory_count": len(context.retrieved_memories),
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "metadata": context.metadata,
        }

        return summary

    async def search_sessions(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search sessions by content."""
        matching_sessions = []
        query_lower = query.lower()

        # Search active sessions
        for session_id, context in self._active_sessions.items():
            score = self._calculate_session_relevance(context, query_lower)
            if score > 0:
                matching_sessions.append({
                    "session_id": session_id,
                    "score": score,
                    "is_active": True,
                    "updated_at": context.updated_at.isoformat(),
                    "preview": self._get_session_preview(context, query_lower),
                })

        # Search saved sessions
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem

            # Skip if already in active sessions
            if session_id in self._active_sessions:
                continue

            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)

                # Create a temporary context for searching
                context = ConversationContext(
                    session_id=session_id,
                    messages=[
                        ChatMessage(**msg) for msg in session_data.get("messages", [])
                    ],
                    metadata=session_data.get("metadata", {}),
                )

                score = self._calculate_session_relevance(context, query_lower)
                if score > 0:
                    matching_sessions.append({
                        "session_id": session_id,
                        "score": score,
                        "is_active": False,
                        "updated_at": session_data.get("updated_at"),
                        "preview": self._get_session_preview(context, query_lower),
                    })

            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by relevance score
        matching_sessions.sort(key=lambda x: x["score"], reverse=True)

        return matching_sessions[:limit]

    async def _save_session(self, context: ConversationContext) -> bool:
        """Save a session context to disk."""
        try:
            session_file = self.sessions_dir / f"{context.session_id}.json"

            session_data = {
                "session_id": context.session_id,
                "messages": [
                    {
                        "id": str(msg.id),
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata,
                        "tool_calls": msg.tool_calls,
                        "tool_call_id": msg.tool_call_id,
                    }
                    for msg in context.messages
                ],
                "retrieved_memories": context.retrieved_memories,
                "active_tools": context.active_tools,
                "metadata": context.metadata,
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
            }

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error saving session {context.session_id}: {e}")
            return False

    async def _load_session(self, session_id: str) -> Optional[ConversationContext]:
        """Load a session context from disk."""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"

            if not session_file.exists():
                return None

            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            messages = []
            for msg_data in session_data.get("messages", []):
                message = ChatMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    metadata=msg_data.get("metadata", {}),
                    tool_calls=msg_data.get("tool_calls"),
                    tool_call_id=msg_data.get("tool_call_id"),
                )
                messages.append(message)

            context = ConversationContext(
                session_id=session_id,
                messages=messages,
                retrieved_memories=session_data.get("retrieved_memories", []),
                active_tools=session_data.get("active_tools", {}),
                metadata=session_data.get("metadata", {}),
                created_at=datetime.fromisoformat(session_data["created_at"]),
                updated_at=datetime.fromisoformat(session_data["updated_at"]),
            )

            return context

        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def _extract_topics_from_context(self, context: ConversationContext) -> List[str]:
        """Extract topics from a conversation context."""
        word_counts = {}

        for message in context.messages:
            if message.role.value in ["user", "assistant"]:
                words = message.content.lower().split()
                for word in words:
                    # Simple filtering
                    if (
                        len(word) > 3
                        and word.isalpha()
                        and word not in [
                            "what", "how", "when", "where", "why", "the", "and", "but",
                            "for", "with", "this", "that", "have", "will", "would",
                            "could", "should", "can", "may", "might", "must", "shall",
                        ]
                    ):
                        word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and return top topics
        sorted_topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics if count > 1]

    def _calculate_session_relevance(
        self, context: ConversationContext, query: str
    ) -> float:
        """Calculate relevance score for a session based on query."""
        score = 0.0
        query_words = query.split()

        for message in context.messages:
            message_lower = message.content.lower()
            for word in query_words:
                if word in message_lower:
                    score += 1

        # Normalize by total message count
        if len(context.messages) > 0:
            score = score / len(context.messages)

        return score

    def _get_session_preview(
        self, context: ConversationContext, query: str
    ) -> str:
        """Get a preview of the session highlighting the query."""
        # Find the first message containing the query
        for message in context.messages:
            if query in message.content.lower():
                preview = message.content[:200]
                if len(message.content) > 200:
                    preview += "..."
                return f"{message.role.value}: {preview}"

        # If no exact match, return first user message
        for message in context.messages:
            if message.role.value == "user":
                preview = message.content[:200]
                if len(message.content) > 200:
                    preview += "..."
                return f"user: {preview}"

        return "No preview available"

    def _calculate_duration_minutes(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Optional[float]:
        """Calculate duration between two times in minutes."""
        if start_time is None or end_time is None:
            return None

        duration = end_time - start_time
        return duration.total_seconds() / 60

    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._active_sessions)

    async def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Clean up old session files."""
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                # Check file modification time
                if session_file.stat().st_mtime < cutoff_time:
                    session_file.unlink()
                    cleaned_count += 1
            except Exception:
                # Skip files we can't process
                continue

        return cleaned_count