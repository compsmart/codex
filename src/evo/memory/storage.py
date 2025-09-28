"""
SQLite storage backend with vector extension for Evo AI memory system.
"""

import json
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import aiosqlite
import numpy as np
from pydantic import ValidationError

from .models import Memory, MemoryType, EpisodicMemory, SemanticMemory, ProceduralMemory


class SQLiteStorage:
    """SQLite storage backend with vector extension support."""

    def __init__(
        self,
        db_path: Union[str, Path],
        enable_wal: bool = True,
        vector_extension_path: Optional[str] = None,
    ):
        self.db_path = Path(db_path).expanduser()
        self.enable_wal = enable_wal
        self.vector_extension_path = vector_extension_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)

        # Enable WAL mode for better concurrent access
        if self.enable_wal:
            await self._connection.execute("PRAGMA journal_mode=WAL")

        # Load vector extension if available
        if self.vector_extension_path:
            try:
                await self._connection.enable_load_extension(True)
                await self._connection.load_extension(self.vector_extension_path)
            except Exception as e:
                print(f"Warning: Could not load vector extension: {e}")

        await self._create_tables()
        await self._migrate_database()
        await self._connection.commit()

    async def _create_tables(self) -> None:
        """Create database tables."""

        # Main memories table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TIMESTAMP,

                -- Temporal dynamics (neuromorphic memory)
                activation_level REAL NOT NULL DEFAULT 1.0,
                dormancy_state TEXT NOT NULL DEFAULT 'active',
                reactivation_count INTEGER NOT NULL DEFAULT 0,
                last_reactivation TIMESTAMP,
                decay_rate REAL NOT NULL DEFAULT 0.1,

                -- Emotional intelligence
                emotions TEXT,
                valence REAL NOT NULL DEFAULT 0.0,
                arousal REAL NOT NULL DEFAULT 0.0,
                dominance REAL NOT NULL DEFAULT 0.0,
                outcome_type TEXT NOT NULL DEFAULT 'neutral',
                outcome_valence REAL NOT NULL DEFAULT 0.0,
                emotional_intensity REAL NOT NULL DEFAULT 0.0,

                -- Conflict resolution
                conflicted BOOLEAN NOT NULL DEFAULT 0,
                conflicts_with TEXT,
                superseded_by TEXT,
                verification_requested BOOLEAN NOT NULL DEFAULT 0,

                -- Type-specific fields
                session_id TEXT,
                user_message TEXT,
                assistant_response TEXT,
                context_summary TEXT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                confidence REAL,
                source TEXT,
                verified BOOLEAN,
                related_concepts TEXT,
                skill_name TEXT,
                steps TEXT,
                prerequisites TEXT,
                success_rate REAL,
                last_used TIMESTAMP,
                proficiency_level TEXT
            )
        """)

        # Create indexes for performance
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_activation_level ON memories(activation_level)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_dormancy_state ON memories(dormancy_state)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_valence ON memories(valence)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_emotional_intensity ON memories(emotional_intensity)
        """)

        # Full-text search
        await self._connection.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                user_message,
                assistant_response,
                context_summary,
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Vector similarity table (if vector extension is available)
        try:
            await self._connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    embedding float[384]
                )
            """)
        except Exception:
            # Fallback: regular table for embeddings
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            """)

    async def _migrate_database(self) -> None:
        """Migrate existing database to include new neuromorphic memory fields."""

        # Check if we need to migrate by looking for new columns
        cursor = await self._connection.execute("PRAGMA table_info(memories)")
        columns = await cursor.fetchall()
        existing_columns = {col[1] for col in columns}  # col[1] is column name

        # Define all the new columns we need
        new_columns = {
            'activation_level': 'REAL NOT NULL DEFAULT 1.0',
            'dormancy_state': 'TEXT NOT NULL DEFAULT "active"',
            'reactivation_count': 'INTEGER NOT NULL DEFAULT 0',
            'last_reactivation': 'TIMESTAMP',
            'decay_rate': 'REAL NOT NULL DEFAULT 0.1',
            'valence': 'REAL NOT NULL DEFAULT 0.0',
            'arousal': 'REAL NOT NULL DEFAULT 0.0',
            'dominance': 'REAL NOT NULL DEFAULT 0.0',
            'outcome_type': 'TEXT NOT NULL DEFAULT "neutral"',
            'outcome_valence': 'REAL NOT NULL DEFAULT 0.0',
            'emotional_intensity': 'REAL NOT NULL DEFAULT 0.0',
            'conflicted': 'BOOLEAN NOT NULL DEFAULT 0',
            'conflicts_with': 'TEXT',
            'superseded_by': 'TEXT',
            'verification_requested': 'BOOLEAN NOT NULL DEFAULT 0'
        }

        # Add missing columns
        for column_name, column_def in new_columns.items():
            if column_name not in existing_columns:
                try:
                    await self._connection.execute(f"ALTER TABLE memories ADD COLUMN {column_name} {column_def}")
                    print(f"Added column: {column_name}")
                except Exception as e:
                    print(f"Warning: Could not add column {column_name}: {e}")

        # Add missing indexes
        missing_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_activation_level ON memories(activation_level)",
            "CREATE INDEX IF NOT EXISTS idx_memories_dormancy_state ON memories(dormancy_state)",
            "CREATE INDEX IF NOT EXISTS idx_memories_valence ON memories(valence)",
            "CREATE INDEX IF NOT EXISTS idx_memories_emotional_intensity ON memories(emotional_intensity)"
        ]

        for index_sql in missing_indexes:
            try:
                await self._connection.execute(index_sql)
            except Exception as e:
                print(f"Warning: Could not create index: {e}")

    async def store_memory(self, memory: Memory) -> None:
        """Store a memory in the database."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        # Serialize the memory data
        memory_data = {
            "id": str(memory.id),
            "type": memory.type.value,
            "content": memory.content,
            "embedding": self._serialize_embedding(memory.embedding) if memory.embedding else None,
            "metadata": json.dumps(memory.metadata),
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "importance": memory.importance,
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed,

            # Temporal dynamics
            "activation_level": memory.activation_level,
            "dormancy_state": memory.dormancy_state.value,
            "reactivation_count": memory.reactivation_count,
            "last_reactivation": memory.last_reactivation,
            "decay_rate": memory.decay_rate,

            # Emotional intelligence
            "emotions": json.dumps(memory.emotions),
            "valence": memory.valence,
            "arousal": memory.arousal,
            "dominance": memory.dominance,
            "outcome_type": memory.outcome_type.value,
            "outcome_valence": memory.outcome_valence,
            "emotional_intensity": memory.emotional_intensity,

            # Conflict resolution
            "conflicted": memory.conflicted,
            "conflicts_with": json.dumps([str(id) for id in memory.conflicts_with]),
            "superseded_by": str(memory.superseded_by) if memory.superseded_by else None,
            "verification_requested": memory.verification_requested,
        }

        # Add type-specific fields
        if isinstance(memory, EpisodicMemory):
            memory_data.update({
                "session_id": memory.session_id,
                "user_message": memory.user_message,
                "assistant_response": memory.assistant_response,
                "context_summary": memory.context_summary,
            })
        elif isinstance(memory, SemanticMemory):
            memory_data.update({
                "subject": memory.subject,
                "predicate": memory.predicate,
                "object": memory.object,
                "confidence": memory.confidence,
                "source": memory.source,
                "verified": memory.verified,
                "related_concepts": json.dumps(memory.related_concepts),
            })
        elif isinstance(memory, ProceduralMemory):
            memory_data.update({
                "skill_name": memory.skill_name,
                "steps": json.dumps(memory.steps),
                "prerequisites": json.dumps(memory.prerequisites),
                "success_rate": memory.success_rate,
                "last_used": memory.last_used,
                "proficiency_level": memory.proficiency_level,
            })

        # Insert or replace memory
        columns = ", ".join(memory_data.keys())
        placeholders = ", ".join("?" * len(memory_data))

        await self._connection.execute(
            f"INSERT OR REPLACE INTO memories ({columns}) VALUES ({placeholders})",
            list(memory_data.values())
        )

        # Update FTS index
        await self._connection.execute("""
            INSERT OR REPLACE INTO memories_fts (rowid, content, user_message, assistant_response, context_summary)
            SELECT rowid, content, user_message, assistant_response, context_summary
            FROM memories WHERE id = ?
        """, (str(memory.id),))

        await self._connection.commit()

    async def retrieve_memory(self, memory_id: UUID) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        cursor = await self._connection.execute(
            "SELECT * FROM memories WHERE id = ?", (str(memory_id),)
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    async def search_memories_text(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """Search memories using full-text search."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        # Handle special case where we want all memories (no text search)
        if query in ("*", ""):
            return await self.get_all_memories(memory_types, limit)

        # Escape FTS5 special characters
        escaped_query = self._escape_fts_query(query)

        type_filter = ""
        params = [escaped_query]

        if memory_types:
            type_placeholders = ", ".join("?" * len(memory_types))
            type_filter = f"AND m.type IN ({type_placeholders})"
            params.extend([t.value for t in memory_types])

        params.append(limit)

        cursor = await self._connection.execute(f"""
            SELECT m.*, fts.rank
            FROM memories_fts fts
            JOIN memories m ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
            {type_filter}
            ORDER BY fts.rank
            LIMIT ?
        """, params)

        rows = await cursor.fetchall()
        results = []

        for row in rows:
            memory = self._row_to_memory(row[:-1])  # Exclude rank column
            score = 1.0 / (1.0 + abs(row[-1]))  # Convert rank to similarity score
            results.append((memory, score))

        return results

    def _escape_fts_query(self, query: str) -> str:
        """Escape FTS5 special characters in search query."""
        # FTS5 special characters that need escaping
        special_chars = ['"', "'", '*', '+', '-', '(', ')', '[', ']', '{', '}', '!', '?', '^', '~', ':', '@', ',', '.', ';']

        # Remove special characters or wrap in quotes
        cleaned_query = query
        for char in special_chars:
            cleaned_query = cleaned_query.replace(char, ' ')

        # Remove extra whitespace and return non-empty query
        cleaned_query = ' '.join(cleaned_query.split())

        # If query becomes empty after cleaning, return a safe default
        if not cleaned_query:
            return "content"  # Search for any content

        return cleaned_query

    async def get_all_memories(
        self,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """Get all memories without text search, ordered by creation time."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        type_filter = ""
        params = []

        if memory_types:
            type_placeholders = ", ".join("?" * len(memory_types))
            type_filter = f"WHERE type IN ({type_placeholders})"
            params.extend([t.value for t in memory_types])

        params.append(limit)

        cursor = await self._connection.execute(f"""
            SELECT * FROM memories
            {type_filter}
            ORDER BY created_at DESC
            LIMIT ?
        """, params)

        rows = await cursor.fetchall()
        results = []

        for row in rows:
            memory = self._row_to_memory(row)
            if memory:
                # Default score of 0.5 for non-search results
                results.append((memory, 0.5))

        return results

    async def search_memories_vector(
        self,
        query_embedding: List[float],
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_similarity: float = 0.3
    ) -> List[Tuple[Memory, float]]:
        """Search memories using vector similarity."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        # Try vector extension first
        try:
            return await self._search_with_vec_extension(
                query_embedding, memory_types, limit, min_similarity
            )
        except Exception:
            # Fallback to manual cosine similarity
            return await self._search_with_cosine_similarity(
                query_embedding, memory_types, limit, min_similarity
            )

    async def _search_with_vec_extension(
        self,
        query_embedding: List[float],
        memory_types: Optional[List[MemoryType]],
        limit: int,
        min_similarity: float
    ) -> List[Tuple[Memory, float]]:
        """Search using sqlite-vec extension."""
        type_filter = ""
        params = [self._serialize_embedding(query_embedding)]

        if memory_types:
            type_placeholders = ", ".join("?" * len(memory_types))
            type_filter = f"AND m.type IN ({type_placeholders})"
            params.extend([t.value for t in memory_types])

        params.extend([limit])

        cursor = await self._connection.execute(f"""
            SELECT m.*, vec.distance
            FROM memories_vec vec
            JOIN memories m ON m.id = vec.rowid
            WHERE vec.embedding MATCH ?
            {type_filter}
            ORDER BY vec.distance
            LIMIT ?
        """, params)

        rows = await cursor.fetchall()
        results = []

        for row in rows:
            memory = self._row_to_memory(row[:-1])
            similarity = 1.0 - row[-1]  # Convert distance to similarity
            if similarity >= min_similarity:
                results.append((memory, similarity))

        return results

    async def _search_with_cosine_similarity(
        self,
        query_embedding: List[float],
        memory_types: Optional[List[MemoryType]],
        limit: int,
        min_similarity: float
    ) -> List[Tuple[Memory, float]]:
        """Fallback search using manual cosine similarity calculation."""
        type_filter = ""
        params = []

        if memory_types:
            type_placeholders = ", ".join("?" * len(memory_types))
            type_filter = f"WHERE type IN ({type_placeholders})"
            params.extend([t.value for t in memory_types])

        cursor = await self._connection.execute(f"""
            SELECT * FROM memories {type_filter}
        """, params)

        rows = await cursor.fetchall()
        results = []
        query_vector = np.array(query_embedding)

        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding:
                memory_vector = np.array(memory.embedding)
                similarity = self._cosine_similarity(query_vector, memory_vector)
                if similarity >= min_similarity:
                    results.append((memory, similarity))

        # Sort by similarity and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def delete_memory(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        cursor = await self._connection.execute(
            "DELETE FROM memories WHERE id = ?", (str(memory_id),)
        )

        # Also delete from FTS index
        await self._connection.execute(
            "DELETE FROM memories_fts WHERE rowid IN (SELECT rowid FROM memories WHERE id = ?)",
            (str(memory_id),)
        )

        await self._connection.commit()
        return cursor.rowcount > 0

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        if self._connection is None:
            raise RuntimeError("Database not initialized")

        cursor = await self._connection.execute("""
            SELECT
                type,
                COUNT(*) as count,
                AVG(importance) as avg_importance,
                AVG(access_count) as avg_access_count
            FROM memories
            GROUP BY type
        """)

        stats_by_type = {}
        total_count = 0

        async for row in cursor:
            type_name, count, avg_importance, avg_access_count = row
            stats_by_type[type_name] = {
                "count": count,
                "avg_importance": avg_importance,
                "avg_access_count": avg_access_count
            }
            total_count += count

        return {
            "total_memories": total_count,
            "by_type": stats_by_type,
            "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0
        }

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding vector to bytes."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _deserialize_embedding(self, data: bytes) -> List[float]:
        """Deserialize embedding vector from bytes."""
        if not data:
            return []
        count = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f"{count}f", data))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _row_to_memory(self, row: Tuple) -> Memory:
        """Convert database row to Memory object."""
        # Column order must match actual database schema
        columns = [
            'id', 'type', 'content', 'embedding', 'metadata', 'created_at', 'updated_at',
            'importance', 'access_count', 'last_accessed',
            # Type-specific fields (original order)
            'session_id', 'user_message', 'assistant_response', 'context_summary', 'emotions',
            'subject', 'predicate', 'object', 'confidence', 'source', 'verified', 'related_concepts',
            'skill_name', 'steps', 'prerequisites', 'success_rate', 'last_used', 'proficiency_level',
            # Neuromorphic fields (added by migration)
            'activation_level', 'dormancy_state', 'reactivation_count', 'last_reactivation', 'decay_rate',
            'valence', 'arousal', 'dominance', 'outcome_type', 'outcome_valence', 'emotional_intensity',
            'conflicted', 'conflicts_with', 'superseded_by', 'verification_requested'
        ]

        data = dict(zip(columns, row))

        # Convert types
        data['id'] = UUID(data['id'])
        data['type'] = MemoryType(data['type'])
        if data['embedding']:
            data['embedding'] = self._deserialize_embedding(data['embedding'])
        if data['metadata']:
            data['metadata'] = json.loads(data['metadata'])
        else:
            data['metadata'] = {}

        # Import enums at runtime to avoid circular imports
        from .models import DormancyState, OutcomeType

        # Convert enum fields (with safety checks for migration)
        if 'dormancy_state' in data and data['dormancy_state']:
            try:
                data['dormancy_state'] = DormancyState(data['dormancy_state'])
            except ValueError:
                # Invalid dormancy state, default to ACTIVE
                data['dormancy_state'] = DormancyState.ACTIVE
        else:
            data['dormancy_state'] = DormancyState.ACTIVE

        if 'outcome_type' in data and data['outcome_type']:
            try:
                data['outcome_type'] = OutcomeType(data['outcome_type'])
            except ValueError:
                # Invalid outcome type, default to NEUTRAL
                data['outcome_type'] = OutcomeType.NEUTRAL
        else:
            data['outcome_type'] = OutcomeType.NEUTRAL

        # Convert JSON fields (with safety checks for migration)
        if 'emotions' in data and data['emotions']:
            data['emotions'] = json.loads(data['emotions'])
        else:
            data['emotions'] = {}

        if 'conflicts_with' in data and data['conflicts_with']:
            data['conflicts_with'] = [UUID(id_str) for id_str in json.loads(data['conflicts_with'])]
        else:
            data['conflicts_with'] = []

        if 'superseded_by' in data and data['superseded_by']:
            data['superseded_by'] = UUID(data['superseded_by'])

        # Set default values for missing neuromorphic fields
        neuromorphic_defaults = {
            'activation_level': 1.0,
            'reactivation_count': 0,
            'decay_rate': 0.1,
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'outcome_valence': 0.0,
            'emotional_intensity': 0.0,
            'conflicted': False,
            'verification_requested': False
        }

        for field, default_value in neuromorphic_defaults.items():
            if field not in data or data[field] is None:
                data[field] = default_value

        # Create appropriate memory type
        memory_type = data['type']

        if memory_type == MemoryType.EPISODIC:
            return EpisodicMemory(**{k: v for k, v in data.items() if v is not None})

        elif memory_type == MemoryType.SEMANTIC:
            if data['related_concepts']:
                data['related_concepts'] = json.loads(data['related_concepts'])
            else:
                data['related_concepts'] = []
            return SemanticMemory(**{k: v for k, v in data.items() if v is not None})

        elif memory_type == MemoryType.PROCEDURAL:
            if data['steps']:
                data['steps'] = json.loads(data['steps'])
            else:
                data['steps'] = []
            if data['prerequisites']:
                data['prerequisites'] = json.loads(data['prerequisites'])
            else:
                data['prerequisites'] = []
            return ProceduralMemory(**{k: v for k, v in data.items() if v is not None})

        else:
            return Memory(**{k: v for k, v in data.items() if v is not None})