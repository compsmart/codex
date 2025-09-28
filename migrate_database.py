#!/usr/bin/env python3
"""
Database migration script for neuromorphic memory fields.
Run this to update existing Evo databases with the new memory features.
"""

import asyncio
import sqlite3
from pathlib import Path

def migrate_database(db_path: str):
    """Migrate database to include neuromorphic memory fields."""

    print(f"Migrating database: {db_path}")

    # Use synchronous sqlite3 for the migration
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get existing columns
        cursor.execute("PRAGMA table_info(memories)")
        columns = cursor.fetchall()
        existing_columns = {col[1] for col in columns}

        print(f"Found {len(existing_columns)} existing columns")

        # Define new columns
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
        added_count = 0
        for column_name, column_def in new_columns.items():
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {column_name} {column_def}")
                    print(f"[OK] Added column: {column_name}")
                    added_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to add column {column_name}: {e}")

        # Create missing indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_activation_level ON memories(activation_level)",
            "CREATE INDEX IF NOT EXISTS idx_memories_dormancy_state ON memories(dormancy_state)",
            "CREATE INDEX IF NOT EXISTS idx_memories_valence ON memories(valence)",
            "CREATE INDEX IF NOT EXISTS idx_memories_emotional_intensity ON memories(emotional_intensity)"
        ]

        index_count = 0
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                index_count += 1
            except Exception as e:
                print(f"Warning: Could not create index: {e}")

        conn.commit()
        print(f"[OK] Migration complete! Added {added_count} columns and {index_count} indexes")

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

    return True

def main():
    """Main migration function."""

    # Default Evo database path
    default_db = Path.home() / ".evo" / "memory.db"

    if default_db.exists():
        print("Found Evo database, running migration...")
        success = migrate_database(str(default_db))
        if success:
            print("Migration completed successfully! You can now run 'evo chat'")
        else:
            print("Migration failed. Please check the error messages above.")
    else:
        print("No existing Evo database found. Migration not needed.")
        print(f"Looked for: {default_db}")

if __name__ == "__main__":
    main()