import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ChatHistoryDB:
    """
    Manages chat history persistence using SQLite.
    Stores sessions and messages.
    """
    def __init__(self, db_path: str = "data/chat_history.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Sessions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Messages table
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                meta_json TEXT, -- flexible storage for judge results etc
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()

    def create_session(self, title: str = "New Chat") -> str:
        """Create a new chat session and return its ID."""
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO sessions (id, title) VALUES (?, ?)', (session_id, title))
        conn.commit()
        conn.close()
        return session_id

    def list_sessions(self) -> List[Tuple[str, str, str]]:
        """List all sessions (id, title, updated_at) ordered by recent."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC')
        sessions = c.fetchall()
        conn.close()
        return sessions

    def rename_session(self, session_id: str, new_title: str):
        """Rename a session."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE sessions SET title = ? WHERE id = ?', (new_title, session_id))
        conn.commit()
        conn.close()

    def delete_session(self, session_id: str):
        """Delete a session and its messages."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Messages cascade delete automatically or we do it manually if sqlite < 3.24 without PRAGMA foreign_keys=ON
        c.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str, meta: Optional[Dict] = None):
        """Add a message to a session."""
        msg_id = str(uuid.uuid4())
        meta_json = json.dumps(meta) if meta else "{}"
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO messages (id, session_id, role, content, meta_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (msg_id, session_id, role, content, meta_json))
        
        # Update session timestamp
        c.execute('UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (session_id,))
        
        conn.commit()
        conn.close()

    def get_session_history(self, session_id: str) -> List[Dict]:
        """
        Get full message history for a session in OpenAI/Gradio 'messages' format.
        Returns: [{"role": "user", "content": "..."}, ...]
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT role, content 
            FROM messages 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (session_id,))
        rows = c.fetchall()
        conn.close()
        
        return [{"role": r[0], "content": r[1]} for r in rows]

    def get_last_session(self) -> Optional[str]:
        """Get the ID of the most recently updated session."""
        sessions = self.list_sessions()
        if sessions:
            return sessions[0][0]
        return None
