"""Server-side per-account conversation history.

Schema supports both the flat append API used by the gateway when recording
chat completions, and the tree-with-branches model used by the stock llama.cpp
webui (parent/children edges, currNode pointer, fork-of relationships).

Tree fields are stored alongside the flat columns; older rows simply have
NULL parent_id / empty children_json and behave as a single linear path.
"""

import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from threading import RLock


DEFAULT_HISTORY_PATH = os.environ.get("GRIMOIRE_HISTORY_PATH", "/var/lib/grimoire/history.sqlite3")
FALLBACK_HISTORY_PATH = os.path.expanduser("~/.local/share/grimoire/history.sqlite3")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def now_ms():
    return int(time.time() * 1000)


def identity_hash(token):
    """Hash a stable identity into a non-secret storage key."""
    if not token:
        token = "anonymous"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class HistoryStore:
    """Small SQLite-backed conversation store keyed by account hash."""

    def __init__(self, path=DEFAULT_HISTORY_PATH):
        self.path = path
        self._lock = RLock()
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except PermissionError:
            self.path = FALLBACK_HISTORY_PATH
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    CONV_TREE_COLUMNS = (
        ("curr_node_id", "TEXT"),
        ("forked_from_id", "TEXT"),
        ("mcp_overrides_json", "TEXT"),
        ("last_modified_ms", "INTEGER"),
    )
    MSG_TREE_COLUMNS = (
        ("parent_id", "TEXT"),
        ("children_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("type", "TEXT"),
        ("timestamp_ms", "INTEGER"),
        ("tool_calls", "TEXT"),
        ("tool_call_id", "TEXT"),
        ("reasoning_content", "TEXT"),
        ("extra_json", "TEXT"),
        ("timings_json", "TEXT"),
        ("model", "TEXT"),
    )

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                    ON conversations(user_hash, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON messages(conversation_id, created_at);
                """
            )
            self._migrate_tree_columns(conn)

    def _migrate_tree_columns(self, conn):
        """Idempotently add tree-aware columns. SQLite's ADD COLUMN has no
        IF NOT EXISTS, so we introspect via PRAGMA table_info first."""
        for table, columns in (
            ("conversations", self.CONV_TREE_COLUMNS),
            ("messages", self.MSG_TREE_COLUMNS),
        ):
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns:
                if name in existing:
                    continue
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv_timestamp ON messages(conversation_id, timestamp_ms)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_forked_from ON conversations(forked_from_id)"
        )

    def _ensure_owner(self, conn, user_hash, conversation_id):
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_hash = ?",
            (conversation_id, user_hash),
        ).fetchone()
        if not row:
            raise KeyError(f"Conversation '{conversation_id}' not found")

    def list_conversations(self, user_hash, limit=100):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, model, created_at, updated_at
                FROM conversations
                WHERE user_hash = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_hash, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def create_conversation(self, user_hash, title="New chat", model=None, messages=None):
        conversation_id = str(uuid.uuid4())
        now = utcnow()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_hash, title, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, user_hash, title or "New chat", model, now, now),
            )
            for message in messages or []:
                self._insert_message(conn, conversation_id, message.get("role", "user"), message.get("content"))
        return self.get_conversation(user_hash, conversation_id)

    def get_conversation(self, user_hash, conversation_id):
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            conversation = conn.execute(
                """
                SELECT id, title, model, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_hash = ?
                """,
                (conversation_id, user_hash),
            ).fetchone()
            messages = conn.execute(
                """
                SELECT id, role, content_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

            result = dict(conversation)
            result["messages"] = [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": json.loads(row["content_json"]),
                    "created_at": row["created_at"],
                }
                for row in messages
            ]
            return result

    def conversation_exists(self, user_hash, conversation_id):
        """Return True if the caller owns the conversation."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_hash = ?",
                (conversation_id, user_hash),
            ).fetchone()
        return row is not None

    def replace_conversation(self, user_hash, conversation_id, title=None, model=None, messages=None):
        now = utcnow()
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            updates = ["updated_at = ?"]
            values = [now]
            if title is not None:
                updates.append("title = ?")
                values.append(title or "New chat")
            if model is not None:
                updates.append("model = ?")
                values.append(model)
            values.append(conversation_id)
            conn.execute(f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?", values)

            if messages is not None:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                for message in messages:
                    self._insert_message(conn, conversation_id, message.get("role", "user"), message.get("content"))
        return self.get_conversation(user_hash, conversation_id)

    def delete_conversation(self, user_hash, conversation_id):
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def append_message(self, user_hash, conversation_id, role, content, model=None):
        now = utcnow()
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            self._insert_message(conn, conversation_id, role, content)
            if model is not None:
                conn.execute(
                    "UPDATE conversations SET model = ?, updated_at = ? WHERE id = ?",
                    (model, now, conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )

    def _insert_message(self, conn, conversation_id, role, content):
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content_json, created_at, timestamp_ms, children_json)
            VALUES (?, ?, ?, ?, ?, ?, '[]')
            """,
            (str(uuid.uuid4()), conversation_id, role, json.dumps(content), utcnow(), now_ms()),
        )

    # =====================================================================
    # Tree-aware API used by the stock llama.cpp webui via DatabaseService.
    # Webui supplies its own UUIDs and timestamps; we honor them so the
    # client/server stay in sync without round-trip ID rewrites.
    # =====================================================================

    @staticmethod
    def _row_to_conv_dict(row):
        out = {
            "id": row["id"],
            "name": row["title"],
            "title": row["title"],
            "model": row["model"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "currNode": row["curr_node_id"],
            "lastModified": row["last_modified_ms"] or 0,
            "forkedFromConversationId": row["forked_from_id"],
        }
        overrides = row["mcp_overrides_json"]
        if overrides:
            try:
                out["mcpServerOverrides"] = json.loads(overrides)
            except json.JSONDecodeError:
                pass
        return out

    @staticmethod
    def _row_to_msg_dict(row):
        try:
            content = json.loads(row["content_json"])
        except json.JSONDecodeError:
            content = row["content_json"]
        try:
            children = json.loads(row["children_json"] or "[]")
        except json.JSONDecodeError:
            children = []
        out = {
            "id": row["id"],
            "convId": row["conversation_id"],
            "role": row["role"],
            "content": content if isinstance(content, str) else json.dumps(content),
            "type": row["type"] or row["role"],
            "timestamp": row["timestamp_ms"] or 0,
            "parent": row["parent_id"],
            "children": children,
        }
        for column, key in (
            ("tool_calls", "toolCalls"),
            ("tool_call_id", "toolCallId"),
            ("reasoning_content", "reasoningContent"),
            ("model", "model"),
        ):
            value = row[column]
            if value is not None:
                out[key] = value
        for column, key in (("extra_json", "extra"), ("timings_json", "timings")):
            value = row[column]
            if not value:
                continue
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
        return out

    def list_conversations_tree(self, user_hash, limit=500):
        """List conversations with tree fields for the webui sidebar."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, user_hash, title, model, created_at, updated_at,
                       curr_node_id, forked_from_id, mcp_overrides_json, last_modified_ms
                FROM conversations
                WHERE user_hash = ?
                ORDER BY COALESCE(last_modified_ms, 0) DESC, updated_at DESC
                LIMIT ?
                """,
                (user_hash, limit),
            ).fetchall()
            return [self._row_to_conv_dict(row) for row in rows]

    def get_conversation_tree(self, user_hash, conversation_id):
        """Return one conversation's metadata + all its messages with tree edges."""
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            conv_row = conn.execute(
                """
                SELECT id, title, model, created_at, updated_at,
                       curr_node_id, forked_from_id, mcp_overrides_json, last_modified_ms
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
            msg_rows = conn.execute(
                """
                SELECT id, conversation_id, role, content_json, created_at,
                       parent_id, children_json, type, timestamp_ms,
                       tool_calls, tool_call_id, reasoning_content, extra_json, timings_json, model
                FROM messages
                WHERE conversation_id = ?
                ORDER BY COALESCE(timestamp_ms, 0) ASC, created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
            result = self._row_to_conv_dict(conv_row)
            result["messages"] = [self._row_to_msg_dict(row) for row in msg_rows]
            return result

    def upsert_conversation_tree(self, user_hash, payload):
        """Create or update a conversation using the webui's flat field set."""
        conv_id = payload.get("id") or str(uuid.uuid4())
        now = utcnow()
        last_modified = int(payload.get("lastModified") or now_ms())
        name = payload.get("name") or payload.get("title") or "New chat"
        model = payload.get("model")
        curr_node = payload.get("currNode")
        forked_from = payload.get("forkedFromConversationId")
        mcp_overrides = payload.get("mcpServerOverrides")
        mcp_json = json.dumps(mcp_overrides) if mcp_overrides is not None else None
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT user_hash FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            if existing and existing["user_hash"] != user_hash:
                raise PermissionError(f"Conversation '{conv_id}' not owned by caller")
            self._validate_fork_parent(conn, user_hash, conv_id, forked_from)
            if curr_node is not None:
                current_message = conn.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND conversation_id = ?",
                    (curr_node, conv_id),
                ).fetchone()
                if not current_message:
                    raise KeyError(
                        f"Current message '{curr_node}' not found in conversation '{conv_id}'"
                    )
            if existing:
                conn.execute(
                    """
                    UPDATE conversations
                    SET title = ?, model = ?, updated_at = ?, last_modified_ms = ?,
                        curr_node_id = ?, forked_from_id = ?, mcp_overrides_json = ?
                    WHERE id = ?
                    """,
                    (name, model, now, last_modified, curr_node, forked_from, mcp_json, conv_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO conversations
                        (id, user_hash, title, model, created_at, updated_at,
                         last_modified_ms, curr_node_id, forked_from_id, mcp_overrides_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (conv_id, user_hash, name, model, now, now,
                     last_modified, curr_node, forked_from, mcp_json),
                )
        return self.get_conversation_tree(user_hash, conv_id)

    def patch_conversation_tree(self, user_hash, conversation_id, updates):
        """Apply a partial update from the webui's updateConversation."""
        if not isinstance(updates, dict):
            raise ValueError("updates must be a JSON object")
        column_map = {
            "name": "title",
            "title": "title",
            "model": "model",
            "currNode": "curr_node_id",
            "forkedFromConversationId": "forked_from_id",
            "lastModified": "last_modified_ms",
        }
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            if "forkedFromConversationId" in updates:
                self._validate_fork_parent(
                    conn, user_hash, conversation_id, updates["forkedFromConversationId"]
                )
            if updates.get("currNode") is not None:
                current_message = conn.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND conversation_id = ?",
                    (updates["currNode"], conversation_id),
                ).fetchone()
                if not current_message:
                    raise KeyError(
                        f"Current message '{updates['currNode']}' not found in conversation '{conversation_id}'"
                    )
            sets = ["updated_at = ?"]
            values = [utcnow()]
            for key, value in updates.items():
                if key == "mcpServerOverrides":
                    sets.append("mcp_overrides_json = ?")
                    values.append(json.dumps(value) if value is not None else None)
                    continue
                column = column_map.get(key)
                if not column:
                    continue
                sets.append(f"{column} = ?")
                values.append(value)
            if "lastModified" not in updates:
                sets.append("last_modified_ms = ?")
                values.append(now_ms())
            values.append(conversation_id)
            conn.execute(f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", values)
        return self.get_conversation_tree(user_hash, conversation_id)

    def delete_conversation_with_options(self, user_hash, conversation_id, delete_with_forks=False):
        """Delete a conversation; optionally cascade through fork descendants."""
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            if delete_with_forks:
                to_delete = [conversation_id]
                queue = [conversation_id]
                visited = {conversation_id}
                while queue:
                    parent = queue.pop()
                    children = conn.execute(
                        "SELECT id FROM conversations WHERE forked_from_id = ? AND user_hash = ?",
                        (parent, user_hash),
                    ).fetchall()
                    for child in children:
                        if child["id"] in visited:
                            continue
                        visited.add(child["id"])
                        to_delete.append(child["id"])
                        queue.append(child["id"])
                placeholders = ",".join("?" * len(to_delete))
                conn.execute(
                    f"DELETE FROM conversations WHERE id IN ({placeholders})",
                    to_delete,
                )
            else:
                row = conn.execute(
                    "SELECT forked_from_id FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                new_parent = row["forked_from_id"] if row else None
                conn.execute(
                    "UPDATE conversations SET forked_from_id = ? WHERE forked_from_id = ? AND user_hash = ?",
                    (new_parent, conversation_id, user_hash),
                )
                conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def create_message_branch(self, user_hash, conversation_id, payload):
        """Create a new message under parent_id and update parent's children + currNode."""
        msg_id = payload.get("id") or str(uuid.uuid4())
        parent_id = payload.get("parent")
        role = payload.get("role") or "user"
        msg_type = payload.get("type") or role
        content = payload.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        timestamp_ms = int(payload.get("timestamp") or now_ms())
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            if parent_id is not None:
                parent_row = conn.execute(
                    "SELECT children_json FROM messages WHERE id = ? AND conversation_id = ?",
                    (parent_id, conversation_id),
                ).fetchone()
                if not parent_row:
                    raise KeyError(f"Parent message '{parent_id}' not found in conversation '{conversation_id}'")
            conn.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content_json, created_at,
                    parent_id, children_json, type, timestamp_ms,
                    tool_calls, tool_call_id, reasoning_content, extra_json, timings_json, model
                )
                VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id, conversation_id, role, json.dumps(content), utcnow(),
                    parent_id, msg_type, timestamp_ms,
                    payload.get("toolCalls"),
                    payload.get("toolCallId"),
                    payload.get("reasoningContent"),
                    json.dumps(payload["extra"]) if payload.get("extra") is not None else None,
                    json.dumps(payload["timings"]) if payload.get("timings") is not None else None,
                    payload.get("model"),
                ),
            )
            if parent_id is not None:
                self._append_child(conn, conversation_id, parent_id, msg_id)
            conn.execute(
                "UPDATE conversations SET curr_node_id = ?, last_modified_ms = ?, updated_at = ? WHERE id = ?",
                (msg_id, now_ms(), utcnow(), conversation_id),
            )
            row = conn.execute(
                """
                SELECT id, conversation_id, role, content_json, created_at,
                       parent_id, children_json, type, timestamp_ms,
                       tool_calls, tool_call_id, reasoning_content, extra_json, timings_json, model
                FROM messages WHERE id = ?
                """,
                (msg_id,),
            ).fetchone()
        return self._row_to_msg_dict(row)

    def update_message_tree(self, user_hash, conversation_id, message_id, updates):
        """Apply a partial update from the webui's updateMessage."""
        column_map = {
            "role": "role",
            "type": "type",
            "content": "content_json",
            "timestamp": "timestamp_ms",
            "parent": "parent_id",
            "toolCalls": "tool_calls",
            "toolCallId": "tool_call_id",
            "reasoningContent": "reasoning_content",
            "model": "model",
        }
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            current = conn.execute(
                "SELECT parent_id FROM messages WHERE id = ? AND conversation_id = ?",
                (message_id, conversation_id),
            ).fetchone()
            if not current:
                raise KeyError(f"Message '{message_id}' not in conversation '{conversation_id}'")
            old_parent_id = current["parent_id"]
            new_parent_id = updates.get("parent", old_parent_id)
            if new_parent_id is not None:
                parent = conn.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND conversation_id = ?",
                    (new_parent_id, conversation_id),
                ).fetchone()
                if not parent:
                    raise KeyError(
                        f"Parent message '{new_parent_id}' not found in conversation '{conversation_id}'"
                    )
                cursor = new_parent_id
                visited = set()
                while cursor is not None:
                    if cursor == message_id or cursor in visited:
                        raise ValueError("Message parent would create a cycle")
                    visited.add(cursor)
                    row = conn.execute(
                        "SELECT parent_id FROM messages WHERE id = ? AND conversation_id = ?",
                        (cursor, conversation_id),
                    ).fetchone()
                    cursor = row["parent_id"] if row else None
            if "children" in updates:
                requested_children = updates["children"] or []
                if (
                    not isinstance(requested_children, list)
                    or any(not isinstance(child, str) for child in requested_children)
                    or len(requested_children) != len(set(requested_children))
                    or message_id in requested_children
                ):
                    raise ValueError("Message children must be unique message IDs and cannot include itself")
                actual_children = {
                    row["id"]
                    for row in conn.execute(
                        "SELECT id FROM messages WHERE parent_id = ? AND conversation_id = ?",
                        (message_id, conversation_id),
                    ).fetchall()
                }
                if set(requested_children) != actual_children:
                    raise ValueError("Message children must match their parent links")
            sets = []
            values = []
            for key, value in updates.items():
                if key == "children":
                    sets.append("children_json = ?")
                    values.append(json.dumps(value or []))
                    continue
                if key == "extra":
                    sets.append("extra_json = ?")
                    values.append(json.dumps(value) if value is not None else None)
                    continue
                if key == "timings":
                    sets.append("timings_json = ?")
                    values.append(json.dumps(value) if value is not None else None)
                    continue
                if key == "content":
                    sets.append("content_json = ?")
                    values.append(json.dumps(value) if not isinstance(value, str) else json.dumps(value))
                    continue
                column = column_map.get(key)
                if not column:
                    continue
                sets.append(f"{column} = ?")
                values.append(value)
            if not sets:
                return
            values.extend([message_id, conversation_id])
            conn.execute(
                f"UPDATE messages SET {', '.join(sets)} WHERE id = ? AND conversation_id = ?",
                values,
            )
            if new_parent_id != old_parent_id:
                if old_parent_id:
                    self._remove_child(conn, conversation_id, old_parent_id, message_id)
                if new_parent_id:
                    self._append_child(conn, conversation_id, new_parent_id, message_id)
            conn.execute(
                "UPDATE conversations SET last_modified_ms = ?, updated_at = ? WHERE id = ?",
                (now_ms(), utcnow(), conversation_id),
            )

    def delete_message_tree(self, user_hash, conversation_id, message_id, cascade=False):
        """Delete a single message (unlinking from parent) or the whole subtree."""
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            row = conn.execute(
                "SELECT parent_id FROM messages WHERE id = ? AND conversation_id = ?",
                (message_id, conversation_id),
            ).fetchone()
            if not row:
                raise KeyError(f"Message '{message_id}' not in conversation '{conversation_id}'")
            parent_id = row["parent_id"]
            conv_row = conn.execute(
                "SELECT curr_node_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            current_node = conv_row["curr_node_id"] if conv_row else None
            if cascade:
                deleted = []
                queue = [message_id]
                visited = set()
                while queue:
                    current = queue.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    deleted.append(current)
                    child_rows = conn.execute(
                        "SELECT id FROM messages WHERE parent_id = ? AND conversation_id = ?",
                        (current, conversation_id),
                    ).fetchall()
                    queue.extend(child["id"] for child in child_rows)
                placeholders = ",".join("?" * len(deleted))
                conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    deleted,
                )
                if parent_id:
                    self._remove_child(conn, conversation_id, parent_id, message_id)
                conn.execute(
                    "UPDATE conversations SET curr_node_id = ?, last_modified_ms = ?, updated_at = ? WHERE id = ?",
                    (None if current_node in deleted else current_node, now_ms(), utcnow(), conversation_id),
                )
                return deleted
            child_rows = conn.execute(
                "SELECT id FROM messages WHERE parent_id = ? AND conversation_id = ?",
                (message_id, conversation_id),
            ).fetchall()
            child_ids = [child["id"] for child in child_rows]
            if child_ids:
                conn.execute(
                    "UPDATE messages SET parent_id = ? WHERE parent_id = ? AND conversation_id = ?",
                    (parent_id, message_id, conversation_id),
                )
            conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            if parent_id:
                self._remove_child(conn, conversation_id, parent_id, message_id)
                for child_id in child_ids:
                    self._append_child(conn, conversation_id, parent_id, child_id)
            conn.execute(
                "UPDATE conversations SET curr_node_id = ?, last_modified_ms = ?, updated_at = ? WHERE id = ?",
                (None if current_node == message_id else current_node, now_ms(), utcnow(), conversation_id),
            )
            return [message_id]

    def fork_conversation(self, user_hash, source_conversation_id, at_message_id, name, include_attachments=True):
        """Create a new conversation containing the path from root to at_message_id."""
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, source_conversation_id)
            target = conn.execute(
                "SELECT id FROM messages WHERE id = ? AND conversation_id = ?",
                (at_message_id, source_conversation_id),
            ).fetchone()
            if not target:
                raise KeyError(f"Message '{at_message_id}' not in conversation '{source_conversation_id}'")

            path_ids = []
            cursor = at_message_id
            visited = set()
            while cursor is not None:
                if cursor in visited:
                    raise ValueError("Conversation contains a cyclic message parent chain")
                visited.add(cursor)
                row = conn.execute(
                    "SELECT parent_id FROM messages WHERE id = ? AND conversation_id = ?",
                    (cursor, source_conversation_id),
                ).fetchone()
                if not row:
                    raise ValueError("Conversation contains a dangling message parent")
                path_ids.append(cursor)
                cursor = row["parent_id"]
            path_ids.reverse()

            placeholders = ",".join("?" * len(path_ids))
            path_rows = conn.execute(
                f"""
                SELECT id, role, content_json, type, timestamp_ms,
                       tool_calls, tool_call_id, reasoning_content, extra_json, timings_json, model
                FROM messages
                WHERE id IN ({placeholders}) AND conversation_id = ?
                """,
                [*path_ids, source_conversation_id],
            ).fetchall()
            by_id = {row["id"]: row for row in path_rows}

            new_conv_id = str(uuid.uuid4())
            now = utcnow()
            now_ms_value = now_ms()
            conn.execute(
                """
                INSERT INTO conversations
                    (id, user_hash, title, model, created_at, updated_at,
                     last_modified_ms, curr_node_id, forked_from_id)
                VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, ?)
                """,
                (new_conv_id, user_hash, name or "Forked chat", now, now, now_ms_value, source_conversation_id),
            )

            id_map = {old_id: str(uuid.uuid4()) for old_id in path_ids}
            previous_new_id = None
            for old_id in path_ids:
                row = by_id[old_id]
                new_id = id_map[old_id]
                extra = row["extra_json"] if include_attachments else None
                conn.execute(
                    """
                    INSERT INTO messages (
                        id, conversation_id, role, content_json, created_at,
                        parent_id, children_json, type, timestamp_ms,
                        tool_calls, tool_call_id, reasoning_content, extra_json, timings_json, model
                    )
                    VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id, new_conv_id, row["role"], row["content_json"], now,
                        previous_new_id, row["type"], row["timestamp_ms"],
                        row["tool_calls"], row["tool_call_id"], row["reasoning_content"],
                        extra, row["timings_json"], row["model"],
                    ),
                )
                if previous_new_id:
                    self._append_child(conn, new_conv_id, previous_new_id, new_id)
                previous_new_id = new_id

            conn.execute(
                "UPDATE conversations SET curr_node_id = ? WHERE id = ?",
                (previous_new_id, new_conv_id),
            )
        return self.get_conversation_tree(user_hash, new_conv_id)

    def import_conversations_tree(self, user_hash, payload):
        """Bulk import in the webui's exported shape: [{conv, messages}, ...]."""
        imported = 0
        skipped = 0
        with self._lock, self._connect() as conn:
            entries = payload or []
            import_ids = [
                entry.get("conv", {}).get("id")
                for entry in entries
                if isinstance(entry, dict) and isinstance(entry.get("conv"), dict)
            ]
            import_id_counts = Counter(import_ids)
            duplicate_ids = {
                conv_id for conv_id, count in import_id_counts.items() if conv_id and count > 1
            }
            pending_parents = {}
            candidate_entries = {}
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("conv"), dict):
                    continue
                candidate = entry["conv"]
                candidate_id = candidate.get("id")
                if not isinstance(candidate_id, str) or not candidate_id or candidate_id in duplicate_ids:
                    continue
                exists = conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ?", (candidate_id,)
                ).fetchone()
                if not exists:
                    pending_parents[candidate_id] = candidate.get("forkedFromConversationId")
                    candidate_entries[candidate_id] = entry

            ordered_ids = []
            children_by_parent = {}
            incoming = {}
            for candidate_id, parent_id in pending_parents.items():
                incoming[candidate_id] = int(parent_id in pending_parents)
                if parent_id in pending_parents:
                    children_by_parent.setdefault(parent_id, []).append(candidate_id)
            ready = deque(
                candidate_id for candidate_id, count in incoming.items() if count == 0
            )
            while ready:
                candidate_id = ready.popleft()
                ordered_ids.append(candidate_id)
                for child_id in children_by_parent.get(candidate_id, []):
                    incoming[child_id] -= 1
                    if incoming[child_id] == 0:
                        ready.append(child_id)
            cyclic_ids = set(pending_parents) - set(ordered_ids)

            non_candidates = [
                entry
                for entry in entries
                if not isinstance(entry, dict)
                or not isinstance(entry.get("conv"), dict)
                or entry["conv"].get("id") not in candidate_entries
            ]
            entries = non_candidates + [
                candidate_entries[candidate_id]
                for candidate_id in ordered_ids
                if candidate_id not in cyclic_ids
            ]
            skipped += len(cyclic_ids)

            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("conv"), dict):
                    skipped += 1
                    continue
                conv = entry.get("conv") or {}
                conv_id = conv.get("id")
                if not isinstance(conv_id, str) or not conv_id or conv_id in duplicate_ids:
                    skipped += 1
                    continue
                row = conn.execute(
                    "SELECT user_hash FROM conversations WHERE id = ?",
                    (conv_id,),
                ).fetchone()
                if row:
                    skipped += 1
                    continue
                forked_from = conv.get("forkedFromConversationId")
                try:
                    self._validate_fork_parent(
                        conn,
                        user_hash,
                        conv_id,
                        forked_from,
                    )
                except (KeyError, PermissionError):
                    # Exported forks are self-contained. If their original
                    # conversation is unavailable to this account, import them
                    # as a detached conversation instead of retaining a dangling
                    # or cross-account relationship.
                    forked_from = None
                except ValueError:
                    skipped += 1
                    continue
                messages = entry.get("messages") or []
                if not isinstance(messages, list) or not all(isinstance(msg, dict) for msg in messages):
                    skipped += 1
                    continue
                if any(msg.get("id") is not None and not isinstance(msg.get("id"), str) for msg in messages):
                    skipped += 1
                    continue
                source_ids = [msg.get("id") or f"__missing_{index}" for index, msg in enumerate(messages)]
                if len(source_ids) != len(set(source_ids)):
                    skipped += 1
                    continue
                id_map = {source_id: str(uuid.uuid4()) for source_id in source_ids}
                references_valid = all(
                    (msg.get("parent") is None or isinstance(msg.get("parent"), str))
                    and isinstance(msg.get("children") or [], list)
                    and all(isinstance(child, str) for child in (msg.get("children") or []))
                    and (msg.get("parent") is None or msg.get("parent") in id_map)
                    and all(child in id_map for child in (msg.get("children") or []))
                    for msg in messages
                )
                if references_valid:
                    parents = {
                        source_id: msg.get("parent")
                        for source_id, msg in zip(source_ids, messages)
                    }
                    for source_id in source_ids:
                        cursor = source_id
                        visited = set()
                        while cursor is not None:
                            if cursor in visited:
                                references_valid = False
                                break
                            visited.add(cursor)
                            cursor = parents.get(cursor)
                        if not references_valid:
                            break
                if references_valid:
                    expected_children = {source_id: [] for source_id in source_ids}
                    for source_id, parent_id in parents.items():
                        if parent_id is not None:
                            expected_children[parent_id].append(source_id)
                    references_valid = all(
                        isinstance(msg.get("children") or [], list)
                        and len(msg.get("children") or []) == len(set(msg.get("children") or []))
                        and set(msg.get("children") or []) == set(expected_children[source_id])
                        for source_id, msg in zip(source_ids, messages)
                    )
                curr_node = conv.get("currNode")
                if not references_valid or (curr_node is not None and curr_node not in id_map):
                    skipped += 1
                    continue
                now = utcnow()
                conn.execute(
                    """
                    INSERT INTO conversations
                        (id, user_hash, title, model, created_at, updated_at,
                         last_modified_ms, curr_node_id, forked_from_id, mcp_overrides_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conv_id,
                        user_hash,
                        conv.get("name") or "New chat",
                        conv.get("model"),
                        now,
                        now,
                        int(conv.get("lastModified") or now_ms()),
                        id_map.get(curr_node),
                        forked_from,
                        json.dumps(conv["mcpServerOverrides"]) if conv.get("mcpServerOverrides") is not None else None,
                    ),
                )
                for index, msg in enumerate(messages):
                    source_id = msg.get("id") or f"__missing_{index}"
                    children = [id_map[child] for child in (msg.get("children") or [])]
                    extra = msg.get("extra")
                    timings = msg.get("timings")
                    content = msg.get("content") or ""
                    if not isinstance(content, str):
                        content = json.dumps(content)
                    conn.execute(
                        """
                        INSERT INTO messages (
                            id, conversation_id, role, content_json, created_at,
                            parent_id, children_json, type, timestamp_ms,
                            tool_calls, tool_call_id, reasoning_content, extra_json, timings_json, model
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            id_map[source_id],
                            conv_id,
                            msg.get("role") or "user",
                            json.dumps(content),
                            utcnow(),
                            id_map.get(msg.get("parent")),
                            json.dumps(children),
                            msg.get("type") or msg.get("role") or "user",
                            int(msg.get("timestamp") or now_ms()),
                            msg.get("toolCalls"),
                            msg.get("toolCallId"),
                            msg.get("reasoningContent"),
                            json.dumps(extra) if extra is not None else None,
                            json.dumps(timings) if timings is not None else None,
                            msg.get("model"),
                        ),
                    )
                imported += 1
        return {"imported": imported, "skipped": skipped}

    def find_message_conversation(self, user_hash, message_id):
        """Return the conversation_id that owns this message_id, or None."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT m.conversation_id FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE m.id = ? AND c.user_hash = ?
                """,
                (message_id, user_hash),
            ).fetchone()
            return row["conversation_id"] if row else None

    @staticmethod
    def _validate_fork_parent(conn, user_hash, conversation_id, parent_id):
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("Fork parent must be a conversation ID")
        cursor = parent_id
        visited = set()
        while cursor is not None:
            if cursor == conversation_id or cursor in visited:
                raise ValueError("Conversation fork parent would create a cycle")
            visited.add(cursor)
            row = conn.execute(
                "SELECT user_hash, forked_from_id FROM conversations WHERE id = ?",
                (cursor,),
            ).fetchone()
            if not row:
                raise KeyError(f"Fork parent conversation '{cursor}' not found")
            if row["user_hash"] != user_hash:
                raise PermissionError(f"Fork parent conversation '{cursor}' not owned by caller")
            cursor = row["forked_from_id"]

    @staticmethod
    def _append_child(conn, conversation_id, parent_id, child_id):
        row = conn.execute(
            "SELECT children_json FROM messages WHERE id = ? AND conversation_id = ?",
            (parent_id, conversation_id),
        ).fetchone()
        if not row:
            raise KeyError(f"Parent message '{parent_id}' not found in conversation '{conversation_id}'")
        try:
            children = json.loads(row["children_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            children = []
        if child_id not in children:
            children.append(child_id)
        conn.execute(
            "UPDATE messages SET children_json = ? WHERE id = ? AND conversation_id = ?",
            (json.dumps(children), parent_id, conversation_id),
        )

    @staticmethod
    def _remove_child(conn, conversation_id, parent_id, child_id):
        row = conn.execute(
            "SELECT children_json FROM messages WHERE id = ? AND conversation_id = ?",
            (parent_id, conversation_id),
        ).fetchone()
        if not row:
            return
        try:
            children = json.loads(row["children_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            children = []
        children = [c for c in children if c != child_id]
        conn.execute(
            "UPDATE messages SET children_json = ? WHERE id = ? AND conversation_id = ?",
            (json.dumps(children), parent_id, conversation_id),
        )


history_store = HistoryStore()
