"""Escalation persistence backed by SQLite.

Generic version: the column that stores the domain entity uses the name
``entity_id`` in the SQL schema, but the :class:`EscalationStore` reads
the configurable ``entity_id_field`` key from escalation dicts so that
domain servers can pass e.g. ``"vehicle_id"`` without renaming.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

_CREATE_SQL = """\
CREATE TABLE IF NOT EXISTS escalations (
    id                TEXT PRIMARY KEY,
    lead_id           TEXT NOT NULL,
    escalation_type   TEXT NOT NULL,
    old_status        TEXT NOT NULL,
    new_status        TEXT NOT NULL,
    score             REAL NOT NULL,
    entity_id         TEXT NOT NULL,
    customer_name     TEXT NOT NULL DEFAULT '',
    customer_contact  TEXT NOT NULL DEFAULT '',
    source_channel    TEXT NOT NULL DEFAULT 'direct',
    triggering_action TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    enriched_payload  TEXT,
    delivered         INTEGER NOT NULL DEFAULT 0,
    delivered_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_escalations_lead_id
    ON escalations(lead_id);
CREATE INDEX IF NOT EXISTS idx_escalations_created_at
    ON escalations(created_at);
CREATE INDEX IF NOT EXISTS idx_escalations_delivered
    ON escalations(delivered);
"""

_MIGRATE_VEHICLE_ID_SQL = """\
ALTER TABLE escalations RENAME COLUMN vehicle_id TO entity_id;
"""


class EscalationStore:
    """Thread-safe escalation persistence sharing an existing SQLite connection.

    Parameters
    ----------
    conn:
        An open ``sqlite3.Connection`` (with ``row_factory = sqlite3.Row``
        recommended).
    lock:
        Optional ``threading.RLock``.  One is created automatically if not
        supplied.
    entity_id_field:
        The key to read from escalation dicts when extracting the domain
        entity identifier (default ``"entity_id"``).  AutoCIP passes
        ``"vehicle_id"`` here.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock | None = None,
        *,
        entity_id_field: str = "entity_id",
    ) -> None:
        self._conn = conn
        self._lock = lock if lock is not None else threading.RLock()
        self._entity_id_field = entity_id_field
        with self._lock:
            self._conn.executescript(_CREATE_SQL)
            self._migrate_if_needed()

    def _migrate_if_needed(self) -> None:
        """Rename legacy ``vehicle_id`` column to ``entity_id`` if present."""
        cursor = self._conn.execute("PRAGMA table_info(escalations)")
        columns = {row[1] for row in cursor.fetchall()}
        if "vehicle_id" in columns and "entity_id" not in columns:
            self._conn.executescript(_MIGRATE_VEHICLE_ID_SQL)

    def _remap_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """If entity_id_field differs from 'entity_id', copy the value under both keys."""
        if self._entity_id_field != "entity_id" and "entity_id" in row:
            row[self._entity_id_field] = row["entity_id"]
        return row

    def save(self, escalation: dict[str, Any]) -> None:
        """Persist an escalation record.  Ignores duplicates by id."""
        if self._entity_id_field in escalation:
            entity_value = escalation[self._entity_id_field]
        elif "entity_id" in escalation:
            entity_value = escalation["entity_id"]
        else:
            raise KeyError(
                f"Escalation dict missing both '{self._entity_id_field}' "
                f"and 'entity_id' keys."
            )
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO escalations
                   (id, lead_id, escalation_type, old_status, new_status,
                    score, entity_id, customer_name, customer_contact,
                    source_channel, triggering_action, created_at,
                    enriched_payload, delivered, delivered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
                (
                    escalation["id"],
                    escalation["lead_id"],
                    escalation["escalation_type"],
                    escalation["old_status"],
                    escalation["new_status"],
                    escalation["score"],
                    entity_value,
                    escalation.get("customer_name", ""),
                    escalation.get("customer_contact", ""),
                    escalation.get("source_channel", "direct"),
                    escalation.get("triggering_action", ""),
                    escalation["created_at"],
                    json.dumps(escalation.get("enriched_payload"))
                    if escalation.get("enriched_payload")
                    else None,
                ),
            )
            self._conn.commit()

    def has_active_escalation(self, lead_id: str, escalation_type: str) -> bool:
        """True if this lead already has an undelivered escalation of this type."""
        with self._lock:
            row = self._conn.execute(
                """SELECT 1 FROM escalations
                   WHERE lead_id = ? AND escalation_type = ? AND delivered = 0
                   LIMIT 1""",
                (lead_id, escalation_type),
            ).fetchone()
            return row is not None

    def get_pending(
        self,
        *,
        limit: int = 50,
        escalation_type: str = "",
    ) -> list[dict[str, Any]]:
        """Return undelivered escalations, newest first."""
        with self._lock:
            if escalation_type:
                rows = self._conn.execute(
                    """SELECT * FROM escalations
                       WHERE delivered = 0 AND escalation_type = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (escalation_type, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM escalations
                       WHERE delivered = 0
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [self._remap_row(dict(r)) for r in rows]

    def get_all(
        self,
        *,
        limit: int = 50,
        days: int = 30,
        escalation_type: str = "",
    ) -> list[dict[str, Any]]:
        """Return all recent escalations regardless of delivery status."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            if escalation_type:
                rows = self._conn.execute(
                    """SELECT * FROM escalations
                       WHERE created_at > ? AND escalation_type = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (since, escalation_type, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM escalations
                       WHERE created_at > ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (since, limit),
                ).fetchall()
            return [self._remap_row(dict(r)) for r in rows]

    def mark_delivered(self, escalation_id: str) -> bool:
        """Mark an escalation as delivered.  Returns True if a row was updated."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE escalations
                   SET delivered = 1, delivered_at = ?
                   WHERE id = ? AND delivered = 0""",
                (now_iso, escalation_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def reset(self) -> None:
        """Clear all escalation records.  Intended for tests."""
        with self._lock:
            self._conn.execute("DELETE FROM escalations")
            self._conn.commit()
