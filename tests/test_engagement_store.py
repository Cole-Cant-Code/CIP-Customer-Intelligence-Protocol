"""Tests for cip_protocol.engagement.store."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from cip_protocol.engagement.store import EscalationStore


def _make_store(conn=None, *, entity_id_field="entity_id"):
    if conn is None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
    return EscalationStore(conn, entity_id_field=entity_id_field)


def _make_escalation(
    esc_id="esc-001",
    lead_id="lead-1",
    escalation_type="cold_to_warm",
    **overrides,
):
    base = {
        "id": esc_id,
        "lead_id": lead_id,
        "escalation_type": escalation_type,
        "old_status": "new",
        "new_status": "engaged",
        "score": 15.0,
        "entity_id": "e-100",
        "customer_name": "Alice",
        "customer_contact": "alice@test.com",
        "source_channel": "web",
        "triggering_action": "viewed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base


class TestEscalationStore:
    def test_save_and_get_pending(self):
        store = _make_store()
        esc = _make_escalation()
        store.save(esc)
        pending = store.get_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == "esc-001"
        assert pending[0]["entity_id"] == "e-100"

    def test_save_ignores_duplicates(self):
        store = _make_store()
        esc = _make_escalation()
        store.save(esc)
        store.save(esc)  # same id
        assert len(store.get_pending()) == 1

    def test_has_active_escalation(self):
        store = _make_store()
        store.save(_make_escalation())
        assert store.has_active_escalation("lead-1", "cold_to_warm") is True
        assert store.has_active_escalation("lead-1", "warm_to_hot") is False
        assert store.has_active_escalation("lead-999", "cold_to_warm") is False

    def test_mark_delivered(self):
        store = _make_store()
        store.save(_make_escalation())
        assert store.mark_delivered("esc-001") is True
        assert store.get_pending() == []
        # Already delivered — should return False
        assert store.mark_delivered("esc-001") is False

    def test_mark_delivered_nonexistent(self):
        store = _make_store()
        assert store.mark_delivered("nope") is False

    def test_get_pending_with_type_filter(self):
        store = _make_store()
        store.save(_make_escalation(esc_id="esc-a", escalation_type="cold_to_warm"))
        store.save(_make_escalation(esc_id="esc-b", escalation_type="warm_to_hot"))
        warm = store.get_pending(escalation_type="cold_to_warm")
        assert len(warm) == 1
        assert warm[0]["escalation_type"] == "cold_to_warm"

    def test_get_all_includes_delivered(self):
        store = _make_store()
        store.save(_make_escalation())
        store.mark_delivered("esc-001")
        all_escs = store.get_all()
        assert len(all_escs) == 1

    def test_get_all_with_type_filter(self):
        store = _make_store()
        store.save(_make_escalation(esc_id="esc-a", escalation_type="cold_to_warm"))
        store.save(_make_escalation(esc_id="esc-b", escalation_type="warm_to_hot"))
        result = store.get_all(escalation_type="warm_to_hot")
        assert len(result) == 1

    def test_reset(self):
        store = _make_store()
        store.save(_make_escalation())
        store.reset()
        assert store.get_pending() == []
        assert store.get_all() == []

    def test_auto_creates_lock(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        store = EscalationStore(conn)
        # threading.RLock() returns _thread.RLock — check duck-type
        assert hasattr(store._lock, "acquire")
        assert hasattr(store._lock, "release")

    def test_custom_entity_id_field_in_save(self):
        store = _make_store(entity_id_field="vehicle_id")
        esc = {
            "id": "esc-v1",
            "lead_id": "lead-1",
            "escalation_type": "cold_to_warm",
            "old_status": "new",
            "new_status": "engaged",
            "score": 12.0,
            "vehicle_id": "car-42",
            "customer_name": "",
            "customer_contact": "",
            "source_channel": "direct",
            "triggering_action": "viewed",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        store.save(esc)
        rows = store.get_pending()
        assert rows[0]["entity_id"] == "car-42"  # stored in entity_id column
        assert rows[0]["vehicle_id"] == "car-42"  # remapped for backward compat

    def test_save_missing_entity_field_raises(self):
        store = _make_store(entity_id_field="vehicle_id")
        esc = _make_escalation()
        # Has "entity_id" but not "vehicle_id" — should still work via fallback
        store.save(esc)
        assert len(store.get_pending()) == 1

    def test_save_missing_both_entity_fields_raises(self):
        store = _make_store(entity_id_field="vehicle_id")
        esc = _make_escalation()
        del esc["entity_id"]
        try:
            store.save(esc)
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_migrate_vehicle_id_column(self):
        """Existing table with vehicle_id column gets migrated to entity_id."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create old-style table with vehicle_id
        conn.executescript("""
            CREATE TABLE escalations (
                id TEXT PRIMARY KEY,
                lead_id TEXT NOT NULL,
                escalation_type TEXT NOT NULL,
                old_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                score REAL NOT NULL,
                vehicle_id TEXT NOT NULL,
                customer_name TEXT NOT NULL DEFAULT '',
                customer_contact TEXT NOT NULL DEFAULT '',
                source_channel TEXT NOT NULL DEFAULT 'direct',
                triggering_action TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                enriched_payload TEXT,
                delivered INTEGER NOT NULL DEFAULT 0,
                delivered_at TEXT
            );
        """)
        # Insert a row using old schema
        conn.execute(
            "INSERT INTO escalations (id, lead_id, escalation_type, old_status, "
            "new_status, score, vehicle_id, created_at) "
            "VALUES ('esc-old', 'lead-1', 'cold_to_warm', 'new', 'engaged', "
            "10.0, 'car-99', '2025-01-01T00:00:00')"
        )
        conn.commit()
        # Creating the store should migrate vehicle_id -> entity_id
        store = EscalationStore(conn, entity_id_field="vehicle_id")
        rows = store.get_pending()
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "car-99"
        assert rows[0]["vehicle_id"] == "car-99"  # remapped

    def test_get_pending_limit(self):
        store = _make_store()
        for i in range(5):
            store.save(_make_escalation(esc_id=f"esc-{i}"))
        assert len(store.get_pending(limit=3)) == 3

    def test_enriched_payload(self):
        store = _make_store()
        esc = _make_escalation(enriched_payload={"extra": "data"})
        store.save(esc)
        rows = store.get_pending()
        assert rows[0]["enriched_payload"] is not None
