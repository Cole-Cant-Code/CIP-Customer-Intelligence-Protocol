"""Lead escalation detection — pure logic, no DB or I/O.

Parameterized version: transitions and entity-ID field name are supplied
via :class:`EscalationConfig` rather than hard-coded module constants.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

EscalationCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class EscalationConfig:
    """Domain-specific escalation settings.

    Parameters
    ----------
    transitions:
        ``{(old_status, new_status): escalation_type}`` map.
    entity_id_field:
        Key name used for the domain entity (e.g. ``"vehicle_id"``,
        ``"property_id"``).  Defaults to ``"entity_id"``.
    """

    transitions: dict[tuple[str, str], str]
    entity_id_field: str = "entity_id"


def check_escalation(
    *,
    config: EscalationConfig,
    lead_id: str,
    old_status: str,
    new_status: str,
    score: float,
    entity_id: str,
    customer_name: str = "",
    customer_contact: str = "",
    source_channel: str = "",
    action: str = "",
    callbacks: list[EscalationCallback] | None = None,
) -> dict[str, Any] | None:
    """Return an escalation record if the transition warrants one, else ``None``.

    Does NOT persist — the caller is responsible for dedup and storage.
    """
    if old_status == new_status:
        return None

    escalation_type = config.transitions.get((old_status, new_status))
    if escalation_type is None:
        return None

    escalation: dict[str, Any] = {
        "id": f"esc-{uuid.uuid4().hex[:12]}",
        "lead_id": lead_id,
        "escalation_type": escalation_type,
        "old_status": old_status,
        "new_status": new_status,
        "score": score,
        config.entity_id_field: entity_id,
        "customer_name": customer_name,
        "customer_contact": customer_contact,
        "source_channel": source_channel,
        "triggering_action": action,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if callbacks:
        for cb in callbacks:
            try:
                cb(escalation)
            except Exception:
                logger.exception("Escalation callback failed")

    return escalation


class EscalationDetector:
    """Stateful wrapper around :func:`check_escalation` with a callback registry."""

    def __init__(self, config: EscalationConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._callbacks: list[EscalationCallback] = []

    @property
    def config(self) -> EscalationConfig:
        return self._config

    def register_callback(self, cb: EscalationCallback) -> None:
        """Register a callback to be fired on every escalation event."""
        with self._lock:
            self._callbacks.append(cb)

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks."""
        with self._lock:
            self._callbacks.clear()

    def check(self, **kwargs: Any) -> dict[str, Any] | None:
        """Delegate to :func:`check_escalation` with this detector's config and callbacks."""
        with self._lock:
            cbs = list(self._callbacks)
        return check_escalation(config=self._config, callbacks=cbs, **kwargs)
