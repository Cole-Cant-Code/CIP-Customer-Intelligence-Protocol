"""Structured telemetry primitives for CIP protocol components."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TelemetryEvent:
    """Single structured telemetry event."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)


@runtime_checkable
class TelemetrySink(Protocol):
    """Telemetry sink protocol."""

    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event."""
        raise NotImplementedError


class NoOpTelemetrySink:
    """Default sink that records nothing."""

    def emit(self, event: TelemetryEvent) -> None:
        _ = event


class InMemoryTelemetrySink:
    """Test-friendly sink that stores events in memory."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class LoggerTelemetrySink:
    """Sink that emits structured events through Python logging."""

    def __init__(self, logger_name: str = "cip_protocol.telemetry") -> None:
        self.logger = logging.getLogger(logger_name)

    def emit(self, event: TelemetryEvent) -> None:
        self.logger.info(
            "telemetry_event",
            extra={
                "event_name": event.name,
                "event_timestamp_ms": event.timestamp_ms,
                "event_attributes": event.attributes,
            },
        )
