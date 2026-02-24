from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class TelemetryEvent:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)


@runtime_checkable
class TelemetrySink(Protocol):
    def emit(self, event: TelemetryEvent) -> None: ...


class NoOpTelemetrySink:
    def emit(self, event: TelemetryEvent) -> None:
        pass


class InMemoryTelemetrySink:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class LoggerTelemetrySink:
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
