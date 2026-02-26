"""Engagement utilities — escalation detection, lead scoring, and data parsing."""

from cip_protocol.engagement.detector import (
    EscalationCallback,
    EscalationConfig,
    EscalationDetector,
    check_escalation,
)
from cip_protocol.engagement.parsing import (
    clean_numeric_string,
    parse_float,
    parse_int,
    parse_price,
)
from cip_protocol.engagement.scoring import (
    LeadEvent,
    LeadScoringConfig,
    compute_lead_score,
    infer_lead_status,
    lead_score_band,
    recency_multiplier,
)
from cip_protocol.engagement.store import EscalationStore

__all__ = [
    "EscalationCallback",
    "EscalationConfig",
    "EscalationDetector",
    "EscalationStore",
    "LeadEvent",
    "LeadScoringConfig",
    "check_escalation",
    "clean_numeric_string",
    "compute_lead_score",
    "infer_lead_status",
    "lead_score_band",
    "parse_float",
    "parse_int",
    "parse_price",
    "recency_multiplier",
]
