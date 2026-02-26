"""Lead scoring algorithms — recency weighting, score bands, status inference.

All functions are pure (no I/O) and parameterized via :class:`LeadScoringConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple


class LeadEvent(NamedTuple):
    """A single engagement action with its timestamp."""

    action: str
    created_at: datetime


@dataclass(frozen=True)
class LeadScoringConfig:
    """Domain-specific scoring parameters.

    Parameters
    ----------
    action_weights:
        ``{action_name: weight}`` — how much each action contributes.
    status_thresholds:
        Ascending ``(score, status)`` pairs.  The *last* pair whose score
        is <= the computed score wins.
    recency_bands:
        ``(max_age_days, multiplier)`` pairs, checked in order.
    recency_default:
        Multiplier for events older than every band.
    score_bands:
        Descending ``(min_score, label)`` pairs for human-readable buckets.
    terminal_statuses:
        Statuses that should never be overridden by scoring (e.g. "won", "lost").
    scoring_window_days:
        Default look-back window for :func:`compute_lead_score`.
    """

    action_weights: dict[str, float]
    status_thresholds: list[tuple[float, str]]
    recency_bands: list[tuple[float, float]]
    recency_default: float = 0.0
    score_bands: list[tuple[float, str]]  = field(default_factory=list)
    terminal_statuses: frozenset[str] = frozenset({"won", "lost"})
    scoring_window_days: int = 30


def recency_multiplier(age_days: float, config: LeadScoringConfig) -> float:
    """Return the time-decay multiplier for an event *age_days* old."""
    for max_age, mult in config.recency_bands:
        if age_days <= max_age:
            return mult
    return config.recency_default


def compute_lead_score(
    events: list[LeadEvent],
    now: datetime,
    config: LeadScoringConfig,
) -> float:
    """Compute a weighted, recency-adjusted lead score from a list of events."""
    score = 0.0
    for ev in events:
        weight = config.action_weights.get(ev.action, 0.0)
        if weight <= 0:
            continue
        age_days = max(0.0, (now - ev.created_at).total_seconds() / 86_400)
        score += weight * recency_multiplier(age_days, config)
    return round(score, 2)


def infer_lead_status(
    score: float,
    existing_status: str,
    config: LeadScoringConfig,
) -> str:
    """Determine lead status from score, preserving terminal statuses."""
    if existing_status in config.terminal_statuses:
        return existing_status

    result = config.status_thresholds[0][1] if config.status_thresholds else "new"
    for threshold, status in config.status_thresholds:
        if score >= threshold:
            result = status
        else:
            break
    return result


def lead_score_band(score: float, config: LeadScoringConfig) -> str:
    """Map a numeric score to a human-readable band label."""
    for min_score, label in config.score_bands:
        if score >= min_score:
            return label
    return config.score_bands[-1][1] if config.score_bands else "cold"
