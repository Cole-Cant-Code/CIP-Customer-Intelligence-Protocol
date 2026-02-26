"""Scaffold health analysis — M-layer scoring, friction/emergence detection."""

from cip_protocol.health.analysis import (
    analyze_portfolio,
    analyze_portfolio_with_backend,
    analyze_scaffold_with_backend,
)
from cip_protocol.health.scoring import score_scaffold_layers

__all__ = [
    "analyze_portfolio",
    "analyze_portfolio_with_backend",
    "analyze_scaffold_with_backend",
    "score_scaffold_layers",
]
