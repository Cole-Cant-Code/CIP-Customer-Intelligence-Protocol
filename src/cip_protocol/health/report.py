"""Output formatters for scaffold health reports: aligned table and JSON."""

from __future__ import annotations

import json
from typing import Any

from cip_protocol.health.analysis import PortfolioHealthResult, ScaffoldHealthResult
from cip_protocol.health.scoring import LAYER_NAMES


def _row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cols, widths))


def format_table(result: PortfolioHealthResult) -> str:
    lines: list[str] = []

    # --- Summary table ---
    lines.append("Scaffold Health Report")
    lines.append("=" * 72)
    hdr = ["Scaffold", "M-score", "Cohere", "Dominant", "Signal"]
    widths = [30, 8, 8, 10, 20]
    lines.append(_row(hdr, widths))
    lines.append("-" * 72)
    for s in result.scaffolds:
        lines.append(
            _row(
                [
                    s.scaffold_id[:30],
                    f"{s.m_score:.3f}",
                    f"{s.coherence:.3f}",
                    s.dominant_layer,
                    s.signal,
                ],
                widths,
            )
        )
    lines.append("-" * 72)
    lines.append("")

    # --- Layer scores ---
    lines.append("Layer Scores")
    lhdr = ["Scaffold", *[n.rjust(8) for n in LAYER_NAMES]]
    lwidths = [30, *[8] * len(LAYER_NAMES)]
    lines.append(_row(lhdr, lwidths))
    lines.append("-" * 72)
    for s in result.scaffolds:
        lines.append(
            _row(
                [s.scaffold_id[:30], *[f"{s.layers[n]:.3f}" for n in LAYER_NAMES]],
                lwidths,
            )
        )
    lines.append("")

    # --- Tension pairs ---
    any_tension = any(s.tension_pairs for s in result.scaffolds)
    if any_tension:
        lines.append("Tension Pairs (agreement < threshold)")
        for s in result.scaffolds:
            for a, b, score in s.tension_pairs:
                lines.append(f"  {s.scaffold_id}: {a} <-> {b}  agreement={score:.3f}")
        lines.append("")

    # --- Cross-scaffold coupling (top 10) ---
    if result.coupling:
        lines.append("Top Cross-Scaffold Coupling")
        for id_a, id_b, layer, score in result.coupling[:10]:
            lines.append(f"  {id_a} <-> {id_b}  [{layer}]  score={score:.3f}")
        lines.append("")

    # --- Portfolio summary ---
    n = len(result.scaffolds)
    lines.append(
        f"Portfolio: {n} scaffold{'s' if n != 1 else ''}"
        f" | coherence: {result.avg_coherence:.3f}"
        f" | signal: {result.portfolio_signal}"
    )

    return "\n".join(lines)


def _scaffold_to_dict(s: ScaffoldHealthResult) -> dict[str, Any]:
    return {
        "scaffold_id": s.scaffold_id,
        "layers": s.layers,
        "m_score": round(s.m_score, 4),
        "coherence": round(s.coherence, 4),
        "dominant_layer": s.dominant_layer,
        "signal": s.signal,
        "tension_pairs": [
            {"layer_a": a, "layer_b": b, "agreement": score}
            for a, b, score in s.tension_pairs
        ],
    }


def format_json(result: PortfolioHealthResult) -> str:
    payload: dict[str, Any] = {
        "scaffolds": [_scaffold_to_dict(s) for s in result.scaffolds],
        "coupling": [
            {"scaffold_a": a, "scaffold_b": b, "layer": layer, "score": score}
            for a, b, layer, score in result.coupling
        ],
        "avg_coherence": result.avg_coherence,
        "portfolio_signal": result.portfolio_signal,
    }
    return json.dumps(payload, indent=2)
