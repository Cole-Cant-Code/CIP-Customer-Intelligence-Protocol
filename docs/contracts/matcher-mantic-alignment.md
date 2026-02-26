# Matcher-Mantic Layer Alignment

CIP's scaffold health system uses four layers — **micro, meso, macro, meta** — to
score scaffold completeness and detect friction/emergence across a portfolio.
Mantic-thinking's hierarchy levels use the same four-tier structure:
**Micro, Meso, Macro, Meta**.

This document records the semantic alignment and confirms that no translation
layer is needed between them.

## Layer Mapping

| CIP Scaffold Layer | Mantic Hierarchy | Semantics |
|--------------------|------------------|-----------|
| `micro` | `Micro` | Applicability — tools, keywords, intent signals |
| `meso` | `Meso` | Reasoning depth — framework steps, domain knowledge |
| `macro` | `Macro` | Output specification — format, constraints, length |
| `meta` | `Meta` | Guardrails — disclaimers, escalation triggers, prohibitions |

## Detection Formula Parity

Both systems use the same core formula:

```
M = sum(W_i * L_i * I_i) * f(t) / k_n
```

**CIP native** uses equal weights (`W = 1/N`), unit interactions (`I = 1.0`),
and `k_n = sqrt(N)`.

**Mantic** accepts caller-defined weights (normalized to sum 1.0), dynamic
interaction coefficients (`I ∈ [0.1, 2.0]`), temporal kernels, and `k_n = 1.0`.

When CIP passes equal weights to mantic, the M-scores will differ by the
normalization constant (`sqrt(N)` vs `1.0`), but **signal classification**
(friction/emergence/baseline) is consistent because it depends on layer spread
and floor, not the absolute M-score.

## Signal Classification

| Condition | CIP Native Signal | Mantic Mode | Mantic Indicator |
|-----------|-------------------|-------------|------------------|
| `max(L) - min(L) > threshold` | `friction_detected` | friction | `alert is not None` |
| `min(L) > threshold` | `emergence_window` | emergence | `window_detected is True` |
| Otherwise | `baseline` | either | no alert, no window |

The `mantic_adapter.py` module normalizes mantic's raw output into CIP's
three-signal vocabulary, ensuring downstream consumers never need to know
which backend produced the result.

## Non-Goals

- The matcher's **scaffold selection** logic (keyword matching, intent scoring,
  tool affinity) is unrelated to M-layer detection. No alignment is needed there.
- Cross-scaffold **coupling** is a CIP concept with no mantic equivalent.
  Coupling analysis always runs in CIP-native code.
