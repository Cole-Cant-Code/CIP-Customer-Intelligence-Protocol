# CIP — Customer Intelligence Protocol

Structured reasoning frameworks and safety boundaries for consumer-facing MCP servers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 478](https://img.shields.io/badge/tests-478_passing-brightgreen.svg)](tests/)

```python
from cip_protocol import CIP, DomainConfig

config = DomainConfig(name="personal_finance", system_prompt="You are a finance expert.")
cip = CIP.from_config(config, "scaffolds/", "anthropic", api_key="sk-...")

result = await cip.run("where is my money going?", data_context=spending_data)
print(result.response.content)

# Override behavior at runtime — plain English, no config changes
result = await cip.run("same but shorter", policy="be concise, bullet points")
```

> **Reasoning protocol; consumes mantic via optional adapter.** CIP owns domain reasoning, scaffold selection, and engagement scoring. Multi-signal detection math is delegated to mantic when installed, with a built-in native fallback.

---

## What is this?

Most MCP servers are built for developers — return JSON, call a function, done.

**CIP is for MCP servers that talk to regular people.** It gives an inner LLM structured reasoning frameworks ("scaffolds") so it can analyze domain data and respond in plain language — with safety boundaries that catch real mistakes on the way out, and a policy layer that lets you adjust behavior per-request without editing config.

The LLM does the thinking. CIP gives it structure to think well and boundaries that matter. The protocol knows nothing about finance, health, legal, or any domain. You bring the domain. CIP brings the framework.

```
                    ┌─────────────────────────────────────────┐
                    │             RunPolicy (optional)         │
                    │  "be creative, skip disclaimers, brief" │
                    └──────┬──────────┬──────────┬────────────┘
                           │          │          │
User query → MCP Tool → Select → Render → Invoke → Safety → Response
                           │          │          │
                        scaffold   assembled   temperature
                        matching    prompt     max_tokens
                         bias      overrides   disclaimers
```

## Why?

| Problem | CIP's answer |
|---|---|
| How does the LLM know what role to play? | [Scaffold YAMLs](#scaffolds) define role, perspective, tone, and reasoning steps — the LLM follows these as a reasoning framework, not a rulebook |
| How do I catch genuine mistakes before they reach the user? | [Safety boundaries](src/cip_protocol/llm/response.py) — pattern matching, regex policies, escalation triggers that intervene only when output crosses a real line |
| How does it pick the right reasoning strategy? | [Scaffold engine](src/cip_protocol/scaffold/engine.py) — layered selection across micro/meso/macro/meta signals, with all parameters tunable per-request via [`SelectionParams`](src/cip_protocol/scaffold/matcher.py) |
| How do disclaimers work? | Missing disclaimers are appended automatically — the LLM focuses on reasoning, the framework handles the boilerplate |
| How do I switch domains? | Swap the [`DomainConfig`](src/cip_protocol/domain.py) — same protocol, different domain |
| How do I change behavior without editing config? | [`RunPolicy`](src/cip_protocol/control.py) — per-request overlay for temperature, format, and safety toggles |

## Getting started

```sh
pip install -e ".[all]"   # core + Anthropic + OpenAI + re2 + mantic-thinking
```

<details>
<summary>Other install options</summary>

```sh
pip install -e "."              # core only (mock provider)
pip install -e ".[anthropic]"   # + Claude
pip install -e ".[openai]"      # + OpenAI
pip install -e ".[re2]"         # + ReDoS-safe regex via google-re2
pip install -e ".[mantic]"      # + mantic-thinking backend
pip install -e ".[dev]"         # + pytest, ruff
```

</details>

### 1. Define your domain and run

```python
from cip_protocol import CIP, DomainConfig

config = DomainConfig(
    name="personal_finance",
    display_name="CIP Personal Finance",
    system_prompt="You are an expert in consumer personal finance.",
    default_scaffold_id="spending_review",
    prohibited_indicators={
        "recommending products": ("i recommend", "sign up for"),
        "making predictions": ("the market will", "guaranteed to"),
    },
)

cip = CIP.from_config(config, "scaffolds/", "anthropic", api_key="sk-...")
result = await cip.run("where is my money going?", data_context=spending_data)
```

The facade handles scaffold loading, selection, prompt assembly, LLM invocation, and safety checks in one call. `result` includes the response, which scaffold was selected, how it was selected, and the scores.

Same structure, different domain:

```python
health = DomainConfig(
    name="health_wellness",
    system_prompt="You are a health information specialist...",
    default_scaffold_id="symptom_overview",
    prohibited_indicators={
        "diagnosing conditions": ("you have", "this is definitely"),
        "prescribing treatment": ("take this medication",),
    },
)
cip = CIP.from_config(health, "scaffolds/", "anthropic", api_key="sk-...")
```

### 2. Multi-turn conversations

```python
conv = cip.conversation(max_history_turns=20)

r1 = await conv.say("where is my money going?", data_context=spending_data)
r2 = await conv.say("break that down by category")   # history carries forward
r3 = await conv.say("focus on the top 3")             # context accumulates

conv.turn_count   # 3
conv.history      # full message list
conv.reset()      # start fresh
```

History is truncated to `max_history_turns` pairs. Context exports from each turn accumulate and merge into subsequent calls automatically.

### 3. Override behavior at runtime

No config changes. No YAML edits. Just tell CIP what you want:

```python
# Plain English — parsed into a RunPolicy
result = await cip.run("analyze my spending", policy="be creative, skip disclaimers")

# Or from a built-in preset
from cip_protocol import RunPolicy
from cip_protocol.control import BUILTIN_PRESETS

policy = RunPolicy.from_preset(BUILTIN_PRESETS["aggressive"])
result = await cip.run("analyze my spending", policy=policy)

# Or construct directly
result = await cip.run("analyze my spending", policy=RunPolicy(temperature=0.8, compact=True))
```

### 4. Explainability

Every `CIPResult` tells you how the scaffold was selected:

```python
result = await cip.run("where is my money going?", data_context=data)

result.scaffold_id          # "spending_review"
result.selection_mode       # "scored" | "tool_match" | "caller_id" | "default"
result.selection_scores     # {"spending_review": 3.50, "budget_overview": 1.00}
result.policy_source        # "constraint:brief+bullet_format"
result.unrecognized_constraints  # ["do a backflip"] — nothing silently dropped
```

### 5. CLI playground

Try scaffolds interactively without writing code:

```sh
python -m cip_protocol playground --scaffold-dir=scaffolds/ --provider=mock

CIP Playground
Provider: mock | Domain: playground
Scaffolds loaded: 3
Type /help for commands, /quit to exit.

you> where is my money going?
[scaffold: spending_review | mode: scored | score: 3.50]
cip> Based on your spending data...

you> /policy be creative, skip disclaimers
Policy set: be creative, skip disclaimers

you> /explain
Selection mode: scored
Selected: spending_review
Scores:
  spending_review: 3.50
  budget_overview: 1.00
```

<details>
<summary>Manual wiring (without facade)</summary>

For full control over each component:

```python
from cip_protocol.scaffold import ScaffoldEngine, ScaffoldRegistry, load_scaffold_directory
from cip_protocol.llm import InnerLLMClient, create_provider

registry = ScaffoldRegistry()
load_scaffold_directory("scaffolds/", registry)
engine = ScaffoldEngine(registry, config=config)
llm = InnerLLMClient(create_provider("anthropic", api_key="sk-..."), config=config)

scaffold = engine.select(tool_name="analyze_spending", user_input=query)
prompt = engine.apply(scaffold, user_query=query, data_context=data)
response = await llm.invoke(prompt, scaffold, data_context=data, policy=policy)
```

</details>

## Scaffolds

Scaffolds externalize prompt engineering as YAML. Each file defines a reasoning framework for a specific type of request — the role, perspective, reasoning steps, output shape, and safety boundaries. The LLM uses these as structure for its thinking, not as rigid constraints:

```yaml
id: symptom_overview
version: "1.0"
domain: health_wellness
display_name: Symptom Overview
description: General health information about reported symptoms.

applicability:
  tools: [check_symptoms]
  keywords: [symptom, feeling, pain]
  intent_signals: [what could this symptom mean]

framing:
  role: Health information specialist
  perspective: Evidence-based, cautious, educational
  tone: Reassuring but direct

reasoning_framework:
  steps:
    - Identify the reported symptoms
    - Note relevant context (duration, severity)
    - Provide general educational information
    - Flag when professional consultation is warranted

output_calibration:
  format: structured_narrative
  must_include: [Disclaimer about not being medical advice]
  never_include: [Specific diagnoses, Medication recommendations]

guardrails:
  disclaimers: [This is general health information, not medical advice.]
  escalation_triggers: [severe or emergency symptoms reported]
  prohibited_actions: [Diagnosing medical conditions, Recommending specific medications]
```

The [scaffold engine](src/cip_protocol/scaffold/engine.py) selects scaffolds using a priority cascade: **explicit ID** > **tool name match** > **layered scoring** (micro/meso/macro/meta signal layers with saturation and cross-layer reinforcement). All scoring parameters are tunable per-request via [`SelectionParams`](src/cip_protocol/scaffold/matcher.py) — the calling LLM decides how selection behaves, not hardcoded constants.

Generate a JSON schema for IDE validation and autocomplete: `make schema` → [`docs/scaffold.schema.json`](docs/scaffold.schema.json)

## Runtime policy

[`RunPolicy`](src/cip_protocol/control.py) is a per-request behavior overlay. It flows through the entire pipeline without modifying your `DomainConfig` or scaffold YAML — think of it as the LLM (or caller) expressing preferences, not a control panel:

| What it adjusts | How |
|---|---|
| LLM temperature and max tokens | `policy.temperature`, `policy.max_tokens` |
| Output format and length | `policy.output_format`, `policy.max_length_guidance` |
| Tone variant | `policy.tone_variant` (selects from scaffold's `tone_variants`) |
| Disclaimers | `policy.skip_disclaimers` suppresses auto-appended disclaimers |
| Safety boundaries | `policy.remove_prohibited_actions` (use `["*"]` to clear all) |
| Must/never include lists | `policy.extra_must_include`, `policy.extra_never_include` |
| Scaffold selection weighting | `policy.scaffold_selection_bias` — per-scaffold score multipliers |
| Prompt compression | `policy.compact` — strip headers, collapse bullets, inline JSON |

Three ways to create a policy:

```python
from cip_protocol import ConstraintParser, RunPolicy
from cip_protocol.control import BUILTIN_PRESETS

# 1. Parse plain English
result = ConstraintParser.parse("be more creative, bullet points, skip disclaimers")
result.policy.temperature   # 0.8
result.policy.output_format # "bullet_points"
result.unrecognized         # [] — everything matched

# 2. Use a built-in preset
policy = RunPolicy.from_preset(BUILTIN_PRESETS["precise"])

# 3. Merge multiple sources — last writer wins for scalars, union for lists
base = RunPolicy.from_preset(BUILTIN_PRESETS["creative"])
override = ConstraintParser.parse("must include risk factors, under 200 words").policy
final = base.merge(override)
```

**Built-in presets:**

| Preset | Temperature | Format | Length | Disclaimers | Notes |
|---|---|---|---|---|---|
| `creative` | 0.8 | — | no limit | on | Exploratory responses |
| `precise` | 0.1 | bullet_points | under 300 words | on | Compact, factual |
| `aggressive` | 0.5 | — | brief | **off** | Removes all prohibited actions |
| `balanced` | 0.3 | — | — | on | Default baseline |

Register custom presets:

```python
from cip_protocol import ControlPreset, PresetRegistry

registry = PresetRegistry()  # includes builtins
registry.register(ControlPreset(
    name="internal_review",
    temperature=0.2,
    skip_disclaimers=True,
    output_format="bullet_points",
    extra_must_include=["confidence score"],
))
```

<details>
<summary>Constraint language reference</summary>

The parser recognizes these patterns (case-insensitive, comma or semicolon separated):

| Pattern | Field | Value |
|---|---|---|
| `more creative` | temperature | 0.8 |
| `more precise` | temperature | 0.1 |
| `more aggressive` | temperature | 0.5 |
| `temperature 0.7` | temperature | 0.7 |
| `bullet points` | output_format | `"bullet_points"` |
| `structured narrative` | output_format | `"structured_narrative"` |
| `under 200 words` | max_length_guidance | `"under 200 words"` |
| `keep it brief` / `be concise` | max_length_guidance | `"concise, under 200 words"` |
| `no length limit` | max_length_guidance | `"no length constraint"` |
| `skip disclaimers` | skip_disclaimers | `True` |
| `skip prohibited actions` | remove_prohibited_actions | `["*"]` |
| `must include X` | extra_must_include | `["X"]` |
| `never include X` | extra_never_include | `["X"]` |
| `compact mode` | compact | `True` |
| `tone: friendly` | tone_variant | `"friendly"` |
| `max 4000 tokens` | max_tokens | `4000` |
| `preset: creative` | (resolves via registry) | preset fields |

Unrecognized clauses are returned in `result.unrecognized` — nothing is silently dropped.

</details>

<details>
<summary>Streaming with incremental safety checks</summary>

Safety checks run on every chunk as it arrives. If a boundary is crossed mid-stream, the client halts and returns sanitized content:

```python
async for event in llm.invoke_stream(prompt, scaffold, data_context=data):
    if event.event == "chunk":
        print(event.text, end="")
    elif event.event == "halted":
        # boundary crossed mid-stream — content sanitized
        final = event.response
    elif event.event == "final":
        final = event.response
```

See [`InnerLLMClient.invoke_stream`](src/cip_protocol/llm/client.py) for the full implementation.

</details>

<details>
<summary>Telemetry</summary>

Structured events are emitted for scaffold selection, LLM latency, token usage, safety interventions, and policy sources. Plug in any sink that implements the [`TelemetrySink`](src/cip_protocol/telemetry.py) protocol:

```python
from cip_protocol import LoggerTelemetrySink

engine = ScaffoldEngine(registry, config=config, telemetry_sink=LoggerTelemetrySink())
llm = InnerLLMClient(provider, config=config, telemetry_sink=LoggerTelemetrySink())
```

</details>

<details>
<summary>Performance tuning</summary>

- **Compact mode** — `policy.compact = True` or `compact mode` in constraint parser. Strips markdown headers, collapses bullets, inlines JSON. Reduces prompt size for cost/speed.
- **Async safety checks** — `check_guardrails_async` runs evaluators concurrently.
- **Matcher cache** — `prepare_matcher_cache(registry)` pre-compiles all scaffold token patterns. Called automatically on `load_scaffold_directory`.
- **google-re2** — `pip install cip-protocol[re2]` for linear-time regex in safety evaluation. Falls back to stdlib `re` if unavailable.
- **CIP_PERF_MODE=1** — relaxes Pydantic validation for production throughput.

</details>

<details>
<summary>Project structure</summary>

```
src/cip_protocol/
├── cip.py                 # CIP facade + CIPResult (3-line entry point)
├── conversation.py        # Multi-turn Conversation + Turn
├── __main__.py            # CLI entry point (python -m cip_protocol)
├── domain.py              # DomainConfig — the protocol-domain boundary
├── control.py             # RunPolicy, presets, constraint parser
├── telemetry.py           # Structured telemetry events and sinks
├── cli/
│   └── playground.py      # Interactive REPL with /policy, /explain, etc.
├── scaffold/
│   ├── models.py          # Pydantic models (Scaffold, AssembledPrompt)
│   ├── registry.py        # In-memory scaffold index (by id/tool/tag)
│   ├── loader.py          # YAML → Scaffold deserialization
│   ├── matcher.py         # Layered selection (micro/meso/macro/meta) + explainability
│   ├── renderer.py        # Scaffold → two-part LLM prompt assembly
│   ├── engine.py          # select() + select_explained() + apply()
│   └── validator.py       # Scaffold YAML validation
└── llm/
    ├── provider.py        # LLMProvider protocol + factory
    ├── providers/         # Anthropic, OpenAI, mock implementations
    ├── client.py          # InnerLLMClient (invoke + invoke_stream)
    └── response.py        # Pluggable safety evaluators + sanitization
```

</details>

## Development

```sh
pip install -e ".[dev]"
make test      # 478 tests
make lint      # ruff
make schema    # regenerate scaffold JSON schema
# Optional CI hard gate for mantic-enabled runs:
CIP_REQUIRE_MANTIC=1 python scripts/check_mantic_runtime.py
```

## Reference implementation

**[CIP-Claude](https://github.com/Cole-Cant-Code/CIP-Claude)** — a personal finance MCP server built on this protocol.

## License

MIT — see [LICENSE](LICENSE).
