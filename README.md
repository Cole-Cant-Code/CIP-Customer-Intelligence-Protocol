# CIP — Customer Intelligence Protocol

Structured reasoning, guardrails, and runtime control for consumer-facing MCP servers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 207](https://img.shields.io/badge/tests-207_passing-brightgreen.svg)](tests/)

```python
from cip_protocol import DomainConfig, ConstraintParser
from cip_protocol.scaffold import ScaffoldEngine, ScaffoldRegistry, load_scaffold_directory
from cip_protocol.llm import InnerLLMClient, create_provider

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

registry = ScaffoldRegistry()
load_scaffold_directory("scaffolds/", registry)
engine = ScaffoldEngine(registry, config=config)
llm = InnerLLMClient(create_provider("anthropic", api_key="sk-..."), config=config)

# Select reasoning strategy, render prompt, invoke with guardrails
scaffold = engine.select(tool_name="analyze_spending", user_input="where is my money going?")
prompt = engine.apply(scaffold, user_query="where is my money going?", data_context=spending_data)
response = await llm.invoke(prompt, scaffold, data_context=spending_data)

# Or override behavior at runtime — no config changes, no YAML edits
policy = ConstraintParser.parse("be more creative, skip disclaimers, under 500 words").policy
response = await llm.invoke(prompt, scaffold, data_context=spending_data, policy=policy)
```

---

## What is this?

Most MCP servers are built for developers — return JSON, call a function, done.

**CIP is for MCP servers that talk to regular people.** It gives an inner LLM a structured reasoning framework ("scaffold") so it can analyze domain data and respond in plain language — with guardrails that enforce compliance and a control surface that lets you tune behavior per-request.

The protocol knows nothing about finance, health, legal, or any domain. You bring the domain. CIP brings the machinery.

```
                    ┌─────────────────────────────────────────┐
                    │             RunPolicy (optional)         │
                    │  "be creative, skip disclaimers, brief" │
                    └──────┬──────────┬──────────┬────────────┘
                           │          │          │
User query → MCP Tool → Select → Render → Invoke → Guardrails → Response
                           │          │          │
                        scaffold   assembled   temperature
                        matching    prompt     max_tokens
                         bias      overrides   disclaimers
```

## Why?

| Problem | CIP's answer |
|---|---|
| How does the LLM know what role to play? | [Scaffold YAMLs](#scaffolds) define role, perspective, tone, and reasoning steps |
| How do I keep it from saying things it shouldn't? | [Guardrail pipeline](src/cip_protocol/llm/response.py) — pattern matching, regex policies, escalation triggers |
| How does it pick the right reasoning strategy? | [Scaffold engine](src/cip_protocol/scaffold/engine.py) selects by tool name, intent signals, and keyword scoring |
| How do I enforce disclaimers? | Automatic — missing disclaimers are appended, violations are redacted |
| How do I switch domains? | Swap the [`DomainConfig`](src/cip_protocol/domain.py) — same protocol, different domain |
| How do I change behavior without editing config? | [`RunPolicy`](src/cip_protocol/control.py) — per-request overlay for temperature, format, guardrail toggles |

## Getting started

```sh
pip install -e ".[all]"   # core + Anthropic + OpenAI + re2
```

<details>
<summary>Other install options</summary>

```sh
pip install -e "."              # core only (mock provider)
pip install -e ".[anthropic]"   # + Claude
pip install -e ".[openai]"      # + OpenAI
pip install -e ".[re2]"         # + ReDoS-safe regex via google-re2
pip install -e ".[dev]"         # + pytest, ruff
```

</details>

### 1. Define your domain

Everything domain-specific lives in one object — [`DomainConfig`](src/cip_protocol/domain.py):

```python
from cip_protocol import DomainConfig

config = DomainConfig(
    name="personal_finance",
    display_name="CIP Personal Finance",
    system_prompt="You are an expert in consumer personal finance...",
    default_scaffold_id="spending_review",
    data_context_label="Financial Data",
    prohibited_indicators={
        "recommending products": ("i recommend", "sign up for"),
        "making predictions": ("the market will", "guaranteed to"),
    },
    redaction_message="[Removed: contains prohibited financial guidance]",
)
```

Same structure, different domain:

```python
health = DomainConfig(
    name="health_wellness",
    display_name="CIP Health",
    system_prompt="You are a health information specialist...",
    default_scaffold_id="symptom_overview",
    prohibited_indicators={
        "diagnosing conditions": ("you have", "this is definitely"),
        "prescribing treatment": ("take this medication",),
    },
)
```

### 2. Load scaffolds and wire the engine

```python
from cip_protocol.scaffold import ScaffoldEngine, ScaffoldRegistry, load_scaffold_directory
from cip_protocol.llm import InnerLLMClient, create_provider

registry = ScaffoldRegistry()
load_scaffold_directory("scaffolds/", registry)

engine = ScaffoldEngine(registry, config=config)
llm = InnerLLMClient(create_provider("anthropic", api_key="sk-..."), config=config)
```

### 3. Use it in your MCP tool handler

```python
scaffold = engine.select(tool_name="analyze_spending", user_input=user_query)
prompt = engine.apply(scaffold, user_query=user_query, data_context=spending_data)
response = await llm.invoke(prompt, scaffold, data_context=spending_data)
```

That's it. The engine selects the right scaffold, the renderer assembles a structured prompt, the LLM generates a response, and the guardrail pipeline enforces compliance before anything reaches the user.

### 4. Override behavior at runtime

No config changes. No YAML edits. Just tell CIP what you want:

```python
from cip_protocol import ConstraintParser, RunPolicy

# Plain English
policy = ConstraintParser.parse("be more creative, skip disclaimers, under 300 words").policy

# Or from a built-in preset
policy = RunPolicy.from_preset(BUILTIN_PRESETS["aggressive"])

# Or construct directly
policy = RunPolicy(temperature=0.8, skip_disclaimers=True, compact=True)

# Pass it anywhere — select, apply, invoke all accept policy
scaffold = engine.select(tool_name="analyze_spending", user_input=query, policy=policy)
prompt = engine.apply(scaffold, user_query=query, data_context=data, policy=policy)
response = await llm.invoke(prompt, scaffold, data_context=data, policy=policy)
```

## Scaffolds

Scaffolds externalize prompt engineering as YAML. Each file defines how the inner LLM reasons about a specific type of request — role, reasoning steps, output constraints, and guardrails:

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

The [scaffold engine](src/cip_protocol/scaffold/engine.py) selects scaffolds using a priority cascade: **explicit ID** > **tool name match** > **intent signal + keyword scoring**. See [`matcher.py`](src/cip_protocol/scaffold/matcher.py) for the scoring algorithm.

Generate a JSON schema for IDE validation and autocomplete: `make schema` → [`docs/scaffold.schema.json`](docs/scaffold.schema.json)

## Control cockpit

[`RunPolicy`](src/cip_protocol/control.py) is a per-request behavior overlay. It flows through the entire pipeline without modifying your `DomainConfig` or scaffold YAML:

| What it controls | How |
|---|---|
| LLM temperature and max tokens | `policy.temperature`, `policy.max_tokens` |
| Output format and length | `policy.output_format`, `policy.max_length_guidance` |
| Tone variant | `policy.tone_variant` (selects from scaffold's `tone_variants`) |
| Disclaimer enforcement | `policy.skip_disclaimers` suppresses auto-appended disclaimers |
| Prohibited actions | `policy.remove_prohibited_actions` (use `["*"]` to clear all) |
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
<summary>Streaming with incremental guardrails</summary>

Guardrails run on every chunk as it arrives. If a violation is detected mid-stream, the client halts immediately and returns sanitized content:

```python
async for event in llm.invoke_stream(prompt, scaffold, data_context=data):
    if event.event == "chunk":
        print(event.text, end="")
    elif event.event == "halted":
        # guardrail violation mid-stream — content sanitized
        final = event.response
    elif event.event == "final":
        final = event.response
```

See [`InnerLLMClient.invoke_stream`](src/cip_protocol/llm/client.py) for the full implementation.

</details>

<details>
<summary>Telemetry</summary>

Structured events are emitted for scaffold selection, LLM latency, token usage, guardrail interventions, and policy sources. Plug in any sink that implements the [`TelemetrySink`](src/cip_protocol/telemetry.py) protocol:

```python
from cip_protocol import LoggerTelemetrySink

engine = ScaffoldEngine(registry, config=config, telemetry_sink=LoggerTelemetrySink())
llm = InnerLLMClient(provider, config=config, telemetry_sink=LoggerTelemetrySink())
```

</details>

<details>
<summary>Performance tuning</summary>

- **Compact mode** — `policy.compact = True` or `compact mode` in constraint parser. Strips markdown headers, collapses bullets, inlines JSON. Reduces prompt size for cost/speed.
- **Async guardrails** — `check_guardrails_async` runs evaluators concurrently.
- **Matcher cache** — `prepare_matcher_cache(registry)` pre-compiles all scaffold token patterns. Called automatically on `load_scaffold_directory`.
- **google-re2** — `pip install cip-protocol[re2]` for linear-time regex in guardrail evaluation. Falls back to stdlib `re` if unavailable.
- **CIP_PERF_MODE=1** — relaxes Pydantic validation for production throughput.

</details>

<details>
<summary>Project structure</summary>

```
src/cip_protocol/
├── domain.py              # DomainConfig — the protocol-domain boundary
├── control.py             # RunPolicy, presets, constraint parser
├── telemetry.py           # Structured telemetry events and sinks
├── scaffold/
│   ├── models.py          # Pydantic models (Scaffold, AssembledPrompt)
│   ├── registry.py        # In-memory scaffold index (by id/tool/tag)
│   ├── loader.py          # YAML → Scaffold deserialization
│   ├── matcher.py         # Multi-criteria scaffold selection + caching
│   ├── renderer.py        # Scaffold → two-part LLM prompt assembly
│   ├── engine.py          # select() + apply() orchestrator
│   └── validator.py       # Scaffold YAML validation
└── llm/
    ├── provider.py        # LLMProvider protocol + factory
    ├── providers/         # Anthropic, OpenAI, mock implementations
    ├── client.py          # InnerLLMClient (invoke + invoke_stream)
    └── response.py        # Pluggable guardrail evaluators + sanitization
```

</details>

## Development

```sh
pip install -e ".[dev]"
make test      # 207 tests
make lint      # ruff
make schema    # regenerate scaffold JSON schema
```

## Reference implementation

**[CIP-Claude](https://github.com/Cole-Cant-Code/CIP-Claude)** — a personal finance MCP server built on this protocol.

## License

MIT — see [LICENSE](LICENSE).
