# CIP — Customer Intelligence Protocol

Give your MCP server a brain that knows how to talk to humans

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What is this?

Most MCP servers are built for developers — return JSON, call a function, done.

**CIP is for building MCP servers that talk to regular people.** It gives an inner LLM a structured reasoning framework ("scaffold") so it can analyze domain data and respond in plain language — with guardrails that actually enforce compliance.

The protocol itself knows nothing about finance, health, legal, or any other domain. You bring the domain. CIP brings the machinery.

```
User query → MCP Tool → Scaffold Engine → Inner LLM → Guardrails → Response
```

## Why?

If you're building a consumer-facing MCP server, you need answers to questions that raw LLM calls don't handle:

| Problem | CIP's answer |
|---|---|
| How does the LLM know what role to play? | [Scaffold YAMLs](#scaffolds) define role, perspective, tone, and reasoning steps |
| How do I keep it from saying things it shouldn't? | [Guardrail pipeline](src/cip_protocol/llm/response.py) — pattern matching, regex policies, escalation triggers |
| How does it pick the right reasoning strategy? | [Scaffold engine](src/cip_protocol/scaffold/engine.py) selects by tool name, intent signals, and keyword scoring |
| How do I enforce disclaimers? | Automatic — missing disclaimers are appended, violations are redacted |
| How do I switch domains? | Swap the [`DomainConfig`](src/cip_protocol/domain.py) — same protocol, different domain |

## Quick start

```sh
pip install -e ".[all]"   # core + Anthropic + OpenAI
```

<details>
<summary>Other install options</summary>

```sh
pip install -e "."              # core only (mock provider)
pip install -e ".[anthropic]"   # + Claude
pip install -e ".[openai]"      # + OpenAI
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

<details>
<summary>Streaming with incremental guardrails</summary>

Guardrails run on every chunk as it arrives. If a violation is detected mid-stream, the client halts immediately and returns sanitized content:

```python
async for event in llm.invoke_stream(prompt, scaffold, data_context=data):
    if event.event == "chunk":
        print(event.text, end="")
    elif event.event == "halted":
        # guardrail violation — stream killed, content sanitized
        final = event.response
    elif event.event == "final":
        final = event.response
```

See [`InnerLLMClient.invoke_stream`](src/cip_protocol/llm/client.py) for the full implementation.

</details>

<details>
<summary>Telemetry</summary>

Structured events are emitted for scaffold selection, LLM latency, token usage, and guardrail interventions. Plug in any sink that implements the [`TelemetrySink`](src/cip_protocol/telemetry.py) protocol:

```python
from cip_protocol import LoggerTelemetrySink, InMemoryTelemetrySink

engine = ScaffoldEngine(registry, config=config, telemetry_sink=LoggerTelemetrySink())
llm = InnerLLMClient(provider, config=config, telemetry_sink=LoggerTelemetrySink())
```

</details>

<details>
<summary>Project structure</summary>

```
src/cip_protocol/
├── domain.py              # DomainConfig — the protocol-domain boundary
├── telemetry.py           # Structured telemetry events and sinks
├── scaffold/
│   ├── models.py          # Pydantic models (Scaffold, AssembledPrompt)
│   ├── registry.py        # In-memory scaffold index (by id/tool/tag)
│   ├── loader.py          # YAML → Scaffold deserialization
│   ├── matcher.py         # Multi-criteria scaffold selection
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
make test      # 72 tests
make lint      # ruff
make schema    # regenerate scaffold JSON schema
```

## Reference implementation

**[CIP-Claude](https://github.com/Cole-Cant-Code/CIP-Claude)** — a personal finance MCP server built on this protocol.

## License

MIT — see [LICENSE](LICENSE).
