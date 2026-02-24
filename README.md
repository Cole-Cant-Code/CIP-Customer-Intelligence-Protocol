# cip-protocol

Structured reasoning and guardrails for MCP servers that talk to real people

Most MCP servers are built for developers — return JSON, call a function, done. CIP is for building MCP servers that talk to *regular people*. It gives an inner LLM a structured reasoning framework (a "scaffold") so it can analyze domain-specific data and respond in plain language with appropriate guardrails.

The protocol knows nothing about finance, health, legal, or any other domain. You tell it what domain it's operating in with a single `DomainConfig`, point it at a directory of scaffold YAMLs, and it handles the rest.

```python
from cip_protocol import DomainConfig
from cip_protocol.scaffold import ScaffoldEngine, ScaffoldRegistry, load_scaffold_directory
from cip_protocol.llm import InnerLLMClient, create_provider

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

registry = ScaffoldRegistry()
load_scaffold_directory("scaffolds/", registry)
engine = ScaffoldEngine(registry, config=config)
provider = create_provider("anthropic", api_key="sk-...")
llm = InnerLLMClient(provider, config=config)

# In your MCP tool handler:
scaffold = engine.select(tool_name="analyze_spending", user_input=user_query)
prompt = engine.apply(scaffold, user_query=user_query, data_context=spending_data)
response = await llm.invoke(prompt, scaffold, data_context=spending_data)
```

- **Domain-agnostic** — one protocol handles finance, health, legal, or anything else via `DomainConfig`
- **Scaffolds as configuration** — prompt engineering lives in YAML files, not buried in code
- **Pluggable guardrails** — pattern matching, regex policies, escalation triggers, and automatic disclaimer enforcement
- **Multi-criteria scaffold selection** — matches by tool name, intent signals, and keywords with weighted scoring
- **Streaming with incremental guardrails** — checks each chunk as it arrives, halts on violation
- **Structured telemetry** — events for scaffold selection, LLM latency, token usage, and guardrail interventions

## How it works

```
User query → MCP Tool → Scaffold Engine → Inner LLM → Guardrails → Response
```

1. **ScaffoldEngine.select()** picks a reasoning framework (YAML) based on which tool was called and what the user asked
2. **ScaffoldEngine.apply()** renders the scaffold into a structured two-part prompt — system message (role, reasoning steps, guardrails) and user message (query + data context)
3. **InnerLLMClient.invoke()** sends the prompt to a provider (Anthropic, OpenAI, or mock) with optional chat history
4. **Guardrail pipeline** runs pluggable evaluators (prohibited patterns, regex policies, escalation triggers), redacts violations, and appends missing disclaimers

## DomainConfig

The single boundary between the protocol and your domain. Same structure, different domain:

```python
from cip_protocol import DomainConfig

# Finance
finance = DomainConfig(
    name="personal_finance",
    display_name="CIP Personal Finance",
    system_prompt="You are an expert in consumer personal finance...",
    default_scaffold_id="spending_review",
    data_context_label="Financial Data",
    prohibited_indicators={
        "recommending products": ("i recommend", "sign up for"),
        "making predictions": ("the market will", "guaranteed to"),
    },
    regex_guardrail_policies={
        "specific_security_reco": r"\b(buy|sell)\b.+\b(stock|ETF|fund)\b",
    },
    redaction_message="[Removed: contains prohibited financial guidance]",
)

# Health
health = DomainConfig(
    name="health_wellness",
    display_name="CIP Health",
    system_prompt="You are a health information specialist...",
    default_scaffold_id="symptom_overview",
    data_context_label="Health Records",
    prohibited_indicators={
        "diagnosing conditions": ("you have", "this is definitely"),
        "prescribing treatment": ("take this medication",),
    },
    redaction_message="[Removed: contains prohibited medical guidance]",
)
```

## Scaffold YAMLs

Scaffolds externalize prompt engineering as configuration. Each YAML file defines how the inner LLM reasons about a specific type of request:

```yaml
id: symptom_overview
version: "1.0"
domain: health_wellness
display_name: Symptom Overview
description: Provides general health information about reported symptoms.

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
  must_include:
    - Disclaimer about not being medical advice
  never_include:
    - Specific diagnoses
    - Medication recommendations

guardrails:
  disclaimers:
    - This is general health information, not medical advice.
  escalation_triggers:
    - severe or emergency symptoms reported
  prohibited_actions:
    - Diagnosing medical conditions
    - Recommending specific medications
```

The engine selects scaffolds using a priority cascade: explicit ID > tool name match > intent signal + keyword scoring.

## Streaming

```python
async for event in llm.invoke_stream(prompt, scaffold, data_context=data):
    if event.event == "chunk":
        print(event.text, end="")
    elif event.event == "halted":
        # Guardrail violation detected mid-stream
        final = event.response
    elif event.event == "final":
        final = event.response
```

Guardrails run incrementally on each chunk. If a violation is detected mid-stream, the client emits a `halted` event with the sanitized content and stops.

## Project structure

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

## Install

```sh
pip install -e "."                  # core only (mock provider)
pip install -e ".[anthropic]"       # + Anthropic Claude
pip install -e ".[openai]"          # + OpenAI
pip install -e ".[all]"             # all providers
```

<details>
<summary>Development setup</summary>

```sh
pip install -e ".[dev]"
make test    # pytest
make lint    # ruff
make schema  # regenerate docs/scaffold.schema.json
```

</details>

## Scaffold JSON schema

Generate a JSON schema for IDE validation and autocomplete in your scaffold YAMLs:

```sh
make schema
# → docs/scaffold.schema.json
```

## Reference implementation

[CIP-Claude](https://github.com/Cole-Cant-Code/CIP-Claude) — a personal finance MCP server built on this protocol. 13 tools, 8 scaffolds, full guardrail pipeline.

## License

MIT
