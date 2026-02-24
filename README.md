# CIP Customer Intelligence Protocol

Domain-agnostic framework for building consumer-facing MCP servers.

Most MCP servers are built for developers — return JSON, call a function, done. CIP (Customer Intelligence Protocol) is for building MCP servers that talk to regular people. It gives an inner specialist LLM a structured reasoning framework (a "scaffold") so it can analyze domain data and respond in plain language with appropriate guardrails.

The protocol doesn't know anything about finance, health, legal, or any other domain. You tell it what domain it's operating in by providing a `DomainConfig`.

## How It Works

```
User query → MCP Tool → Data Provider → Scaffold Engine → Inner LLM → Guardrails → Response
```

1. **Scaffold Engine** selects a reasoning framework (YAML) based on which tool was called and what the user asked
2. **Renderer** assembles the scaffold into a structured LLM prompt — role, reasoning steps, domain knowledge, output format, guardrails
3. **Inner LLM Client** sends the prompt to a provider (Anthropic, OpenAI, or mock) with optional chat history
4. **Guardrail Pipeline** runs pluggable evaluators (pattern, regex, escalation), redacts violations, and appends missing disclaimers
5. **Telemetry Hooks** emit structured events for scaffold selection, latency, token use, and guardrail interventions

All domain-specific behavior comes from one object: `DomainConfig`.

## DomainConfig

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
    regex_guardrail_policies={
        "specific_security_reco": r"\\b(buy|sell)\\b.+\\b(stock|ETF|fund)\\b",
    },
    redaction_message="[Removed: contains prohibited financial guidance]",
)
```

Same structure, different domain:

```python
config = DomainConfig(
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

## Using the Protocol

```python
from cip_protocol import DomainConfig
from cip_protocol.scaffold import ScaffoldEngine, ScaffoldRegistry, load_scaffold_directory
from cip_protocol.llm import InnerLLMClient, create_provider
from cip_protocol.telemetry import LoggerTelemetrySink

# 1. Define your domain
config = DomainConfig(
    name="my_domain",
    display_name="My CIP Server",
    system_prompt="You are a domain expert...",
    default_scaffold_id="default_analysis",
    data_context_label="Domain Data",
)

# 2. Load scaffolds
registry = ScaffoldRegistry()
load_scaffold_directory("path/to/scaffolds/", registry)

# 3. Create engine and LLM client
telemetry = LoggerTelemetrySink()
engine = ScaffoldEngine(registry, config=config, telemetry_sink=telemetry)
provider = create_provider("anthropic", api_key="sk-...")
llm_client = InnerLLMClient(provider, config=config, telemetry_sink=telemetry)

# 4. In your MCP tool handler:
scaffold = engine.select(tool_name="analyze_data", user_input=user_query)
prompt = engine.apply(
    scaffold,
    user_query=user_query,
    data_context=data,
    chat_history=[
        {"role": "user", "content": "Prior user turn"},
        {"role": "assistant", "content": "Prior assistant turn"},
    ],
)
response = await llm_client.invoke(prompt, scaffold, data_context=data)
```

### Streaming responses

```python
async for event in llm_client.invoke_stream(prompt, scaffold, data_context=data):
    if event.event == "chunk":
        print(event.text, end="")
    elif event.event in {"halted", "final"}:
        final_response = event.response
```

### Scaffold JSON Schema

```bash
make schema
# writes docs/scaffold.schema.json for IDE validation/autocomplete
```

## Scaffold YAMLs

Scaffolds are YAML files that define how the inner LLM reasons about a specific type of request. They externalize prompt engineering as configuration.

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

## What's in the Box

```
src/cip_protocol/
├── domain.py              # DomainConfig — the protocol-domain boundary
├── telemetry.py           # Structured telemetry events/sinks
├── scaffold/
│   ├── models.py          # Pydantic scaffold + prompt models
│   ├── registry.py        # In-memory scaffold index (by id/tool/tag)
│   ├── loader.py          # YAML → Scaffold deserialization
│   ├── matcher.py         # Multi-criteria scaffold selection
│   ├── renderer.py        # Scaffold → LLM prompt assembly
│   ├── engine.py          # select() + apply() orchestrator
│   └── validator.py       # Scaffold YAML validation
└── llm/
    ├── provider.py        # LLMProvider protocol + factory
    ├── providers/         # Anthropic, OpenAI, mock implementations
    ├── client.py          # InnerLLMClient (invoke + invoke_stream)
    └── response.py        # Pluggable guardrail evaluators + sanitization
```

## Install

```bash
pip install -e "."                  # core (mock provider only)
pip install -e ".[anthropic]"       # + Anthropic Claude
pip install -e ".[openai]"          # + OpenAI
pip install -e ".[all]"             # everything
pip install -e ".[dev]"             # + test/lint tools
```

## Test

```bash
pip install -e ".[dev]"
make test    # run full test suite
make lint    # ruff
make schema  # regenerate scaffold JSON schema
```

## Reference Implementation

[CIP-Claude](https://github.com/Cole-Cant-Code/CIP-Claude) — a personal finance MCP server built on this protocol. 13 tools, 8 scaffolds, full guardrail pipeline.
