# CIP — Customer Intelligence Protocol

Structured reasoning and guardrails for MCP servers that talk to real people

```
User query → MCP Tool → Scaffold Engine → Inner LLM → Guardrails → Response
```

Most MCP servers return JSON and call it a day. CIP is for building MCP servers that talk to *regular people* — it gives an inner LLM a structured reasoning framework ("scaffold") so it can analyze domain data and respond in plain language with guardrails that actually enforce compliance.

The protocol knows nothing about finance, health, legal, or any other domain. You define one `DomainConfig` and point it at a directory of scaffold YAMLs.

## Quick start

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
llm = InnerLLMClient(create_provider("anthropic", api_key="sk-..."), config=config)
```

Then in your MCP tool handler:

```python
scaffold = engine.select(tool_name="analyze_spending", user_input=user_query)
prompt = engine.apply(scaffold, user_query=user_query, data_context=spending_data)
response = await llm.invoke(prompt, scaffold, data_context=spending_data)
```

Same structure works for any domain:

```python
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

## Scaffolds

Scaffolds externalize prompt engineering as YAML. Each file defines how the inner LLM reasons about a specific type of request — its role, reasoning steps, output constraints, and guardrails:

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

The engine selects scaffolds using a priority cascade: explicit ID > tool name match > intent signal + keyword scoring.

## Streaming

Guardrails run incrementally on each chunk. If a violation is detected mid-stream, the client halts and returns sanitized content.

```python
async for event in llm.invoke_stream(prompt, scaffold, data_context=data):
    if event.event == "chunk":
        print(event.text, end="")
    elif event.event in ("halted", "final"):
        final = event.response
```

## Install

```sh
pip install -e "."              # core only (mock provider)
pip install -e ".[anthropic]"   # + Claude
pip install -e ".[openai]"      # + OpenAI
pip install -e ".[all]"         # all providers
pip install -e ".[dev]"         # + pytest, ruff
```

## Reference implementation

[CIP-Claude](https://github.com/Cole-Cant-Code/CIP-Claude) — a personal finance MCP server built on this protocol.

## License

MIT — see [LICENSE](LICENSE).
