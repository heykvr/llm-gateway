# llm-gateway

A production-grade LLM API gateway built with FastAPI. Sits in front of any LLM provider (Ollama locally, OpenAI/Anthropic-ready) and adds the production concerns raw APIs lack — token tracking, cost estimation, personas, resilience, and conversation management.

Built as a hands-on deep dive into LLM engineering fundamentals: chat completions, tokens, context windows, sampling, and system prompts.

## Features

- **Provider abstraction** — swap Ollama for OpenAI or Anthropic by changing one line of config, not every endpoint (Strategy pattern)
- **Personas** — server-controlled system prompts (`concise`, `engineer`, `teacher`) that shape model behavior per request
- **Token tracking** — real prompt/completion token counts on every response
- **Cost estimation** — see what each call costs, plus a cross-provider comparison: *"this exact conversation would have cost $0.0015 on GPT-4o — it ran free locally"*
- **Strict API contract** — Pydantic validation on both requests and responses; invalid roles, temperatures, or empty messages are rejected with clear errors
- **Health checks** — the gateway verifies its Ollama dependency instead of assuming it's up

### Roadmap

- [ ] Resilience layer — retries with exponential backoff, timeouts, model fallback chain
- [ ] Conversation manager — history with sliding-window trimming and token budgets
- [ ] SSE streaming endpoint
- [ ] `/benchmark` — run one prompt across models, compare tokens/sec, latency, and cost

## Architecture

```
                        ┌──────────────────────────────────┐
                        │           llm-gateway            │
                        │                                  │
 client ──── HTTP ────▶ │  FastAPI endpoints               │
                        │    /chat                         │
                        │    /chat/cost-comparison         │
                        │    /health                       │
                        │        │                         │
                        │        ▼                         │
                        │  Pydantic schemas (contract)     │
                        │        │                         │
                        │        ▼                         │
                        │  LLMProvider (abstract)          │
                        │        │                         │
                        │        ├──▶ OllamaProvider ──────┼───▶ localhost:11434
                        │        ├──▶ OpenAIProvider*      │
                        │        └──▶ AnthropicProvider*   │
                        └──────────────────────────────────┘
                                                * planned
```

## Quick start

**Prerequisites:** Python 3.12+, [Ollama](https://ollama.com) with a model pulled (`ollama pull llama3.2`)

```bash
# Clone and set up
git clone https://github.com/<your-username>/llm-gateway.git
cd llm-gateway
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama serve

# Run the gateway
uvicorn app.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

## Usage

### Basic chat with a persona

```bash
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is a token in an LLM?"}],
    "persona": "concise",
    "temperature": 0
  }'
```

```json
{
  "content": "A token is the smallest unit of text an LLM processes...",
  "model": "llama3.2",
  "usage": {
    "prompt_tokens": 47,
    "completion_tokens": 66,
    "total_tokens": 113,
    "estimated_cost_usd": 0.0
  }
}
```

### Cross-provider cost comparison

```bash
curl -s http://localhost:8000/chat/cost-comparison \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Explain SAR imaging in one paragraph"}],
    "persona": "teacher"
  }'
```

```json
{
  "usage": {
    "prompt_tokens": 47,
    "completion_tokens": 139,
    "cost_breakdown": {
      "llama3.2": 0.0,
      "gpt-4o": 0.0015075,
      "gpt-4o-mini": 0.00009045,
      "claude-sonnet": 0.002226
    }
  }
}
```

Only the local model runs — the comparison multiplies measured token counts against each provider's published pricing. Estimates are within ~±10% since tokenizers differ across providers.

### Personas

| Persona | Behavior |
|---|---|
| `default` | Standard helpful assistant |
| `concise` | Maximum 2 sentences |
| `engineer` | Precise, technical, code examples |
| `teacher` | Simple explanations with analogies |

Same question, same temperature — `concise` used 66 completion tokens, `teacher` used 239. System prompts directly drive token count, latency, and cost.

## Project structure

```
llm-gateway/
├── app/
│   ├── main.py         # FastAPI app, endpoints, personas
│   ├── llm.py          # LLMProvider interface + OllamaProvider
│   ├── models.py       # Pydantic request/response schemas
│   └── tokenizer.py    # Token counting + cost estimation
├── tests/
└── requirements.txt
```

## Design decisions

- **Endpoints never call Ollama directly.** They depend on the `LLMProvider` interface. Each provider normalizes its response shape (Ollama's `eval_count` → standard `completion_tokens`), so the app is provider-agnostic.
- **Prices live in one config dict**, never scattered through code. Adding a provider to the cost comparison is one line.
- **Validation is declarative.** `Literal["system", "user", "assistant"]` and `Field(ge=0.0, le=2.0)` encode the rules — zero hand-written validation code.
- **Response shape mirrors OpenAI's** (`usage.prompt_tokens` etc.) so the gateway feels familiar to anyone who has used commercial LLM APIs.

## License

MIT