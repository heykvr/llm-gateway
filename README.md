# llm-gateway

A production-grade LLM API gateway built with FastAPI. Sits in front of any LLM provider (Ollama locally, OpenAI/Anthropic-ready) and adds the production concerns raw APIs lack — token tracking, cost estimation, personas, resilience, and conversation memory.

Built as a hands-on deep dive into LLM engineering fundamentals: chat completions, tokens, context windows, sampling, and system prompts.

## Features

- **Provider abstraction** — swap Ollama for OpenAI or Anthropic by changing one line of config, not every endpoint (Strategy pattern)
- **Personas** — server-controlled system prompts (`concise`, `engineer`, `teacher`) that shape model behavior per request
- **Token tracking + cost estimation** — real token counts on every response, plus a cross-provider comparison: *"this exact conversation would have cost $0.0015 on GPT-4o — it ran free locally"*
- **Resilience** — retries with exponential backoff, timeouts, and automatic model fallback (llama3.1 → llama3.2); clean 503 when all backends are down
- **Conversation memory** — server-side sessions with a token budget, enforced by two selectable strategies: sliding window or LLM summarization
- **SSE streaming** — token-by-token responses via `/chat/stream`
- **Benchmarking** — `/benchmark` runs one prompt across models and compares throughput and cost
- **Strict API contract** — Pydantic validation on requests and responses; invalid roles, temperatures, or empty messages rejected with clear errors

## Architecture

```
                        ┌──────────────────────────────────┐
                        │           llm-gateway            │
                        │                                  │
 client ──── HTTP ────▶ │  FastAPI endpoints               │
                        │    /chat  /chat/stream           │
                        │    /chat/cost-comparison         │
                        │    /session/chat  /benchmark     │
                        │    /health                       │
                        │        │                         │
                        │        ▼                         │
                        │  Pydantic schemas (contract)     │
                        │        │                         │
                        │        ▼                         │
                        │  ResilientLLM                    │
                        │   (retry / timeout / fallback)   │
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

**Prerequisites:** Python 3.12+, [Ollama](https://ollama.com) with models pulled (`ollama pull llama3.1 llama3.2`)

```bash
git clone https://github.com/heykvr/llm-gateway.git
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

### Chat with a persona

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
  "model": "llama3.1",
  "usage": { "prompt_tokens": 47, "completion_tokens": 66, "total_tokens": 113 }
}
```

Personas: `default`, `concise` (max 2 sentences), `engineer` (technical, code examples), `teacher` (simple, analogies). Same question at temperature 0 — `concise` used 66 completion tokens, `teacher` used 239. System prompts directly drive token count, latency, and cost.

### Cross-provider cost comparison

```bash
curl -s http://localhost:8000/chat/cost-comparison \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Explain SAR imaging in one paragraph"}], "persona": "teacher"}'
```

```json
{
  "usage": {
    "cost_breakdown": {
      "llama3.2": 0.0,
      "gpt-4o": 0.0015075,
      "gpt-4o-mini": 0.00009045,
      "claude-sonnet": 0.002226
    }
  }
}
```

Only the local model runs — measured token counts are multiplied against each provider's published pricing. Estimates within ~±10% since tokenizers differ.

### Sessions with memory strategies

```bash
curl -s http://localhost:8000/session/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "message": "My name is Venkat.", "strategy": "summarize"}'
```

The gateway stores history server-side and enforces a token budget with a selectable strategy:

| Strategy | How it trims | Tradeoff |
|---|---|---|
| `sliding` (default) | Drops oldest messages | Fast and free — but early facts are lost completely |
| `summarize` | Compresses old messages into an LLM-written summary, keeps recent messages verbatim, falls back to sliding if still over budget | Preserves facts across long conversations — costs an extra LLM call and can blur who said what |

Tested head-to-head with the same conversation (intro → two 1,000+ token essays → "what is my name?"):

- `sliding`: *"You didn't mention your name earlier"* — the intro was dropped
- `summarize`: *"Your name is Venkat..."* — the summary preserved it, budget still enforced

One observed failure mode worth knowing: the summarizer can compress the **assistant's** speculation into the summary as if the user said it — provenance gets lost. Mitigable via the summarization prompt.

### Streaming

```bash
curl -N http://localhost:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Tell me about the moon"}], "persona": "concise"}'
```

Emits SSE events: `{"type": "token", ...}` per token, a final `{"type": "stats", ...}`, then `[DONE]`.

### Benchmark

```bash
curl -s -X POST http://localhost:8000/benchmark
```

| Model | Latency | Tokens/sec | Cost (local) | Would cost on GPT-4o |
|---|---|---|---|---|
| llama3.1 (8B) | 5.56s | 25.5 | $0 | $0.0015 |
| llama3.2 (3B) | 4.19s | 36.7 | $0 | $0.0016 |

*Measured on Apple M5. The smaller model trades quality for ~44% higher throughput.*

## Project structure

```
llm-gateway/
├── app/
│   ├── main.py          # FastAPI app, endpoints, personas
│   ├── llm.py           # LLMProvider interface + OllamaProvider (chat + streaming)
│   ├── resilience.py    # ResilientLLM: retries, timeouts, fallback
│   ├── conversation.py  # Session store, token budget, sliding window + summarization
│   ├── models.py        # Pydantic request/response schemas
│   └── tokenizer.py     # Token counting + cost estimation
├── tests/
└── requirements.txt
```

## Design decisions

- **Endpoints never call Ollama directly.** They depend on the `LLMProvider` interface; each provider normalizes its response shape (Ollama's `eval_count` → standard `completion_tokens`). Adding resilience required zero endpoint changes.
- **Only transient errors are retried.** Timeouts and connection failures get exponential backoff; a 400 is never retried — resending malformed input just wastes time.
- **Trimming happens before the model call**, not after a crash. System messages are never dropped — losing the persona mid-conversation would silently change behavior.
- **Streaming bypasses the resilience layer.** Retrying a half-delivered stream is a genuinely hard problem (the client already received partial tokens) — the streaming endpoint fails fast instead.
- **Sessions are in-memory** — swapping the dict for Redis is the intended production path; the interface stays identical.
- **Prices live in one config dict.** Adding a provider to the cost comparison is one line.

## License

MIT