import tiktoken

# Prices per 1M tokens (USD). Ollama is free — the comparison is the feature.
PRICING = {
    "llama3.2":       {"input": 0.0,  "output": 0.0},
    "llama3.1":       {"input": 0.0,  "output": 0.0},
    "gpt-4o":         {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":    {"input": 0.15, "output": 0.60},
    "claude-sonnet":  {"input": 3.00, "output": 15.00},
}

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count. Exact counts vary per model's tokenizer,
    but cl100k_base is close enough for budgeting."""
    return len(_enc.encode(text))


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Cost in USD for a given token usage on a given model."""
    prices = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (
        prompt_tokens * prices["input"] / 1_000_000
        + completion_tokens * prices["output"] / 1_000_000
    )
    return round(cost, 8)


def cost_comparison(prompt_tokens: int, completion_tokens: int) -> dict:
    """The killer feature: what would this call cost on every provider?"""
    return {
        model: estimate_cost(prompt_tokens, completion_tokens, model)
        for model in PRICING
    }