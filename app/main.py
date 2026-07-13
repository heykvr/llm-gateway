from fastapi import FastAPI, HTTPException
import httpx

from app.llm import OllamaProvider
from app.models import ChatRequest, ChatResponse, Usage
from app.tokenizer import estimate_cost, cost_comparison
app = FastAPI(title="llm-gateway", version="0.1.0")

llm = OllamaProvider()

# ── Personas: server-controlled system prompts (Topic 9) ──────────────────
PERSONAS = {
    "default":  "You are a helpful assistant.",
    "concise":  "You are a helpful assistant. Answer in 2 sentences maximum.",
    "engineer": "You are a senior software engineer. Be precise and technical. Use code examples where helpful.",
    "teacher":  "You explain concepts simply, using analogies, as if teaching a beginner.",
}


@app.get("/health")
async def health():
    """Is ollama reachable? Production services check their dependencies."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:11434/api/version")
            r.raise_for_status()
        return {"status": "ok", "ollama": r.json()["version"]}
    except Exception:
        raise HTTPException(status_code=503, detail="ollama unreachable")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Build final message list: persona system prompt goes first
    messages = [m.model_dump() for m in req.messages]

    if req.persona:
        if req.persona not in PERSONAS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown persona. Available: {list(PERSONAS.keys())}"
            )
        # Prepend system prompt (unless client already sent one)
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": PERSONAS[req.persona]})

    result = await llm.chat(messages, temperature=req.temperature)

    return ChatResponse(
        content=result["content"],
        model=result["model"],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
            estimated_cost_usd=estimate_cost(
                result["prompt_tokens"], result["completion_tokens"], result["model"]
            ),
        ),
    )


@app.post("/chat/cost-comparison", response_model=ChatResponse)
async def chat_with_comparison(req: ChatRequest):
    """Same as /chat but shows what the call would cost on every provider."""
    response = await chat(req)
    response.usage.cost_breakdown = cost_comparison(
        response.usage.prompt_tokens, response.usage.completion_tokens
    )
    return response