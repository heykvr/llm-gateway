from fastapi import FastAPI, HTTPException
import httpx
import time
from app.llm import OllamaProvider
from app.models import ChatRequest, ChatResponse, Usage
from app.tokenizer import estimate_cost, cost_comparison
from app.resilience import ResilientLLM
from app.conversation import store
from app.models import SessionChatRequest, SessionChatResponse
from fastapi.responses import StreamingResponse
import json as jsonlib
app = FastAPI(title="llm-gateway", version="0.1.0")
llm = ResilientLLM(
    primary=OllamaProvider(model="llama3.1"),
    fallback=OllamaProvider(model="llama3.2"),
)

# llm = OllamaProvider()

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

    try:
        result = await llm.chat(messages, temperature=req.temperature)
    except HTTPException:
        raise                                    # don't swallow our own 400s
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="All LLM backends unavailable. Try again shortly."
        )

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

@app.post("/session/chat", response_model=SessionChatResponse)
async def session_chat(req: SessionChatRequest):
    # 1. Persona system prompt — only on a brand-new session
    if req.persona and not store.get(req.session_id):
        if req.persona not in PERSONAS:
            raise HTTPException(status_code=400,
                detail=f"Unknown persona. Available: {list(PERSONAS.keys())}")
        store.append(req.session_id, "system", PERSONAS[req.persona])

    # 2. Add the user's new message
    store.append(req.session_id, "user", req.message)

 
    # 3. Enforce token budget BEFORE calling the model
    dropped = 0
    compressed = 0
    if req.strategy == "summarize":
        compressed = await store.summarize_and_trim(req.session_id, llm)
    else:
        dropped = store.trim(req.session_id)

    # 4. Call the model with managed history
    try:
        result = await llm.chat(store.get(req.session_id), temperature=req.temperature)
    except Exception:
        raise HTTPException(status_code=503,
            detail="All LLM backends unavailable. Try again shortly.")

    # 5. Store the assistant's reply — it's part of history now
    store.append(req.session_id, "assistant", result["content"])

    return SessionChatResponse(
        content=result["content"],
        model=result["model"],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
            estimated_cost_usd=estimate_cost(
                result["prompt_tokens"], result["completion_tokens"], result["model"]),
        ),
        session_tokens=store.history_tokens(req.session_id),
        messages_in_history=len(store.get(req.session_id)),
        dropped_messages=dropped,
        compressed_messages=compressed
    )


@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    store.reset(session_id)
    return {"status": "cleared", "session_id": session_id}



@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    if req.persona and req.persona in PERSONAS and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": PERSONAS[req.persona]})

    provider = llm.primary   # stream from primary directly

    async def generate():
        try:
            async for event in provider.chat_stream(messages, req.temperature):
                if event["type"] == "token":
                    yield f"data: {jsonlib.dumps(event)}\n\n"
                else:
                    yield f"data: {jsonlib.dumps(event)}\n\n"
                    yield "data: [DONE]\n\n"
        except Exception:
            yield f'data: {jsonlib.dumps({"type": "error", "detail": "stream failed"})}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")



@app.post("/benchmark")
async def benchmark(prompt: str = "Explain what a token is in one paragraph"):
    """Run the same prompt on every model, compare speed and cost."""
    results = []
    for model_name in ["llama3.1", "llama3.2"]:
        provider = OllamaProvider(model=model_name)
        start = time.perf_counter()
        try:
            r = await provider.chat(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
            elapsed = time.perf_counter() - start
            results.append({
                "model": model_name,
                "latency_s": round(elapsed, 2),
                "completion_tokens": r["completion_tokens"],
                "tokens_per_sec": round(r["completion_tokens"] / elapsed, 1),
                "hypothetical_cost": cost_comparison(
                    r["prompt_tokens"], r["completion_tokens"]
                ),
            })
        except Exception as e:
            results.append({"model": model_name, "error": str(e)})
    return {"prompt": prompt, "results": results}