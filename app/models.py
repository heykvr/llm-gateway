from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    persona: str | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float = 0.0
    cost_breakdown: dict[str, float] | None = None


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage


class SessionChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    persona: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    strategy: Literal["sliding", "summarize"] = "sliding"


class SessionChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    session_tokens: int          # how full is this session's context
    messages_in_history: int
    dropped_messages: int        # sliding window at work
    compressed_messages: int = 0