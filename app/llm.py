from abc import ABC, abstractmethod
import httpx


class LLMProvider(ABC):
    """Abstract interface — every provider must implement this."""

    @abstractmethod
    async def chat(self, messages: list[dict], temperature: float = 0.7) -> dict:
        """Send messages, return {content, prompt_tokens, completion_tokens, model}."""
        ...


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.base_url}/api/chat", json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            })
            r.raise_for_status()
            data = r.json()

        return {
            "content": data["message"]["content"],
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "model": self.model,
        }