import logging
from tenacity import (  # type: ignore[import-not-found]
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import httpx

from app.llm import LLMProvider

logger = logging.getLogger("resilience")
logging.basicConfig(level=logging.INFO)

# Errors worth retrying — transient network/server issues.
# NOT worth retrying: 400 bad request (retrying won't fix your JSON)
RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)


class ResilientLLM:
    """Wraps a primary + fallback provider with retries and timeouts."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider | None = None):
        self.primary = primary
        self.fallback = fallback

    @retry(
        stop=stop_after_attempt(3),                          # try 3 times max
        wait=wait_exponential(multiplier=1, min=1, max=8),   # wait 1s, 2s, 4s...
        retry=retry_if_exception_type(RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # log each retry
        reraise=True,                                         # re-raise final failure
    )
    async def _call_with_retry(self, provider: LLMProvider, messages, temperature):
        return await provider.chat(messages, temperature=temperature)

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> dict:
        try:
            return await self._call_with_retry(self.primary, messages, temperature)
        except RETRYABLE as e:
            if self.fallback is None:
                raise
            logger.warning(
                f"Primary model failed after all retries ({e!r}) — "
                f"falling back to {self.fallback.model}"
            )
            result = await self._call_with_retry(self.fallback, messages, temperature)
            result["fallback_used"] = True
            return result