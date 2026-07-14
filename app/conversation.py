from app.tokenizer import count_tokens

MAX_HISTORY_TOKENS = 2000   # budget for conversation history (deliberately small for testing)
SUMMARY_PREFIX = "Summary of earlier conversation:"
KEEP_RECENT = 4   # last N messages always survive verbatim

class ConversationStore:
    """In-memory session store with token-budget enforcement.
    Swap the dict for Redis in production — the interface stays identical."""

    def __init__(self, max_history_tokens: int = MAX_HISTORY_TOKENS):
        self._sessions: dict[str, list[dict]] = {}
        self.max_history_tokens = max_history_tokens

    def get(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )

    def history_tokens(self, session_id: str) -> int:
        return sum(
            count_tokens(m["content"]) + 4          # +4 ≈ per-message role overhead
            for m in self.get(session_id)
        )

    def trim(self, session_id: str) -> int:
        """Sliding window: drop oldest messages until under budget.
        Never drops a system message. Returns how many were dropped."""
        history = self._sessions.get(session_id, [])
        dropped = 0
        while self.history_tokens(session_id) > self.max_history_tokens and len(history) > 1:
            # find oldest non-system message
            for i, m in enumerate(history):
                if m["role"] != "system":
                    history.pop(i)
                    dropped += 1
                    break
            else:
                break   # only system messages left — nothing more to drop
        return dropped
    
    async def summarize_and_trim(self, session_id: str, llm) -> int:
        """Compress old messages into a summary instead of dropping them.
        Returns number of messages compressed."""
        history = self._sessions.get(session_id, [])

        if self.history_tokens(session_id) <= self.max_history_tokens:
            return 0   # under budget, nothing to do

        # Split: [protected system msgs] [old msgs → compress] [recent → keep]
        system_msgs = [m for m in history if m["role"] == "system"
                       and not m["content"].startswith(SUMMARY_PREFIX)]
        old_summary = next((m for m in history
                            if m["content"].startswith(SUMMARY_PREFIX)), None)
        conversational = [m for m in history if m["role"] != "system"]

        if len(conversational) <= KEEP_RECENT:
            return 0   # nothing old enough to compress

        to_compress = conversational[:-KEEP_RECENT]
        recent = conversational[-KEEP_RECENT:]

        # Build the summarization prompt
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
        if old_summary:
            transcript = old_summary["content"] + "\n\n" + transcript

        result = await llm.chat([
            {"role": "system", "content":
                "Summarize this conversation in under 150 words. "
                "Preserve ALL names, facts, numbers, and decisions. "
                "Write in third person."},
            {"role": "user", "content": transcript},
        ], temperature=0.0)

        summary_msg = {
            "role": "system",
            "content": f"{SUMMARY_PREFIX} {result['content']}"
        }


        # Rebuild: persona system msgs + summary + recent verbatim
        self._sessions[session_id] = system_msgs + [summary_msg] + recent

        # Safety net: if recent messages alone still bust the budget,
        # fall back to sliding-window on them
        compressed = len(to_compress)
        if self.history_tokens(session_id) > self.max_history_tokens:
            self.trim(session_id)

        return compressed

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


store = ConversationStore()