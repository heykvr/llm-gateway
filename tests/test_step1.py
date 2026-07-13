import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import OllamaProvider

async def main():
    llm = OllamaProvider()
    result = await llm.chat(
        [{"role": "user", "content": "Say hello in 5 telugu words."}],
        temperature=1.5
    )
    print(result)

asyncio.run(main())