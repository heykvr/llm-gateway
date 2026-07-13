from pydantic import ValidationError
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import ChatRequest, Message

# Valid request — should work
req = ChatRequest(messages=[Message(role="user", content="hi")], temperature=0.5)
print("Valid request OK:", req.temperature)

# Invalid role — should fail
try:
    Message(role="admin", content="hi")
except ValidationError as e:
    print("Rejected bad role ✓")

# Invalid temperature — should fail
try:
    ChatRequest(messages=[Message(role="user", content="hi")], temperature=5.0)
except ValidationError as e:
    print("Rejected bad temperature ✓")