import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent.context_engine import ContextEngine

from .chat import Chat
from .prompt import system_prompt

logger = logging.getLogger(__name__)

# Hermes merges adjacent user messages; this keeps the view its own message, so its
# last whole block carries a cache mark.
ACK = "(the chat continues below)"
WAIT = 120

_chats = {}
_chats_lock = threading.Lock()

ZOOM = {
    "name": "zoom",
    "description": "Open the line id+n of the view into the two lines of n/2 under it; n = 1 gives the message whole.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "The line's first message; a multiple of n."},
            "n": {"type": "integer", "description": "How many messages the line covers; a power of 2."},
        },
        "required": ["id", "n"],
    },
}
DATE = {
    "name": "date",
    "description": "The date and time of message id.",
    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
}


def _open_chat(root):
    with _chats_lock:
        if root not in _chats:
            _chats[root] = Chat(root)
        return _chats[root]


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")


def _entries(message):
    """(kind, text) log entries for one OpenAI-format message; thoughts are never logged."""
    role = message.get("role")
    if role == "user":
        return [("user", _text(message.get("content")))]
    if role == "tool":
        return [("echo", _text(message.get("content")))]
    if role != "assistant":
        return []
    entries = [("unii", text)] if (text := _text(message.get("content")).strip()) else []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        entries.append(("tool", f"{fn.get('name')} {fn.get('arguments')}"))
    return entries


class OptChatEngine(ContextEngine):
    """One chat that never ends: each request is [system] [view] [this turn's messages]."""

    emit_automatic_compaction_status = False

    def __init__(self):
        self.chat = None
        self.instructions = ""
        self.platform = self.model = ""
        self.turn = None
        self.logged = 0
        self.now = lambda: datetime.now(timezone.utc)

    @property
    def name(self):
        return "optchat"

    def on_session_start(self, session_id, hermes_home=None, platform="cli", model="", conversation_id=None, **kwargs):
        home = Path(hermes_home) / "optchat"
        key = re.sub(r"[^A-Za-z0-9_.-]", "_", conversation_id or "cli")
        self.chat = _open_chat(home / "chats" / key)
        path = home / "instructions.md"
        self.instructions = path.read_text() if path.exists() else ""
        self.platform, self.model = platform, model

    def clone_for_agent(self):
        return type(self)()

    def update_model(self, model, context_length, *args, **kwargs):
        super().update_model(model, context_length, *args, **kwargs)
        self.model = model

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, *args, **kwargs):
        return messages

    def select_context(self, request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0):
        if self.chat is None or not incoming_message or not conversation_messages:
            return None
        k = max(j for j, m in enumerate(conversation_messages) if m == incoming_message)
        turn = conversation_messages[k:]
        sent = request_messages[len(request_messages) - len(turn) :]
        if self.turn != k:
            self._start_turn(k, incoming_message)
        self._log(turn)
        blocks, tail = self.view
        state = f"[{self.now().isoformat(timespec='minutes')} · {self.platform} · {self.model}]"
        head = f"{tail}\n\n{state}\n\n"
        content = sent[0].get("content")
        user = {**sent[0], "content": head + content if isinstance(content, str) else [{"type": "text", "text": head}, *content]}
        view = [{"role": "user", "content": [{"type": "text", "text": b} for b in blocks]}, {"role": "assistant", "content": ACK}] if blocks else []
        return [{"role": "system", "content": system_prompt(self.instructions)}, *view, user, *sent[1:]]

    def _start_turn(self, k, incoming):
        if not self.chat.wait_summarized(WAIT):
            logger.warning("optchat: earlier messages still unsummarized after %ss", WAIT)
        self.view = self.chat.render_view()
        self.turn, self.logged = k, 0

    def _log(self, turn):
        for message in turn[self.logged :]:
            for kind, text in _entries(message):
                self.chat.log(kind, text, self.now())
        self.logged = max(self.logged, len(turn))

    def on_turn_complete(self, messages, usage=None, **kwargs):
        if self.chat is not None and self.turn is not None:
            self._log(messages[self.turn :])

    def get_tool_schemas(self):
        return [ZOOM, DATE]

    def handle_tool_call(self, name, args, **kwargs):
        try:
            if name == "zoom":
                return json.dumps({"result": self.chat.zoom(int(args["id"]), int(args["n"]))})
            if name == "date":
                return json.dumps({"result": self.chat.date(int(args["id"]))})
        except (KeyError, ValueError) as e:
            return json.dumps({"error": str(e)})
        return super().handle_tool_call(name, args, **kwargs)
