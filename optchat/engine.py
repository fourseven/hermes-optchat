import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent.context_engine import ContextEngine

from .chat import Chat
from .compactor import ACK, MARK, Compactor
from .prompt import system_prompt, turn_prompt

logger = logging.getLogger(__name__)

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


def _caller():
    from .openrouter import caller

    return caller()


def _open_chat(root, system):
    """One Chat and one Compactor per chat directory, shared by every agent in the process."""
    with _chats_lock:
        if root not in _chats:
            chat = Chat(root)
            Compactor(chat, _caller(), system, lambda: datetime.now(timezone.utc))
            _chats[root] = chat
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


def compose(system, view, state, turn, marks=()):
    """[system] [view as 4-line blocks] [ack] [view tail, state and this turn's messages].

    marks: indexes of view blocks to carry a cache mark.
    """
    blocks, tail = view
    head = f"{tail}\n\n{state}\n\n"
    content = turn[0].get("content")
    user = {**turn[0], "content": head + content if isinstance(content, str) else [{"type": "text", "text": head}, *content]}
    parts = [{"type": "text", "text": b} for b in blocks]
    for k in marks:
        parts[k]["cache_control"] = MARK
    view = [{"role": "user", "content": parts}, {"role": "assistant", "content": ACK}] if blocks else []
    return [{"role": "system", "content": system}, *view, user, *turn[1:]]


def view_marks(chat, blocks):
    """Mark the last block, and where the previous turn's view ended so this turn finds its entry.

    OpenRouter showed no lookback from a mark to an earlier entry, for Luna and for Claude.
    """
    previous = chat.turn_started(blocks)
    return sorted({k for k in (previous - 1, blocks - 1) if 0 <= k < blocks})


def install_cache_mark_patch():
    """Monkey patch: Hermes strips every cache_control and re-applies its own on each attempt
    (agent/turn_api_request.py:113), adding none on routes without Anthropic-style caching
    (Luna on OpenRouter). After it runs, put the engine's view marks back on those routes.
    Routes Hermes marks itself are left alone: it already spends all 4 breakpoints there.
    """
    import agent.conversation_loop as loop

    original = loop._redecorate_prompt_cache_for_provider
    if getattr(original, "optchat", False):
        return

    def redecorate(agent, api_messages, **kwargs):
        messages, prepared, tools = original(agent, api_messages, **kwargs)
        engine = getattr(agent, "context_compressor", None)
        if isinstance(engine, OptChatEngine) and not agent._use_prompt_caching:
            messages = engine.restore_marks(messages)
        return messages, prepared, tools

    redecorate.optchat = True
    loop._redecorate_prompt_cache_for_provider = redecorate


class OptChatEngine(ContextEngine):
    """One chat that never ends: each request is [system] [view] [this turn's messages]."""

    emit_automatic_compaction_status = False

    def __init__(self):
        self.chat = None
        self.system = ""
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
        path = home / "instructions.md"
        instructions = path.read_text() if path.exists() else ""
        self.system = turn_prompt(instructions)
        self.chat = _open_chat(home / "chats" / key, system_prompt(instructions))
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
        if self.chat is not None:
            self.chat.record_usage({"kind": "turn", "model": self.model, "usage": usage})

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
        state = f"[turn · {self.now().isoformat(timespec='minutes')} · {self.platform} · {self.model}]"
        return compose(self.system, self.view, state, sent, self.marks)

    def _start_turn(self, k, incoming):
        if not self.chat.wait_summarized(WAIT):
            logger.warning("optchat: earlier messages still unsummarized after %ss", WAIT)
        self.view = self.chat.render_view()
        self.marks = view_marks(self.chat, len(self.view[0]))
        self.turn, self.logged = k, 0

    def _log(self, turn):
        for message in turn[self.logged :]:
            for kind, text in _entries(message):
                self.chat.log(kind, text, self.now())
        self.logged = max(self.logged, len(turn))

    def on_turn_complete(self, messages, usage=None, **kwargs):
        if self.chat is not None and self.turn is not None:
            self._log(messages[self.turn :])

    def restore_marks(self, messages):
        """Put the view's cache marks back on a request Hermes stripped them from."""
        blocks = self.view[0] if self.turn is not None else []
        for k, message in enumerate(messages):
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, list) and len(content) == len(blocks) \
                    and content and content[0].get("text") == blocks[0]:
                parts = [dict(part) for part in content]
                for j in self.marks:
                    parts[j]["cache_control"] = MARK
                return [*messages[:k], {**message, "content": parts}, *messages[k + 1 :]]
        return messages

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
