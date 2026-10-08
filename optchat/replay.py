"""Replay a Hermes state.db through OptChat and measure the cache.

Each real message is logged with its real date. Before each assistant message the turn
call it answered is sent again, built as the engine builds it, with max_tokens small:
only the prompt side is measured. Writes calls.jsonl next to the chat and prints totals.
"""

import argparse
import copy
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .chat import Chat
from .compactor import Compactor
from .engine import DATE, ZOOM, _entries, compose
from .openrouter import MODEL, caller
from .prompt import system_prompt, turn_prompt

TOOLS = [{"type": "function", "function": schema} for schema in (ZOOM, DATE)]
MARK = {"type": "ephemeral"}


def load(db, sources, limit):
    query = f"""
        select m.role, m.content, m.tool_calls, m.tool_call_id, m.timestamp
        from messages m join sessions s on s.id = m.session_id
        where s.parent_session_id is null and s.source in ({','.join('?' * len(sources))})
          and coalesce(m._compressed_summary, 0) = 0 and m.role in ('user', 'assistant', 'tool')
        order by m.timestamp, m.id"""
    messages = []
    for role, content, tool_calls, tool_call_id, ts in sqlite3.connect(db).execute(query, sources):
        message = {"role": role, "content": content or ""}
        if tool_calls:
            message["tool_calls"] = json.loads(tool_calls)
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        messages.append((message, datetime.fromtimestamp(ts, timezone.utc)))
    turns, logged = [], 0
    for message, when in messages:
        if message["role"] == "user" or not turns:
            if limit and logged >= limit:
                break
            turns.append([])
        turns[-1].append((message, when))
        logged += len(_entries(message))
    return turns


def mark(messages):
    """Anthropic cache marks where Hermes puts them: the view's last block and the request's end."""
    messages = copy.deepcopy(messages)
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            message["content"][-1]["cache_control"] = MARK
            break
    last = messages[-1]
    if isinstance(last["content"], str) and last["content"]:
        last["content"] = [{"type": "text", "text": last["content"], "cache_control": MARK}]
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    parser.add_argument("out")
    parser.add_argument("--sources", default="telegram")
    parser.add_argument("--limit", type=int, default=0, help="stop after about this many logged messages")
    parser.add_argument("--turn-model", default="openai/gpt-6-luna")
    parser.add_argument("--compact-model", default=MODEL)
    parser.add_argument("--instructions", default="")
    args = parser.parse_args()

    out = Path(args.out)
    instructions = Path(args.instructions).read_text() if args.instructions else ""
    chat = Chat(out / "chat")
    compactor = Compactor(chat, caller(args.compact_model), system_prompt(instructions), lambda: datetime.now(timezone.utc))
    # One session id for the whole replay: OpenRouter scopes OpenAI's cache to it.
    turn_call = caller(args.turn_model, {"session_id": "optchat-replay"})
    system = turn_prompt(instructions)
    anthropic = args.turn_model.startswith("anthropic/")
    turns = load(args.db, args.sources.split(","), args.limit)
    previous = []

    with open(out / "calls.jsonl", "a") as calls:
        for t, turn in enumerate(turns):
            started = time.monotonic()
            chat.wait_summarized(600)
            wait = time.monotonic() - started
            view = chat.render_view()
            lines = list(chat.view.lines)
            kept = next((k for k, (a, b) in enumerate(zip(previous, lines)) if a != b), min(len(previous), len(lines)))
            previous = lines
            sent, first = [], True
            for c, (message, when) in enumerate(turn):
                if message["role"] == "assistant" and sent:
                    state = f"[turn · {when.isoformat(timespec='minutes')} · telegram · {args.turn_model}]"
                    request = compose(system, view, state, sent)
                    _, usage = turn_call(mark(request) if anthropic else request, max_tokens=16, tools=TOOLS)
                    calls.write(json.dumps({"turn": t, "call": c, "messages": len(chat.log_.messages), "wait": round(wait, 2),
                                            "view_lines": len(lines), "rewritten": len(lines) - kept if first else 0,
                                            "usage": usage}) + "\n")
                    calls.flush()
                    first = False
                sent.append(message)
                for kind, text in _entries(message):
                    chat.log(kind, text, when)
            print(f"turn {t + 1}/{len(turns)}: {len(chat.log_.messages)} messages, view {len(lines)} lines, wait {wait:.1f}s", flush=True)
    compactor.idle(600)


if __name__ == "__main__":
    main()
