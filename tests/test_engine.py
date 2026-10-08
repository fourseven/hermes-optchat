import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("agent.context_engine")

import optchat.engine  # noqa: E402
from optchat.engine import ACK, OptChatEngine, chat_key, view_marks  # noqa: E402


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(optchat.engine, "_caller", lambda: lambda messages: ("line", {}))


def start(tmp_path, model="m1", day=8):
    engine = OptChatEngine()
    engine.now = lambda: datetime(2026, 10, day, 9, 30, tzinfo=timezone.utc)
    engine.on_session_start("s", hermes_home=str(tmp_path), platform="cli", model=model, conversation_id=None)
    return engine


def turn(engine, history, user, *tail):
    incoming = {"role": "user", "content": user}
    conversation = [*history, incoming, *tail]
    request = [{"role": "system", "content": "hermes prompt with a date"}, *conversation]
    return conversation, engine.select_context(request, conversation_messages=conversation, incoming_message=incoming)


def test_request_is_system_view_and_this_turns_messages(tmp_path):
    (tmp_path / "optchat").mkdir()
    (tmp_path / "optchat" / "instructions.md").write_text("I am Mathew.")
    engine = start(tmp_path)
    for k in range(5):
        engine.chat.log("user", f"old {k}", engine.now())
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "zoom", "arguments": '{"id": 0, "n": 1}'}}]}
    result = {"role": "tool", "tool_call_id": "c1", "content": "user: old 0"}
    _, request = turn(engine, [{"role": "user", "content": "stale history"}], "hi", call, result)
    system, view, ack, user, *rest = request
    assert system["role"] == "system" and system["content"].endswith("I am Mathew.")
    assert "Compaction" not in system["content"]
    assert view == {"role": "user", "content": [{"type": "text", "text": "<chat>\n0+1|user: old 0\n1+1|user: old 1\n2+1|user: old 2\n3+1|user: old 3\n", "cache_control": {"type": "ephemeral"}}]}
    assert ack == {"role": "assistant", "content": ACK}
    assert user["content"].startswith("4+1|user: old 4\n</chat>\n\n[turn · 2026-10-08T09:30+00:00 · cli · m1]\n\nhi")
    assert rest == [call, result]


def test_system_prompt_is_byte_identical_across_days_and_models(tmp_path):
    _, first = turn(start(tmp_path, "m1", 8), [], "a")
    engine = start(tmp_path / "other", "m2", 9)
    _, second = turn(engine, [], "b")
    assert first[0] == second[0]


def test_turn_messages_are_logged_as_they_happen(tmp_path):
    engine = start(tmp_path)
    call = {"role": "assistant", "content": "looking", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "date", "arguments": '{"id": 0}'}}]}
    result = {"role": "tool", "tool_call_id": "c1", "content": "2026"}
    turn(engine, [], "when?")
    turn(engine, [], "when?", call, result)
    turn(engine, [], "when?", call, result)
    engine.on_turn_complete([{"role": "user", "content": "when?"}, call, result, {"role": "assistant", "content": "today"}])
    assert [(m["kind"], m["text"]) for m in engine.chat.log_.messages] == [
        ("user", "when?"), ("unii", "looking"), ("tool", 'date {"id": 0}'), ("echo", "2026"), ("unii", "today")]


def test_zoom_and_date_tools(tmp_path):
    engine = start(tmp_path)
    engine.chat.log("user", "hello", engine.now())
    assert {s["name"] for s in engine.get_tool_schemas()} == {"zoom", "date"}
    assert json.loads(engine.handle_tool_call("zoom", {"id": 0, "n": 1})) == {"result": "user: hello"}
    assert json.loads(engine.handle_tool_call("date", {"id": 0})) == {"result": "2026-10-08T09:30:00+00:00"}
    assert "error" in json.loads(engine.handle_tool_call("zoom", {"id": 1, "n": 1}))


def test_clones_share_one_chat_and_never_compress(tmp_path):
    engine = start(tmp_path)
    clone = engine.clone_for_agent()
    clone.on_session_start("s2", hermes_home=str(tmp_path), platform="cli", model="m1", conversation_id=None)
    assert clone.chat is engine.chat
    assert not engine.should_compress(10**9)


def test_view_marks_the_last_block_and_where_the_previous_turn_ended(tmp_path):
    engine = start(tmp_path)
    assert view_marks(engine.chat, 3) == [2]
    assert view_marks(engine.chat, 5) == [2, 4]
    assert view_marks(engine.chat, 5) == [4]


def test_restore_marks_puts_the_view_marks_back_after_hermes_strips_them(tmp_path):
    engine = start(tmp_path)
    for k in range(9):
        engine.chat.log("user", f"old {k}", engine.now())
    _, request = turn(engine, [], "hi")
    stripped = [{**m, "content": [{k: v for k, v in p.items() if k != "cache_control"} for p in m["content"]]}
                if isinstance(m["content"], list) else m for m in request]
    restored = engine.restore_marks(stripped)
    assert restored == request
    assert "cache_control" not in stripped[1]["content"][-1]


def test_one_chat_per_user_or_channel_across_threads():
    assert chat_key("agent:main:telegram:dm:123456789:229149") == "agent:main:telegram:dm:123456789"
    assert chat_key("agent:main:telegram:dm:123456789") == "agent:main:telegram:dm:123456789"
    assert chat_key(None) == "cli"


def test_cron_and_subagents_keep_hermess_own_context(tmp_path):
    for platform in ("cron", "subagent"):
        engine = OptChatEngine()
        engine.on_session_start("s", hermes_home=str(tmp_path), platform=platform, model="m", conversation_id=None)
        incoming = {"role": "user", "content": "run the job"}
        assert engine.select_context([incoming], conversation_messages=[incoming], incoming_message=incoming) is None
        assert "error" in json.loads(engine.handle_tool_call("zoom", {"id": 0, "n": 1}))
    assert not (tmp_path / "optchat").exists()
