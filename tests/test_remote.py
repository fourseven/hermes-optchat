import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from optchat.chat import Chat
from optchat.remote import LeaseLost, RemoteChat

# Runs against the TypeScript service; point OPTCHAT_MEMORY_SRC at an optchat-memory checkout.
SRC = os.environ.get("OPTCHAT_MEMORY_SRC")
pytestmark = pytest.mark.skipif(not SRC, reason="set OPTCHAT_MEMORY_SRC")

WHEN = datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc)
TOKEN = "test-token"


@pytest.fixture(scope="module")
def url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen(
        ["node", str(Path(SRC) / "scripts" / "serve.ts")],
        env={**os.environ, "MEMORY_TOKEN": TOKEN, "PORT": str(port)},
        stdout=subprocess.PIPE,
    )
    assert server.stdout.readline().strip() == b"listening"
    yield f"http://127.0.0.1:{port}"
    server.terminate()
    server.wait()


def test_remote_reads_match_a_local_chat_given_the_same_writes(url, tmp_path):
    remote = RemoteChat(url, TOKEN, "same-writes", "hermes", "Be brief.")
    local = Chat(tmp_path)
    texts = ["a" * 300, "b\nc", "c", "x" * 600, "d", "é" * 100]
    for text in texts:
        remote.log("user", text, WHEN)
        local.log("user", text, WHEN)
    assert remote.render_view() == local.render_view()
    assert remote.zoom(0, 4) == local.zoom(0, 4)
    assert remote.zoom(3, 1) == local.zoom(3, 1)
    assert remote.date(5) == local.date(5)
    assert remote.wait_summarized(0) is False
    assert remote.turn_started(3) == local.turn_started(3) == 0
    assert remote.turn_started(1) == 3
    remote.close()


def test_bad_zoom_is_a_value_error_as_locally(url):
    remote = RemoteChat(url, TOKEN, "bad-zoom", "hermes", "")
    remote.log("user", "a", WHEN)
    with pytest.raises(ValueError, match="no line 0\\+2"):
        remote.zoom(0, 2)
    remote.close()


def test_a_second_holder_cannot_take_the_chat(url):
    first = RemoteChat(url, TOKEN, "two-holders", "hermes", "")
    with pytest.raises(LeaseLost):
        RemoteChat(url, TOKEN, "two-holders", "other", "")
    first.log("user", "still mine", WHEN)
    first.close()


def test_a_wrong_token_is_refused(url):
    with pytest.raises(Exception, match="401"):
        RemoteChat(url, "wrong", "refused", "hermes", "")


def test_the_engine_runs_a_turn_against_the_service(url, tmp_path, monkeypatch):
    pytest.importorskip("agent.context_engine")
    from optchat import engine

    monkeypatch.setenv("OPTCHAT_REMOTE_URL", url)
    monkeypatch.setenv("OPTCHAT_REMOTE_TOKEN", TOKEN)
    monkeypatch.setattr(engine, "_chats", {})
    e = engine.OptChatEngine()
    e.now = lambda: WHEN
    e.on_session_start("s", hermes_home=str(tmp_path), platform="cli", model="m", conversation_id=None)
    assert isinstance(e.chat, RemoteChat)
    incoming = {"role": "user", "content": "hello"}
    request = e.select_context([incoming], conversation_messages=[incoming], incoming_message=incoming)
    # The view is rendered before the turn is logged; the turn follows it.
    assert request[-1]["content"] == "<chat>\n</chat>\n\n[turn · 2026-10-08T09:30+00:00 · cli · m]\n\nhello"
    e.on_turn_complete([incoming, {"role": "assistant", "content": "hi there"}])
    assert [e.chat.zoom(0, 1), e.chat.zoom(1, 1)] == ["user: hello", "unii: hi there"]
    e.chat.close()
