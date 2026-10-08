import json
import threading
from datetime import datetime, timezone

from optchat.chat import Chat
from optchat.compactor import Compactor
from optchat.prompt import RULER

WHEN = datetime(2026, 10, 8, tzinfo=timezone.utc)
LONG = "x" * 600


class Fake:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.lock = threading.Lock()

    def __call__(self, messages):
        with self.lock:
            self.requests.append(messages)
            reply = self.replies.pop(0) if self.replies else "line"
        if isinstance(reply, Exception):
            raise reply
        return reply, {"prompt_tokens": 10}


def run(tmp_path, fake, *texts):
    chat = Chat(tmp_path)
    compactor = Compactor(chat, fake, "SYSTEM", lambda: WHEN)
    for text in texts:
        chat.log("unii", text, WHEN)
    assert chat.wait_summarized(5)
    compactor.idle(5)
    return chat


def test_a_long_message_is_compressed_with_the_ruler(tmp_path):
    fake = Fake("unii: summary")
    chat = run(tmp_path, fake, "short", LONG)
    assert chat.log_.nodes[(0, 1)] == "unii: summary"
    system, user = fake.requests[0]
    assert system["content"][0] == {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    assert user["content"].startswith("<chat>\n0+1|unii: short\n</chat>\n\nCompaction: compress message 1")
    assert RULER in user["content"] and f"unii: {LONG}" in user["content"]


def test_a_reply_over_512_bytes_gets_the_cut_and_the_shortest_line_is_kept(tmp_path):
    fake = Fake("a" * 700, "b" * 610, "c" * 650, "d" * 601, "e" * 900)
    chat = run(tmp_path, fake, LONG)
    assert chat.log_.nodes[(0, 0)] == "d" * 601
    assert len(fake.requests) == 5
    retry = fake.requests[1]
    assert retry[-2] == {"role": "assistant", "content": "a" * 700}
    assert retry[-1]["content"].startswith("Too long: your line is 700 bytes")
    assert retry[-1]["content"].endswith("a" * 512 + "| ← LIMIT")


def test_a_merge_takes_both_halves_and_their_range(tmp_path):
    fake = Fake("y" * 300, "y" * 300, "merged")
    chat = run(tmp_path, fake, LONG, LONG)
    assert chat.log_.nodes[(1, 0)] == "merged"
    task = fake.requests[2][-1]["content"]
    assert "merge lines 0+1 and 1+1" in task and "messages, 0 to 1," in task
    assert task.endswith(f"<input>\n{'y' * 300}\n{'y' * 300}\n</input>")


def test_a_failed_call_is_tried_again_at_the_next_message(tmp_path):
    fake = Fake(RuntimeError("down"))
    chat = Chat(tmp_path)
    compactor = Compactor(chat, fake, "SYSTEM", lambda: WHEN)
    chat.log("unii", LONG, WHEN)
    compactor.idle(5)
    assert (0, 0) not in chat.log_.nodes
    chat.log("user", "hi", WHEN)
    assert chat.wait_summarized(5)
    assert chat.log_.nodes[(0, 0)] == "line"


def test_usage_is_logged_per_call(tmp_path):
    run(tmp_path, Fake("s"), LONG)
    [row] = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert row["kind"] == "compaction" and row["node"] == [0, 0] and row["usage"] == {"prompt_tokens": 10}


def test_an_id_head_the_model_wrote_is_dropped(tmp_path):
    chat = run(tmp_path, Fake("240+16|unii: summary"), LONG)
    assert chat.log_.nodes[(0, 0)] == "unii: summary"
