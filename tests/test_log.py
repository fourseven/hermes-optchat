import json
from datetime import datetime, timezone

import pytest

from optchat.log import CLIP, SPLIT, Log, LogLocked

WHEN = datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc)


def test_append_is_on_disk_before_close(tmp_path):
    log = Log(tmp_path)
    assert log.append("user", "hi", WHEN) == [0]
    line = (tmp_path / "main" / "2026-10-08.jsonl").read_text()
    assert json.loads(line) == {"i": 0, "kind": "user", "text": "hi", "size": 2, "date": "2026-10-08T09:30:00+00:00"}


def test_reload_keeps_messages_nodes_and_view(tmp_path):
    log = Log(tmp_path)
    log.append("user", "hi", WHEN)
    log.append("unii", "héllo", WHEN)
    log.add_node(1, 0, "user: hi\nunii: héllo", WHEN)
    log.save_view([(1, 0)], draining=True)
    log.close()
    log = Log(tmp_path)
    assert [m["text"] for m in log.messages] == ["hi", "héllo"]
    assert log.messages[1]["size"] == 6
    assert log.nodes == {(1, 0): "user: hi\nunii: héllo"}
    assert log.load_view() == ([(1, 0)], True)


def test_second_writer_is_refused(tmp_path):
    first = Log(tmp_path)
    with pytest.raises(LogLocked):
        Log(tmp_path)
    first.close()


def test_echo_is_clipped_to_head_and_tail(tmp_path):
    log = Log(tmp_path)
    text = "a" * CLIP + "b" * CLIP
    [i] = log.append("echo", text, WHEN)
    clipped = log.messages[i]["text"]
    assert clipped.startswith("a" * (CLIP // 2)) and clipped.endswith("b" * (CLIP // 2))
    assert len(clipped) < CLIP + 100


def test_other_long_text_is_split_into_consecutive_messages(tmp_path):
    log = Log(tmp_path)
    text = "x" * (SPLIT * 2 + 1)
    assert log.append("user", text, WHEN) == [0, 1, 2]
    assert "".join(m["text"] for m in log.messages) == text
