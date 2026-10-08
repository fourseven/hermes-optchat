from datetime import datetime, timezone

import pytest

from optchat.chat import Chat

WHEN = datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc)


def chat_with(tmp_path, texts):
    chat = Chat(tmp_path)
    for text in texts:
        chat.log("user", text, WHEN)
    return chat


def test_view_comes_as_whole_4_line_blocks_and_a_tail(tmp_path):
    chat = chat_with(tmp_path, ["a" * 300 for _ in range(6)])
    blocks, tail = chat.render_view()
    assert len(blocks) == 1
    assert blocks[0].startswith("<chat>\n0+1|user: aaa")
    assert blocks[0].count("\n") == 5
    assert tail == f"4+1|user: {'a' * 300}\n5+1|user: {'a' * 300}\n</chat>"


def test_newlines_in_a_line_become_spaces(tmp_path):
    chat = chat_with(tmp_path, ["one\ntwo"])
    assert chat.render_view() == ([], "<chat>\n0+1|user: one two\n</chat>")


def test_zoom_opens_a_line_down_to_the_whole_message(tmp_path):
    chat = chat_with(tmp_path, ["a" * 300, "b\nc", "c", "d"])
    assert chat.zoom(0, 4) == f"0+2|user: {'a' * 300} user: b c\n2+2|user: c user: d"
    assert chat.zoom(0, 2) == f"0+1|user: {'a' * 300}\n1+1|user: b c"
    assert chat.zoom(1, 1) == "user: b\nc"


def test_zoom_rejects_a_line_that_does_not_exist(tmp_path):
    chat = chat_with(tmp_path, ["a", "b", "c"])
    with pytest.raises(ValueError):
        chat.zoom(1, 2)
    with pytest.raises(ValueError):
        chat.zoom(0, 3)
    with pytest.raises(ValueError):
        chat.zoom(2, 2)


def test_date_gives_a_messages_time(tmp_path):
    chat = chat_with(tmp_path, ["a"])
    assert chat.date(0) == "2026-10-08T09:30:00+00:00"


def test_unbuilt_leaf_shows_a_placeholder_and_blocks_summarized(tmp_path):
    chat = chat_with(tmp_path, ["x" * 600])
    assert chat.render_view() == ([], "<chat>\n0+1|(not summarized yet: zoom it)\n</chat>")
    assert not chat.wait_summarized(timeout=0)
