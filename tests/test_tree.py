from datetime import datetime, timezone

from optchat.log import Log
from optchat.tree import Tree

WHEN = datetime(2026, 10, 8, tzinfo=timezone.utc)
LONG = "x" * 600


def make(tmp_path):
    log = Log(tmp_path)
    return log, Tree(log)


def add(log, tree, kind, text):
    for i in log.append(kind, text, WHEN):
        tree.add_message(i, WHEN)


def test_short_message_is_its_own_node(tmp_path):
    log, tree = make(tmp_path)
    add(log, tree, "user", "hi")
    assert log.nodes[(0, 0)] == "user: hi"
    assert tree.take() is None


def test_two_short_lines_join_with_a_newline_up_the_tree(tmp_path):
    log, tree = make(tmp_path)
    for text in "abcd":
        add(log, tree, "user", text)
    assert log.nodes[(1, 0)] == "user: a\nuser: b"
    assert log.nodes[(2, 0)] == "user: a\nuser: b\nuser: c\nuser: d"


def test_long_message_waits_for_a_compaction(tmp_path):
    log, tree = make(tmp_path)
    add(log, tree, "unii", LONG)
    assert tree.take() == (0, 0)
    assert tree.take() is None
    tree.put(0, 0, "unii: summary", WHEN)
    assert log.nodes[(0, 0)] == "unii: summary"


def test_a_merge_too_long_to_join_is_queued_once_both_halves_are_built(tmp_path):
    log, tree = make(tmp_path)
    add(log, tree, "unii", LONG)
    add(log, tree, "unii", LONG)
    assert [tree.take(), tree.take()] == [(0, 0), (0, 1)]
    tree.put(0, 1, "y" * 300, WHEN)
    assert tree.take() is None
    tree.put(0, 0, "z" * 300, WHEN)
    assert tree.take() == (1, 0)


def test_a_leaf_starts_once_fewer_than_8_lines_before_it_are_unbuilt(tmp_path):
    log, tree = make(tmp_path)
    for _ in range(9):
        add(log, tree, "unii", LONG)
    assert [tree.take() for _ in range(9)] == [(0, i) for i in range(8)] + [None]
    tree.put(0, 3, "s", WHEN)
    assert tree.take() == (0, 8)


def test_a_failed_call_is_tried_again_at_the_next_message(tmp_path):
    log, tree = make(tmp_path)
    add(log, tree, "unii", LONG)
    task = tree.take()
    tree.fail(task)
    assert tree.take() is None
    add(log, tree, "user", "hi")
    assert tree.take() == task


def test_restart_requeues_unbuilt_work(tmp_path):
    log, tree = make(tmp_path)
    add(log, tree, "unii", LONG)
    add(log, tree, "unii", LONG)
    tree.take()
    tree.put(0, 0, "z" * 300, WHEN)
    log.close()
    log = Log(tmp_path)
    tree = Tree(log)
    assert tree.take() == (0, 1)
    assert not tree.all_leaves_built()
