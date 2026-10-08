from optchat.view import View, merge_most_due


def push(new_state, states):
    # Taelin's rollback_state_list.js push, life always 0.
    if states is None:
        return [0, new_state, None]
    keep, state, older = states
    if keep == 0:
        return [1, state, older]
    return [0, new_state, push(state, older)]


def push_view(states, total):
    starts = []
    while states is not None:
        starts.append(states[1])
        states = states[2]
    starts.reverse()
    ends = starts[1:] + [total]
    return [((end - start).bit_length() - 1, start // (end - start)) for start, end in zip(starts, ends)]


def test_merge_order_matches_push():
    states, lines = None, []
    for t in range(20_001):
        states = push(t, states)
        lines.append((0, t))
        expected = push_view(states, t + 1)
        while len(lines) > len(expected):
            assert merge_most_due(lines, t + 1, lambda l, i: True)
        assert lines == expected, t


def test_due_measures_from_the_pairs_last_message():
    lines = [(2, 0), (2, 1), (0, 8), (0, 9)]
    merge_most_due(lines, 10, lambda l, i: True)
    assert lines == [(2, 0), (2, 1), (1, 4)]


def test_merges_only_pairs_whose_parent_is_built():
    lines = [(2, 0), (2, 1), (0, 8), (0, 9)]
    merge_most_due(lines, 10, lambda l, i: l == 3)
    assert lines == [(3, 0), (0, 8), (0, 9)]
    assert not merge_most_due(lines, 10, lambda l, i: l == 3)


def test_view_grows_one_line_per_message_until_over_high():
    view = View(high=10, low=5)
    for i in range(10):
        view.append(i, line_bytes=lambda l, i: 1, built=lambda l, i: True)
    assert view.lines == [(0, i) for i in range(10)]


def test_view_batch_merges_down_to_low_once_over_high():
    view = View(high=10, low=5)
    for i in range(11):
        view.append(i, line_bytes=lambda l, i: 1, built=lambda l, i: True)
    assert len(view.lines) == 5
    view.append(11, line_bytes=lambda l, i: 1, built=lambda l, i: True)
    assert len(view.lines) == 6


def test_view_keeps_draining_at_each_message_until_low():
    built = {(1, 0)}
    view = View(high=4, low=2)
    for i in range(5):
        view.append(i, line_bytes=lambda l, i: 1, built=lambda l, i: (l, i) in built)
    assert view.lines == [(1, 0), (0, 2), (0, 3), (0, 4)]
    built.add((1, 1))
    built.add((2, 0))
    view.append(5, line_bytes=lambda l, i: 1, built=lambda l, i: (l, i) in built)
    assert view.lines == [(2, 0), (0, 4), (0, 5)]
    assert view.draining
    built.add((1, 2))
    view.append(6, line_bytes=lambda l, i: 1, built=lambda l, i: (l, i) in built)
    assert view.lines == [(2, 0), (1, 2), (0, 6)]
    assert view.draining
