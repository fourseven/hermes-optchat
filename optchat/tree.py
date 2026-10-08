from collections import deque

from .log import size

LINE = 512
LEAD = 8


class Tree:
    """Builds node(l, i) once each; ready work is queued, never found by scanning.

    node(0, i) is message i in at most 512 bytes; node(l, i) merges node(l-1, 2i)
    and node(l-1, 2i+1). A source that already fits is its own node.
    """

    def __init__(self, log):
        self.log = log
        self.next_leaf = 0
        self.started = set()
        self.failed = []
        self.retry = deque()
        self.merges = deque(
            (l + 1, i // 2)
            for l, i in sorted(log.nodes)
            if i % 2 == 0 and (l, i + 1) in log.nodes and (l + 1, i // 2) not in log.nodes
        )
        self.leaves_built = sum(1 for l, _ in log.nodes if l == 0)

    def all_leaves_built(self):
        return self.leaves_built == len(self.log.messages)

    def add_message(self, i, when):
        self.retry.extend(self.failed)
        self.failed.clear()
        message = self.log.messages[i]
        line = f"{message['kind']}: {message['text']}"
        if size(line) <= LINE:
            self.put(0, i, line, when)

    def take(self):
        if self.retry:
            return self.retry.popleft()
        messages = len(self.log.messages)
        while self.next_leaf < messages and (0, self.next_leaf) in self.log.nodes:
            self.next_leaf += 1
        if self.next_leaf < messages and len(self.started) < LEAD:
            self.started.add(self.next_leaf)
            self.next_leaf += 1
            return (0, self.next_leaf - 1)
        if self.merges:
            return self.merges.popleft()
        return None

    def fail(self, task):
        self.failed.append(task)

    def put(self, l, i, text, when):
        self.log.add_node(l, i, text, when)
        if l == 0:
            self.started.discard(i)
            self.leaves_built += 1
        left, right = self.log.nodes.get((l, i & ~1)), self.log.nodes.get((l, i | 1))
        if left is None or right is None:
            return
        joined = f"{left}\n{right}"
        if size(joined) <= LINE:
            self.put(l + 1, i // 2, joined, when)
        else:
            self.merges.append((l + 1, i // 2))
