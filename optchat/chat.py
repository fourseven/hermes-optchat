import json
import threading

from .log import Log, size
from .tree import Tree
from .view import View

BLOCK = 4
COMPACTION_HIGH = 32_000
COMPACTION_LOW = 16_000
UNBUILT = "(not summarized yet: zoom it)"


class Chat:
    """One chat that never ends: the log, its tree and its view, behind one lock."""

    def __init__(self, root):
        self.log_ = Log(root)
        self.tree = Tree(self.log_)
        lines, draining, self.last_blocks = self.log_.load_view()
        self.view = View(lines, draining)
        self.changed = threading.Condition()
        self.version = 0
        self._reset_compaction_view()

    def _reset_compaction_view(self):
        """Compactions see the chat's view merged further, to a 16-32 KB sawtooth."""
        self.cview = View(self.view.lines, True, COMPACTION_HIGH, COMPACTION_LOW)
        self.cview.drain(len(self.log_.messages), self._line_bytes, self._built)

    def line(self, l, i):
        text = self.log_.nodes.get((l, i), UNBUILT).replace("\n", " ")
        return f"{i << l}+{1 << l}|{text}"

    def _line_bytes(self, l, i):
        return size(self.line(l, i)) + 1

    def _built(self, l, i):
        return (l, i) in self.log_.nodes

    def log(self, kind, text, when):
        with self.changed:
            for i in self.log_.append(kind, text, when):
                self.tree.add_message(i, when)
                before = self.view.lines + [(0, i)]
                self.view.append(i, self._line_bytes, self._built)
                if self.view.lines == before:
                    self.cview.append(i, self._line_bytes, self._built)
                else:
                    self._reset_compaction_view()
            self.log_.save_view(self.view.lines, self.view.draining, self.last_blocks)
            self.version += 1
            self.changed.notify_all()

    def turn_started(self, blocks):
        """Record how many whole view blocks this turn sent; returns the previous turn's count."""
        with self.changed:
            previous, self.last_blocks = self.last_blocks, blocks
            self.log_.save_view(self.view.lines, self.view.draining, self.last_blocks)
            return previous

    def put(self, l, i, text, when):
        with self.changed:
            self.tree.put(l, i, text, when)
            self.version += 1
            self.changed.notify_all()

    def record_usage(self, row):
        with open(self.log_.root / "usage.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def wait_summarized(self, timeout):
        with self.changed:
            return self.changed.wait_for(self.tree.all_leaves_built, timeout)

    def render_view(self):
        with self.changed:
            return self._render(self.view.lines)

    def compaction_view(self, l, i):
        """The compaction view up to node(l, i)'s last message, stopping at the first unbuilt line."""
        end = (i + 1) << l if l else i
        lines = []
        for line in self.cview.lines:
            if ((line[1] + 1) << line[0]) > end or not self._built(*line):
                break
            lines.append(line)
        return self._render(lines)

    def _render(self, view_lines):
        """A view as whole 4-line blocks, and the tail that closes it."""
        lines = ["<chat>\n"] + [self.line(l, i) + "\n" for l, i in view_lines]
        whole = (len(lines) - 1) // BLOCK * BLOCK
        blocks = ["".join(lines[k : k + BLOCK]) for k in range(1, whole + 1, BLOCK)]
        if blocks:
            blocks[0] = lines[0] + blocks[0]
            tail = lines[whole + 1 :]
        else:
            tail = lines
        return blocks, "".join(tail) + "</chat>"

    def zoom(self, id, n):
        if n < 1 or n & (n - 1) or id % n or id < 0 or id + n > len(self.log_.messages):
            raise ValueError(f"no line {id}+{n}")
        with self.changed:
            if n == 1:
                message = self.log_.messages[id]
                return f"{message['kind']}: {message['text']}"
            l, i = n.bit_length() - 1, id // n
            return f"{self.line(l - 1, 2 * i)}\n{self.line(l - 1, 2 * i + 1)}"

    def date(self, id):
        if not 0 <= id < len(self.log_.messages):
            raise ValueError(f"no message {id}")
        return self.log_.messages[id]["date"]
