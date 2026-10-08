import threading

from .log import Log, size
from .tree import Tree
from .view import View

BLOCK = 4
UNBUILT = "(not summarized yet: zoom it)"


class Chat:
    """One chat that never ends: the log, its tree and its view, behind one lock."""

    def __init__(self, root):
        self.log_ = Log(root)
        self.tree = Tree(self.log_)
        lines, draining = self.log_.load_view()
        self.view = View(lines, draining)
        self.changed = threading.Condition()

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
                self.view.append(i, self._line_bytes, self._built)
            self.log_.save_view(self.view.lines, self.view.draining)
            self.changed.notify_all()

    def put(self, l, i, text, when):
        with self.changed:
            self.tree.put(l, i, text, when)
            self.changed.notify_all()

    def wait_summarized(self, timeout):
        with self.changed:
            return self.changed.wait_for(self.tree.all_leaves_built, timeout)

    def render_view(self):
        """The view as whole 4-line blocks, and the tail that closes it."""
        with self.changed:
            lines = [self.line(l, i) + "\n" for l, i in self.view.lines]
        lines[:0] = ["<chat>\n"]
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
