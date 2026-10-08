import fcntl
import json
import os
from pathlib import Path

CLIP = 30_000
SPLIT = 30_000


class LogLocked(Exception):
    pass


def size(text):
    return len(text.encode())


def clip(text):
    if len(text) <= CLIP:
        return text
    cut = len(text) - CLIP
    return f"{text[: CLIP // 2]}\n[{cut} characters clipped]\n{text[-CLIP // 2 :]}"


class Log:
    """One chat's append-only store: main/*.jsonl, tree/*.jsonl and view.json."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "main").mkdir(parents=True, exist_ok=True)
        (self.root / "tree").mkdir(exist_ok=True)
        self._lock = open(self.root / "lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise LogLocked(self.root) from None
        self.messages = [json.loads(line) for line in self._lines("main")]
        self.nodes = {(n["l"], n["i"]): n["text"] for n in map(json.loads, self._lines("tree"))}

    def _lines(self, folder):
        for path in sorted((self.root / folder).glob("*.jsonl")):
            yield from path.read_text().splitlines()

    def _write(self, folder, when, record):
        with open(self.root / folder / f"{when.date().isoformat()}.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, kind, text, when):
        """Log a message, split over several if long; returns the new ids."""
        if kind == "echo":
            parts = [clip(text)]
        else:
            parts = [text[k : k + SPLIT] for k in range(0, len(text), SPLIT)] or [""]
        ids = []
        for part in parts:
            message = {"i": len(self.messages), "kind": kind, "text": part, "size": size(part), "date": when.isoformat()}
            self._write("main", when, message)
            self.messages.append(message)
            ids.append(message["i"])
        return ids

    def add_node(self, l, i, text, when):
        self._write("tree", when, {"l": l, "i": i, "text": text, "size": size(text)})
        self.nodes[(l, i)] = text

    def load_view(self):
        path = self.root / "view.json"
        if not path.exists():
            return [], False
        view = json.loads(path.read_text())
        return [tuple(line) for line in view["lines"]], view["draining"]

    def save_view(self, lines, draining):
        tmp = self.root / "view.json.tmp"
        tmp.write_text(json.dumps({"lines": lines, "draining": draining}))
        os.replace(tmp, self.root / "view.json")

    def close(self):
        self._lock.close()
