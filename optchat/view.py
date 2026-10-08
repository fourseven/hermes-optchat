from fractions import Fraction

HIGH = 128_000
LOW = 64_000


def merge_most_due(lines, total, built):
    """Merge the most due sibling pair whose parent is built, oldest on ties.

    due = (total - last) / 2^l, with last the pair's last message.
    """
    best = None
    for k in range(len(lines) - 1):
        l, i = lines[k]
        if i % 2 or lines[k + 1] != (l, i + 1) or not built(l + 1, i // 2):
            continue
        due = Fraction(total - ((i + 2) << l) + 1, 1 << l)
        if best is None or due > best[0]:
            best = (due, k)
    if best is None:
        return False
    k = best[1]
    l, i = lines[k]
    lines[k : k + 2] = [(l + 1, i // 2)]
    return True


class View:
    """Nodes covering the whole chat, oldest first; a 64-128 KB sawtooth."""

    def __init__(self, lines=(), draining=False, high=HIGH, low=LOW):
        self.lines = [tuple(line) for line in lines]
        self.draining = draining
        self.high = high
        self.low = low

    def size(self, line_bytes):
        return sum(line_bytes(l, i) for l, i in self.lines)

    def append(self, i, line_bytes, built):
        self.lines.append((0, i))
        if self.size(line_bytes) > self.high:
            self.draining = True
        self.drain(i + 1, line_bytes, built)

    def drain(self, total, line_bytes, built):
        while self.draining and self.size(line_bytes) > self.low:
            if not merge_most_due(self.lines, total, built):
                return
        self.draining = False
