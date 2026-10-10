import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from .log import size
from .prompt import leaf_task, merge_task, too_long
from .tree import LINE

logger = logging.getLogger(__name__)

# Haiku 5.5 began 4 of 20 spot-checked lines with an id head ("12|", "240+16|") despite the prompt,
# and about 1 in 7 wrapped them in a tag (<line>, <compaction-output>).
HEAD = re.compile(r"^\d+(?:[+-]\d+)?\||^<[\w-]+>\s*|\s*</[\w-]+>$")
LANES = 8
TRIES = 5
MARK = {"type": "ephemeral"}
# Hermes merges adjacent user messages; this keeps the view its own message, so its
# last whole block carries a cache mark. Turns and compactions both use it.
ACK = "(the chat continues below)"


class Compactor:
    """Builds the tree in the background, up to 8 calls at once.

    call(messages) -> (text, usage) sends OpenAI-format messages to the cheap model.
    Compactions send no tools: they could not share the turns' cache anyway, since
    Hermes puts its tools ahead of the system prompt.
    """

    def __init__(self, chat, call, system, now):
        self.chat, self.call, self.system, self.now = chat, call, system, now
        self.inflight = 0
        self.seen = -1
        self.writing = {}
        self.writing_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(LANES)
        threading.Thread(target=self._dispatch, daemon=True).start()

    def idle(self, timeout):
        """Wait until no call is running and nothing is ready to build."""
        with self.chat.changed:
            return self.chat.changed.wait_for(lambda: self.inflight == 0 and self.seen == self.chat.version, timeout)

    def _dispatch(self):
        changed = self.chat.changed
        with changed:
            while True:
                task = self.chat.tree.take() if self.inflight < LANES else None
                if task is None:
                    self.seen = self.chat.version
                    changed.notify_all()
                    changed.wait()
                    continue
                self.inflight += 1
                try:
                    self.pool.submit(self._build, task, self._request(*task))
                except RuntimeError:
                    # The interpreter is exiting (a one-shot CLI turn); unbuilt nodes requeue at the next start.
                    return

    def _request(self, l, i):
        blocks, tail = self.chat.compaction_view(l, i)
        if l == 0:
            message = self.chat.log_.messages[i]
            task = leaf_task(i, message["kind"], message["text"])
        else:
            half = 1 << (l - 1)
            a, b = (l - 1, 2 * i), (l - 1, 2 * i + 1)
            task = merge_task(f"{2 * i * half}+{half}", f"{(2 * i + 1) * half}+{half}", i << l, ((i + 1) << l) - 1,
                              self.chat.log_.nodes[a], self.chat.log_.nodes[b])
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system, "cache_control": MARK}]}]
        if blocks:
            parts = [{"type": "text", "text": block} for block in blocks]
            # Each compaction's view ends at its own node, so marks also sit at block counts
            # aligned to 8 and 32: those positions recur across compactions.
            n = len(parts)
            for k in {n - 1, n // 8 * 8 - 1, n // 32 * 32 - 1}:
                if k >= 0:
                    parts[k]["cache_control"] = MARK
            messages += [{"role": "user", "content": parts}, {"role": "assistant", "content": ACK}]
        messages.append({"role": "user", "content": f"{tail}\n\n{task}"})
        return messages, tuple(blocks)

    def _build(self, task, request):
        try:
            messages, prefix = request
            best = None
            for attempt in range(TRIES):
                text, usage = self._call_once(messages, prefix if attempt == 0 else None)
                self.chat.record_usage({"kind": "compaction", "node": list(task), "attempt": attempt, "usage": usage})
                text = HEAD.sub("", text.strip())
                if best is None or size(text) < size(best):
                    best = text
                if size(text) <= LINE:
                    break
                messages = [*messages, {"role": "assistant", "content": text},
                            {"role": "user", "content": too_long(size(text), text)}]
            self.chat.put(*task, best, self.now())
        except Exception:
            logger.warning("optchat: compaction %s failed; retrying at the next message", task, exc_info=True)
            with self.chat.changed:
                self.chat.tree.fail(task)
        finally:
            with self.chat.changed:
                self.inflight -= 1
                self.chat.changed.notify_all()

    def _call_once(self, messages, prefix):
        """A call whose marked prefix another call is writing waits for that call first."""
        if prefix is None:
            return self.call(messages)
        with self.writing_lock:
            writer = self.writing.get(prefix)
            if writer is None:
                self.writing[prefix] = done = threading.Event()
        if writer is not None:
            writer.wait(60)
            return self.call(messages)
        try:
            return self.call(messages)
        finally:
            with self.writing_lock:
                del self.writing[prefix]
            done.set()
