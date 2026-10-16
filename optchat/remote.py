import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

LEASE_EVERY = 20
TRIES = 3
WAIT_STEP = 60


class LeaseLost(Exception):
    pass


class RemoteChat:
    """The chat in the optchat-memory service: the methods the engine calls on a local Chat.

    The service compacts; this process only logs, renders and reads. It holds the chat's write
    lease while open, so a second harness can't answer the same chat.
    """

    def __init__(self, url, token, chat, holder, instructions):
        self.base = f"{url.rstrip('/')}/v1/chats/home/{urllib.parse.quote(chat, safe='')}/"
        self.token, self.holder = token, holder
        self._lease()
        self._post("instructions", {"text": instructions}, lease=True)
        self._stop = threading.Event()
        threading.Thread(target=self._keep_lease, daemon=True).start()

    def _request(self, method, path, body=None, lease=False, timeout=30):
        # Cloudflare answers urllib's default user agent with 403 (error 1010).
        headers = {"authorization": f"Bearer {self.token}", "content-type": "application/json",
                   "user-agent": "optchat-remote/1"}
        if lease:
            headers["x-lease"] = self.holder
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        for attempt in range(TRIES):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                error = json.loads(e.read() or b"{}").get("error", str(e))
                if e.code == 400:
                    raise ValueError(error) from None
                if e.code == 409:
                    raise LeaseLost(error) from None
                if e.code < 500 or attempt == TRIES - 1:
                    raise
            except urllib.error.URLError:
                if attempt == TRIES - 1:
                    raise
            time.sleep(attempt + 1)

    def _post(self, path, body, lease=False):
        try:
            return self._request("POST", path, body, lease)
        except LeaseLost:
            if not lease:
                raise
            # The lease lapsed (a missed renewal); take it back if nobody else has, and go again.
            self._lease()
            return self._request("POST", path, body, lease)

    def _get(self, path, timeout=30):
        return self._request("GET", path, timeout=timeout)

    def _lease(self):
        self._post("lease", {"holder": self.holder})

    def _keep_lease(self):
        while not self._stop.wait(LEASE_EVERY):
            try:
                self._lease()
            except Exception:
                pass

    def log(self, kind, text, when):
        # One request id for every try, so a retry after a lost response logs once.
        self._post("messages", {"kind": kind, "text": text, "date": when.isoformat(), "request_id": uuid.uuid4().hex}, lease=True)

    def turn_started(self, blocks):
        return self._post("turn", {"blocks": blocks}, lease=True)["previous"]

    def render_view(self):
        blocks, tail = self._get("view")
        return blocks, tail

    def wait_summarized(self, timeout):
        left = timeout
        while True:
            step = min(left, WAIT_STEP)
            if self._get(f"summarized?wait={int(step * 1000)}", timeout=step + 30)["summarized"]:
                return True
            left -= step
            if left <= 0:
                return False

    def zoom(self, id, n):
        return self._get(f"zoom?id={int(id)}&n={int(n)}")["result"]

    def date(self, id):
        return self._get(f"date?id={int(id)}")["result"]

    def record_usage(self, row):
        self._post("usage", {"row": row})

    def close(self):
        self._stop.set()
