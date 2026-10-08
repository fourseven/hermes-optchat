import json
import subprocess
import tempfile
import time

MODEL = "claude-haiku-5-5"


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


def caller(model=MODEL):
    """call(messages) -> (text, usage) through the Claude Code CLI on the user's subscription.

    The CLI takes one prompt, so the conversation after the system message is flattened;
    a "Too long" retry sees its earlier reply quoted. Caching is left to Claude Code.
    """
    workdir = tempfile.mkdtemp(prefix="optchat-claude-")

    def call(messages, max_tokens=1024, tools=None):
        system = _text(messages[0]["content"])
        prompt = "\n\n".join(
            _text(m["content"]) if m["role"] == "user" else f"(Your previous reply:)\n{_text(m['content'])}"
            for m in messages[1:])
        args = ["claude", "-p", "--model", model, "--system-prompt", system, "--tools", "",
                "--output-format", "json", "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config"]
        for attempt in range(3):
            try:
                out = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=300, cwd=workdir, check=True)
                break
            except FileNotFoundError:
                # Claude Code auto-updates replace the binary; a call during the swap finds nothing.
                if attempt == 2:
                    raise
                time.sleep(10)
        events = json.loads(out.stdout)
        result = [e for e in events if e.get("type") == "result"][-1] if isinstance(events, list) else events
        if result.get("is_error") or model not in (result.get("modelUsage") or {}):
            raise RuntimeError(f"claude -p failed: {result.get('result')!r} {list(result.get('modelUsage') or {})}")
        return result["result"], {**result.get("usage", {}), "duration_ms": result.get("duration_ms")}

    return call
