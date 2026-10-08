"""Totals for a replay: python -m optchat.report OUT"""

import json
import sys
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def cost(usage):
    # BYOK calls report cost 0 and the provider's charge as upstream_inference_cost.
    return usage.get("cost") or (usage.get("cost_details") or {}).get("upstream_inference_cost") or 0


def tokens(usage):
    """(prompt, cached) for OpenRouter usage or Anthropic usage from the Claude Code CLI."""
    if "prompt_tokens" in usage:
        return usage["prompt_tokens"], (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cached = usage.get("cache_read_input_tokens", 0)
    return usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + cached, cached


def share(calls):
    pairs = [tokens(c["usage"]) for c in calls]
    prompt = sum(p for p, _ in pairs)
    cached = sum(c for _, c in pairs)
    return cached / prompt if prompt else 0, prompt


def main():
    out = Path(sys.argv[1])
    turns = rows(out / "calls.jsonl")
    compactions = rows(out / "chat" / "usage.jsonl")
    messages = sum(1 for _ in (out / "chat" / "main").glob("*.jsonl") for _ in _.open())
    turn_share, turn_prompt = share(turns)
    comp_share, comp_prompt = share(compactions)
    firsts = list({t["turn"]: t for t in reversed(turns)}.values())
    spent = sum(cost(c["usage"]) for c in turns + compactions)
    print(f"messages logged            {messages}")
    print(f"turn calls                 {len(turns)}  prefix read from cache {turn_share:.1%}  ({turn_prompt:,} prompt tokens)")
    print(f"compaction calls           {len(compactions)}  prefix read from cache {comp_share:.1%}  ({comp_prompt:,} prompt tokens)")
    print(f"view lines rewritten/msg   {sum(t['rewritten'] for t in firsts) / max(messages, 1):.1f}")
    print(f"wait before a turn         max {max((t['wait'] for t in turns), default=0):.1f}s  mean {sum(t['wait'] for t in firsts) / max(len(firsts), 1):.1f}s")
    print(f"largest turn prompt        {max((t['usage'].get('prompt_tokens', 0) for t in turns), default=0):,} tokens")
    print(f"cost                       ${spent:.4f} total, ${spent / max(messages, 1):.6f} per message")


if __name__ == "__main__":
    main()
