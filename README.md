# OptChat for Hermes

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) context-engine plugin that implements Victor Taelin's [OptChat / UniiChat design](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449): one chat that never ends, a binary tree of 512-byte summaries, a constant-size view, and a `zoom` tool. The design is his; this is a prototype port, built against the gist's 08/10/2026 revision.

**Status: prototype.** Tested against Hermes `08165d5` on two months of real chat history.

## How it works

- Every message is logged word for word in `$HERMES_HOME/optchat/chats/<chat>/` (`main/*.jsonl`), append-only, one writer per chat.
- A background compactor builds the summary tree (`tree/*.jsonl`) with a cheap model, up to 8 calls at once. Short messages and short pairs are their own nodes, with no model call.
- Each request Hermes sends becomes `[system] [view] [this turn's messages]`. The view (`view.json`) covers the whole chat in 64-128 KB, recent lines fine and old lines coarse, merged in the gist's `due` order (checked against Taelin's `push` for t = 0..20,000).
- The model gets `zoom(id, n)` to open a line, down to a whole message, and `date(id)`.
- Hermes's own compaction is off for the chat. `/new` and restarts keep the same chat.

## Install

```sh
mkdir -p "$HERMES_HOME/plugins"
ln -s /path/to/hermes-optchat/optchat "$HERMES_HOME/plugins/optchat"
hermes config set context.engine optchat
```

- Compactions go through OpenRouter with `OPENROUTER_API_KEY` from the Hermes process environment. The model defaults to `anthropic/claude-haiku-5.5`; set `OPTCHAT_COMPACT_MODEL` to change it.
- Your own instructions (who you are, how you like work done) go in `$HERMES_HOME/optchat/instructions.md`. They follow the system prompt and should stay stable: any change re-writes the cache.
- Turn off memory-provider plugins; the chat is the memory.

**Monkey patch.** Hermes strips every `cache_control` and re-applies its own on each request. On routes where it adds none (for example `openai/*` on OpenRouter), the view would never cache across turns. The plugin wraps `agent.conversation_loop._redecorate_prompt_cache_for_provider` to put the view's marks back. It depends on a private Hermes function, so re-check it on every Hermes upgrade.

**Privacy.** The log keeps messages and tool output unredacted, including any secrets that appeared in tool calls, and summaries can copy them. Protect `$HERMES_HOME/optchat`.

## Tests

```sh
HERMES_SRC=/path/to/hermes-agent uv run pytest -q
```

Engine tests need a Hermes checkout on `HERMES_SRC`; the rest are standard library only. No test calls a model.

## Replay

Replays a Hermes `state.db` through the plugin and measures the cache. It makes real model calls through OpenRouter.

```sh
PYTHONPATH=. python -m optchat.replay state.db out/ --sources telegram --limit 1000
python -m optchat.report out/
```
