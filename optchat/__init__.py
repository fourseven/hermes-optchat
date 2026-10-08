"""OptChat: one chat that never ends, as a Hermes ContextEngine."""


def register(ctx):
    from .engine import OptChatEngine, install_cache_mark_patch

    install_cache_mark_patch()
    ctx.register_context_engine(OptChatEngine())
