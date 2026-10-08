"""OptChat: one chat that never ends, as a Hermes ContextEngine."""


def register(ctx):
    from .engine import OptChatEngine

    ctx.register_context_engine(OptChatEngine())
