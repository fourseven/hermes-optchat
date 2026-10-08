import os

from openai import OpenAI

MODEL = os.environ.get("OPTCHAT_COMPACT_MODEL", "anthropic/claude-haiku-5.5")


def caller(model=MODEL):
    """call(messages) -> (text, usage) through OpenRouter, which passes cache marks to Anthropic."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ["OPENROUTER_KEY"])

    def call(messages, max_tokens=1024):
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                  extra_body={"usage": {"include": True}})
        usage = response.usage.model_dump() if response.usage else {}
        return response.choices[0].message.content or "", usage

    return call
