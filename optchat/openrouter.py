import os

from openai import OpenAI

MODEL = os.environ.get("OPTCHAT_COMPACT_MODEL", "qwen/qwen3.7-flash")


def caller(model=MODEL, extra_body=None):
    """call(messages) -> (text, usage) through OpenRouter, which passes cache marks to Anthropic."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ["OPENROUTER_KEY"])

    def call(messages, max_tokens=1024, tools=None):
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, **({"tools": tools} if tools else {}),
            extra_body={**(extra_body or {}), "usage": {"include": True}})
        usage = response.usage.model_dump() if response.usage else {}
        return response.choices[0].message.content or "", usage

    return call
