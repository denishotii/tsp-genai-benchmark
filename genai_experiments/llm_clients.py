"""Thin wrappers over the OpenAI and Anthropic APIs for code generation.

Each generation uses the provider's **default** configuration, with no custom
sampling or thinking settings, so the only experimental variable is the
prompt itself (the prompting strategy). API keys are loaded from a local
``.env`` file; see ``.env.example``.

Note on reproducibility: LLM generation is inherently non-deterministic, so
the harness logs every full response verbatim. Whether the *generated code*
itself runs deterministically depends on what the model chose to do. In this
study, GPT-5.5 seeded its RNG in most (but not all) SA and GA solvers,
while Claude Opus 4.7 never seeded its RNG, so re-evaluating a Claude solver
on the same instance can give a slightly different tour each time.
"""

import anthropic
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Friendly model key -> (provider, API model id). This is the current
# flagship-vs-flagship pairing; see CLAUDE.md §5.1.
MODELS = {
    "gpt": ("openai", "gpt-5.5"),
    "claude": ("anthropic", "claude-opus-4-7"),
}

# Anthropic requires an explicit output cap; generous headroom for a single
# algorithm implementation plus any chain-of-thought reasoning.
_ANTHROPIC_MAX_TOKENS = 16000


@dataclass
class LLMResponse:
    """A single model reply plus token accounting for the experiment log."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


def generate(model_key: str, messages: list[dict]) -> LLMResponse:
    """Send a chat-style message list to ``model_key`` and return its reply.

    Args:
        model_key: one of ``MODELS`` ("gpt" or "claude").
        messages: list of ``{"role": "user" | "assistant", "content": str}``,
            beginning with a user turn (the iterative strategy appends more).

    Returns:
        An :class:`LLMResponse` with the reply text and token counts.
    """
    if model_key not in MODELS:
        raise ValueError(
            f"unknown model key {model_key!r}; expected one of {list(MODELS)}"
        )
    provider, model_id = MODELS[model_key]
    if provider == "openai":
        return _generate_openai(model_id, messages)
    return _generate_anthropic(model_id, messages)


def _generate_openai(model_id: str, messages: list[dict]) -> LLMResponse:
    client = OpenAI()  # reads OPENAI_API_KEY
    response = client.chat.completions.create(model=model_id, messages=messages)
    usage = response.usage
    return LLMResponse(
        text=response.choices[0].message.content or "",
        model=model_id,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


def _generate_anthropic(model_id: str, messages: list[dict]) -> LLMResponse:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    response = client.messages.create(
        model=model_id,
        max_tokens=_ANTHROPIC_MAX_TOKENS,
        messages=messages,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResponse(
        text=text,
        model=model_id,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
