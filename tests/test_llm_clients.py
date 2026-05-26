"""Offline tests for the LLM client layer (no network / API keys required)."""

import pytest

from genai_experiments.llm_clients import MODELS, generate


def test_model_registry_has_expected_pairing():
    assert MODELS["gpt"] == ("openai", "gpt-5.5")
    assert MODELS["claude"] == ("anthropic", "claude-opus-4-7")


def test_generate_rejects_unknown_model_key():
    with pytest.raises(ValueError, match="unknown model key"):
        generate("gemini", [{"role": "user", "content": "hi"}])
