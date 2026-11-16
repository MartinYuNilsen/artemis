# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,

"""The Flash step summarizer follows the configured provider, not always Google."""

from types import SimpleNamespace

import pytest

from artemis.agents.flash.summarizer import _summarizer_rides_google


def _ctx(provider):
    summarizer = None if provider is None else SimpleNamespace(provider=provider)
    return SimpleNamespace(llm_config=SimpleNamespace(summarizer=summarizer))


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("google", True),
        ("vertexai", False),  # the raw Google client cannot serve Vertex AI
        (None, True),  # no summarizer role configured: keep the Google behaviour
        ("anthropic", False),
        ("openai", False),
    ],
)
def test_explicit_summarizer_model_uses_google_only_when_configured_for_it(provider, expected):
    assert _summarizer_rides_google(_ctx(provider)) is expected


def _build(provider, monkeypatch, *, model_name="explicit-model"):
    """Construct the summarizer with both model factories mocked; return (summarizer, factories)."""
    from unittest.mock import Mock

    import artemis.agents.flash.summarizer as module

    raw_model = SimpleNamespace(model="configured-model")  # what the resumable wrapper wraps
    configured = Mock(return_value=SimpleNamespace(base_model=raw_model))
    google = Mock(return_value=SimpleNamespace(model="explicit-model"))
    monkeypatch.setattr(module, "get_llm", configured)
    monkeypatch.setattr(module, "get_google_llm", google)
    ctx = _ctx(provider)
    ctx.data_engine = None
    return module.VisualStepSummarizer(ctx, model_name=model_name), configured, google, raw_model, ctx


@pytest.mark.parametrize("provider", ["anthropic", "openai", "vertexai"])
def test_explicit_model_with_a_non_google_summarizer_uses_the_configured_endpoint(
    provider, monkeypatch
):
    summarizer, configured, google, raw_model, ctx = _build(provider, monkeypatch)

    configured.assert_called_once_with(ctx, name="summarizer", temperature=0.0)
    google.assert_not_called()  # a keyless Google client could never succeed here
    # The raw model, never the resumable wrapper: its retry-exhaustion path pauses the whole task.
    assert summarizer._llm is raw_model


@pytest.mark.parametrize("provider", ["google", None])
def test_explicit_model_with_a_google_or_unset_summarizer_keeps_the_google_client(
    provider, monkeypatch
):
    summarizer, configured, google, _raw, _ctx_unused = _build(provider, monkeypatch)

    google.assert_called_once_with(model_name="explicit-model", temperature=0.0)
    configured.assert_not_called()
    assert summarizer._llm is google.return_value
