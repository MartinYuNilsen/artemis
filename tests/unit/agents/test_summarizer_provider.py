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
