# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic factory: Claude 5+ models reject an explicit ``temperature``."""

import pytest

from artemis.llm.router import ModelEndpoint, ModelFactory, ModelProvider, _anthropic_rejects_temperature


@pytest.mark.parametrize(
    ("model", "rejects"),
    [
        ("claude-sonnet-5-5", True),
        ("claude-haiku-5-5", True),
        ("claude-opus-5-5", True),
        ("claude-fable-5-2", True),
        ("claude-sonnet-4-5", False),
        ("claude-3-5-sonnet-20241022", False),
    ],
)
def test_temperature_support_by_model_family(model, rejects):
    assert _anthropic_rejects_temperature(model) is rejects


@pytest.mark.parametrize(("model", "sent"), [("claude-sonnet-5-5", False), ("claude-sonnet-4-5", True)])
def test_factory_omits_temperature_only_for_claude_5(model, sent, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    chat = ModelFactory.create_model(
        ModelEndpoint(provider=ModelProvider.ANTHROPIC, model_name=model, temperature=0.0)
    )
    base = getattr(chat, "base_model", chat)
    assert ("temperature" in base._get_request_payload("hi")) is sent


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        (ModelProvider.ANTHROPIC, "claude-sonnet-5-5", {"method": "json_schema"}),
        (ModelProvider.ANTHROPIC, "claude-sonnet-4-5", {}),
        (ModelProvider.OPENAI, "gpt-4o", {}),
    ],
)
def test_structured_output_avoids_forced_tool_choice_on_claude_5(provider, model, expected):
    from artemis.services.llm import RobustChatModelWrapper

    base = _RecordingModel()
    endpoint = ModelEndpoint(provider=provider, model_name=model)
    RobustChatModelWrapper(base, None, endpoint=endpoint).with_structured_output(dict)
    assert base.calls == [expected]


def test_structured_output_keeps_an_explicit_method():
    from artemis.services.llm import RobustChatModelWrapper

    base = _RecordingModel()
    endpoint = ModelEndpoint(provider=ModelProvider.ANTHROPIC, model_name="claude-sonnet-5-5")
    RobustChatModelWrapper(base, None, endpoint=endpoint).with_structured_output(
        dict, method="function_calling"
    )
    assert base.calls == [{"method": "function_calling"}]
