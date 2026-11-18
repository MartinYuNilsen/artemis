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

"""An empty model reply is retried before it can count as a stalled turn."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from artemis.agents.flash.runner import MAX_EMPTY_REPLY_RETRIES, FlashRunner

EMPTY = AIMessage(content="")
TOOL_CALL = AIMessage(content="", tool_calls=[{"name": "press_key", "args": {"key": "DPAD_RIGHT"}, "id": "t1"}])
TEXT_ONLY = AIMessage(content="I am thinking about it.")


@pytest.fixture
def runner():
    return FlashRunner.__new__(FlashRunner)  # the retry logic needs no run state


@pytest.mark.parametrize(
    ("reply", "empty"),
    [(EMPTY, True), (AIMessage(content="   \n"), True), (TOOL_CALL, False), (TEXT_ONLY, False)],
)
def test_only_a_reply_with_nothing_in_it_is_empty(runner, reply, empty):
    assert runner._is_empty_reply(reply) is empty


@pytest.mark.asyncio
async def test_a_usable_reply_is_never_retried(runner):
    runner._invoke_model = AsyncMock()
    assert await runner._recover_empty_reply(TOOL_CALL, "llm", [], []) is TOOL_CALL
    runner._invoke_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_empty_reply_is_retried_and_the_first_usable_reply_wins(runner):
    runner._invoke_model = AsyncMock(side_effect=[EMPTY, TOOL_CALL])
    history = [HumanMessage(content="observation")]

    result = await runner._recover_empty_reply(EMPTY, "llm", ["tools"], history)

    assert result is TOOL_CALL
    assert runner._invoke_model.await_count == 2
    first_retry, second_retry = (call.args[2] for call in runner._invoke_model.await_args_list)
    assert first_retry == history  # the first retry repeats the request unchanged
    assert len(second_retry) == len(history) + 1 and "empty" in second_retry[-1].content  # then a one-off nudge
    assert history == [HumanMessage(content="observation")]  # nothing extra is stored in the history


@pytest.mark.asyncio
async def test_persistent_emptiness_is_handed_back_for_the_normal_stall_guards(runner):
    runner._invoke_model = AsyncMock(return_value=EMPTY)

    result = await runner._recover_empty_reply(EMPTY, "llm", [], [])

    assert result is EMPTY
    assert runner._invoke_model.await_count == MAX_EMPTY_REPLY_RETRIES


@pytest.mark.asyncio
async def test_a_missing_reply_passes_through(runner):
    runner._invoke_model = AsyncMock()
    assert await runner._recover_empty_reply(None, "llm", [], []) is None


def test_the_main_loop_routes_every_model_reply_through_the_retry():
    import inspect

    source = inspect.getsource(FlashRunner.run)
    assert source.index("_invoke_model(") < source.index("_recover_empty_reply(") < source.index("messages.append(response)")
