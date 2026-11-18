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

"""App guides: per-app facts appended to the Flash/Pro platform guidance."""

from types import SimpleNamespace

import pytest

from artemis.agents.app_guidance import GUIDES_DIR_ENV, MAX_GUIDE_CHARS, app_guidance, parse_guide
from artemis.agents.platform_guidance import device_action_guidance


@pytest.fixture
def guides(tmp_path, monkeypatch):
    monkeypatch.setenv(GUIDES_DIR_ENV, str(tmp_path))
    return tmp_path


def test_header_selects_platforms_and_apps_and_is_stripped_from_the_text():
    platforms, apps, body = parse_guide("<!-- platforms: TVOS, ios; apps: a.b, c.d -->\n\nTabs: Home, Film.")
    assert platforms == {"tvos", "ios"} and apps == {"a.b", "c.d"}
    assert body == "Tabs: Home, Film."


def test_a_guide_without_a_header_applies_everywhere():
    assert parse_guide("Just facts.") == (set(), set(), "Just facts.")


def test_only_guides_for_the_platform_apply(guides):
    (guides / "tv.md").write_text("<!-- platforms: tvos -->\nTV FACTS", encoding="utf-8")
    (guides / "phone.md").write_text("<!-- platforms: android -->\nPHONE FACTS", encoding="utf-8")

    text = app_guidance("tvos")

    assert "TV FACTS" in text and "PHONE FACTS" not in text
    assert text.startswith("App guide") and text.endswith("\n\n")


def test_a_locked_app_must_be_one_the_guide_names(guides):
    (guides / "a.md").write_text("<!-- platforms: tvos; apps: no.tv2.sumo.qa -->\nQA FACTS", encoding="utf-8")

    assert "QA FACTS" in app_guidance("tvos", "no.tv2.sumo.qa")
    assert app_guidance("tvos", "some.other.app") == ""
    assert "QA FACTS" in app_guidance("tvos", None)  # no lock: the platform match is enough


def test_guides_are_ordered_by_file_name_and_capped(guides):
    (guides / "b.md").write_text("SECOND", encoding="utf-8")
    (guides / "a.md").write_text("FIRST", encoding="utf-8")
    (guides / "c.md").write_text("x" * (MAX_GUIDE_CHARS + 500), encoding="utf-8")

    text = app_guidance("android")

    assert text.index("FIRST") < text.index("SECOND")
    assert "x" * MAX_GUIDE_CHARS in text and "x" * (MAX_GUIDE_CHARS + 1) not in text


def test_a_missing_folder_or_empty_guide_adds_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(GUIDES_DIR_ENV, str(tmp_path / "nope"))
    assert app_guidance("tvos") == ""
    monkeypatch.setenv(GUIDES_DIR_ENV, str(tmp_path))
    (tmp_path / "empty.md").write_text("<!-- platforms: tvos -->\n", encoding="utf-8")
    assert app_guidance("tvos") == ""


def test_the_guide_follows_the_platform_guidance_in_the_prompt(guides):
    (guides / "tv.md").write_text("<!-- platforms: tvos -->\nTV FACTS", encoding="utf-8")
    from artemis.agents import platform_guidance

    ctx = SimpleNamespace(device=SimpleNamespace(mobile_platform="ios", device_id="x"), agent_config=None)
    original = platform_guidance._is_tvos_target
    platform_guidance._is_tvos_target = lambda _ctx: True
    try:
        text = device_action_guidance(ctx)
    finally:
        platform_guidance._is_tvos_target = original

    assert text.startswith("Target platform: tvOS")
    assert text.index("Target platform: tvOS") < text.index("TV FACTS")


def test_no_guide_leaves_the_prompt_exactly_as_before(guides):
    ctx = SimpleNamespace(device=SimpleNamespace(mobile_platform="android"), agent_config=None)
    assert device_action_guidance(ctx) == ""
