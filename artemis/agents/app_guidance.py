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

"""App-specific knowledge for the Flash and Pro prompts.

A general agent re-discovers the same facts about an app on every run: which tab does what, where
focus lands, which dialogs appear, which buttons must never be pressed. Runs that discover them
differently behave differently. An *app guide* writes those facts down once.

Guides are markdown files in ``config/app_guides/`` (override the folder with
``ARTEMIS_APP_GUIDES_DIR``). An optional first line selects where a guide applies::

    <!-- platforms: tvos; apps: com.example.app, com.example.app.dev -->

``platforms`` is any of ``android``, ``ios``, ``tvos``; ``apps`` (optional) restricts the guide to
those package or bundle ids when the run is locked to an app. Without a header a guide applies
everywhere. Matching guides are appended to the platform guidance, in file-name order.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from artemis.config.paths import CONFIG_DIR

GUIDES_DIR_ENV = "ARTEMIS_APP_GUIDES_DIR"
MAX_GUIDE_CHARS = 8000
_HEADER = re.compile(r"^\s*<!--(?P<body>.*?)-->\s*", re.DOTALL)


def guides_dir() -> Path:
    override = os.environ.get(GUIDES_DIR_ENV)
    return Path(override) if override else CONFIG_DIR / "app_guides"


def _split_list(value: str) -> set[str]:
    return {item.strip().casefold() for item in value.split(",") if item.strip()}


def parse_guide(text: str) -> tuple[set[str], set[str], str]:
    """``(platforms, apps, body)``; empty sets mean "no restriction"."""
    platforms: set[str] = set()
    apps: set[str] = set()
    match = _HEADER.match(text)
    if match:
        for part in match.group("body").split(";"):
            key, _, value = part.partition(":")
            if key.strip().casefold() == "platforms":
                platforms = _split_list(value)
            elif key.strip().casefold() == "apps":
                apps = _split_list(value)
        text = text[match.end() :]
    return platforms, apps, text.strip()


def app_guidance(platform: str, locked_app: str | None = None) -> str:
    """The guides that apply to this platform (and locked app, when there is one)."""
    folder = guides_dir()
    if not folder.is_dir():
        return ""
    chosen: list[str] = []
    for path in sorted(folder.glob("*.md")):
        try:
            platforms, apps, body = parse_guide(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not body:
            continue
        if platforms and platform.casefold() not in platforms:
            continue
        if apps and locked_app and locked_app.casefold() not in apps:
            continue
        chosen.append(body[:MAX_GUIDE_CHARS])
    if not chosen:
        return ""
    return "App guide (facts about this app, verified on real runs; trust them over guessing):\n" + "\n\n".join(chosen) + "\n\n"
