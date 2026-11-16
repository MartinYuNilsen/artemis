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

"""tvOS simulator support: discovery, focus observation, and remote input."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from artemis.agents.platform_guidance import device_action_guidance
from artemis.drivers.ios.discovery import parse_simctl_devices
from artemis.drivers.ios.hierarchy import parse_hierarchy
from artemis.drivers.ios.xcode_driver import XcodeSimulatorDriver

HIERARCHY = """Device orientation: Unknown
------------------------
Application bundle identifier: com.example.tv
Application, pid: 1, label: 'Example'
 Window, {{0.0, 0.0}, {1920.0, 1080.0}}, hitPoint: {960.0, 540.0}
  Button, {{100.0, 50.0}, {200.0, 80.0}}, label: 'Home, Focused', hitPoint: {200.0, 90.0}
  Button, {{400.0, 50.0}, {200.0, 80.0}}, label: 'Search', Focused, hitPoint: {500.0, 90.0}
  TextField, {{700.0, 50.0}, {200.0, 80.0}}, label: 'Query', Keyboard Focused, hitPoint: {800.0, 90.0}
"""


def test_simctl_parsing_includes_tvos_with_os_family():
    payload = {
        "devices": {
            "com.apple.CoreSimulator.SimRuntime.iOS-27-0": [
                {"udid": "A", "name": "iPhone", "state": "Booted", "isAvailable": True}
            ],
            "com.apple.CoreSimulator.SimRuntime.tvOS-27-0": [
                {"udid": "B", "name": "Apple TV 4K", "state": "Shutdown", "isAvailable": True}
            ],
            "com.apple.CoreSimulator.SimRuntime.watchOS-27-0": [
                {"udid": "C", "name": "Watch", "state": "Shutdown", "isAvailable": True}
            ],
        }
    }
    families = {d["udid"]: d["os_family"] for d in parse_simctl_devices(payload)}
    assert families == {"A": "ios", "B": "tvos"}


def test_hierarchy_marks_only_the_real_focused_element():
    elements, _ = parse_hierarchy(HIERARCHY, 1920, 1080)
    focused = [e["text"] for e in elements if e.get("focused") == "true"]
    # A label merely containing ", Focused" and iOS "Keyboard Focused" are not tvOS focus.
    assert focused == ["Search"]


def _driver(os_family: str) -> XcodeSimulatorDriver:
    driver = XcodeSimulatorDriver(device_id="udid")
    driver._os_family = os_family
    driver._session_key = "session"
    driver._synthesize = AsyncMock(return_value={})
    return driver


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "command"),
    [
        ("KEYCODE_DPAD_UP", "r up"),
        ("dpad_down", "r down"),
        ("DPAD_LEFT", "r left"),
        ("dpad_right", "r right"),
        ("enter", "r select"),
        ("dpad_center", "r select"),
        ("back", "r menu"),
        ("home", "r home"),
        ("media_play_pause", "r playpause"),
    ],
)
async def test_tvos_keys_map_to_siri_remote_buttons(key, command):
    driver = _driver("tvos")
    assert await driver.press_key(key) is True
    driver._synthesize.assert_awaited_once_with(command)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["power", "volume_up", "app_switch"])
async def test_tvos_rejects_touch_device_keys(key):
    driver = _driver("tvos")
    with pytest.raises(NotImplementedError, match="not available on tvOS"):
        await driver.press_key(key)
    driver._synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_tvos_rejects_touch_input_with_remote_guidance():
    driver = _driver("tvos")
    for call in (driver.tap(10, 10), driver.swipe(1, 1, 5, 5), driver.swipe_direction("up")):
        with pytest.raises(NotImplementedError, match="DPAD_UP"):
            await call
    driver._synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_ios_keys_are_unchanged():
    driver = _driver("ios")
    await driver.press_key("home")
    driver._synthesize.assert_awaited_once_with("b h")


def test_tvos_guidance_is_selected_by_active_driver():
    ctx = SimpleNamespace(
        device=SimpleNamespace(mobile_platform="ios", device_id="udid"),
        _active_driver=_driver("tvos"),
    )
    guidance = device_action_guidance(ctx)
    assert guidance.startswith("Target platform: tvOS") and "DPAD_UP" in guidance
    ctx._active_driver = _driver("ios")
    assert device_action_guidance(ctx).startswith("Target platform: iOS")


# --- Screenshot freshness / resolution ------------------------------------------------

TV_HIERARCHY = """Device orientation: Unknown
------------------------
Application bundle identifier: com.example.tv
Application, pid: 1, label: 'Example'
 Window, {{0.0, 0.0}, {1920.0, 1080.0}}, hitPoint: {960.0, 540.0}
  Button, {{100.0, 50.0}, {200.0, 80.0}}, label: 'Play', Focused, hitPoint: {200.0, 90.0}
"""


def _capture_driver(tmp_path, monkeypatch, os_family, simctl):
    from io import BytesIO

    from PIL import Image

    from artemis.drivers.ios import xcode_driver

    stale = tmp_path / "stale.png"  # Xcode's half-size, possibly lagging frame
    Image.new("RGB", (960, 540), (1, 2, 3)).save(stale)
    hierarchy = tmp_path / "h.txt"
    hierarchy.write_text(TV_HIERARCHY)
    driver = _driver(os_family)
    driver._synthesize = AsyncMock(
        return_value={"screenshotPath": str(stale), "hierarchyPath": str(hierarchy)}
    )
    monkeypatch.setattr(xcode_driver, "run_xcrun", simctl)
    return driver, BytesIO, Image


@pytest.mark.asyncio
async def test_tvos_capture_uses_fresh_full_resolution_frame(tmp_path, monkeypatch):
    calls = []

    async def simctl(*args, **kwargs):
        from PIL import Image

        calls.append(args)
        Image.new("RGB", (3840, 2160), (200, 100, 50)).save(args[-1])
        return b""

    driver, BytesIO, Image = _capture_driver(tmp_path, monkeypatch, "tvos", simctl)
    data = await driver.get_screen_data()
    assert calls and calls[0][:3] == ("simctl", "io", "udid")
    assert (data.width, data.height) == (1920, 1080)
    with Image.open(BytesIO(data.screenshot_bytes)) as image:
        assert image.format == "JPEG" and image.size == (1920, 1080)
        assert image.getpixel((10, 10))[0] > 150  # the fresh frame, not Xcode's stale one
    # Hierarchy coordinates now map 1:1 onto the screenshot.
    focused = next(e for e in data.ui_elements if e.get("focused") == "true")
    assert focused["parsed_bounds"] == {"left": 100, "top": 50, "right": 300, "bottom": 130}


@pytest.mark.asyncio
async def test_tvos_capture_falls_back_to_xcode_frame_when_simctl_fails(tmp_path, monkeypatch):
    async def simctl(*args, **kwargs):
        raise RuntimeError("simctl failed")

    driver, BytesIO, Image = _capture_driver(tmp_path, monkeypatch, "tvos", simctl)
    data = await driver.get_screen_data()
    assert (data.width, data.height) == (960, 540)
    assert any(e.get("focused") == "true" for e in data.ui_elements)


@pytest.mark.asyncio
async def test_ios_capture_does_not_call_simctl(tmp_path, monkeypatch):
    async def simctl(*args, **kwargs):
        raise AssertionError("iOS captures must keep using Xcode's frame")

    driver, BytesIO, Image = _capture_driver(tmp_path, monkeypatch, "ios", simctl)
    data = await driver.get_screen_data()
    assert (data.width, data.height) == (960, 540)
