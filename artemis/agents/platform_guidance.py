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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared platform constraints for the Flash and Pro device-action prompts."""


def _is_tvos_target(ctx) -> bool:
    """Whether the iOS-platform target is an Apple TV (tvOS) simulator."""
    driver = getattr(ctx, "_active_driver", None)
    if driver is not None and getattr(driver, "is_tvos", False) is True:
        return True
    device_id = getattr(getattr(ctx, "device", None), "device_id", None)
    if not isinstance(device_id, str) or not device_id:
        return False
    from artemis.drivers.ios.discovery import device_matches_identifier, list_ios_simulators_sync

    return any(
        device.get("os_family") == "tvos" and device_matches_identifier(device, device_id)
        for device in list_ios_simulators_sync() or []
    )


_TVOS_GUIDANCE = (
    "Target platform: tvOS (Apple TV). There is no touchscreen: tap, swipe, "
    "long_press, and coordinate input are unavailable. The UI is focus-based and "
    "exactly one element is focused (marked focused=true in the element list). "
    "Move focus with press_key DPAD_UP/DPAD_DOWN/DPAD_LEFT/DPAD_RIGHT, activate the "
    "focused element with ENTER (or DPAD_CENTER), go back with BACK (the Siri "
    "Remote Menu button), go to the Home screen with HOME, and toggle playback "
    "with MEDIA_PLAY_PAUSE. Read the focused element after every move; count the "
    "steps from it to the target using the element positions and repeat the "
    "directional press that many times. Text entry only works while a text field "
    "or the on-screen keyboard has focus; call input_text with clear_exist=false. "
    "Android keycodes, ADB commands, power/volume keys, and app locking are "
    "unavailable. manage_app accepts installed app display names or bundle "
    "identifiers.\n\n"
)


def device_action_guidance(ctx) -> str:
    if getattr(getattr(ctx, "device", None), "mobile_platform", None) != "ios":
        return ""
    if _is_tvos_target(ctx):
        return _TVOS_GUIDANCE
    return (
        "Target platform: iOS. For input_text, explicitly set clear_exist=false; "
        "type into an empty field or at its existing cursor. After typing into a URL "
        "or search field, submit it with press_key(enter) or by tapping the "
        "keyboard's Go/Search button; typing alone does not navigate. Whole-field "
        "clearing, Android keycodes, Back/Delete keys, ADB commands, and app "
        "locking are unavailable. iOS has no Back button: never call press_key "
        "with BACK; return to the previous screen with the app's back chevron "
        "(top-left) instead. press_key supports enter, home, power, volume_up, "
        "volume_down, and app_switch. manage_app accepts installed iOS display "
        "names or bundle identifiers. Action coordinates use the screenshot's "
        "normalized 0-1000 space.\n\n"
    )
