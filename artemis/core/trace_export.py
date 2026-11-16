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

"""Structured export of a recorded run, read from the DataEngine SQLite store.

The agent's own prose report is free text and differs from run to run. The store,
however, already holds what actually happened: the exact action dispatched at every
step and the element tree of every screen the agent saw. This module turns one
session into a stable, machine-readable document (``artemis.trace-export/1``):

* ``actions``      the dispatched actions in one canonical vocabulary, whether the run
                   was Flash (one dict with ``args``) or Pro (a list of dicts);
* ``focus_before`` / ``focus_after``  the focused element's label on a focus-driven UI
                   (tvOS, Android TV), read from the screen trees;
* ``confirm_texts`` texts that appear after the step but not before it, the screen's
                   identifying texts, ranked by on-screen area;
* ``outcome``      whether the dispatch succeeded and any execution incident.

It is the source for generating deterministic tests (XCUITest, Playwright, plain
scripts) and for comparing runs. Nothing here writes to the database.
"""

from __future__ import annotations

import datetime
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

EXPORT_SCHEMA = "artemis.trace-export/1"

# Dispatched actions that are bookkeeping, not part of the path a test would replay.
_NON_PATH_ACTIONS = frozenset({"report_task_status"})
_WS = re.compile(r"[\s\u00a0]+")
_MAX_CONFIRM_CHARS = 80


class TraceExportError(RuntimeError):
    """A session could not be found or read."""


def open_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open the store without any ability to modify it (safe beside a live run)."""
    path = Path(db_path)
    if not path.is_file():
        raise TraceExportError(f"No DataEngine database at {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def resolve_session(con: sqlite3.Connection, session: str) -> sqlite3.Row:
    """Find a session by full id, unique id prefix, or ``latest``."""
    if session == "latest":
        row = con.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 1").fetchone()
        if row is None:
            raise TraceExportError("The database has no sessions.")
        return row
    rows = con.execute(
        "SELECT * FROM sessions WHERE session_id = ? OR session_id LIKE ? ORDER BY start_time DESC",
        (session, f"{session}%"),
    ).fetchall()
    exact = [r for r in rows if r["session_id"] == session]
    if exact:
        return exact[0]
    if not rows:
        raise TraceExportError(f"No session matches {session!r}.")
    if len(rows) > 1:
        ids = ", ".join(r["session_id"][:12] for r in rows[:6])
        raise TraceExportError(f"{session!r} matches {len(rows)} sessions ({ids}...); use more characters.")
    return rows[0]


def _loads(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _clean(text: Any) -> str:
    return _WS.sub(" ", str(text or "")).strip()


# --- actions -----------------------------------------------------------------------------


def _key_name(raw: Any) -> str:
    name = str(raw or "").strip()
    if name.upper().startswith("KEYCODE_"):
        name = name[len("KEYCODE_") :]
    return name.upper()


def normalize_action(raw: dict[str, Any]) -> dict[str, Any]:
    """One dispatched action in a vocabulary shared by Flash and Pro recordings."""
    name = str(raw.get("action") or "")
    args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
    if name == "press_key":
        key = _key_name(args.get("key") or raw.get("keycode") or raw.get("key"))
        return {"action": "press_key", "key": key}
    if name == "input_text":
        out: dict[str, Any] = {
            "action": "input_text",
            "text": args.get("text", raw.get("text")),
            "clear_exist": args.get("clear_exist", raw.get("clear_exist")),
        }
        if raw.get("target_text"):
            out["target_label"] = _clean(raw["target_text"])
        return out
    if name == "manage_app":
        return {
            "action": "manage_app",
            "operation": args.get("action", raw.get("operation")),
            "app": args.get("app_name", raw.get("app_name")),
        }
    if name == "wait_for_delay":
        return {"action": "wait_for_delay", "ms": args.get("time_in_ms", raw.get("time_in_ms"))}
    if name == "report_task_status":
        return {"action": name, "status": args.get("status"), "explanation": args.get("explanation")}
    # click, swipe, long_press, ...: keep what the recording has, without guessing semantics.
    out = {"action": name}
    for key in ("coordinates", "coordinate_space", "target_text"):
        if raw.get(key) is not None:
            out[key] = raw[key]
    if args:
        out["args"] = args
    return out


def parse_actions(action_taken: str | None) -> list[dict[str, Any]]:
    """Normalize a step's ``action_taken`` (a JSON list for Pro, a JSON dict for Flash)."""
    data = _loads(action_taken)
    items: Iterable[Any] = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    return [normalize_action(item) for item in items if isinstance(item, dict) and item.get("action")]


def action_signature(action: dict[str, Any]) -> str | None:
    """A short stable token for comparing paths; ``None`` for non-path actions and pure waits."""
    kind = action.get("action")
    if kind in _NON_PATH_ACTIONS or kind == "wait_for_delay":
        return None
    if kind == "press_key":
        return f"press_key:{action.get('key')}"
    if kind == "input_text":
        return f"input_text:{action.get('text')}"
    if kind == "manage_app":
        return f"manage_app:{action.get('operation')}:{action.get('app')}"
    return str(kind)


# --- screens ---------------------------------------------------------------------------------


def _screen(con: sqlite3.Connection, image_name: str | None) -> list[dict[str, Any]]:
    if not image_name:
        return []
    row = con.execute("SELECT ui_tree FROM images WHERE image_name = ?", (image_name,)).fetchone()
    tree = _loads(row["ui_tree"]) if row else None
    return [e for e in tree if isinstance(e, dict)] if isinstance(tree, list) else []


def _area(element: dict[str, Any]) -> int:
    box = element.get("parsed_bounds") or {}
    try:
        return max(0, int(box["right"]) - int(box["left"])) * max(0, int(box["bottom"]) - int(box["top"]))
    except (KeyError, TypeError, ValueError):
        return 0


def focused_label(elements: list[dict[str, Any]]) -> str | None:
    """Label of the element that has focus, on UIs that report one (tvOS, Android TV)."""
    for element in elements:
        if str(element.get("focused")).lower() == "true":
            return _clean(element.get("text") or element.get("resource_id")) or None
    return None


def screen_texts(elements: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for element in elements:
        text = _clean(element.get("text"))
        if text:
            seen.setdefault(text)
    return list(seen)


def confirm_texts(before: list[dict[str, Any]], after: list[dict[str, Any]], limit: int = 4) -> list[str]:
    """Texts that appear after a step but not before it, the largest on-screen first."""
    known = set(screen_texts(before))
    ranked: dict[str, int] = {}
    for element in after:
        text = _clean(element.get("text"))
        # Long strings are body copy or catalog descriptions, never a screen's identity.
        if text and len(text) <= _MAX_CONFIRM_CHARS and text not in known:
            ranked[text] = max(ranked.get(text, 0), _area(element))
    ordered = sorted(ranked, key=lambda t: (-ranked[t], -len(t)))
    return ordered[:limit]


# --- session -------------------------------------------------------------------------------


def _usage(con: sqlite3.Connection, session_id: str) -> dict[str, int]:
    prompt = completion = calls = 0
    for row in con.execute(
        "SELECT payload FROM traces WHERE session_id = ? AND name = 'llm_usage'", (session_id,)
    ):
        data = _loads(row["payload"]) or {}
        prompt += int(data.get("prompt_tokens") or 0)
        completion += int(data.get("completion_tokens") or 0)
        calls += 1
    return {"llm_calls": calls, "prompt_tokens": prompt, "completion_tokens": completion}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat(timespec="seconds")


def session_meta(con: sqlite3.Connection, session: sqlite3.Row, images_dir: Path | None) -> dict[str, Any]:
    device = _loads(session["device_info"]) or {}
    start, end = session["start_time"], session["end_time"]
    return {
        "session_id": session["session_id"],
        "goal": session["initial_goal"],
        "status": session["status"],
        "platform": device.get("mobile_platform"),
        "device_id": device.get("device_id"),
        "started_at": _iso(start),
        "ended_at": _iso(end),
        "duration_s": round(end - start, 1) if start is not None and end is not None else None,
        "usage": _usage(con, session["session_id"]),
        "images_dir": str(images_dir) if images_dir else None,
    }


def export_session(
    con: sqlite3.Connection, session: str, *, images_dir: Path | None = None
) -> dict[str, Any]:
    """The full ``artemis.trace-export/1`` document for one session."""
    row = resolve_session(con, session)
    meta = session_meta(con, row, images_dir)
    step_rows = con.execute(
        "SELECT * FROM steps WHERE session_id = ? ORDER BY step_number", (row["session_id"],)
    ).fetchall()
    started = row["start_time"]

    steps: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for index, step in enumerate(step_rows):
        actions = parse_actions(step["action_taken"])
        report = next((a for a in actions if a["action"] == "report_task_status"), None)
        if report is not None:
            final = {"status": report.get("status"), "explanation": report.get("explanation")}
        path_actions = [a for a in actions if a["action"] not in _NON_PATH_ACTIONS]
        if not path_actions:
            continue  # the closing observation, or a step that only reported status
        # The state after a step is what the next step started from.
        after_name = step["post_image_name"] or (
            step_rows[index + 1]["pre_image_name"] if index + 1 < len(step_rows) else None
        )
        before = _screen(con, step["pre_image_name"])
        after = _screen(con, after_name)
        result = _loads(step["last_execution_result"]) or {}
        steps.append(
            {
                "step": step["step_number"],
                "t_offset_s": round(step["timestamp"] - started, 1)
                if step["timestamp"] is not None and started is not None
                else None,
                "actions": path_actions,
                "outcome": {
                    "status": result.get("status"),
                    "incident": result.get("incident"),
                },
                "focus_before": focused_label(before),
                "focus_after": focused_label(after),
                "confirm_texts": confirm_texts(before, after),
                "screenshot_before": step["pre_image_name"],
                "screenshot_after": after_name,
            }
        )
    return {
        "schema": EXPORT_SCHEMA,
        "session": meta,
        "steps": steps,
        "final": final,
        "path": [sig for s in steps for a in s["actions"] if (sig := action_signature(a))],
    }
