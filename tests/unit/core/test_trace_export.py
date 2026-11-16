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

"""Structured trace export and the run-consistency baseline, on a synthetic DataEngine store."""

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from artemis.core.trace_consistency import (
    collect_runs,
    goal_label,
    path_agreement,
    summarize,
)
from artemis.core.trace_export import (
    EXPORT_SCHEMA,
    TraceExportError,
    action_signature,
    confirm_texts,
    export_session,
    normalize_action,
    open_readonly,
    resolve_session,
)

# The real DataEngine tables (columns only matter, not constraints).
SCHEMA = """
CREATE TABLE sessions (session_id TEXT PRIMARY KEY, initial_goal TEXT, start_time REAL,
    end_time REAL, status TEXT, device_info TEXT, video_filepath TEXT, pid INTEGER);
CREATE TABLE steps (step_id TEXT PRIMARY KEY, session_id TEXT, step_number INTEGER, timestamp REAL,
    pre_image_name TEXT, post_image_name TEXT, summary TEXT, action_taken TEXT,
    operator_raw_thinking TEXT, operator_native_thinking TEXT, last_execution_result TEXT,
    extra_metadata TEXT);
CREATE TABLE images (image_name TEXT PRIMARY KEY, timestamp REAL, ocr_result TEXT, ui_tree TEXT,
    extra_metadata TEXT);
CREATE TABLE traces (trace_id TEXT PRIMARY KEY, session_id TEXT, step_id TEXT, parent_trace_id TEXT,
    type TEXT, name TEXT, timestamp REAL, duration REAL, status TEXT, payload TEXT);
"""


def element(text, *, focused=False, box=(0, 0, 100, 50), cls="Button"):
    left, top, right, bottom = box
    out = {
        "text": text,
        "class": cls,
        "parsed_bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
    }
    if focused:
        out["focused"] = "true"
        out["focusable"] = "true"
    return out


class Store:
    """Builds a throwaway store; every helper returns what it inserted."""

    def __init__(self, path):
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.executescript(SCHEMA)
        self._image = 0

    def image(self, elements):
        self._image += 1
        name = f"img{self._image}.jpg"
        self.con.execute(
            "INSERT INTO images VALUES (?,?,?,?,?)", (name, 0.0, None, json.dumps(elements), None)
        )
        return name

    def session(self, sid, goal="go", status="completed", start=1000.0, end=1060.0, platform="ios"):
        self.con.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
            (sid, goal, start, end, status, json.dumps({"mobile_platform": platform, "device_id": "D1"}), None, None),
        )

    def step(self, sid, n, action, pre=None, post=None, result=None, ts=None):
        self.con.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"{sid}-{n}", sid, n, ts if ts is not None else 1000.0 + n * 5, pre, post, None,
                json.dumps(action) if action is not None else None, None, None,
                json.dumps(result) if result is not None else None, None,
            ),
        )

    def usage(self, sid, prompt, completion):
        self.con.execute(
            "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"u{sid}{prompt}", sid, None, None, "llm_call", "llm_usage", 0.0, 0.0, "ok",
             json.dumps({"prompt_tokens": prompt, "completion_tokens": completion})),
        )

    def commit(self):
        self.con.commit()
        self.con.close()
        return open_readonly(self.path)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data_engine.db")


def pro_run(store, sid="aaaa1111", keys=("DPAD_RIGHT", "ENTER"), **session):
    """A Pro-shaped run (a JSON list per step) over a two-screen focus UI."""
    store.session(sid, **session)
    home = store.image([element("Hjem", focused=True), element("Film")])
    film = store.image([element("Film", focused=True), element("Interstellar", box=(0, 0, 400, 300))])
    detail = store.image([element("Trailer", focused=True), element("Medvirkende")])
    screens = [home, film, detail]
    for i, key in enumerate(keys, start=1):
        store.step(
            sid, i, [{"action": "press_key", "keycode": f"KEYCODE_{key}"}], pre=screens[i - 1],
            result={"status": "dispatched", "incident": None},
        )
    store.step(sid, len(keys) + 1, None, pre=screens[len(keys)])


# --- action normalization ----------------------------------------------------------------


def test_pro_and_flash_actions_normalize_to_the_same_vocabulary():
    pro = normalize_action({"action": "press_key", "keycode": "KEYCODE_DPAD_RIGHT"})
    flash = normalize_action(
        {"action": "press_key", "coordinates": None, "args": {"key": "DPAD_RIGHT"}}
    )
    assert pro == flash == {"action": "press_key", "key": "DPAD_RIGHT"}


def test_flash_text_and_app_actions_keep_their_meaning():
    text = normalize_action(
        {"action": "input_text", "args": {"text": "peppa", "clear_exist": False}, "target_text": "Keyboard"}
    )
    assert text == {"action": "input_text", "text": "peppa", "clear_exist": False, "target_label": "Keyboard"}
    app = normalize_action({"action": "manage_app", "args": {"action": "stop", "app_name": "a.b"}})
    assert app == {"action": "manage_app", "operation": "stop", "app": "a.b"}


def test_unknown_actions_are_kept_not_guessed():
    click = normalize_action({"action": "click", "coordinates": [10, 20], "coordinate_space": "normalized"})
    assert click == {"action": "click", "coordinates": [10, 20], "coordinate_space": "normalized"}


def test_waits_and_status_reports_are_not_part_of_the_path():
    assert action_signature({"action": "wait_for_delay", "ms": 4000}) is None
    assert action_signature({"action": "report_task_status", "status": "completed"}) is None
    assert action_signature({"action": "press_key", "key": "ENTER"}) == "press_key:ENTER"


# --- export ----------------------------------------------------------------------------------


def test_export_describes_each_step_from_what_was_recorded(store):
    pro_run(store)
    con = store.commit()

    doc = export_session(con, "aaaa1111")

    assert doc["schema"] == EXPORT_SCHEMA
    assert doc["path"] == ["press_key:DPAD_RIGHT", "press_key:ENTER"]
    first, second = doc["steps"]
    assert first["actions"] == [{"action": "press_key", "key": "DPAD_RIGHT"}]
    assert (first["focus_before"], first["focus_after"]) == ("Hjem", "Film")
    assert (second["focus_before"], second["focus_after"]) == ("Film", "Trailer")
    # What appeared after the step but was not there before, the largest element first.
    assert first["confirm_texts"][0] == "Interstellar"
    assert first["outcome"] == {"status": "dispatched", "incident": None}
    assert doc["session"]["platform"] == "ios" and doc["session"]["duration_s"] == 60.0


def test_the_closing_observation_is_not_an_action_step(store):
    pro_run(store)
    con = store.commit()
    assert [s["step"] for s in export_session(con, "aaaa1111")["steps"]] == [1, 2]


def test_a_flash_final_report_is_kept_apart_from_the_path(store):
    store.session("flash001")
    img = store.image([element("Hjem", focused=True)])
    store.step("flash001", 1, {"action": "press_key", "args": {"key": "DPAD_LEFT"}}, pre=img,
               result={"status": "dispatched"})
    store.step("flash001", 2, {"action": "report_task_status",
                               "args": {"status": "completed", "explanation": "[]"}}, pre=img)
    con = store.commit()

    doc = export_session(con, "flash001")

    assert doc["final"] == {"status": "completed", "explanation": "[]"}
    assert doc["path"] == ["press_key:DPAD_LEFT"]
    assert len(doc["steps"]) == 1


def test_usage_is_summed_from_the_llm_usage_rows(store):
    pro_run(store)
    store.usage("aaaa1111", 1000, 50)
    store.usage("aaaa1111", 2000, 70)
    con = store.commit()
    assert export_session(con, "aaaa1111")["session"]["usage"] == {
        "llm_calls": 2, "prompt_tokens": 3000, "completion_tokens": 120,
    }


def test_screens_without_a_focus_marker_give_no_focus_label(store):
    store.session("plain01")
    a = store.image([element("Login")])
    store.step("plain01", 1, [{"action": "click", "coordinates": [5, 5]}], pre=a)
    con = store.commit()
    step = export_session(con, "plain01")["steps"][0]
    assert step["focus_before"] is None and step["focus_after"] is None


def test_confirm_texts_skip_known_and_long_texts_and_rank_by_area():
    before = [element("Hjem")]
    after = [
        element("Hjem"),
        element("Small", box=(0, 0, 10, 10)),
        element("Big", box=(0, 0, 100, 100)),
        element("x" * 200, box=(0, 0, 900, 900)),
    ]
    assert confirm_texts(before, after) == ["Big", "Small"]


# --- finding a session ------------------------------------------------------------------------


def test_session_lookup_by_prefix_and_latest(store):
    pro_run(store, "abc12345", start=1000.0, end=1010.0)
    pro_run(store, "abd99999", start=2000.0, end=2010.0)
    con = store.commit()

    assert resolve_session(con, "abc")["session_id"] == "abc12345"
    assert resolve_session(con, "latest")["session_id"] == "abd99999"
    with pytest.raises(TraceExportError, match="matches 2 sessions"):
        resolve_session(con, "ab")
    with pytest.raises(TraceExportError, match="No session matches"):
        resolve_session(con, "zzz")


def test_the_store_is_opened_read_only(store):
    pro_run(store)
    con = store.commit()
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM steps")


def test_a_missing_database_is_a_clear_error(tmp_path):
    with pytest.raises(TraceExportError, match="No DataEngine database"):
        open_readonly(tmp_path / "nope.db")


# --- consistency ---------------------------------------------------------------------------------


def test_goal_labels_prefer_what_differs_between_goals():
    shared = "You are documenting a journey.\nRules...\n"
    assert goal_label(shared + 'Journey: Search (flow id "search"). Steps...') == "search"
    assert goal_label("first line\nlast line here") == "last line here"
    assert goal_label("x" * 100).endswith("...")


def test_path_agreement_needs_two_runs_and_measures_similarity():
    assert path_agreement([["a"]]) == {"comparable": False, "runs": 1}
    same = path_agreement([["a", "b"], ["a", "b"], ["a", "c"]])
    assert same["distinct_paths"] == 2
    assert same["identical_to_modal"] == 0.67
    assert same["modal_path"] == ["a", "b"]
    assert 0.8 < same["mean_similarity_to_modal"] < 1.0


def test_setup_failures_are_counted_apart_from_agent_failures(store):
    pro_run(store, "ok000001", goal="same goal")
    store.usage("ok000001", 100, 10)
    store.session("setup001", goal="same goal", status="failed", start=3000.0, end=3001.0)  # never reached the model
    store.session("agent001", goal="same goal", status="failed", start=4000.0, end=4050.0)
    store.usage("agent001", 100, 10)
    con = store.commit()

    [group] = summarize(collect_runs(con))

    assert (group["runs"], group["completed"], group["setup_failures"]) == (3, 1, 1)
    assert group["completion_rate"] == 0.33
    assert group["agent_completion_rate"] == 0.5  # 1 of the 2 runs that actually tried


def test_grouping_by_label_joins_edited_goals_of_one_journey(store):
    pro_run(store, "one00001", goal='A\nJourney (flow id "search")', keys=("DPAD_LEFT", "ENTER"))
    pro_run(store, "two00001", goal='Edited wording\nJourney (flow id "search")', keys=("DPAD_LEFT", "ENTER"))
    con = store.commit()
    runs = collect_runs(con)

    assert len(summarize(runs, group_by="goal")) == 2
    [joined] = summarize(runs, group_by="label")
    assert joined["runs"] == 2 and joined["path_agreement"]["identical_to_modal"] == 1.0
    with pytest.raises(ValueError):
        summarize(runs, group_by="nope")


# --- CLI --------------------------------------------------------------------------------------------


def test_cli_exports_json_and_reports_consistency(store, tmp_path):
    from artemis.interfaces.cli.commands.trace import trace_app

    pro_run(store, "cli00001", goal='G (flow id "x")')
    store.commit()
    db = str(tmp_path / "data_engine.db")
    runner = CliRunner()

    exported = runner.invoke(trace_app, ["export", "cli0", "--db", db])
    assert exported.exit_code == 0, exported.output
    assert json.loads(exported.stdout)["path"] == ["press_key:DPAD_RIGHT", "press_key:ENTER"]

    out = tmp_path / "run.json"
    assert runner.invoke(trace_app, ["export", "latest", "--db", db, "-o", str(out)]).exit_code == 0
    assert json.loads(out.read_text())["session"]["session_id"] == "cli00001"

    report = runner.invoke(trace_app, ["consistency", "--db", db, "--json"])
    assert report.exit_code == 0
    assert json.loads(report.stdout)[0]["goal"] == "x"

    missing = runner.invoke(trace_app, ["export", "nope", "--db", db])
    assert missing.exit_code == 1
