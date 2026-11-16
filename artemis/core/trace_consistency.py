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

"""How consistently does Artemis do the same job? A baseline over recorded runs.

Runs are grouped by their goal text. For every group this reports the completion
rate, run size (actions, time, tokens) and, above all, *path agreement*: do repeated
runs press the same keys in the same order? That agreement is what decides whether a
test generated from one run will also describe the next one.

Read-only; works on whatever the DataEngine store already holds.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import sqlite3
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from artemis.core.trace_export import _clean, _usage, action_signature, parse_actions


@dataclass
class RunSummary:
    session_id: str
    goal_key: str
    goal_label: str
    status: str
    started_at: float | None
    duration_s: float | None
    actions: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    path: list[str] = field(default_factory=list)


def goal_label(goal: str | None, width: int = 48) -> str:
    """A short label for a goal. Task goals often share a long preamble, so prefer what differs."""
    text = goal or ""
    flow = re.search(r'flow id "([^"]+)"', text)
    if flow:
        return flow.group(1)
    lines = [_clean(line) for line in text.splitlines() if _clean(line)]
    chosen = lines[-1] if lines else ""
    return chosen[:width] + ("..." if len(chosen) > width else "")


def goal_key(goal: str | None) -> str:
    return hashlib.sha1(_clean(goal).encode("utf-8")).hexdigest()[:10]


def session_path(con: sqlite3.Connection, session_id: str) -> list[str]:
    """The ordered action signatures of a run (waits and status reports excluded)."""
    path: list[str] = []
    for row in con.execute(
        "SELECT action_taken FROM steps WHERE session_id = ? ORDER BY step_number", (session_id,)
    ):
        path.extend(sig for a in parse_actions(row["action_taken"]) if (sig := action_signature(a)))
    return path


def collect_runs(
    con: sqlite3.Connection, *, goal_contains: str | None = None, since: float | None = None
) -> list[RunSummary]:
    runs: list[RunSummary] = []
    for row in con.execute("SELECT * FROM sessions ORDER BY start_time"):
        goal = row["initial_goal"] or ""
        if goal_contains and goal_contains.casefold() not in goal.casefold():
            continue
        if since is not None and (row["start_time"] or 0) < since:
            continue
        usage = _usage(con, row["session_id"])
        path = session_path(con, row["session_id"])
        start, end = row["start_time"], row["end_time"]
        runs.append(
            RunSummary(
                session_id=row["session_id"],
                goal_key=goal_key(goal),
                goal_label=goal_label(goal),
                status=row["status"] or "unknown",
                started_at=start,
                duration_s=round(end - start, 1) if start is not None and end is not None else None,
                actions=len(path),
                llm_calls=usage["llm_calls"],
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                path=path,
            )
        )
    return runs


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def path_agreement(paths: list[list[str]]) -> dict[str, Any]:
    """How alike repeated runs are. Needs at least two runs to mean anything."""
    if len(paths) < 2:
        return {"comparable": False, "runs": len(paths)}
    counts = Counter(tuple(p) for p in paths)
    modal, modal_count = counts.most_common(1)[0]
    similarity = [difflib.SequenceMatcher(a=list(modal), b=p).ratio() for p in paths]
    return {
        "comparable": True,
        "runs": len(paths),
        "distinct_paths": len(counts),
        "identical_to_modal": round(modal_count / len(paths), 2),
        "mean_similarity_to_modal": round(statistics.fmean(similarity), 2),
        "modal_path": list(modal),
    }


def summarize(
    runs: list[RunSummary], *, min_runs: int = 1, group_by: str = "goal"
) -> list[dict[str, Any]]:
    """One entry per goal, most-run first.

    ``group_by="goal"`` compares runs of the *exact* same goal text; ``"label"`` also groups goals whose
    wording was edited between attempts but which name the same journey (``flow id "..."``).
    """
    if group_by not in ("goal", "label"):
        raise ValueError("group_by must be 'goal' or 'label'")
    groups: dict[str, list[RunSummary]] = {}
    for run in runs:
        groups.setdefault(run.goal_label if group_by == "label" else run.goal_key, []).append(run)
    summaries: list[dict[str, Any]] = []
    for key, members in groups.items():
        if len(members) < min_runs:
            continue
        completed = [r for r in members if r.status == "completed"]
        # A run that never reached the model (no LLM call) failed in setup (credentials, device, config),
        # which says nothing about how well the agent does the job.
        setup_failures = [r for r in members if r.llm_calls == 0 and r.status != "completed"]
        attempted = len(members) - len(setup_failures)
        with_path = [r.path for r in completed if r.path]
        summaries.append(
            {
                "goal_key": key,
                "goal": members[0].goal_label,
                "runs": len(members),
                "completed": len(completed),
                "completion_rate": round(len(completed) / len(members), 2),
                "setup_failures": len(setup_failures),
                "agent_completion_rate": round(len(completed) / attempted, 2) if attempted else None,
                "actions_median": _median([r.actions for r in completed]),
                "actions_range": [min(r.actions for r in completed), max(r.actions for r in completed)]
                if completed
                else None,
                "duration_s_median": _median([r.duration_s for r in members if r.duration_s is not None]),
                "tokens_median": _median([r.prompt_tokens + r.completion_tokens for r in members]),
                "path_agreement": path_agreement(with_path),
                "sessions": [asdict(r) | {"path": r.path} for r in members],
            }
        )
    return sorted(summaries, key=lambda s: -s["runs"])
