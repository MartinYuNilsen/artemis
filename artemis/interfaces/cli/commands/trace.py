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

"""Trace inspection and replay commands (artemis trace)."""

import datetime
import json
from pathlib import Path
from typing import Annotated

from artemis.config import settings
from third_party.mobile_use.utils.logger import get_logger
from rich.console import Console
from rich.table import Table
import typer

logger = get_logger(__name__)
trace_app = typer.Typer(help="Inspect, list, and query task execution traces.")


@trace_app.command("list")
def list_traces(
    traces_path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help="Custom traces directory path."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max sessions to list.")] = 20,
) -> None:
    """List recorded execution trace sessions."""
    base_dir = traces_path or settings.TRACES_PATH
    if not base_dir.exists():
        typer.secho(f"No traces found at: {base_dir}", fg=typer.colors.YELLOW)
        return

    sessions = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    console = Console()
    table = Table(title=f"Recorded Traces ({len(sessions)} total)")
    table.add_column("Session Name", style="cyan")
    table.add_column("Last Modified", style="green")
    table.add_column("Artifacts", style="white")

    for s in sessions[:limit]:
        artifacts = []
        if (s / "recording.mp4").exists() or (s / "recording.mkv").exists():
            artifacts.append("🎬 Video")
        if (s / "steps.json").exists():
            artifacts.append("📋 Steps")
        if (s / "notes").exists():
            artifacts.append("📝 Notes")
        mtime = time_str = time_to_str(s.stat().st_mtime)
        table.add_row(s.name, mtime, ", ".join(artifacts) or "Screenshots")

    console.print(table)


def time_to_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


@trace_app.command("view")
def view_trace(
    session_name: Annotated[str, typer.Argument(help="Name of the trace session directory.")],
    traces_path: Annotated[Path | None, typer.Option("--path", "-p")] = None,
) -> None:
    """View step-by-step summary of a recorded trace session."""
    base_dir = traces_path or settings.TRACES_PATH
    session_dir = base_dir / session_name

    if not session_dir.exists():
        typer.secho(f"Session '{session_name}' not found in {base_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    steps_file = session_dir / "steps.json"
    console = Console()

    if steps_file.exists():
        try:
            steps_data = json.loads(steps_file.read_text(encoding="utf-8"))
            table = Table(title=f"Trace Steps: {session_name}")
            table.add_column("Step", justify="right", style="cyan")
            table.add_column("Action", style="green")
            table.add_column("Reasoning / Motivation", style="white")

            for step in steps_data:
                table.add_row(
                    str(step.get("step", "-")),
                    step.get("action", "-"),
                    step.get("motivation", step.get("thought", "-")),
                )
            console.print(table)
            return
        except Exception as e:
            logger.warning(f"Could not parse steps.json: {e}")

    # If steps.json not present, list files
    typer.secho(f"Session files in {session_dir}:", fg=typer.colors.CYAN)
    for f in session_dir.iterdir():
        typer.echo(f"  - {f.name}")


def _open_trace_db(db: Path | None):
    from artemis.config import get_data_engine_db_path
    from artemis.core.trace_export import TraceExportError, open_readonly

    path = db or get_data_engine_db_path()
    try:
        return open_readonly(path), path
    except TraceExportError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc


@trace_app.command("export")
def export_trace(
    session: Annotated[
        str, typer.Argument(help="Session id, a unique id prefix, or 'latest'.")
    ],
    db: Annotated[
        Path | None,
        typer.Option("--db", envvar="ARTEMIS_DATA_ENGINE_DB", help="Path to data_engine.db."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the JSON here instead of stdout.")
    ] = None,
) -> None:
    """Export a recorded run as structured steps (artemis.trace-export/1).

    Actions, focus before/after, texts that identify each screen and the outcome of every
    dispatch are read from the DataEngine store, not from the agent's own prose report.
    """
    from artemis.core.trace_export import TraceExportError, export_session

    con, path = _open_trace_db(db)
    try:
        document = export_session(con, session, images_dir=path.parent / "images")
    except TraceExportError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    finally:
        con.close()
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if out is None:
        typer.echo(text)
        return
    out.write_text(text + "\n", encoding="utf-8")
    typer.secho(
        f"Exported {len(document['steps'])} steps to {out}", fg=typer.colors.GREEN, err=True
    )


@trace_app.command("consistency")
def trace_consistency(
    db: Annotated[
        Path | None,
        typer.Option("--db", envvar="ARTEMIS_DATA_ENGINE_DB", help="Path to data_engine.db."),
    ] = None,
    goal: Annotated[
        str | None, typer.Option("--goal", "-g", help="Only goals containing this text.")
    ] = None,
    min_runs: Annotated[int, typer.Option("--min-runs", help="Hide goals run fewer times.")] = 1,
    group_by: Annotated[
        str,
        typer.Option(
            "--group-by",
            help="'goal' = identical goal text; 'label' = same journey even if the wording was edited.",
        ),
    ] = "goal",
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Baseline: how consistently were repeated runs of the same goal completed?

    Groups recorded runs by goal and reports completion, size and path agreement, i.e.
    whether repeated runs press the same keys in the same order.
    """
    from artemis.core.trace_consistency import collect_runs, summarize

    con, _ = _open_trace_db(db)
    try:
        groups = summarize(
            collect_runs(con, goal_contains=goal), min_runs=min_runs, group_by=group_by
        )
    finally:
        con.close()
    if as_json:
        typer.echo(json.dumps(groups, ensure_ascii=False, indent=2))
        return
    if not groups:
        typer.secho("No recorded runs match.", fg=typer.colors.YELLOW)
        return
    table = Table(title="Run consistency by goal")
    for column, justify in (
        ("Goal", "left"),
        ("Runs", "right"),
        ("Done", "right"),
        ("Setup fails", "right"),
        ("Agent rate", "right"),
        ("Actions", "right"),
        ("Median s", "right"),
        ("Median tokens", "right"),
        ("Same path", "right"),
    ):
        table.add_column(  # type: ignore[arg-type]
            column, justify=justify, no_wrap=True, min_width=len(column), overflow="ellipsis"
        )
    for g in groups:
        agree = g["path_agreement"]
        table.add_row(
            g["goal"],
            str(g["runs"]),
            str(g["completed"]),
            str(g["setup_failures"]),
            "-" if g["agent_completion_rate"] is None else f"{g['agent_completion_rate']:.0%}",
            "-" if g["actions_median"] is None else f"{g['actions_median']:g}",
            "-" if g["duration_s_median"] is None else f"{g['duration_s_median']:g}",
            "-" if g["tokens_median"] is None else f"{int(g['tokens_median']):,}",
            f"{agree['identical_to_modal']:.0%}" if agree["comparable"] else "n/a",
        )
    Console().print(table)
