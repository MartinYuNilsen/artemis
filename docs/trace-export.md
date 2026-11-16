# Trace export and run-consistency baseline

Artemis records every run in `traces/data_engine.db`. Two read-only commands turn that
record into something a person (or a test generator) can use. Neither changes the store,
so both are safe to run beside a live task.

## `artemis trace export <session|prefix|latest>`

Writes one run as structured JSON (`artemis.trace-export/1`):

```bash
artemis trace export ffbfe18d --out run.json
artemis trace export latest --db /path/to/data_engine.db
```

| Field | Meaning |
|---|---|
| `path` | The ordered action signatures (`press_key:DPAD_RIGHT`, ...). Waits and the final status report are left out. Use it to compare runs. |
| `steps[].actions` | The dispatched actions in one vocabulary, whether the run was Flash (one dict with `args`) or Pro (a list of dicts). |
| `steps[].focus_before` / `focus_after` | The focused element's label on a focus-driven UI (tvOS, Android TV); `null` where the UI reports no focus or the element has no text. |
| `steps[].confirm_texts` | Texts that appear after the step but not before it, largest on-screen first (at most 4, none longer than 80 characters). |
| `steps[].outcome` | The dispatch status and any execution incident. |
| `steps[].screenshot_before` / `screenshot_after` | Image names under `session.images_dir`. The "after" frame is the next step's "before" frame. |
| `final` | The status and explanation the agent reported, if it reported one. |
| `session` | Goal, status, platform, device, duration and LLM usage (calls, prompt and completion tokens). |

The agent's own prose report is free text and varies from run to run; this export is read
from what was actually dispatched and observed.

Limits to know:

* `confirm_texts` from a *single* run include catalog content (video titles, carousel
  tiles) that changes over time. Only a comparison across runs can tell stable identifying
  texts from incidental ones.
* An element without text, for example a logo tab, has no focus label.
* Only the post-action frame is derived; Artemis does not store one per action.

## `artemis trace consistency`

A baseline over recorded runs: how consistently is the same job done?

```bash
artemis trace consistency                       # runs of the identical goal text
artemis trace consistency --group-by label      # also join goals whose wording was edited
artemis trace consistency --goal "Film" --min-runs 3 --json
```

Per goal it reports runs, completed runs, **setup failures** (the run never reached the
model: credentials, device or config problems, counted apart so they do not hide how well
the agent does its job), the agent completion rate, median actions, time and tokens, and
**same path**: the share of completed runs whose key sequence equals the most common one.
A test generated from one run only describes the next run if that share is high.

Point either command at another store with `--db` or `ARTEMIS_DATA_ENGINE_DB`.
