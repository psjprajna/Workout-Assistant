"""How a tool call reads on the chat page: its arguments, then its status and result.

Plain text and markdown only (no Chainlit import), so the tests call it directly. Long
results are cut here, for the page; the evidence record keeps them whole.
"""

import json
from typing import Any

STATUS_TEXT = {"ok": "✓ ok"}
MAX_SHOWN_CHARS = 1500
MAX_TABLE_ROWS = 12


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")


def _table(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows[:MAX_TABLE_ROWS]:
        lines.append("| " + " | ".join(_cell(row.get(c)) for c in columns) + " |")
    if len(rows) > MAX_TABLE_ROWS:
        lines.append(f"\n… {len(rows) - MAX_TABLE_ROWS} more rows")
    return "\n".join(lines)


def _is_flat_rows(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(r, dict) and r.keys() == value[0].keys() for r in value)
        and all(not isinstance(v, (dict, list)) for r in value for v in r.values())
    )


def format_args(args: dict[str, Any]) -> str:
    """Plain JSON: the step's input is shown as a json code block by the page itself."""
    return json.dumps(args, indent=2, ensure_ascii=False)


def _planned_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "exercise": e.get("exercise_name"),
            "sets": len(e.get("target_reps", [])),
            "targets": ", ".join(str(t) for t in e.get("target_reps", [])),
            "last_on": e.get("last_on"),
        }
        for e in result.get("exercises", [])
    ]


def format_result(result: dict[str, Any]) -> str:
    """Status line first, then the result as a table (one list of flat rows) or JSON."""
    status = str(result.get("status", ""))
    head = STATUS_TEXT.get(status, status)
    rest = {k: v for k, v in result.items() if k != "status"}
    lists = [k for k, v in rest.items() if isinstance(v, list)]
    if "exercises" in rest:  # planned_day: nested last sets, so a summary table
        scalars = {k: v for k, v in rest.items() if k != "exercises"}
        parts = [_json_block(scalars)] if scalars else []
        if rest["exercises"]:
            parts.append(_table(_planned_rows(result)))
        body = "\n\n".join(parts)
    elif len(lists) == 1 and _is_flat_rows(rest[lists[0]]):
        scalars = {k: v for k, v in rest.items() if k != lists[0]}
        body = "\n\n".join(([_json_block(scalars)] if scalars else []) + [_table(rest[lists[0]])])
    else:
        body = _json_block(rest) if rest else ""
    text = f"{head}\n\n{body}" if body else head
    if len(text) > MAX_SHOWN_CHARS:
        text = text[:MAX_SHOWN_CHARS] + "\n… (cut for the page)"
    return text
