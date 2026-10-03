"""The tools the model can call, and the dispatcher that runs them.

One module per capability, with `base` holding what they share. Everything a caller
needs is re-exported here, so `from interfaze_lite import tools` remains the only
import written outside this package.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts import InputFetchError
from ..filerefs import FileRefError
from .base import _MAX_TOOL_ATTEMPTS, Tool, ToolContext, ToolResult, _memo_key
from .detection import GUI_DETECTION, OBJECT_DETECTION
from .forecast import FORECAST
from .ocr import OCR
from .speech import SPEECH_TO_TEXT
from .translate import TRANSLATE

__all__ = ["REGISTRY", "Tool", "ToolContext", "ToolResult", "dispatch", "schemas"]

# Tools that read a file the caller supplied. Offered only when there is one: offered
# anyway, a text-only request got an OCR call on a made-up reference.
FILE_TOOLS = (OCR, SPEECH_TO_TEXT, OBJECT_DETECTION, GUI_DETECTION)
# Tools whose input can be the prompt itself. Offered when the request asks for them.
TEXT_TOOLS = (TRANSLATE, FORECAST)

REGISTRY: dict[str, Tool] = {t.name: t for t in (*FILE_TOOLS, *TEXT_TOOLS)}


def schemas(files: bool = True, text: tuple[str, ...] | None = None,
            series_in_prompt: bool = False) -> list[dict[str, Any]]:
    """The file tools when there is a file, and the text tools named in `text` (all when None).

    With the series already written in the message, forecast is offered without
    `dataset`, which it reads from the message instead. Offered, the model copied a
    year of daily rows into the call one by one: most of a minute before the forecast,
    which itself takes under a second.
    """
    offered = [t for t in TEXT_TOOLS if text is None or t.name in text]
    out = [t.as_openai() for t in (*(FILE_TOOLS if files else ()), *offered)]
    if series_in_prompt:
        for schema in out:
            fn = schema["function"]
            if fn["name"] == FORECAST.name:
                params = fn["parameters"]
                fn["parameters"] = {**params, "properties": {
                    k: v for k, v in params["properties"].items() if k != "dataset"}}
    return out
def memo_key(name: str | None, raw_args) -> str | None:
    """What a call of ours asks, as `_memo_key` states it; None for another's or bad JSON."""
    tool = REGISTRY.get(name or "")
    if tool is None:
        return None
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError:
        return None
    return _memo_key(tool, args) if isinstance(args, dict) else None


async def dispatch(name: str, raw_args: str, ctx: ToolContext) -> ToolResult:
    """Run one tool call, converting every failure into a result the model can act on.

    A raised exception here would abort the whole request. Interfaze instead hands the
    model an `{error, message}` object so it can correct itself and retry, which is why
    even a malformed-JSON argument list comes back as a result rather than a 500.
    """
    tool = REGISTRY.get(name)
    if tool is None:
        return ToolResult(model_facing={
            "error": f"unknown tool {name!r}",
            "message": f"Use one of: {', '.join(sorted(REGISTRY))}.",
        })

    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError as exc:
        return ToolResult(model_facing={
            "error": f"arguments were not valid JSON: {exc}",
            "message": "Retry the tool call with valid JSON arguments.",
        })

    key = _memo_key(tool, args)
    cached = ctx.results.get(key)
    if cached is not None:
        return cached

    # Telling the model to retry an identical call is only useful when the failure
    # depends on the arguments. A backend that is down or out of memory fails the same
    # way every time, and the model will happily burn the entire step budget
    # rediscovering that -- six attempts and five minutes, in the OOM case. One retry,
    # then the call is refused without touching the backend again.
    if ctx.failures.get(key, 0) >= _MAX_TOOL_ATTEMPTS:
        return ToolResult(model_facing={
            "error": f"{tool.name} already failed {_MAX_TOOL_ATTEMPTS} times with these "
                     f"arguments and was not retried.",
            "message": ("This capability is unavailable right now. Tell the user so and "
                        "do not call it again."),
        })

    try:
        result = await tool.execute(args, ctx)
        ctx.results[key] = result
        return result
    except FileRefError as exc:
        return ToolResult(model_facing={"error": str(exc), "message": "Retry with a valid ref id."})
    except InputFetchError as exc:
        # The caller's file is unreachable; no retry changes that, and it is not an
        # outage of this capability.
        return ToolResult(model_facing={
            "error": str(exc),
            "message": "The file could not be fetched from that address. Tell the user so; "
                       "do not call a tool on it again."})
    except KeyError as exc:
        return ToolResult(model_facing={
            "error": f"missing required argument {exc}",
            "message": "Retry the tool call with all required arguments.",
        })
    except Exception as exc:
        ctx.failures[key] = ctx.failures.get(key, 0) + 1
        if ctx.failures[key] >= _MAX_TOOL_ATTEMPTS:
            return ToolResult(model_facing={
                "error": str(exc),
                "message": (f"{tool.name} has now failed {_MAX_TOOL_ATTEMPTS} times with "
                            "these arguments and will not be retried. Tell the user this "
                            "capability is unavailable right now."),
            })
        return ToolResult(model_facing={"error": str(exc), "message": "Retry the tool call"})
