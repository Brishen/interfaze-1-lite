"""Request validation for /v1/chat/completions.

Rejecting a malformed request costs a few microseconds here; letting it through costs
a GPU-seconds-long round trip that ends in a 500. Every check below returns a 4xx with
a message naming the offending field, because a caller debugging against a remote
endpoint has nothing else to go on.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any

TASKS = ("ocr", "stt", "object_detection", "gui_detection", "translate", "forecast")

# A routed task may be named differently from the tool that serves it. Transcription
# is routed as `speech_to_text` but reports itself as `stt` in precontext, and both
# names are published -- so accepting only one of them breaks callers written against
# the other. The alias resolves to a tool; what the caller is told it ran is the name
# the caller used.
TASK_ALIASES = {"speech_to_text": "stt"}

_TASK_TAG = re.compile(r"<task>(.*?)</task>", re.IGNORECASE | re.DOTALL)

# `developer` is OpenAI's newer name for `system`, and the SDKs send it by default for
# recent models -- which is how it reaches this service. Accepting the same four roles
# interfaze does, and collapsing the alias the same way, keeps one wire contract rather
# than two that differ by which SDK version the caller happens to run.
ROLE_ALIASES = {"developer": "system"}
ROLES = ("system", "user", "assistant", "tool", *ROLE_ALIASES)


def normalise_roles(messages: list[dict]) -> list[dict]:
    """Collapse role aliases before anything downstream reads a role.

    Widening the accepted set alone is not enough: the task tag and the caller's own
    system prompt are both found by testing `role == "system"`, so a `developer` turn
    would validate and then be silently ignored -- no error, and the instructions in it
    simply absent from the request.
    """
    if not any(m.get("role") in ROLE_ALIASES for m in messages if isinstance(m, dict)):
        return messages
    return [{**m, "role": ROLE_ALIASES[m["role"]]}
            if isinstance(m, dict) and m.get("role") in ROLE_ALIASES else m
            for m in messages]


class RequestError(Exception):
    """A bad request. `status` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _texts(message: dict) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"]
    return []


def extract_task(messages: list[dict]) -> str | None:
    """Pull a single `<task>name</task>` directive out of the system turns.

    The SDK's typed `task=` argument lowers to this tag, so it is the wire format even
    though callers rarely write it by hand. Only one task may be routed per request:
    the tag names the tool to force, and "force two tools" has no meaning.
    """
    found: list[str] = []
    for message in messages:
        if message.get("role") != "system":
            continue
        for text in _texts(message):
            found.extend(_TASK_TAG.findall(text))

    if not found:
        return None
    if len(found) > 1:
        raise RequestError("only one task may be specified per request")

    names = [n.strip() for n in found[0].split(",") if n.strip()]
    if len(names) > 1:
        raise RequestError(
            f"only one task may be specified per request, got {len(names)}: "
            + ", ".join(names))
    if not names:
        # `<task></task>` names nothing, which is a request to route nothing -- the
        # same state as omitting the tag. Rejecting it turned a harmless empty
        # template slot into a 400 on an otherwise valid request.
        return None

    task = names[0].lower()
    if task not in TASKS and task not in TASK_ALIASES:
        raise RequestError(
            f"invalid task '{task}'. Supported tasks: "
            + ", ".join((*TASKS, *TASK_ALIASES)))
    return task


def resolve_task(task: str) -> str:
    """The tool that serves a routed task name."""
    return TASK_ALIASES.get(task, task)


def strip_task_tags(messages: list[dict]) -> list[dict]:
    """Remove the directive once it has been read, so it never reaches the model."""
    out: list[dict] = []
    for message in messages:
        if message.get("role") != "system":
            out.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            out.append({**message, "content": _TASK_TAG.sub("", content).strip()})
        elif isinstance(content, list):
            out.append({**message, "content": [
                {**p, "text": _TASK_TAG.sub("", p["text"]).strip()}
                if isinstance(p, dict) and p.get("type") == "text" and "text" in p else p
                for p in content
            ]})
        else:
            out.append(message)
    return out


def _check_data_uri(url: str) -> None:
    """A data: URI that is not decodable is a client bug, not a perception failure.

    Left unchecked it surfaces much later as an unreadable-image error from whichever
    tool happened to receive it, which points the caller at the wrong thing entirely.
    """
    if not url.startswith("data:"):
        return
    header, _, payload = url.partition(",")
    if not _:
        raise RequestError("malformed data URI: missing ',' separator")
    if "base64" not in header:
        return
    payload = payload.strip()
    if not payload:
        raise RequestError("malformed data URI: empty base64 payload")
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError("invalid base64 payload in data URI") from None


def validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise RequestError("`messages` must be a non-empty array")

    has_content = False
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        if message.get("role") not in ROLES:
            raise RequestError(f"unsupported message role: {message.get('role')!r}")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            has_content = True
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise RequestError("each content part must be an object")
                kind = part.get("type")
                if kind == "text" and str(part.get("text", "")).strip():
                    has_content = True
                elif kind in ("image_url", "input_image"):
                    has_content = True
                    url = (part.get("image_url") or {}).get("url") if kind == "image_url" \
                        else part.get("image_url")
                    if isinstance(url, str):
                        _check_data_uri(url)
                elif kind in ("input_audio", "audio_url", "file", "file_url", "video_url"):
                    has_content = True
        if message.get("tool_calls"):
            has_content = True

    if not has_content:
        raise RequestError("request has no text content: every message is empty")


def check_api_key(header: str | None, expected: str | None,
                  admin_header: str | None = None, admin_expected: str | None = None) -> None:
    """Enforce a shared key only when one is configured.

    interfaze-lite ships open by default -- it is a container you run yourself, and
    demanding a key from someone's own localhost is friction with no security value.
    Setting API_KEY turns enforcement on for a deployment that is actually exposed, as a
    Bearer token. ADMIN_KEY is how the hosted deployment is called: by the interfaze API, in
    `x-api-admin-key`, as every one of its model services is. With either set, a request
    must carry one that matches.
    """
    import hmac

    if not expected and not admin_expected:
        return
    token = ""
    if header and header.lower().startswith("bearer "):
        token = header[7:].strip()
    elif header:
        token = header.strip()
    if expected and token and hmac.compare_digest(token.encode(), expected.encode()):
        return
    admin = (admin_header or "").strip()
    if admin_expected and admin and hmac.compare_digest(admin.encode(), admin_expected.encode()):
        return
    raise RequestError("invalid API key", status=401)


# OpenAI's own documented ranges. Out-of-range values were previously passed straight
# through to the inference server, which either clamped them silently or failed deep in
# the engine with a message about a field the caller never named -- so an obvious typo
# surfaced as a 500 from somewhere else entirely, or as output that quietly ignored
# what was asked for.
_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "frequency_penalty": (-2.0, 2.0),
    "presence_penalty": (-2.0, 2.0),
}


def validate_sampling(body: dict, *, max_output_tokens: int) -> None:
    """Reject sampling parameters the server cannot honour, before generating."""
    for name, (low, high) in _RANGES.items():
        value = body.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestError(f"{name} must be a number")
        if not low <= float(value) <= high:
            raise RequestError(f"{name} must be between {low} and {high}, got {value}")

    tokens = body.get("max_tokens")
    if body.get("max_completion_tokens") is not None:
        tokens = body["max_completion_tokens"]
    if tokens is None:
        return
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise RequestError("max_tokens must be an integer")
    if tokens < 1:
        raise RequestError(f"max_tokens must be at least 1, got {tokens}")
    if tokens > max_output_tokens:
        raise RequestError(
            f"max_tokens must be at most {max_output_tokens}, got {tokens}")
