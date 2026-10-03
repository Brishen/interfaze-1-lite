"""Guardrails: the `<guard>` directive, and turning the classifiers' output into a verdict.

Ported from interfaze `helpers/validate.ts` (`validateGuardAndNSFW`) and
`helpers/nsfw/index.ts`. The contract is theirs: which codes exist, what "ALL" expands
to, when a request is blocked, what `content` says when it is ("unsafe S1 S12_IMAGE"),
and the two precontext entries (`text_guardrail_classifier`,
`image_guardrail_classifier`). Pure: the classifiers are called by the orchestrator.
"""

from __future__ import annotations

import re
from typing import Any

TEXT_CODES = tuple(f"S{i}" for i in range(1, 15))
IMAGE_CODES = ("S1_IMAGE", "S12_IMAGE", "S15_IMAGE")
ALL_CODES = (*TEXT_CODES, *IMAGE_CODES)

# Llama Guard 4's category names, which is what interfaze's guard model is prompted with.
CATEGORIES = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex-Related Crimes.",
    "S4": "Child Sexual Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Suicide & Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "S14": "Code Interpreter Abuse.",
}

_GUARD_TAG = re.compile(r"<guard>([\s\S]*?)</guard>")


def extract(system_text: str | None) -> tuple[list[str] | None, str | None]:
    """The requested codes, and the system text with the directive removed.

    None when there is no directive -- including an empty `<guard></guard>`, which
    interfaze reads as no guard at all (`match(...)?.[1]?.trim() || undefined`).
    """
    if not system_text:
        return None, system_text
    match = _GUARD_TAG.search(system_text)
    stripped = _GUARD_TAG.sub("", system_text).strip() if match else system_text
    if not match or not match.group(1).strip():
        return None, stripped
    inner = match.group(1).strip()
    if inner.upper() == "ALL":
        codes = list(ALL_CODES)
    else:
        codes = [c.strip().upper() for c in inner.split(",") if c.strip()]
    return codes, stripped


def weighed_average(scores: list[float]) -> float:
    """Each score weighted by its own square, as interfaze averages image scores."""
    if not scores:
        return 0.0
    squares = sum(s * s for s in scores)
    if squares == 0:
        return 0.0
    return sum(s * s * s for s in scores) / squares


def image_result(adult: float, racy: float, gore: float, threshold: float = 0.5) -> dict[str, Any]:
    """One image's classification in interfaze's shape, from adult/racy/gore scores.

    interfaze reads these three off Azure's adult analysis: nudity is adult content,
    gore is gory content, NSFW is any of the three, and the NSFW score is their
    weighted average.
    """
    is_adult, is_racy, is_gore = adult >= threshold, racy >= threshold, gore >= threshold
    return {
        "nsfw": is_adult or is_gore or is_racy,
        "nudity": is_adult,
        "gore": is_gore,
        "nsfw_score": weighed_average([adult, gore, racy]),
        "nudity_score": adult,
        "gore_score": gore,
    }


def wants_images(codes: list[str]) -> dict[str, bool]:
    return {"nudity": "S12_IMAGE" in codes, "gore": "S1_IMAGE" in codes, "nsfw": "S15_IMAGE" in codes}


def verdict(guard_text: str, codes: list[str],
            images: list[dict[str, Any]] | None = None) -> tuple[bool, str, list[dict[str, Any]]]:
    """(is_safe, content if blocked, precontext) exactly as interfaze computes them.

    `guard_text` is the text classifier's raw answer ("safe", or "unsafe\\nS1,S10").
    A verdict naming none of the requested codes is safe; with no codes requested,
    any unsafe verdict blocks.
    """
    is_safe = True
    text = re.sub(r"[\n,]+", " ", guard_text or "").strip()
    warnings = [w.strip() for w in re.split(r"[\n,]", guard_text or "") if w.strip()]

    if text != "safe" and (any(c in warnings for c in codes) or not codes):
        is_safe = False
        matched = ([w for w in warnings if w in codes] if codes
                   else [w for w in warnings if w != "unsafe"])
        text = f"unsafe {' '.join(matched)}".strip()

    matched_warnings = ([w for w in warnings if w in codes] if codes
                        else [w for w in warnings if w != "unsafe"])
    precontext: list[dict[str, Any]] = [
        {"name": "text_guardrail_classifier", "result": "safe" if is_safe else matched_warnings},
    ]

    if images:
        checks = wants_images(codes)
        precontext.append({"name": "image_guardrail_classifier",
                           "result": images[0] if len(images) == 1 else images})
        flagged = [
            code for code, key in (("S1_IMAGE", "gore"), ("S12_IMAGE", "nudity"), ("S15_IMAGE", "nsfw"))
            if checks[key] and any(r.get(key) for r in images)
        ]
        if flagged:
            if is_safe:
                is_safe = False
                text = f"unsafe {' '.join(flagged)}"
            else:
                text = f"{text} {' '.join(flagged)}"
    return is_safe, text, precontext


def extract_from_messages(messages: list[dict]) -> tuple[list[str] | None, list[dict]]:
    """The requested codes from any system message, and the messages with every directive removed.

    Read from every system turn, in string or array form. Read from one string system
    prompt only, a `<guard>` sent as a content part, or in a second system message, was
    forwarded to the model unenforced.
    """
    codes: list[str] | None = None
    out: list[dict] = []

    def take(text: str) -> str:
        nonlocal codes
        found, rest = extract(text)
        if found is not None and codes is None:
            codes = found
        return rest

    for message in messages:
        content = message.get("content")
        if message.get("role") != "system":
            out.append(message)
        elif isinstance(content, str):
            out.append({**message, "content": take(content)})
        elif isinstance(content, list):
            out.append({**message, "content": [
                {**part, "text": take(part["text"])}
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                else part
                for part in content]})
        else:
            out.append(message)
    return codes, out
