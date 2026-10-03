"""Whether a request asks for a translation or a forecast, read off its instruction.

interfaze decides which tools a request may use with a routing model before the
tool-calling one runs. Lite has no router, and offering these two tools on every text
request cost the benchmark answers it touched: given a question over an English
passage, the model "translated" its own English answer into English, 53 times in the
first 1,400 structured-extraction requests.

A request's instruction sits at its start or its end -- "Translate the following...",
"...in Spanish please", "...what can we expect over the next month?" -- and the
material it works on sits between. Passages mention languages all the time ("known in
English as..."), so only the instruction's end of the message is read: across 5,000
extraction prompts that offered translation to 88, where reading the whole message
offered it to 1,031.
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    from .translation import languages
except ImportError:  # flat, from a checkout
    from translation import languages  # type: ignore

# How much of each end of the message counts as the instruction.
WINDOW = 300

_FORECAST = re.compile(
    r"\b(?:forecast|predict|projection|extrapolat)"
    r"|\b(?:expect|next|coming|upcoming|future)\b.{0,40}"
    r"\b(?:days?|weeks?|months?|quarters?|years?|periods?|steps?|values?|points?)\b",
    re.I | re.S)


@lru_cache(maxsize=1)
def _translation() -> re.Pattern:
    names = sorted({entry["name"] for entry in languages().values()}, key=len, reverse=True)
    names_pattern = "|".join(re.escape(n) for n in names)
    return re.compile(
        rf"\btranslat|\b(?:in|into|to)\s+(?:{names_pattern})\b"
        r"|\bhow (?:do|would|can) (?:you|i|we) say\b",
        re.I)


def instruction(text: str) -> str:
    """The ends of a message, where its instruction is."""
    text = text or ""
    if len(text) <= 2 * WINDOW:
        return text
    return text[:WINDOW] + "\n" + text[-WINDOW:]


def wants_translation(text: str) -> bool:
    return bool(_translation().search(instruction(text)))


def wants_forecast(text: str) -> bool:
    return bool(_FORECAST.search(instruction(text)))
