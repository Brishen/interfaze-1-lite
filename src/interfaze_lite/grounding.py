"""Locating things with the brain: the question it is asked, and how its answer is read.

Shared by the service, which asks the brain over vLLM, and the transformers model, which
runs it in-process. Both ask the same question and read the reply the same way, so a box
found by one is the box the other finds.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:  # the interfaze_lite package, and the HF repo as trust_remote_code imports it
    from .contracts import _bounds_from_dict, iou, to_bounds
except ImportError:  # the files run flat, from a checkout
    from contracts import _bounds_from_dict, iou, to_bounds  # type: ignore

# A box list for a photograph is a few hundred tokens. 8192 did not make good answers
# longer, it gave a degenerate one room to run: one oversized screenshot produced 233
# boxes before stopping.
MAX_TOKENS = 1536
# A screenshot asked for "all buttons" is every one on the page, listed top to bottom at
# ~33 tokens each. At 1536 the list stopped after 46, in the menu bar, before reaching
# the page itself. interfaze's grounding service allows 8192 on this same model; repeats
# are suppressed, so a looping reply is cut down, not returned.
MAX_TOKENS_UI = 8192

# interfaze's GUI grounding instruction, word for word (object-detection-truss,
# dev_qwen38_gui_main.py), where it runs on this same model. Asking instead for each
# element's visible text turned "all buttons" into every piece of text on the page, and
# left the buy box -- Add to cart, Buy Now -- unboxed.
UI_INSTRUCTION = (
    "Locate every {phrase} in this UI screenshot.\n"
    "Respond with ONLY a JSON array and nothing else. Each element must be "
    '{{"bbox_2d": [x1, y1, x2, y2], "label": "<short name of the element>"}}.\n'
    "Coordinates are normalized to a 0-1000 grid, where (0,0) is the top-left corner "
    "and (1000,1000) is the bottom-right corner, regardless of the image's pixel size.\n"
    "Return every distinct match, not just the best one. "
    "If the screenshot contains no such element, respond with exactly []."
)


def max_tokens(domain: str) -> int:
    return MAX_TOKENS_UI if domain == "ui" else MAX_TOKENS


_OBJECT = re.compile(r'\{[^{}]*?"bbox_2d"\s*:\s*\[[^\]]*\][^{}]*?\}', re.S)


def prompt(prompts: list[str], domain: str = "ui") -> str:
    """The grounding question for `prompts` in a screenshot ("ui") or a photograph.

    The domain picks the wording, and it matters more than it looks. Asked to find
    elements "in this UI screenshot", the model pointed at a photograph frequently
    returns [] because nothing in it is a UI element.
    """
    phrase = ", ".join(prompts)
    if domain == "ui":
        return UI_INSTRUCTION.format(phrase=phrase)
    # Ranking helps when the request names one thing and several candidates come back:
    # RefCOCOg went 77.5% -> 81.2% with it. It hurt UI grounding badly -- ScreenSpot-v2
    # fell to 51.2% and empty replies rose from 1.2% to 18.8% -- so screenshots skip it.
    return (
        f"Locate every {phrase} in this image.\n"
        "Label each match with the kind of object it is.\n"
        'Respond with ONLY a JSON array, each element '
        '{"bbox_2d": [x1,y1,x2,y2], "label": "<label>"}.\n'
        "Coordinates are normalized to a 0-1000 grid, (0,0) top-left, "
        "(1000,1000) bottom-right.\n"
        "Order the array so the element that best answers the request "
        "comes first; callers that want one answer take that one.\n"
        "Return every distinct match. If none, respond with exactly []."
    )


def _element(text: str, fallback_label: str) -> dict[str, Any] | None:
    """One `{"bbox_2d": ..., "label": ...}` object from a reply, or None if it is no box."""
    try:
        obj = json.loads(text)
        bounds = to_bounds(obj["bbox_2d"], src="norm1000", img_w=1000, img_h=1000)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    if bounds.width <= 0 or bounds.height <= 0:
        return None
    return {"bounds": bounds.as_dict(), "label": obj.get("label") or fallback_label,
            "coordinate_space": "normalized_1000"}


def parse(raw: str, fallback_label: str, *, looped: bool = False) -> list[dict[str, Any]]:
    """The boxes in a grounding reply, on the 0-1000 grid, repeats removed.

    The grid is resolution-independent: `to_pixels` places the boxes on the image.
    A reply that `looped` also loses every box it repeated: those are the loop.
    """
    elements = [e for m in _OBJECT.finditer(raw or "") if (e := _element(m.group(0), fallback_label))]
    if looped:
        elements = _unrepeated(elements)
    return suppress_duplicates(cut_ladders(elements))


def _unrepeated(elements: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """The boxes a looping reply gave once. What it went round and round is not a find:
    the screenshot demo's 15 boxes across a search bar, over and over."""
    repeated: set[int] = set()
    for i in range(len(elements)):
        j = next((j for j in range(i) if _same_box(elements[i], elements[j], threshold)), None)
        if j is not None:
            repeated.update((i, j))
    return [e for i, e in enumerate(elements) if i not in repeated]


# Boxes a reply may waste -- repeating one it already gave, or falling off the grid --
# before it counts as looping. A healthy list repeats a box now and then; a loop repeats
# them until the token cap, which on a dense screenshot is a minute of nothing.
LOOP_REPEATS = 10


class LoopWatch:
    """Reads a grounding reply as it streams and says when it has started looping.

    The screenshot demo's "buttons" came back as 15 boxes stepped across the search bar,
    then the same boxes again, for 8192 tokens and 68 seconds. Stopping at the tenth
    wasted box gives the minute back.
    """

    def __init__(self, repeats: int = LOOP_REPEATS):
        self.repeats = repeats
        self.raw = ""
        self.wasted = 0
        self._pos = 0
        self._kept: list[dict[str, Any]] = []

    def feed(self, text: str) -> bool:
        """Add the next piece of the reply; True once it is looping."""
        self.raw += text
        for match in _OBJECT.finditer(self.raw, self._pos):
            self._pos = match.end()
            element = _element(match.group(0), "")
            if element is None or any(_same_box(element, k) for k in self._kept):
                self.wasted += 1
            else:
                self._kept.append(element)
        return self.wasted >= self.repeats


def tiles(width: int, height: int, overlap: float = 0.1) -> list[tuple[int, int, int, int]]:
    """A screenshot as four overlapping quarters, (x1, y1, x2, y2) in its pixels.

    What a looping screenshot is grounded in instead. A quarter has a quarter of the
    elements and is seen at up to four times the resolution, and a crowded page is
    where the model loops. The overlap keeps an element on a seam whole in one of them.
    """
    half_w, half_h = width / 2, height / 2
    pad_w, pad_h = width * overlap / 2, height * overlap / 2
    xs = [(0, round(half_w + pad_w)), (round(half_w - pad_w), width)]
    ys = [(0, round(half_h + pad_h)), (round(half_h - pad_h), height)]
    return [(x1, y1, x2, y2) for y1, y2 in ys for x1, x2 in xs]


def from_tile(elements: list[dict[str, Any]], tile: tuple[int, int, int, int],
              width: int, height: int, margin: int = 5) -> list[dict[str, Any]]:
    """Boxes on a tile's 0-1000 grid, moved onto the whole screenshot's grid.

    A box against a seam is dropped: it is an element the crop cut in half, and the
    overlap shows it whole in the neighbouring tile. Kept, the screenshot demo's top-right
    quarter put a column of slivers down its left edge.
    """
    x1, y1, x2, y2 = tile
    tw, th = x2 - x1, y2 - y1

    def on_seam(b):
        return ((x1 > 0 and b["top_left"]["x"] <= margin)
                or (x2 < width and b["bottom_right"]["x"] >= 1000 - margin)
                or (y1 > 0 and b["top_left"]["y"] <= margin)
                or (y2 < height and b["bottom_right"]["y"] >= 1000 - margin))

    def point(p):
        return {"x": round((x1 + p["x"] / 1000 * tw) / width * 1000),
                "y": round((y1 + p["y"] / 1000 * th) / height * 1000)}

    out = []
    for element in elements:
        if on_seam(element["bounds"]):
            continue
        tl, br = point(element["bounds"]["top_left"]), point(element["bounds"]["bottom_right"])
        out.append({**element, "bounds": {
            "top_left": tl, "top_right": {"x": br["x"], "y": tl["y"]},
            "bottom_right": br, "bottom_left": {"x": tl["x"], "y": br["y"]},
            "width": br["x"] - tl["x"], "height": br["y"] - tl["y"]}})
    return out


def cut_ladders(elements: list[dict[str, Any]], run: int = 6) -> list[dict[str, Any]]:
    """Drop a stack of identical boxes each starting exactly where the last one ends.

    The other shape a looping reply takes: not one box repeated, which
    suppress_duplicates catches, but the same box stepped down the page by its own
    height -- 40 boxes of 144x47 at x=1847 on a product page, down a column with
    nothing in it. A real list leaves gaps between its rows; the thumbnails on the same
    page stepped 100 with a height of 94. The first box of the stack is kept.

    "The same box" allows a quarter of jitter, or three grid units on a thin box: along
    a screenshot's dock, 18 boxes 104 to 108 px wide, each starting where the last ended,
    ran across empty wallpaper, and in a street photo 40 "person" boxes 11 to 13 units
    wide ran to the edge. A stack may step any way -- right, left, down or up -- and may
    drift sideways as it goes: that one climbed 2 to 5 units with every step, and
    another ran leftwards, each box ending where the last began.
    """
    def edges(e):
        b = e["bounds"]
        return (b["top_left"]["x"], b["top_left"]["y"], b["bottom_right"]["x"], b["bottom_right"]["y"])

    def alike(u, v):
        return abs(u - v) <= max(3, 0.25 * max(u, v))

    def stacked(a, b):
        ax1, ay1, ax2, ay2 = edges(a)
        bx1, by1, bx2, by2 = edges(b)
        w, h = ax2 - ax1, ay2 - ay1
        if not (alike(w, bx2 - bx1) and alike(h, by2 - by1)):
            return False
        sideways = abs(bx1 - ax2) <= 1 or abs(bx2 - ax1) <= 1
        upright = abs(by1 - ay2) <= 1 or abs(by2 - ay1) <= 1
        return ((sideways and abs(by1 - ay1) <= max(1, 0.15 * h))
                or (upright and abs(bx1 - ax1) <= max(1, 0.15 * w)))

    drop: set[int] = set()
    start = 0
    for i in range(1, len(elements) + 1):
        if i < len(elements) and stacked(elements[i - 1], elements[i]):
            continue
        if i - start >= run:
            drop.update(range(start + 1, i))
        start = i
    return [e for i, e in enumerate(elements) if i not in drop]


def suppress_duplicates(elements: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """Drop boxes that repeat an earlier box of the same label, as detectors do.

    A grounding reply that falls into a repetition loop emits the same box over and
    over -- 44 "sword" boxes, nearly all one region -- until the token cap. The earlier
    box is kept: the reply is ordered best-first.
    """
    kept: list[dict[str, Any]] = []
    for element in elements:
        if not any(_same_box(element, k, threshold) for k in kept):
            kept.append(element)
    return kept


def _same_box(a: dict[str, Any], b: dict[str, Any], threshold: float = 0.85) -> bool:
    """One element given twice: the same label on nearly the same box."""
    return a["label"] == b["label"] and iou(
        _bounds_from_dict(a["bounds"]), _bounds_from_dict(b["bounds"])) >= threshold


def trim_to_content(elements: list[dict[str, Any]], image, tolerance: int = 12,
                    share: float = 0.98) -> list[dict[str, Any]]:
    """Screenshot boxes with their plain-background margins cut off, on the 0-1000 grid.

    The background is the colour just outside a box. A row or column inside it that is
    `share` that colour is margin, trimmed from each side until something is there:
    the screenshot demo's quarters boxed "Add to cart" from the white page beside it,
    480 px too wide. A box only shrinks, so one already on its element is untouched,
    and a box with nothing in it at all -- blank page -- is dropped.

    `share` is 0.98, not lower: at 0.9 an input's thin border no longer stopped the trim,
    and a Quantity dropdown shrank to its label.
    """
    import numpy as np

    pixels = np.asarray(image.convert("RGB"))
    h, w = pixels.shape[:2]
    out = []
    for element in elements:
        b = element["bounds"]
        x1, y1 = max(0, round(b["top_left"]["x"] / 1000 * w)), max(0, round(b["top_left"]["y"] / 1000 * h))
        x2, y2 = min(w, round(b["bottom_right"]["x"] / 1000 * w)), min(h, round(b["bottom_right"]["y"] / 1000 * h))
        background = _ring_colour(pixels, x1, y1, x2, y2, tolerance) if x2 - x1 > 2 and y2 - y1 > 2 else None
        if background is None:
            out.append(element)
            continue
        plain = np.abs(pixels[y1:y2, x1:x2].astype(np.int16) - background).max(axis=2) <= tolerance
        cols = np.flatnonzero(plain.mean(axis=0) < share)
        rows = np.flatnonzero(plain.mean(axis=1) < share)
        if not len(cols) or not len(rows):
            continue
        if (cols[0], rows[0], cols[-1], rows[-1]) == (0, 0, x2 - x1 - 1, y2 - y1 - 1):
            out.append(element)  # nothing to trim: returned as it came, not re-rounded
            continue
        tl = {"x": round((x1 + cols[0]) / w * 1000), "y": round((y1 + rows[0]) / h * 1000)}
        br = {"x": round((x1 + cols[-1] + 1) / w * 1000), "y": round((y1 + rows[-1] + 1) / h * 1000)}
        out.append({**element, "bounds": {
            "top_left": tl, "top_right": {"x": br["x"], "y": tl["y"]},
            "bottom_right": br, "bottom_left": {"x": tl["x"], "y": br["y"]},
            "width": br["x"] - tl["x"], "height": br["y"] - tl["y"]}})
    return out


def _ring_colour(pixels, x1: int, y1: int, x2: int, y2: int, tolerance: int, ring: int = 4):
    """The colour of a thin ring just outside the box, if one colour holds most of it."""
    import numpy as np

    h, w = pixels.shape[:2]
    bands = [pixels[max(0, y1 - ring):y1, x1:x2], pixels[y2:min(h, y2 + ring), x1:x2],
             pixels[y1:y2, max(0, x1 - ring):x1], pixels[y1:y2, x2:min(w, x2 + ring)]]
    around = np.concatenate([band.reshape(-1, 3) for band in bands]).astype(np.int16)
    if not len(around):
        return None
    colour = np.median(around, axis=0)
    held = (np.abs(around - colour).max(axis=1) <= tolerance).mean()
    return colour if held >= 0.6 else None


def to_pixels(elements: list[dict], width: int | None, height: int | None) -> list[dict]:
    """Move boxes from the 0-1000 grid onto a `width` x `height` image.

    Every caller gets pixels: a grid box read as pixels lands in the wrong place with
    nothing to say so -- gui_detection once scored 3.8% on ScreenSpot-v2 instead of
    67.5% for exactly that.
    """
    if not width or not height:
        return elements
    out = []
    for element in elements:
        top_left = element["bounds"]["top_left"]
        bottom_right = element["bounds"]["bottom_right"]
        x1, y1 = round(top_left["x"] / 1000 * width), round(top_left["y"] / 1000 * height)
        x2, y2 = round(bottom_right["x"] / 1000 * width), round(bottom_right["y"] / 1000 * height)
        out.append({k: v for k, v in element.items() if k != "coordinate_space"} | {"bounds": {
            "top_left": {"x": x1, "y": y1}, "top_right": {"x": x2, "y": y1},
            "bottom_right": {"x": x2, "y": y2}, "bottom_left": {"x": x1, "y": y2},
            "width": x2 - x1, "height": y2 - y1,
        }})
    return out
