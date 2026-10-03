"""Normalise every backend's native output into the shapes interfaze clients expect.

Three jobs live here, in increasing order of how much they matter:

1. Geometry conversion. Every model emits boxes differently -- absolute xyxy
   (an alternative document model, DocLayout-YOLO), quad polygons (the line detector), 0-1000 normalised
   (the vision tower grounding), 0-1 normalised (Grounding DINO). Interfaze clients expect
   exactly one shape: a four-corner polygon with derived width/height.

2. Composition. Turning per-line detector output into sections/lines/words, and
   joining a VLM's text to a detector's geometry, are real algorithms rather than
   schema maps.

3. Speaker attribution. Joining ASR word timestamps to diarization turns. The
   correct rule is maximum TOTAL temporal overlap over each word's full span --
   not midpoint distance, and not IoU. See `attribute_speakers`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Literal

from rapidfuzz import fuzz

# Lines join a section when their heights match and their top and bottom edges agree,
# both within this many pixels.
ROW_TOLERANCE_PX = 5.0

BoxFormat = Literal["xyxy", "quad", "norm1000", "norm01", "cxcywh"]


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class Bounds:
    """The one box shape every client consumes.

    Four explicit corners plus derived extents, so a consumer never has to guess
    whether a pair is (x, y) or (width, height), nor recompute the extents itself.
    """

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point
    width: int
    height: int

    def as_dict(self) -> dict[str, object]:
        return {
            "top_left": self.top_left.as_dict(),
            "top_right": self.top_right.as_dict(),
            "bottom_left": self.bottom_left.as_dict(),
            "bottom_right": self.bottom_right.as_dict(),
            "width": self.width,
            "height": self.height,
        }

    @property
    def x1(self) -> int:
        return self.top_left.x

    @property
    def y1(self) -> int:
        return self.top_left.y

    @property
    def x2(self) -> int:
        return self.bottom_right.x

    @property
    def y2(self) -> int:
        return self.bottom_right.y


def _axis_aligned(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Bounds:
    # Sort so a model that emits corners in the wrong order still yields a positive box,
    # then clamp to the image. Rounding happens once, at the end, so width/height always
    # equal the rounded corner difference rather than a separately rounded float.
    lo_x, hi_x = sorted((x1, x2))
    lo_y, hi_y = sorted((y1, y2))
    rx1 = max(0, round(lo_x))
    ry1 = max(0, round(lo_y))
    rx2 = min(img_w, round(hi_x))
    ry2 = min(img_h, round(hi_y))
    return Bounds(
        top_left=Point(rx1, ry1),
        top_right=Point(rx2, ry1),
        bottom_right=Point(rx2, ry2),
        bottom_left=Point(rx1, ry2),
        width=max(0, rx2 - rx1),
        height=max(0, ry2 - ry1),
    )


def to_bounds(raw: Sequence[float] | Sequence[Sequence[float]], *, src: BoxFormat,
              img_w: int, img_h: int) -> Bounds:
    """Convert any backend's box into interfaze `Bounds`.

    `img_w`/`img_h` are required even for absolute formats, because every path clamps
    to the image -- a detector that runs on a padded or downscaled copy can otherwise
    emit coordinates past the edge.

    Raises ValueError on malformed input rather than silently producing a degenerate
    box; a NaN coordinate that survives into a response serialises as `null` and
    breaks the client far from the cause.
    """
    if src == "quad":
        pts = [(float(p[0]), float(p[1])) for p in raw]  # type: ignore[index]
        if len(pts) != 4:
            raise ValueError(f"quad needs 4 points, got {len(pts)}")
        if not all(math.isfinite(v) for pt in pts for v in pt):
            raise ValueError(f"non-finite coordinate in quad: {raw!r}")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # the line detector quads are near-rectangular but rotated by a degree or two. We take the
        # enclosing axis-aligned box, which is what the interfaze contract has always
        # carried -- Azure's polygons were reduced the same way by `getBounds`.
        return _axis_aligned(min(xs), min(ys), max(xs), max(ys), img_w, img_h)

    vals = [float(v) for v in raw]  # type: ignore[arg-type]
    if len(vals) != 4:
        raise ValueError(f"{src} needs 4 values, got {len(vals)}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"non-finite coordinate in {src}: {raw!r}")

    if src == "xyxy":
        x1, y1, x2, y2 = vals
    elif src == "norm1000":
        # the vision tower grounding: a 0-1000 grid, origin top-left. interfaze already does this
        # rescale in detection/modal.ts.
        x1, y1, x2, y2 = (vals[0] / 1000 * img_w, vals[1] / 1000 * img_h,
                          vals[2] / 1000 * img_w, vals[3] / 1000 * img_h)
    elif src == "norm01":
        x1, y1, x2, y2 = (vals[0] * img_w, vals[1] * img_h, vals[2] * img_w, vals[3] * img_h)
    elif src == "cxcywh":
        cx, cy, w, h = vals
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        raise ValueError(f"unknown box format {src!r}")

    return _axis_aligned(x1, y1, x2, y2, img_w, img_h)


# --------------------------------------------------------------------------- OCR


@dataclass
class OCRWord:
    text: str
    bounds: Bounds
    confidence: float

    def as_dict(self) -> dict[str, object]:
        # Two places, as interfaze reports confidence: 0.99, not 0.9973307847976685.
        return {"text": self.text, "bounds": self.bounds.as_dict(), "confidence": round(self.confidence, 2)}


def split_words(text: str, bounds: Bounds, confidence: float) -> list[OCRWord]:
    """A line's words, each with its share of the line's box, as interfaze returns them.

    Interfaze's words join back into exactly the line's text. The line detector's own
    word boxes cannot give that: they segment its recognised text, which drops the
    spaces between English words, while the line text a caller sees is the document
    reader's. So the line's quad is divided along its length in proportion to character
    count -- the words always match the text, and each box is within a few pixels of
    the glyphs on proportional fonts. Top and bottom edges are interpolated separately,
    so a tilted line keeps tilted word boxes.
    """
    tokens = text.split()
    if len(tokens) <= 1:
        return [OCRWord(text=text.strip() or text, bounds=bounds, confidence=confidence)]

    total = sum(len(t) for t in tokens) + len(tokens) - 1
    tl, tr, br, bl = bounds.top_left, bounds.top_right, bounds.bottom_right, bounds.bottom_left

    def along(a: Point, b: Point, f: float) -> Point:
        return Point(round(a.x + (b.x - a.x) * f), round(a.y + (b.y - a.y) * f))

    words: list[OCRWord] = []
    offset = 0
    for token in tokens:
        f0, f1 = offset / total, (offset + len(token)) / total
        top0, top1 = along(tl, tr, f0), along(tl, tr, f1)
        bot0, bot1 = along(bl, br, f0), along(bl, br, f1)
        words.append(OCRWord(text=token, confidence=confidence, bounds=Bounds(
            top_left=top0, top_right=top1, bottom_right=bot1, bottom_left=bot0,
            width=round(((top1.x - top0.x) ** 2 + (top1.y - top0.y) ** 2) ** 0.5),
            height=round(((bot0.x - top0.x) ** 2 + (bot0.y - top0.y) ** 2) ** 0.5),
        )))
        offset += len(token) + 1
    return words


@dataclass
class OCRLine:
    text: str
    bounds: Bounds
    average_confidence: float
    words: list[OCRWord] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "bounds": self.bounds.as_dict(),
            "average_confidence": round(self.average_confidence, 2),
            "words": [w.as_dict() for w in self.words],
        }


@dataclass
class OCRSection:
    text: str
    lines: list[OCRLine] = field(default_factory=list)
    # Which page this section came from, 1-based, and THAT page's pixel frame.
    # Bounds are page-local, and the top-level width/height is a max/sum across pages
    # that matches no single page -- so without these a consumer cannot turn a bound
    # back into a position on a page.
    page: int | None = None
    width: int | None = None
    height: int | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"text": self.text, "lines": [ln.as_dict() for ln in self.lines]}
        if self.page is not None:
            out["page"] = self.page
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        return out


def merge_lines_into_sections(lines: Sequence[OCRLine],
                              tolerance: float = ROW_TOLERANCE_PX) -> list[OCRSection]:
    """Group lines that share a visual row into one section.

    Two lines belong together when their heights
    match within `tolerance` AND every top edge sits within `tolerance` of the group's
    mean top, and likewise for bottoms. That collapses side-by-side columns on the same
    row into a single section, which is what downstream reading-order logic expects.

    Lines are consumed greedily in input order; each unclaimed line seeds a new section.
    """
    remaining = list(lines)
    sections: list[OCRSection] = []

    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        claimed: list[int] = []

        for idx, cand in enumerate(remaining):
            if abs(cand.bounds.height - seed.bounds.height) > tolerance:
                continue
            top_ys = [seed.bounds.top_left.y, seed.bounds.top_right.y,
                      cand.bounds.top_left.y, cand.bounds.top_right.y]
            bottom_ys = [seed.bounds.bottom_left.y, seed.bounds.bottom_right.y,
                         cand.bounds.bottom_left.y, cand.bounds.bottom_right.y]
            avg_top = sum(top_ys) / 4
            avg_bottom = sum(bottom_ys) / 4
            if all(abs(y - avg_top) <= tolerance for y in top_ys) and \
               all(abs(y - avg_bottom) <= tolerance for y in bottom_ys):
                group.append(cand)
                claimed.append(idx)

        for idx in reversed(claimed):
            remaining.pop(idx)

        group.sort(key=lambda ln: ln.bounds.x1)
        sections.append(OCRSection(text=" ".join(ln.text for ln in group), lines=group))

    return sections


def collapse_to_page(sections: Sequence[OCRSection]) -> OCRSection:
    """Flatten many regions into the single section that represents one page.

    interfaze returns exactly one section per page and consumers rely on it twice: they
    read `sections[0]` as the page's text, and they compute the per-page height as
    `result.height / len(sections)` to turn stitched multi-page coordinates back into
    page-local ones. Both break silently if a page is split across several sections --
    the boxes are simply drawn in the wrong place, with no error anywhere.

    Lines keep their own geometry and are ordered top-to-bottom, then left-to-right.
    """
    lines: list[OCRLine] = []
    for section in sections:
        lines.extend(section.lines)
    lines.sort(key=lambda ln: (ln.bounds.y1, ln.bounds.x1))
    return OCRSection(
        text="\n".join(s.text for s in sections if s.text.strip()),
        lines=lines,
    )


@dataclass
class VLMRegion:
    """A document VLM's output: text and structure, box quality varies, no confidence."""

    text: str
    bounds: Bounds
    category: str = ""
    # The reader's own HTML for the block, so a table can be returned as produced.
    markup: str = ""


def iou(a: Bounds, b: Bounds) -> float:
    inter_w = min(a.x2, b.x2) - max(a.x1, b.x1)
    inter_h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def contained_fraction(inner: Bounds, outer: Bounds) -> float:
    """What fraction of `inner` lies inside `outer`.

    This, not IoU, is the right relation for matching a detector line to the VLM region
    that encloses it. A region covering a five-line paragraph has an IoU of roughly 0.2
    against each individual line -- so an IoU threshold matches none of them and the
    paragraph loses all its geometry. Containment is ~1.0 for every one of those lines.
    """
    inner_area = inner.width * inner.height
    if inner_area <= 0:
        return 0.0
    inter_w = min(inner.x2, outer.x2) - max(inner.x1, outer.x1)
    inter_h = min(inner.y2, outer.y2) - max(inner.y1, outer.y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    return (inter_w * inter_h) / inner_area


def _fold(c: str) -> str:
    """`c` lowercased, unless lowercasing would change its length."""
    low = c.lower()
    return low if len(low) == 1 else c


class PageText:
    """The document reader's text for one page, and where each detector line is in it.

    The reader's boxes cannot place anything. On portrait pages its x-coordinates came
    back squeezed toward the left; on a landscape photo of a form they came back
    transposed -- the clinic's letterhead at x=509, y=44-1056 -- so matching lines to
    regions by position dropped 62 of a memo's 98 lines, its handwriting among them.
    Lines are found by what they say instead. Geometry is the detector's alone.
    """

    def __init__(self, regions: Sequence[VLMRegion]):
        self.text = " ".join(" ".join(r.text for r in regions).split())
        self.pos = [i for i, c in enumerate(self.text) if c.isalnum()]
        # One character in, one out: a match's offsets in `flat` index `pos`. 'İ'.lower()
        # is two characters, and each one shifted everything after it.
        self.flat = "".join(_fold(self.text[i]) for i in self.pos)
        self.tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", self.text)]
        self.token_at = [0] * len(self.text)
        for k, (a, b) in enumerate(self.tokens):
            self.token_at[a:b] = [k] * (b - a)

    def read(self, lines: Sequence[OCRLine], accept: float = 88) -> list[tuple[OCRLine, tuple[int, int] | None]]:
        """Each line with the reader's words for it, and where in the text they are.

        The reader wraps nothing and drops no spaces, so its words are the better
        reading wherever a line is found in them. Each stretch of text goes to one line,
        closest match first -- a form says "Dependent 1" once, and three "Dependent 2"
        style near misses must not all take it. A line not found keeps the detector's
        reading: its box is right, and a line with no text is worse than one with the
        recogniser's.
        """
        found = [self._find(ln.text, (), accept) for ln in lines]
        order = sorted(range(len(lines)), key=lambda i: (
            -(found[i][2] if found[i] else -1), -len(lines[i].text)))
        used: list[tuple[int, int]] = []
        out: list[tuple[OCRLine, tuple[int, int] | None]] = [(ln, None) for ln in lines]
        for i in order:
            hit = found[i]
            if hit and any(lo < b and a < hi for a, b in used for lo, hi in [hit[1]]):
                hit = self._find(lines[i].text, used, accept)
            if not hit:
                continue
            text, span, _ = hit
            used.append(span)
            line = lines[i]
            out[i] = (OCRLine(text=text, bounds=line.bounds,
                              average_confidence=line.average_confidence,
                              words=split_words(text, line.bounds, line.average_confidence)), span)
        return out

    def _find(self, line: str, used, accept: float):
        wanted = "".join(c for c in line if c.isalnum())
        if not wanted or not self.flat:
            return None
        need = 100 if len(wanted) <= 4 else accept
        searched = self.flat
        for a, b in used:
            searched = searched[:a] + "\0" * (b - a) + searched[b:]
        for _ in range(4):
            hit = fuzz.partial_ratio_alignment("".join(map(_fold, wanted)), searched, score_cutoff=need)
            if hit is None or hit.dest_end <= hit.dest_start:
                return None
            text = self._words(line.strip(), self.pos[hit.dest_start], self.pos[hit.dest_end - 1] + 1)
            if text is not None:
                got = "".join(c for c in text if c.isalnum())
                score = fuzz.ratio(wanted, got)
                # The same letters in another case is another occurrence of them: the
                # detector reads case off the glyphs.
                if score >= need and not (got.lower() == wanted.lower() and got != wanted):
                    return text, (hit.dest_start, hit.dest_end), score
            searched = (searched[:hit.dest_start] + "\0" * (hit.dest_end - hit.dest_start)
                        + searched[hit.dest_end:])
        return None

    def _words(self, line: str, lo: int, hi: int) -> str | None:
        """The reader's text over [lo, hi), widened to whole words.

        Without that, a price found inside a barcode ("150.02" in 005150025499) is a
        match. A line may begin or end inside a word only where the word broke across
        lines at a hyphen: "de-" and "terministic".
        """
        t, tokens = self.text, self.tokens
        first, last = self.token_at[lo], self.token_at[hi - 1]
        start, end = tokens[first][0], tokens[last][1]
        if lo > start and _alnum(t[start:lo]):
            if not t[lo:tokens[first][1]].replace("-", "").isalpha():
                return None
            start = lo
        if hi < end and _alnum(t[hi:end]):
            if not line.endswith("-"):
                return None
            end = hi
        text = t[start:end]
        if line.endswith("-") and not text.endswith("-"):
            text += "-"
        # Punctuation at either end only where the detector saw some: a bullet, a
        # percent sign. Not the underscores of the blank after "Medication:".
        if line[:1].isalnum():
            text = text.lstrip("".join({c for c in text if not c.isalnum()}))
        elif first > 0 and not _alnum(t[slice(*tokens[first - 1])]) and set(t[slice(*tokens[first - 1])]) & set(line[:3]):
            text = t[slice(*tokens[first - 1])] + " " + text
        if line[-1:].isalnum():
            text = text.rstrip("".join({c for c in text if not c.isalnum()}))
        elif last + 1 < len(tokens) and not _alnum(t[slice(*tokens[last + 1])]) and set(t[slice(*tokens[last + 1])]) & set(line[-3:]):
            text = text + " " + t[slice(*tokens[last + 1])]
        return text.strip() or None

    def between(self, spans: Sequence[tuple[int, int]]) -> str | None:
        """The reader's text from the first span to the last, if they sit together.

        A block's lines, found in the text, give its prose as the reader wrote it --
        hyphenation resolved, nothing wrapped. Only if the stretch is not much longer
        than the lines themselves; otherwise they came from all over the page.
        """
        if not spans:
            return None
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        if hi - lo > 1.3 * sum(b - a for a, b in spans) + 8:
            return None
        first, last = self.token_at[self.pos[lo]], self.token_at[self.pos[hi - 1]]
        return self.text[self.tokens[first][0]:self.tokens[last][1]]


def _alnum(text: str) -> str:
    return "".join(c for c in text if c.isalnum())


def stitch(regions: Sequence[VLMRegion], detector_lines: Sequence[OCRLine]) -> list[OCRSection]:
    """One page: the document reader's text, on the detector's lines.

    Division of labour: the reader owns the page's text -- reading order, tables,
    structure -- and the detector owns every box and confidence. Each detector line
    keeps its box and takes the reader's words for it (`PageText.read`).
    """
    lines = [line for line, _ in PageText(regions).read(detector_lines)]
    lines.sort(key=lambda ln: (ln.bounds.y1, ln.bounds.x1))
    return [OCRSection(text="\n".join(r.text for r in regions if r.text.strip()), lines=lines)]


# --------------------------------------------------------------------- audio


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"text": self.text, "timestamp": [self.start, self.end]}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out


@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float


def attribute_speakers(words: Sequence[Word], turns: Sequence[SpeakerTurn],
                       fill_nearest: bool = False) -> list[Word]:
    """Assign each word a speaker by MAXIMUM TOTAL TEMPORAL OVERLAP.

    For every word, sum the overlap of its full [start, end] span against each
    speaker's turns, and take the speaker with the largest total. This is what
    a common alignment toolkit's `assign_word_speakers` implements, and it is deliberately neither of
    the two things people reach for first:

    - NOT midpoint assignment. A word straddling a turn boundary is attributed by how
      much of it falls on each side, not by where its centre happens to land.
    - NOT IoU. Turn duration must not be in the denominator, or a long turn is
      penalised purely for being long.

    Do not copy the fast recogniser's toolkit's legacy path, which reduces each word to a single anchor and
    scans turns with a pointer that only moves forward; it mis-assigns whenever turns
    interleave.

    `fill_nearest` mirrors that toolkit's opt-in fallback for words overlapping no turn at
    all: the turn whose edge is closest. Interfaze never returns a null speaker, so the
    orchestrator turns it on; with no turns at all every word is SPEAKER_00.
    """
    out: list[Word] = []

    for word in words:
        totals: dict[str, float] = {}
        for turn in turns:
            overlap = min(turn.end, word.end) - max(turn.start, word.start)
            if overlap > 0:
                totals[turn.speaker] = totals.get(turn.speaker, 0.0) + overlap

        speaker: str | None = None
        if totals:
            speaker = max(totals.items(), key=lambda kv: kv[1])[0]
        elif fill_nearest and turns:
            # Distance to the turn's span, not its centre: a long turn just before the
            # word must beat a short one whose midpoint happens to sit nearer.
            mid = (word.start + word.end) / 2
            speaker = min(turns, key=lambda t: max(0.0, t.start - mid, mid - t.end)).speaker
        elif fill_nearest:
            speaker = "SPEAKER_00"

        out.append(Word(text=word.text, start=word.start, end=word.end, speaker=speaker))

    return out


def split_long_chunks(chunks: Sequence[dict], max_s: float = 3.0) -> list[dict]:
    """Transcript chunks cut to at most `max_s` seconds, as interfaze chunks them.

    A longer segment becomes equal-length pieces, its words divided between them by
    count: "The little tales they tell are false" over 0-4.78 s is "The little tales"
    over 0-2.39 s and "they tell are false" over 2.39-4.78 s, exactly as interfaze returns
    it. Whole segments came back about 7 s long.
    """
    out: list[dict] = []
    for chunk in chunks:
        start, end = (list(chunk.get("timestamp") or []) + [None, None])[:2]
        words = (chunk.get("text") or "").split()
        if start is None or end is None or end - start <= max_s or len(words) < 2:
            out.append(chunk)
            continue
        pieces = min(len(words), math.ceil((end - start) / max_s - 1e-9))
        step = (end - start) / pieces
        for i in range(pieces):
            part = words[i * len(words) // pieces:(i + 1) * len(words) // pieces]
            out.append({**chunk, "text": " ".join(part),
                        "timestamp": [round(start + i * step, 2), round(start + (i + 1) * step, 2)]})
    return out


def group_by_speaker(words: Sequence[Word]) -> list[dict[str, object]]:
    """Collapse speaker-attributed words into contiguous per-speaker chunks.

    This is the `by_speaker=true` shape: consecutive words sharing a speaker become one
    chunk carrying the joined text and the span from first word start to last word end.
    """
    chunks: list[dict[str, object]] = []
    for word in words:
        if chunks and chunks[-1]["speaker"] == word.speaker:
            chunks[-1]["text"] = f"{chunks[-1]['text']} {word.text}"
            chunks[-1]["timestamp"] = [chunks[-1]["timestamp"][0], word.end]  # type: ignore[index]
        else:
            chunks.append({
                "speaker": word.speaker,
                "text": word.text,
                "timestamp": [word.start, word.end],
            })
    return chunks


def estimate_words(segments: Sequence[Word]) -> list[Word]:
    """Word timing estimated from segment timing: each segment's span shared among its
    words by their length.

    For joining speakers to a recording too long to decode word timing for. A word near
    a speaker change may land a fraction of a second off; joined segment by segment
    instead, a segment that spans the change would go wholly to one speaker. Text with
    no spaces stays one unit per segment.
    """
    words: list[Word] = []
    for segment in segments:
        tokens = segment.text.split()
        if not tokens:
            continue
        weights = [len(token) + 1 for token in tokens]
        total, span, start = sum(weights), max(0.0, segment.end - segment.start), segment.start
        for token, weight in zip(tokens, weights, strict=True):
            end = start + span * weight / total
            words.append(Word(text=token, start=start, end=end))
            start = end
    return words


def speech_windows(samples, rate: int = 16_000, *, max_s: float = 30.0, min_s: float = 20.0,
                   frame_s: float = 0.03) -> list[tuple[int, int]]:
    """(start, end) sample ranges covering the audio, each at most `max_s` seconds.

    Each cut falls on the quietest frame between `min_s` and `max_s` after the last,
    so a word is not split between windows: fixed-stride chunks stitched back together
    glued words at the seams ("websitesince") and looped through a card number.
    Silence is where the recogniser invents text ("Thank you."), so it is trimmed away.
    """
    import numpy as np

    n = len(samples)
    frame = max(1, int(frame_s * rate))
    count = n // frame
    if count == 0:
        return [(0, n)] if n else []
    rms = np.sqrt(np.mean(np.asarray(samples[: count * frame], dtype=np.float64)
                          .reshape(count, frame) ** 2, axis=1))
    # Loudness averaged over ~0.3 s: the raw minimum is the first silent frame, right
    # against the end of a word; the averaged one lies inside the pause.
    span = max(1, int(0.3 / frame_s))
    smooth = np.convolve(rms, np.ones(span) / span, mode="same")
    longest, shortest = int(max_s * rate), int(min_s * rate)
    windows, start = [], 0
    while n - start > longest:
        lo, hi = (start + shortest) // frame, min(count, (start + longest) // frame)
        cut = (lo + int(np.argmin(smooth[lo:hi]))) * frame
        windows.append((start, cut))
        start = cut
    windows.append((start, n))

    # Each window is trimmed to its speech, give or take 0.2 s, and one with under 0.3 s
    # of it is dropped: the silence after a last sentence came back as "Thank you very
    # much." Timestamps are offset from each window's own start, so trimming moves none.
    floor = max(1e-3, 0.05 * float(np.percentile(rms, 95)))
    pad, minimum = int(0.2 * rate), max(1, int(0.3 / frame_s))
    kept = []
    for s, e in windows:
        first_frame = s // frame
        loud = np.nonzero(rms[first_frame: max(first_frame + 1, e // frame)] >= floor)[0]
        if loud.size < minimum:
            continue
        kept.append((max(s, (first_frame + int(loud[0])) * frame - pad),
                     min(e, (first_frame + int(loud[-1]) + 1) * frame + pad)))
    return kept


# ------------------------------------------------- document-VLM output
UA = "interfaze-lite/0.1 (+https://github.com/InterfazeAI/interfaze-lite)"

BBOX_ATTR_RE = re.compile(
    r'data-bbox\s*=\s*["\']\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)\s*["\']',
    re.I)


# How upstream writes a figure: ![description](<hash>_img.webp).
_FIGURE_LINK = re.compile(r"!\[([^\]]*)\]\([^)\s]*\)")


def _figure_as_text(match: re.Match) -> str:
    description = match.group(1).strip()
    return f"[Image: {description}]" if description else ""


def describes_a_photo(detector_lines: Sequence, markdown: str) -> bool:
    """True when a page has no text: the line detector read none, and all the document
    reader wrote is a description of a picture.

    Asked for the text of a giraffe photo, the reader answered "[Image: A photograph of a
    giraffe and its calf ...]" and that was returned as the text. A page whose reader
    output does not open with a picture -- a handwritten note the detector missed -- keeps it.
    """
    return not detector_lines and markdown.lstrip().startswith("[Image:")


def vlm_markdown(html: str) -> str:
    """Render the document reader's HTML to markdown using upstream's parser when available."""
    try:
        from chandra.output import parse_markdown

        # The link goes, the description stays. The link names a file upstream saves
        # beside its own output and this service never returns: rendered, as the landing
        # renders extracted_text, every figure was a broken image. The description can
        # be the only reading of a logo -- "Walmart logo" was a receipt's only "Walmart".
        return _FIGURE_LINK.sub(_figure_as_text, parse_markdown(html))
    except Exception:
        # Upstream absent or unhappy with this output -- strip tags rather than lose
        # the text entirely.
        return re.sub(r"<[^>]+>", " ", html)


_INLINE_TAG = re.compile(r"</?(?:b|i|u|s|em|strong|span|sup|sub|a|mark|small|code|del|ins)\b[^>]*>", re.I)


def _plain(markup: str) -> str:
    """Markup as text. Inline tags join ("<i>precontext</i>." is "precontext.", not
    "precontext ."); the rest separate; entities decode ("&amp;" was reaching callers)."""
    text = re.sub(r"<[^>]+>", " ", _INLINE_TAG.sub("", markup))
    return " ".join(unescape(text).split())


def parse_vlm_regions(html: str, img) -> list[VLMRegion]:
    """Pull (text, bounds) out of the document reader's data-bbox-annotated HTML.

    Coordinates are on a 0-1000 grid, the convention the document reader inherits from
    its base model. That is checked rather than assumed: if any value exceeds 1000 the
    boxes are already absolute pixels, and treating them as normalised would collapse
    every region into the top-left corner.

    The page's own size says nothing about which of the two it is. Requiring the page to
    be larger than 1000px before reading the grid meant every page smaller than that --
    a 720x960 receipt, a cropped scan -- had its grid coordinates read as pixels, which
    does not fail or drop anything. It silently moves each region: on that receipt a
    block centred on the page landed at (500,500) instead of (360,480).
    """
    matches = list(BBOX_ATTR_RE.finditer(html))
    if not matches:
        return []

    values = [float(g) for m in matches for g in m.groups()]
    peak = max(values)
    if peak <= 1.0:
        src = "norm01"
    elif peak <= 1000.0:
        src = "norm1000"
    else:
        src = "xyxy"

    regions: list[VLMRegion] = []
    for i, match in enumerate(matches):
        try:
            bounds = to_bounds(
                [float(g) for g in match.groups()], src=src,
                img_w=img.width, img_h=img.height)
        except ValueError:
            continue
        if bounds.width <= 0 or bounds.height <= 0:
            continue
        # The match ends inside the opening tag, just past the attribute, so content
        # starts after that tag closes. It ends where the NEXT annotated element's tag
        # begins -- not at that element's attribute, which would drag its opening tag in.
        open_end = html.find(">", match.end())
        if open_end == -1:
            continue
        if i + 1 < len(matches):
            end = html.rfind("<", open_end, matches[i + 1].start())
            if end == -1:
                end = matches[i + 1].start()
        else:
            end = len(html)

        inner = html[open_end + 1:end].strip()
        text = _plain(inner)
        tag = html[html.rfind("<", 0, match.start()):open_end]
        label = re.search(r'data-label\s*=\s*["\']([^"\']*)', tag)
        if text:
            regions.append(VLMRegion(text=text, bounds=bounds,
                                     category=label.group(1) if label else "", markup=inner))
    return regions


# ------------------------------------------------ document ingestion
PDF_MAGIC = b"%PDF-"
# .docx is a zip; "PK" plus the word/ entry distinguishes it from other zips.
ZIP_MAGIC = b"PK\x03\x04"


class InputFetchError(Exception):
    """The caller's file could not be fetched from the address they gave.

    Kept apart from every other failure because the remedy is the caller's, not ours:
    reported as a service outage, a 404 image had the model tell the user object
    detection was down.
    """


def fetch_url(url: str, *, timeout: float, headers: dict | None = None):
    """GET a caller's URL, raising InputFetchError when it cannot be read."""
    import httpx

    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": UA, **(headers or {})})
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise InputFetchError(f"could not fetch {url}: HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise InputFetchError(f"could not fetch {url}: {type(exc).__name__}") from exc
    return resp


def fetch_bytes(source) -> bytes:
    """Read a URL, path or raw bytes into memory."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return fetch_url(source, timeout=120).content
    with open(source, "rb") as fh:
        return fh.read()


def document_kind(source) -> str:
    """Classify by CONTENT, falling back to extension.

    Sniffing first matters because these arrive as `ref-N` temp files whose suffix came
    from a guessed MIME type, and a PDF handed to PIL fails with "cannot identify image
    file" -- an error that says nothing about the real problem.
    """
    if hasattr(source, "convert"):  # already a PIL image
        return "image"
    try:
        head = fetch_bytes(source)[:4096]
    except Exception:
        head = b""

    if head.startswith(PDF_MAGIC):
        return "pdf"
    if head.startswith(ZIP_MAGIC) and b"word/" in head:
        return "docx"

    name = source.lower() if isinstance(source, str) else ""
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(TEXT_SUFFIXES) or _looks_like_text(head):
        return "text"
    return "image"


TEXT_SUFFIXES = (".csv", ".tsv", ".txt", ".md", ".json", ".log", ".yaml", ".yml")


def _looks_like_text(head: bytes) -> bool:
    """Whether these bytes are readable characters rather than an encoded image.

    Extension alone is not enough: uploads arrive as `ref-N` temp files whose suffix
    came from a guessed MIME type, and a CSV that guessed wrong used to fall through to
    the image branch and fail with "cannot identify image file" -- a 500 naming a
    problem the caller did not have. Image and archive formats put non-UTF-8 bytes in
    their first few hundred, so decoding is a reliable discriminator; a NUL rules out
    text outright.
    """
    if not head or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte character can straddle the end of the slice, which is not a
        # reason to call the whole file binary.
        try:
            head[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def backfill_sections(sections: list[dict], *, page: int, width: int, height: int) -> list[dict]:
    """Stamp each section with its page and that page's pixel frame.

    Note what this does NOT do: it does not translate any coordinate. Bounds stay page-local, which is the whole
    reason each section has to carry its own page and frame -- the top-level
    width/height is a max/sum across pages and matches no single page.
    """
    return [{**s, "page": page, "width": width, "height": height} for s in sections]




def rescale_page(page: dict, width: int, height: int) -> dict:
    """One OCR'd page with every box moved into a `width` x `height` frame.

    A page is read at whatever resolution reads it best, but its boxes have to be in
    the frame the caller measures against: the image's own pixels, or a PDF page at
    interfaze's render scale. The two differ whenever a page is resized to be read.
    """
    sx, sy = width / (page.get("width") or width), height / (page.get("height") or height)
    if (sx, sy) == (1.0, 1.0):
        return page

    def box(b: dict) -> dict:
        corners = {k: {"x": round(b[k]["x"] * sx), "y": round(b[k]["y"] * sy)}
                   for k in ("top_left", "top_right", "bottom_right", "bottom_left") if k in b}
        return {**b, **corners, "width": round(b["width"] * sx), "height": round(b["height"] * sy)}

    def line(ln: dict) -> dict:
        return {**ln, "bounds": box(ln["bounds"]),
                "words": [{**w, "bounds": box(w["bounds"])} for w in ln.get("words") or []]}

    sections = [{**s, "lines": [line(ln) for ln in s.get("lines") or []],
                 **({"width": width, "height": height} if "width" in s else {})}
                for s in page.get("sections") or []]
    layout = [{**b, "bounds": box(b["bounds"])} for b in page.get("layout") or []]
    return {**page, "sections": sections, "layout": layout, "width": width, "height": height}


def speaker_tracks(output: Any):
    """Yield (segment, track, label) across the diarization pipeline versions.

    3.x returns an Annotation directly. 4.x / community-1 returns a DiarizeOutput
    dataclass instead, and we deliberately prefer its `exclusive_speaker_diarization`
    view: overlap is collapsed to one speaker per instant, which is precisely the
    non-overlapping timeline that word-level attribution needs. Using the overlapping
    view would let a single word accrue overlap against two speakers at the same
    instant and bias the argmax.
    """
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, attr, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return annotation.itertracks(yield_label=True)
    if hasattr(output, "itertracks"):
        return output.itertracks(yield_label=True)
    raise RuntimeError(
        f"cannot read speaker turns from {type(output).__name__}; "
        f"attributes: {sorted(a for a in dir(output) if not a.startswith('_'))[:20]}"
    )


def pdf_render_dpi(width_pt: float, height_pt: float, *, dpi: int, max_pixels: int,
                   small_pixels: int) -> int:
    """The DPI to rasterise a `width_pt` x `height_pt` PDF page at for reading.

    A fixed DPI is the wrong control in both directions. On an oversized or
    already-rasterised page it lands at 16+ MP -- one page cost 223 seconds -- so the
    page is fitted to the pixel budget. On a small page it lands far under the budget:
    a 4.3 x 2.8 in page of tiny print, its scan 2592x1703, rendered at 864x567 and the
    reader invented sentences. A page that `dpi` leaves under `small_pixels` -- the
    threshold under which images are read at twice their size -- is read at the budget.
    """
    inches = (width_pt / 72.0) * (height_pt / 72.0)
    if inches <= 0:
        return dpi
    budget = (max_pixels / inches) ** 0.5
    target = budget if inches * dpi * dpi < small_pixels else dpi
    return int(max(72, min(target, budget)))


def enhance_scan(img):
    """Normalise contrast on a page that reads like a degraded scan.

    olmOCR-Bench's old_scans set -- typewritten letters from the 1910s -- scored 44.7%
    with a median fuzzy-match ratio of 0.915 against a ~0.966 threshold. That shape
    means the text is nearly right and fails on a handful of characters, which is a
    legibility problem rather than a reading-order or layout one.

    Autocontrast stretches a faded, yellowed page back to full dynamic range. It is
    applied only when the histogram says the page is actually low-contrast, so a clean
    digital PDF -- already black on white -- is returned untouched rather than having
    its antialiasing crushed.
    """
    try:
        from PIL import ImageOps
    except Exception:
        return img
    if img.mode not in ("L", "RGB"):
        return img

    grey = img.convert("L")
    histogram = grey.histogram()
    pixels = sum(histogram)
    if not pixels:
        return img

    # Ink and paper as the 5th and 95th percentile, so a few black specks or a bright
    # margin do not decide it.
    cumulative, low, high = 0, 0, 255
    for level, count in enumerate(histogram):
        cumulative += count
        if low == 0 and cumulative >= pixels * 0.05:
            low = level
        if cumulative >= pixels * 0.95:
            high = level
            break

    if high - low > 200:
        return img  # already full-range; leave it alone
    return ImageOps.autocontrast(img.convert("RGB"), cutoff=1)


def normalise_error(text: str, vendors: tuple[str, ...] = ()) -> str:
    """Rewrite an error so it names the capability that failed, not the component.

    A sidecar traceback quotes whatever library raised it, and that string is handed
    to the model and then to the caller verbatim. The public contract is a set of
    capabilities, so error text is expressed in those terms.

    `vendors` is supplied by the caller from its own configuration; nothing is
    hardcoded here.
    """
    if not text or not vendors:
        return text

    tokens: set[str] = set()
    for vendor in vendors:
        for part in re.split(r"[/_\-\s]+", vendor.lower()):
            if len(part) > 3 and not part.isdigit():
                tokens.add(part)
        tokens.add(vendor.lower())

    cleaned = re.sub(
        r"\b[\w.-]+/[\w.-]+\b",
        lambda m: "<component>" if any(t in m.group(0).lower() for t in tokens)
        else m.group(0),
        text,
    )
    for token in sorted(tokens, key=len, reverse=True):
        cleaned = re.sub(re.escape(token), "<component>", cleaned, flags=re.I)
    return re.sub(r"(<component>[\s./-]*)+", "<component>", cleaned)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def attach_text_to_layout(blocks: list[dict], lines: list[OCRLine],
                          min_containment: float = 0.5,
                          regions: Sequence[VLMRegion] = ()) -> list[dict]:
    """Give every layout block the text that falls inside it.

    The layout detector says where a title or paragraph is and what kind of thing it
    is; the line detector says which lines lie in it -- the two share a frame, so
    containment is reliable. What those lines say comes from the document reader
    (`PageText`): a block's text is the reader's own stretch across its lines, so prose
    reads as written -- hyphenation resolved, the spaces the recogniser drops restored.
    A table block is the reader's table whose words it shares, kept as its HTML. The
    reader's boxes place nothing: they come back squeezed or transposed.

    Lines outside every block are not dropped: a page always has text the layout
    model did not box, and losing it silently would be worse than an unclassified block.
    """
    page = PageText(regions)
    read = page.read(lines) if regions else [(line, None) for line in lines]
    boxes = [_bounds_from_dict(b["bounds"]) for b in blocks]
    tables = [r for r in regions if "<table" in r.markup.lower()]

    def reading_order(indices):
        return sorted(indices, key=lambda i: (lines[i].bounds.y1, lines[i].bounds.x1))

    # Each line to the smallest block holding it. Blocks nest -- a form boxed whole as
    # a table, its sections boxed again inside -- and handed out in list order, the
    # outer box took 88 of a memo's 98 lines and left every section inside it empty.
    owner: dict[int, int] = {}
    for i, line in enumerate(lines):
        holding = [k for k, box in enumerate(boxes)
                   if contained_fraction(line.bounds, box) >= min_containment]
        if holding:
            owner[i] = min(holding, key=lambda k: (boxes[k].width * boxes[k].height, k))
    claimed = set(owner)
    out: list[dict] = []
    for k, block in enumerate(blocks):
        mine = reading_order(i for i, o in owner.items() if o == k)
        spans = [read[i][1] for i in mine if read[i][1]]
        text = ((page.between(spans) if len(spans) >= 0.7 * len(mine) else None)
                or " ".join(read[i][0].text for i in mine))
        if tables and mine and "table" in str(block.get("type", "")):
            said = _words(text)
            best = max(tables, key=lambda r: len(said & _words(r.text)))
            if said and len(said & _words(best.text)) >= 0.5 * len(said):
                text = best.markup
        out.append({**block, "text": text.strip(), "line_count": len(mine)})

    spare = reading_order(i for i in range(len(lines)) if i not in claimed)
    if spare:
        out.append({
            "type": "unclassified",
            "score": 0.0,
            "bounds": _union_of([lines[i].bounds for i in spare]),
            "text": "\n".join(read[i][0].text for i in spare),
            "line_count": len(spare),
        })
    return out


def _union_of(boxes: list[Bounds]) -> dict:
    xs = [b.top_left.x for b in boxes] + [b.bottom_right.x for b in boxes]
    ys = [b.top_left.y for b in boxes] + [b.bottom_right.y for b in boxes]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return Bounds(
        top_left=Point(x1, y1), top_right=Point(x2, y1),
        bottom_right=Point(x2, y2), bottom_left=Point(x1, y2),
        width=x2 - x1, height=y2 - y1,
    ).as_dict()


def _bounds_from_dict(raw: dict) -> Bounds:
    tl, br = raw["top_left"], raw["bottom_right"]
    return Bounds(
        top_left=Point(tl["x"], tl["y"]), top_right=Point(br["x"], tl["y"]),
        bottom_right=Point(br["x"], br["y"]), bottom_left=Point(tl["x"], br["y"]),
        width=br["x"] - tl["x"], height=br["y"] - tl["y"],
    )
