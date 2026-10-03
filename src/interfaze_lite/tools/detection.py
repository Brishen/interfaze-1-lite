"""Locating things: objects in photographs, elements in screenshots."""

from __future__ import annotations

import asyncio
import re

from ..grounding import to_pixels as _to_pixels
from . import pages as _pages
from .base import Tool, ToolContext, ToolResult, _post


async def _run_detection(args: dict, ctx: ToolContext) -> ToolResult:
    """Boxes, labels and masks.

    Boxes come from the brain by default rather than a dedicated detector. That is not
    a compromise: the brain is a vision-language model already resident and served by
    vLLM, so grounding costs no extra VRAM and runs at vLLM speed -- the gui_detection
    path proves the mechanism at ~3s. It also sidesteps the transformers-version skew
    that makes the alternate detector's remote code unloadable alongside a 5.x-requiring brain.

    Masks are a second call to the segmenter, which is plain PyTorch and therefore immune to
    that skew.
    """
    url = ctx.refs.resolve(args["file_ref_id"])
    prompts = [p for p in (args.get("prompts") or []) if p and p.strip()]
    if not prompts:
        return ToolResult(model_facing={
            "error": "object_detection needs at least one prompt naming what to find.",
            "message": "Retry the tool call with concrete prompts, e.g. ['person', 'car'].",
        })

    if ctx.ground is None:
        raise RuntimeError("object_detection requires a grounding callable on the context")
    want_masks = bool(args.get("return_masks", False))

    pages = await _pages.pdf_pages(url, ctx, args.get("page_range"))
    if pages is None:
        objects = await _detect_via_brain(url, prompts, ctx, want_masks=want_masks)
        extra: dict = {}
    else:
        # A PDF is searched page by page, all at once; each box is in its page's pixels.
        try:
            found = await asyncio.gather(*(
                _detect_via_brain(page.path, prompts, ctx, want_masks=want_masks)
                for page in pages))
        finally:
            _pages.cleanup(pages)
        objects = [{**o, "page": page.number}
                   for page, page_objects in zip(pages, found, strict=True) for o in page_objects]
        extra = {"pages": [{"page": p.number, "width": p.width, "height": p.height}
                           for p in pages]}

    # Found nothing, the model asked again with near-identical prompts, to the step cap.
    nothing = {} if objects else {"message": (
        "Nothing matching these prompts was found. Do not call object_detection again with "
        "the same or similar prompts; tell the user none were found.")}
    return ToolResult(
        # Masks are megabytes of base64. The model cannot use them and they would blow
        # the context window, so they reach the caller through precontext only.
        model_facing={"detected_objects": [
            # The outline is small enough to read and answers questions a box cannot
            # -- which way something faces, whether two things touch.
            {k: o[k] for k in ("page", "bounds", "label", "polygon") if k in o}
            for o in objects
        ], **extra, **nothing},
        # Same key as the model sees, and the same key the response contract uses.
        # Calling it `objects` here meant a consumer reading `detected_objects` found
        # nothing -- the boxes were present the whole time under another name.
        full={"detected_objects": objects, **extra},
    )
async def _detect_via_brain(url: str, prompts: list[str], ctx: ToolContext, *,
                            want_masks: bool = False) -> list[dict]:
    """Ground with the brain, then put the boxes in pixel space.

    Segmentation is off unless asked for. The masks never reach the model -- they are
    stripped before it sees the result, because megabytes of base64 would swamp the
    context -- so running SAM on every call spent its image encoder on an artefact
    almost no caller reads. The same request still rescales the boxes, which is the
    part that is actually needed and costs nothing.
    """
    grounded = await ctx.ground(url, prompts, domain="object")
    elements = grounded.get("gui_elements") or []
    if not elements:
        return []
    pixel_boxes = _to_pixels(elements, grounded.get("width"), grounded.get("height"))

    # The brain returns a normalised 0-1000 grid, which is resolution independent.
    # Perception rescales and segments in one call: it already has the image, so asking
    # it for dimensions first would mean downloading twice and duplicating the maths.
    boxes_norm = [
        [e["bounds"]["top_left"]["x"], e["bounds"]["top_left"]["y"],
         e["bounds"]["bottom_right"]["x"], e["bounds"]["bottom_right"]["y"]]
        for e in elements
    ]

    masks: list[str] = []
    outlines: list = []
    if want_masks or ctx.settings.detection_outlines:
        try:
            seg = await _post(ctx, ctx.settings.perception_url, "/segment",
                              {"url": url, "boxes": boxes_norm, "normalized": True,
                               "return_masks": want_masks,
                               "return_outlines": ctx.settings.detection_outlines})
            masks = seg.get("masks") or []
            outlines = seg.get("outlines") or []
        except Exception:
            # Outlines and masks are extras; losing them must not cost the detections.
            pass

    objects: list[dict] = []
    for i, element in enumerate(pixel_boxes):
        entry: dict = {"label": element.get("label"), "bounds": element["bounds"]}
        # The outline goes out whenever there is one, the mask only when it was
        # asked for. They describe the same thing at wildly different prices: a few
        # dozen coordinates against a megabyte of base64, and only the coordinates are
        # small enough to put in front of the model.
        if i < len(outlines) and outlines[i]:
            entry["polygon"] = outlines[i]
        if i < len(masks):
            entry["mask"] = masks[i]
        objects.append(entry)
    return objects
OBJECT_DETECTION = Tool(
    name="object_detection",
    description=("Detect specific objects in images using text prompts in non GUI images. "
                 "Returns detected objects with bounding boxes and labels. Works on images "
                 "and on the pages of a PDF, where each object also carries its page number."),
    parameters={
        "type": "object",
        "properties": {
            "file_ref_id": {
                "type": "string",
                "description": "The file reference id or URL of the image or PDF.",
            },
            "page_range": {
                "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                "description": ("PDF only: the 1-based [first, last] pages to search. Omit to "
                                f"search the first {_pages.DEFAULT_PAGES} pages."),
            },
            "return_masks": {
                "type": "boolean", "default": False,
                "description": ("Set true only when the request asks for the raster "
                                "segmentation mask IMAGE of each object (e.g. 'cut out', "
                                "'transparent PNG', 'mask image'). Regardless of this "
                                "flag, each object carries a `polygon` -- the outline of "
                                "its largest region as coordinates -- whenever a contour "
                                "is available; prefer `bounds` when you need the object's "
                                "full extent, since `polygon` covers only the largest "
                                "region. Leave this false unless the caller specifically "
                                "needs the mask bitmap. Defaults to false."),
            },
            "prompts": {
                "type": "array", "items": {"type": "string"},
                "description": ("Objects to detect, e.g. ['person', 'dog', 'car']. This is pure "
                                "grounding -- break complex requests into simple noun phrases. "
                                "When the request names a kind of thing ('vehicles', 'animals'), "
                                "pass that word itself, e.g. ['vehicle']: a list of its members "
                                "misses whatever the list leaves out."),
            },
        },
        "required": ["file_ref_id", "prompts"],
        "additionalProperties": False,
    },
    execute=_run_detection,
)
async def _run_gui(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.ground is None:
        raise RuntimeError("gui_detection requires a grounding callable on the context")
    url = ctx.refs.resolve(args["file_ref_id"])
    prompts = [p for p in (args.get("prompts") or []) if p and p.strip()]

    everything = bool(args.get("all_elements")) or not prompts
    prompts = [_ALL_ELEMENTS] if everything else [_target(p) for p in prompts]

    # As interfaze grounds them: one call per phrase, each box labelled with the phrase
    # that found it (dev_qwen38_gui_main.py, label_as_phrase). Given a question-shaped
    # phrase, the model's own label echoes the instruction back. Every element on screen
    # is interfaze's detector path instead, whose elements are all typed "icon".
    grounded = await asyncio.gather(*(ctx.ground(url, [phrase]) for phrase in prompts))
    elements: list[dict] = []
    for phrase, data in zip(prompts, grounded, strict=True):
        for e in _to_pixels(data.get("gui_elements") or [], data.get("width"), data.get("height")):
            # Interfaze's element is {type, bounds} (helpers/detection/detection.ts).
            elements.append({"type": "icon" if everything else phrase, "bounds": e.get("bounds")})
    return ToolResult(model_facing={"gui_elements": elements})


_ALL_ELEMENTS = "interactive element (buttons, links, inputs, icons)"

# A request's own verb and quantifier, ahead of what it asks for: "Detect all buttons".
_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:detect|find|locate|identify|show(?:\s+me)?|list|highlight|mark|"
    r"get|spot|box|label|point\s+out)\s+(?:all\s+(?:of\s+)?(?:the\s+)?|every\s+|each\s+|the\s+|any\s+)?",
    re.I)


def _target(prompt: str) -> str:
    """What to locate, without the request around it: "Detect all buttons" -> "buttons".

    Grounded as "Locate every Detect all buttons in this UI screenshot", the page's
    buttons came back as 12 boxes; interfaze's tool caller hands its grounding model
    "buttons", and 49 came back. Only the leading verb and quantifier go: the head noun
    is kept as written, since singularising it changes the referent ("check the
    letters") and cost interfaze half its ScreenSpot hits on the prompts it touched.
    """
    target = _REQUEST.sub("", prompt, count=1).strip()
    return target or prompt


GUI_DETECTION = Tool(
    name="gui_detection",
    # interfaze's wording, word for word (tools/index.ts). With a shorter all_elements
    # description the model read "Detect all buttons" as every element on screen and
    # never grounded "buttons".
    description=("Detect all GUI elements on a screen or UI screenshot. Returns all interactive "
                 "elements (buttons, inputs, links, etc.) with bounding boxes and labels. Use this "
                 "for GUI detection, computer use, UI automation, and screen interaction tasks. "
                 "Works with image URLs only."),
    parameters={
        "type": "object",
        "properties": {
            "file_ref_id": {
                "type": "string",
                "description": ("The file reference id or URL of the screenshot/UI image to detect "
                                "GUI elements in. This can only support image urls"),
            },
            "prompts": {
                "type": "array", "items": {"type": "string"},
                "description": ("Pass the user prompt directly here without any edits. The "
                                "grounding model will handle the prompt."),
            },
            "all_elements": {
                "type": "boolean", "default": False,
                "description": ("True only when the user wants every element on the screen and "
                                "named nothing specific ('map this UI', 'what can I click here'). "
                                "Returns all elements with generic labels and skips the grounding "
                                "model entirely. Anything the user does name — even a bare noun "
                                "like 'trash' — means false, with that wording in prompts."),
            },
        },
        "required": ["file_ref_id"],
        "additionalProperties": False,
    },
    execute=_run_gui,
)
