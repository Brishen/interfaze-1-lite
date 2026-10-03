"""Config for the Interfaze Lite composite model.

Interfaze Lite is not a single set of weights. It is a router plus five perception
models that together answer one request. This config names the component checkpoints
and the policy for loading them.

Two modes:

- composite (default, `bundle_weights=False`) -- the repo carries only config and
  modeling code; each component is pulled from its own upstream checkpoint on first
  use. Small repo, no weight redistribution, upstream licences stay where they are.
- bundled (`bundle_weights=True`) -- component weights live in this repo under the
  prefixes below. Self-contained and offline-capable, at roughly 45-50 GB and with
  the obligation to carry every upstream licence.
"""

from __future__ import annotations

import os

from transformers import PretrainedConfig

# Component defaults. Each was chosen against a specific constraint; see the model card.
# Capability -> component, resolved from the environment at import time. The mapping
# itself is supplied by the deployment (COMPONENT_<CAPABILITY>), so this file declares
# which capabilities exist without naming what provides them.
_CAPABILITIES = (
    "brain", "segmenter", "ocr_vlm", "asr", "asr_fallback", "diarizer",
    "layout", "line_detector", "line_recognizer", "guard", "forecaster",
)

DEFAULT_COMPONENTS: dict[str, str] = {
    name: os.environ[f"COMPONENT_{name.upper()}"]
    for name in _CAPABILITIES
    if os.environ.get(f"COMPONENT_{name.upper()}")
}


class InterfazeLiteConfig(PretrainedConfig):
    model_type = "interfaze_lite"

    def __init__(
        self,
        components: dict[str, str] | None = None,
        bundle_weights: bool = False,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_context: int = 262_144,
        lazy_load: bool = True,
        detection_score_threshold: float = 0.5,
        detection_mask_threshold: float = 0.5,
        # Fraction of a detector line that must sit inside a layout block to belong to
        # it. Containment, not IoU -- see contracts.contained_fraction.
        ocr_stitch_min_containment: float = 0.5,
        # Layout classification runs as a small detector rather than a structured LLM
        # call. The LLM route costs a full generation per page (~60s measured) to
        # answer a question a purpose-built model answers in milliseconds on CPU, and
        # it has to be told the geometry it is classifying in the first place.
        # "-M" is the balanced checkpoint; "-S" is faster, "-L" more accurate.
        # ~4.2 MP. The previous 1.05 MP was enough for body text and demonstrably not
        # enough for anything finer: a dense arXiv page came in at 896x1152, which
        # leaves subscripts a few pixels tall. On olmOCR-Bench that showed up as 664
        # of 715 failing math tests having the content simply absent from the output
        # rather than mistranscribed -- the same signature as long_tiny_text at 68.6%
        # and old_scans at 44.7%. Prefill and line detection both scale with pixel count, so
        # this is a real cost; it buys detail the recogniser evidently does use.
        ocr_max_pixels: int = 4_194_304,
        # Floor for a page that is mostly small text. A flat budget is wrong in both
        # directions: generous for a sparse scanned letter, starving for a two-column
        # paper. Pages whose detected lines are short in pixels earn the larger budget.
        ocr_dense_max_pixels: int = 8_388_608,
        # An image smaller than this is read at twice its size. A 720x960 phone photo of
        # a receipt read at its own size came back with "EARBUDDS", "DIL" for OIL and
        # 6.68 for 6.58; at 2x it read all ten checked fields right, on both runs, and
        # faster (6.7s against 8.6s). 1.5x fixed eight of the ten.
        ocr_upscale_below_pixels: int = 2_097_152,
        # line detection's detector is the fragile one: above roughly this it aborts in C++
        # with a bare "RuntimeError: std::exception". It only supplies geometry, so it
        # can read a smaller copy and have its boxes scaled up to the VLM's frame.
        line_detector_max_pixels: int = 1_500_000,
        # A line shorter than this (as a fraction of page height) counts as small text.
        ocr_small_line_ratio: float = 0.014,
        # Mean line detection confidence below which a page is re-read at the dense budget.
        ocr_retry_confidence: float = 0.75,
        # Raising ocr_max_pixels without raising this truncates: the budget function
        # asks for 16k tokens on a 4 MP page and gets clamped, so a dense page stops
        # mid-sentence ("Define an equivalence relation $\\sim$ on $I(S_g)^n$ by").
        # More pixels only pays off if there is room to write down what they show.
        ocr_max_new_tokens: int = 8192,
        # Matches what interfaze rasterises PDFs at. Lower loses small type;
        # higher just costs prefill the recogniser cannot use.
        pdf_render_dpi: int = 200,
        # The frame PDF boxes are reported in: interfaze renders pages at scale 2
        # (144 DPI), so an A4 page is 1190x1684 and clients draw on that.
        pdf_frame_dpi: int = 144,
        detect_max_new_tokens: int = 2048,
        # the alternate detector is multi-task and selects the task from question text alone.
        # "mtp" uses its trained-in multi-token prediction head.
        lam_generation_mode: str = "mtp",
        # When set, the document VLM is reached over HTTP at a co-located vLLM server
        # instead of being loaded in-process. Unset falls back to transformers, which
        # is correct but roughly an order of magnitude slower.
        ocr_vlm_url: str | None = None,
        ocr_vlm_served_name: str = "document-reader",
        # "html" for a reader that emits markup carrying data-bbox attributes,
        # "json" for one that emits a list of (bbox, category, text).
        ocr_vlm_format: str = "html",
        # Pages OCR'd at once. Matches the OCR server's --max-num-seqs; higher only
        # deepens its queue, and the line detector is competing for CPU at the same time.
        ocr_page_concurrency: int = 16,
        line_detector_pool_size: int = 4,
        max_pdf_pages: int = 50,
        # What the brain sees when grounding, as the service's settings of the same
        # names: screenshots up to ScreenSpot's ~2 MP untouched, photographs at 1 MP
        # (at 1.48 MP a sword's box landed on the armour's chest).
        ground_max_pixels: int = 4_194_304,
        object_ground_max_pixels: int = 1_048_576,
        asr_languages: list[str] | None = None,
        # Audio longer than this is cut at its quiet points into windows of at most
        # 30 s and decoded in batches, as interfaze's recogniser decodes it. Decoded
        # window after window, 95 minutes took 27 at ~3.5x realtime and outran every
        # deadline; shorter audio keeps the sequential decoder its accuracy was
        # measured on.
        asr_window_after_s: float = 120.0,
        asr_batch_size: int = 16,
        # Speakers are joined to words. Past this length the words' timing is estimated
        # from the segments instead of decoded: decoding it took 6.6 minutes for a
        # 95-minute recording, past the ~350 s a relayed request may wait for an answer.
        asr_exact_words_up_to_s: float = 1800.0,
        **kwargs,
    ):
        # Merge rather than replace, so overriding one component does not silently drop
        # the other five.
        self.components = {**DEFAULT_COMPONENTS, **(components or {})}
        self.asr_window_after_s = asr_window_after_s
        self.asr_batch_size = asr_batch_size
        self.asr_exact_words_up_to_s = asr_exact_words_up_to_s
        self.bundle_weights = bundle_weights
        self.device_map = device_map
        self.max_context = max_context
        # Off means every component loads at __init__. On (default) means a caller who
        # only wants OCR never pulls the 28 GB brain.
        self.lazy_load = lazy_load
        self.detection_score_threshold = detection_score_threshold
        self.detection_mask_threshold = detection_mask_threshold
        self.ocr_stitch_min_containment = ocr_stitch_min_containment
        self.ocr_max_pixels = ocr_max_pixels
        self.ocr_dense_max_pixels = ocr_dense_max_pixels
        self.ocr_upscale_below_pixels = ocr_upscale_below_pixels
        self.line_detector_max_pixels = line_detector_max_pixels
        self.ocr_small_line_ratio = ocr_small_line_ratio
        self.ocr_retry_confidence = ocr_retry_confidence
        self.ocr_max_new_tokens = ocr_max_new_tokens
        self.pdf_render_dpi = pdf_render_dpi
        self.pdf_frame_dpi = pdf_frame_dpi
        self.detect_max_new_tokens = detect_max_new_tokens
        self.lam_generation_mode = lam_generation_mode
        self.ocr_vlm_url = ocr_vlm_url or os.environ.get("OCR_VLM_URL") or None
        self.ocr_vlm_served_name = ocr_vlm_served_name
        self.ocr_vlm_format = ocr_vlm_format
        self.ocr_page_concurrency = ocr_page_concurrency
        # Independent the line detector instances, so page workers stop queueing on one engine.
        # Each carries its own detection + recognition weights, hence a bound.
        # Now the number of detection-runtime worker threads, each keeping its own
        # predictors. It was a queue of shared instances, which could not work: a
        # predictor belongs to the thread that ran it.
        self.line_detector_pool_size = line_detector_pool_size
        self.ground_max_pixels = ground_max_pixels
        self.object_ground_max_pixels = object_ground_max_pixels
        self.max_pdf_pages = max_pdf_pages
        # The 25 languages the fast recogniser TDT v3 covers; anything else routes to the speech recogniser.
        self.asr_languages = asr_languages or [
            "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
            "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
        ]
        super().__init__(torch_dtype=torch_dtype, **kwargs)

    def component(self, name: str) -> str:
        try:
            return self.components[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown component {name!r}; known: {sorted(self.components)}"
            ) from exc
