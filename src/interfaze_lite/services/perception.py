"""Perception sidecar: OCR, open-vocabulary detection, and ASR on one GPU.

Every model here shares a torch build and therefore a process. Diarization does not --
it lives in its own virtualenv behind `services/diarize.py`, because the diarization
stack pins torch versions that cannot co-resolve with vLLM's.

The heavy lifting is delegated to `InterfazeLiteModel` rather than reimplemented, so
this service and the `transformers` model repo cannot drift apart. This file is
transport: HTTP in, interfaze shapes out.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..contracts import InputFetchError

log = logging.getLogger("interfaze.perception")

_model: Any = None
_model_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(_app):
    preload()
    yield


app = FastAPI(title="interfaze-lite perception", docs_url=None, redoc_url=None, lifespan=_lifespan)


def _free_vram() -> None:
    """Release cached CUDA blocks after a failure.

    Perception holds several models and a failed request can leave a large allocation
    cached, so the next request OOMs on memory nothing is actually using. Torch frees
    it on demand within a process but not across the fragmentation this creates.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Set by a request that failed. What it allocated outlives it: FastAPI's threadpool
# keeps the exception in a reference cycle, and through its traceback every frame and
# tensor of the failed call, which refcounting never frees. Measured: a transcription
# that ran out of memory still held 16 GB after it answered; one collection freed it.
# Collecting inside the failed request frees nothing -- the exception is still live.
_failed = threading.Event()


def _collect_failure() -> None:
    """Free what a failed request left allocated, once, before the next request runs."""
    if _failed.is_set():
        _failed.clear()
        gc.collect()
        _free_vram()


def _describe(exc: BaseException) -> str:
    """Always include the exception type. Several exceptions -- StopIteration most
    painfully -- stringify to the empty string, turning a 500 into "ocr failed: "."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def model() -> Any:
    """Instantiate the composite model once, lazily.

    Lazily on purpose: components load on first use, so a deployment that only ever
    receives OCR traffic never pulls the ASR weights.
    """
    global _model
    if _model is None:
        with _model_lock:
            # Re-checked inside the lock. Three requests reaching a fresh container at once
            # each built their own model, each loading its own recogniser on its own thread,
            # while transformers flips the process-wide default dtype for every load. The
            # one kept was part bf16, part float32: "expected scalar type BFloat16 but found
            # Float", then garbage for every transcription after it.
            if _model is None:
                from ..configuration_interfaze_lite import InterfazeLiteConfig
                from ..modeling_interfaze_lite import InterfazeLiteModel

                log.info("initialising composite model (components load on first use)")
                _model = InterfazeLiteModel(InterfazeLiteConfig(lazy_load=True))
    return _model


def preload() -> None:
    """Load the components named in PERCEPTION_PRELOAD before taking any request.

    Loaded on first use, a component came in while other requests were running on the
    same GPU process, and a load changes process-wide torch state as it goes. It also
    made the first request after every deploy pay the load. Unset, everything stays
    lazy: a deployment that only ever reads documents never pulls the speech weights.
    """
    names = [n.strip() for n in os.environ.get("PERCEPTION_PRELOAD", "").split(",") if n.strip()]
    for name in names:
        log.info("preloading %s", name)
        model()._component(name)


class OCRRequest(BaseModel):
    url: str
    return_markdown: bool = False
    page_range: list[int] | None = None


class SegmentRequest(BaseModel):
    url: str
    boxes: list[list[float]] = Field(default_factory=list)
    # Boxes on a 0-1000 grid are rescaled here rather than by the caller. The caller
    # would otherwise have to ask for the image dimensions first, and that extra probe
    # is both a second download and a chance to get the pixel space wrong.
    normalized: bool = False
    # Rescaling boxes is arithmetic; producing masks runs the segmenter's image
    # encoder, which is the entire cost of this endpoint. Callers that only want the
    # boxes in pixel space should not pay for the encoder.
    return_masks: bool = True
    # Outlines need the same encoder pass, so they cost the same; they are separate
    # because what they cost to *return* is nothing alike -- a few dozen coordinates
    # against a megabyte of base64 per object.
    return_outlines: bool = False


class TranscribeRequest(BaseModel):
    url: str
    language: str = "auto"
    word_timestamps: bool = False
    # Words for joining speakers to: decoded, or estimated on a long recording.
    words_for_speakers: bool = False
    # Speaker attribution is *not* done here. This service returns words with
    # timestamps; the orchestrator joins them against the diarization sidecar's turns.
    # Putting the join here would force this process to call the sidecar, coupling two
    # deployments that are deliberately independent.


class ForecastRequest(BaseModel):
    # interfaze's prediction payload: the horizon, and the series as date -> value.
    fh: int
    y: dict[str, float]


CAPABILITY = {
    "asr": "stt",
    "asr_fallback": "stt",
    "line_detector": "text_detection",
    "segmenter": "segmentation",
    "ocr_vlm": "document_reading",
    "diarizer": "diarization",
    "brain": "reasoning",
    "forecaster": "forecasting",
}


@app.get("/health")
def health() -> dict[str, Any]:
    import os

    # Capability names, not component names. `asr_fallback` and `segmenter` say which
    # third-party checkpoints are wired in and in what role, which is not the caller's
    # business and not something we want to promise either -- the model behind a
    # capability is an implementation detail that changes.
    loaded = sorted({CAPABILITY.get(name, name)
                     for name in (_model._cache if _model is not None else ())})
    token_vars = [k for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                              "hf_token", "HUGGINGFACE_HUB_TOKEN")
                  if os.environ.get(k)]
    return {
        "ok": True,
        "initialised": _model is not None,
        "components_loaded": loaded,
        # Names only, never values.
        "hf_token_vars": token_vars,
    }


@app.post("/ocr")
def ocr(req: OCRRequest) -> dict[str, Any]:
    _collect_failure()
    try:
        return model().ocr(req.url, return_markdown=req.return_markdown,
                           page_range=req.page_range)
    except InputFetchError as exc:
        # The caller's file, not this service: 422, so the tool layer can say so.
        raise HTTPException(status_code=422, detail={"input_fetch": str(exc)}) from exc
    except Exception as exc:  # surfaced to the tool layer, which reports it to the model
        _free_vram()
        _failed.set()
        log.exception("ocr failed")
        raise HTTPException(status_code=500, detail=f"ocr failed: {_describe(exc)}") from exc


@app.post("/segment")
def segment(req: SegmentRequest) -> dict[str, Any]:
    """Box-prompted the segmenter masks.

    The boxes come from the vLLM-served brain; this turns them into masks and outlines.
    SAM 2 is plain PyTorch, with no transformers dependency to skew against the brain's.
    """
    _collect_failure()
    try:
        model_ = model()
        img = model_._fit_for_ocr(req.url)
        if not req.boxes:
            # Dimensions are still useful to a caller holding normalised boxes.
            return {"masks": [], "outlines": [], "boxes": [],
                    "width": img.width, "height": img.height}

        if req.normalized:
            boxes = [[x1 / 1000 * img.width, y1 / 1000 * img.height,
                      x2 / 1000 * img.width, y2 / 1000 * img.height]
                     for x1, y1, x2, y2 in req.boxes]
        else:
            boxes = [[float(v) for v in b] for b in req.boxes]

        # Outlines come from the same segmentation pass as the masks, so asking for
        # them costs nothing extra once it has run. They are what a caller and the
        # model can actually use -- a few dozen coordinates rather than a megabyte of
        # base64 -- so the mask itself is returned only when it was asked for.
        masks: list[str] = []
        outlines: list[Any] = []
        if req.return_masks or req.return_outlines:
            masks, outlines = model_._segment_boxes_with_outlines(img, boxes)
            if not req.return_masks:
                masks = []

        return {
            "masks": masks,
            "outlines": outlines,
            # Echoed back in pixel space so the caller does not have to redo the maths.
            "boxes": [[int(round(v)) for v in b] for b in boxes],
            "width": img.width,
            "height": img.height,
        }
    except InputFetchError as exc:
        # The caller's file, not this service: 422, so the tool layer can say so.
        raise HTTPException(status_code=422, detail={"input_fetch": str(exc)}) from exc
    except Exception as exc:
        _free_vram()
        _failed.set()
        log.exception("segment failed")
        raise HTTPException(status_code=500, detail=f"segment failed: {_describe(exc)}") from exc


@app.post("/transcribe")
def transcribe(req: TranscribeRequest) -> dict[str, Any]:
    # the recogniser's word-timestamp pass is the largest transient allocation this service
    # makes, and it runs on a card three processes already share. Returning cached
    # blocks first costs microseconds and removes the case where it OOMs against
    # memory that nothing is actually using.
    _collect_failure()
    _free_vram()
    try:
        # Words only when asked for: they cost the recogniser four times what segments
        # do, and forcing them on every request is what put a transcript's timing at
        # the mercy of word alignment. Segments come back otherwise.
        # The model takes the recogniser one call at a time itself -- per batch, not per
        # recording -- so a long file does not hold up every other transcription.
        result = model().transcribe(
            req.url, by_speaker=False, language=req.language,
            word_timestamps=req.word_timestamps, words_for_speakers=req.words_for_speakers,
        )
    except NotImplementedError as exc:
        # Log it. Returning the message to the caller and not writing it down meant a
        # hard model-loading failure showed up in the logs as a bare "501" with no
        # traceback, and the cause had to be read back off the tool result instead.
        log.exception("transcribe unsupported")
        raise HTTPException(status_code=501, detail=_describe(exc)) from exc
    except InputFetchError as exc:
        # The caller's file, not this service: 422, so the tool layer can say so.
        raise HTTPException(status_code=422, detail={"input_fetch": str(exc)}) from exc
    except Exception as exc:
        _free_vram()
        _failed.set()
        log.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=f"transcribe failed: {_describe(exc)}") from exc
    # Given back now, not at the next transcription: speaker turns are found next, by the
    # diarization sidecar on the same card. After a 95-minute transcription this process
    # still held what it had cached, and diarization ran out of memory on 158 MB.
    _free_vram()
    return result


# One at a time, like transcription: a short GPU pass on a card other models share, and
# serialising it keeps its peak allocation to one request's.
_forecast_lock = threading.Lock()


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict[str, Any]:
    from ..forecasting import DatasetError

    _collect_failure()
    try:
        with _forecast_lock:
            return model().forecast(req.y, req.fh)
    except (DatasetError, ValueError) as exc:
        # The series, not this service: interfaze returns these as the caller's error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _free_vram()
        _failed.set()
        log.exception("forecast failed")
        raise HTTPException(status_code=500, detail=f"forecast failed: {_describe(exc)}") from exc
