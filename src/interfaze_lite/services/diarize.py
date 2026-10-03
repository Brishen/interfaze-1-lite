"""Diarization sidecar. Runs in its own virtualenv, for a measured reason.

the diarization library currently co-installs with vLLM, but the accuracy leader for this slot
does not: an alternative diarizer hard-pins torch==2.1.1 and numpy==1.26.4 against vLLM's torch, and
`uv pip compile` returns "No solution found". Isolating diarization from the start
means swapping in an alternative diarizer -- roughly six DER points better on AMI-SDM -- is a change
to one Dockerfile stanza rather than a re-architecture.

Returns speaker turns only. No words, no transcript. The orchestrator joins these
against ASR word timestamps using maximum total temporal overlap; see
`contracts.attribute_speakers`.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..contracts import speaker_tracks

log = logging.getLogger("interfaze.diarize")

# Resolved from the deploying environment, like every other capability. No default:
# a wrong identifier here fails at load with an opaque validation error from the hub,
# so an unset one should say that plainly instead of guessing.
DEFAULT_PIPELINE = (os.environ.get("DIARIZER_MODEL")
                    or os.environ.get("COMPONENT_DIARIZER") or "")

app = FastAPI(title="interfaze-lite diarization", docs_url=None, redoc_url=None)

_pipeline: Any = None


def pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        token = (os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if not token:
            # community-1 is gated; an anonymous fetch returns 401. Saying so here is
            # far clearer than the auth error the diarization pipeline would surface.
            raise RuntimeError(
                "the diarization model is a gated repository and HF_TOKEN is unset. "
                "Accept the terms on HuggingFace and set HF_TOKEN."
            )

        if not DEFAULT_PIPELINE:
            raise RuntimeError(
                "no diarization component is configured; set COMPONENT_DIARIZER "
                "in the deployment environment.")

        log.info("loading diarization component")
        # the diarization library 4.x renamed use_auth_token to token and rejects the old name
        # outright, so try the current signature first.
        try:
            loaded = Pipeline.from_pretrained(DEFAULT_PIPELINE, token=token)
        except TypeError:
            loaded = Pipeline.from_pretrained(
                DEFAULT_PIPELINE, use_auth_token=token)
        if loaded is None:
            raise RuntimeError(
                "the diarization model loaded as None. This usually means the token is "
                "valid but the gated repository's terms have not been accepted."
            )
        if torch.cuda.is_available():
            loaded.to(torch.device("cuda"))
        _pipeline = loaded
    return _pipeline


def _to_wav(source: str) -> str | None:
    """Decode to 16 kHz mono WAV, which is what the diarization pipeline wants anyway.

    Compressed formats do not have an exact sample count. MP3 carries encoder delay and
    padding, so decoders legitimately disagree on length, and the diarization pipeline treats the
    disagreement as fatal: "resulted in 439895 samples instead of the expected 441000".
    Decoding to PCM once removes the ambiguity, and 16 kHz mono is the rate the
    pipeline resamples to regardless, so nothing is lost.
    """
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", source,
             "-ac", "1", "-ar", "16000", "-f", "wav", out.name],
            check=True, capture_output=True, timeout=300)
        return out.name
    except Exception as exc:
        log.warning("ffmpeg could not normalise %s (%s); using it as-is", source, exc)
        with contextlib.suppress(OSError):

            os.unlink(out.name)
        return None


@contextmanager
def _local_copy(url: str):
    """Give the diarization pipeline a decoded file on disk.

    the diarization pipeline resolves its input through `Audio.validate_file`, which understands local
    paths, file objects and pre-loaded waveforms -- but not URLs. Handing it an https
    string does not download anything; it fails validation, which surfaced as every
    diarization request 500ing and every transcript coming back with zero speakers.
    """
    if not url.startswith(("http://", "https://")):
        wav = _to_wav(url)
        try:
            yield wav or url
        finally:
            if wav:
                with contextlib.suppress(OSError):

                    os.unlink(wav)
        return

    suffix = pathlib.PurePosixPath(urlparse(url).path).suffix or ".wav"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    # urllib identifies itself as "Python-urllib/3.x", which a good number of CDNs
    # refuse outright -- the audio fixtures here answer it with 403. Nothing about the
    # request is unusual apart from that header, so send a normal browser one.
    request = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
        "Accept": "*/*",
    })
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            shutil.copyfileobj(response, handle)
        handle.close()
        wav = _to_wav(handle.name)
        try:
            yield wav or handle.name
        finally:
            if wav:
                with contextlib.suppress(OSError):

                    os.unlink(wav)
    finally:
        handle.close()
        with contextlib.suppress(OSError):

            os.unlink(handle.name)


class DiarizeRequest(BaseModel):
    url: str
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    # No model name: which checkpoint provides diarization is an implementation
    # detail, and naming it in a public health payload turns it into a promise.
    return {"ok": True, "loaded": _pipeline is not None, "capability": "diarization"}


# One recording at a time through the pipeline, which is not documented as thread-safe.
# The segfaults (exit -11) first put down to concurrent calls were in fact the 2 MiB
# thread stacks the container's unlimited stack limit left: see THREAD_STACK_BYTES in
# modal_app.py. Downloads still overlap -- only the model call is serialised.
_pipeline_lock = threading.Lock()

# Set by a recording that failed. What it allocated outlives it in a reference cycle
# through the exception, as in perception, until the cycle collector runs.
_failed = threading.Event()


def _collect_failure() -> None:
    """Free what a failed recording left allocated, before the next one runs."""
    if not _failed.is_set():
        return
    _failed.clear()
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_BATCH_SIZES = ("segmentation_batch_size", "embedding_batch_size")


def _apply(pipe: Any, path: str, hints: dict) -> Any:
    """The pipeline on one recording, its batches halved while they do not fit.

    It shares the card with the recogniser and with whatever else is running; at its
    default 32 a 95-minute recording ran out of memory on 158 MB. The sizes are put back
    afterwards, for the next recording.
    """
    import torch

    defaults = {name: getattr(pipe, name) for name in _BATCH_SIZES
                if isinstance(getattr(pipe, name, None), int)}
    try:
        while True:
            try:
                return pipe(path, **hints)
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # pyannote restates its out-of-memory error as a MemoryError.
                if not defaults or all(getattr(pipe, name) <= 1 for name in defaults):
                    raise
            # Past the except clause, so the failed attempt's tensors are freed first.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for name in defaults:
                setattr(pipe, name, max(1, getattr(pipe, name) // 2))
            log.warning("diarization did not fit in memory; retrying at %s",
                        {name: getattr(pipe, name) for name in defaults})
    finally:
        for name, size in defaults.items():
            setattr(pipe, name, size)


@app.post("/diarize")
def diarize(req: DiarizeRequest) -> dict[str, Any]:
    """Speaker turns for one recording.

    Long recordings should be chunked upstream. Diarization memory scales with audio
    length rather than parameter count -- NVIDIA's offline Sortformer is only 123M
    parameters yet OOMs at roughly twelve minutes on a 48 GB card -- so a multi-hour
    file will exhaust VRAM regardless of how small the model looks.
    """
    hints = {k: v for k, v in (
        ("num_speakers", req.num_speakers),
        ("min_speakers", req.min_speakers),
        ("max_speakers", req.max_speakers),
    ) if v is not None}

    _collect_failure()
    try:
        with _local_copy(req.url) as path, _pipeline_lock:
            annotation = _apply(pipeline(), path, hints)
    except RuntimeError as exc:
        _failed.set()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        _failed.set()
        log.exception("diarization failed")
        raise HTTPException(status_code=500, detail=f"diarization failed: {exc}") from exc

    turns = [
        {"speaker": str(label), "start": float(seg.start), "end": float(seg.end)}
        for seg, _, label in speaker_tracks(annotation)
    ]
    return {"turns": turns, "num_speakers": len({t["speaker"] for t in turns})}


